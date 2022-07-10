import torch
import torch.nn as nn
from data.cil_data_load import CILDatasetLoader
import time
import os
import copy
import pandas as pd
from utils.calc_score import AverageMeter, ProgressMeter, accuracy
from utils.logger import convert_secs2time
import numpy as np
from utils.onehot import get_one_hot
from implementor.eeil import EEIL
import torch.nn.functional as F


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x):
        return self.alpha * x + self.beta


class BiC(EEIL):
    def __init__(self, model, time_data, save_path, device, configs):
        super().__init__(
            model, time_data, save_path, device, configs)
        self.bias_layers = []

    def bias_forward(self, x, task_num, bias_layer):
        if task_num == 1:
            return x
        else:
            idx = self.task_step
            x_old = x[:, :idx*(task_num-1)]
            x_new = bias_layer(x[:, idx*(task_num-1):])
            return torch.cat((x_old, x_new), dim=1)

    def run(self, dataset_path):
        self.datasetloader = CILDatasetLoader(
            self.configs, dataset_path, self.device)
        train_loader, valid_loader = self.datasetloader.get_settled_dataloader()  # init for once

        ## Hyper Parameter setting ##
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.none_reduction_criterion = nn.CrossEntropyLoss(
            reduction='none').to(self.device)

        ## training ##
        tik = time.time()
        learning_time = AverageMeter('Time', ':6.3f')
        tasks_acc = []
        after_bic_tasks_acc = [0.0]

        # Task Init loader #
        self.model.eval()

        # saving=True
        for task_num in range(1, self.configs['task_size']+1):
            task_tik = time.time()
            if self.configs['task_size'] > 0:
                self.incremental_weight(task_num)
                self.model.train()
                self.model.to(self.device)

            ## training info ##
            optimizer = torch.optim.SGD(self.model.parameters(
            ), self.configs['lr'], self.configs['momentum'], weight_decay=self.configs['weight_decay'], nesterov=self.configs['nesterov'])
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.configs['lr_steps'], self.configs['gamma'])
            
            # Task Init loader #
            self.construct_task_dataset(task_num,valid_loader)
            self.bias_layers.append(BiasLayer().to(self.device))

            ## for BiC split 9:1 ##
            if task_num > 1:
                dataset = self.datasetloader.train_data
                bic_images = []
                bic_labels = []
                train_images = []
                train_labels = []
                for cls_idx in range(0, self.current_num_classes):  # for old class
                    if cls_idx==0:
                        len_cls_data = len(
                            dataset.data[dataset.targets == cls_idx])
                    bic_images.append(dataset.data[dataset.targets == cls_idx][:int(
                        len_cls_data*(1-self.configs['split_ratio']))])
                    bic_labels.append(dataset.targets[dataset.targets == cls_idx][:int(
                        len_cls_data*(1-self.configs['split_ratio']))])
                    train_images.append(dataset.data[dataset.targets == cls_idx][int(
                        len_cls_data*(1-self.configs['split_ratio'])):])
                    train_labels.append(dataset.targets[dataset.targets == cls_idx][int(
                        len_cls_data*(1-self.configs['split_ratio'])):])
                bic_images = np.concatenate(bic_images, axis=0)
                bic_labels = np.concatenate(bic_labels, axis=0)
                self.datasetloader.train_data.data = np.concatenate(
                    train_images, axis=0)
                self.datasetloader.train_data.targets = np.concatenate(
                    train_labels, axis=0)
                print("Train dataset shape:",self.datasetloader.train_data.data.shape,"Bic dataset Shape:",bic_images.shape)
                bic_dataset = self.dataset_class(
                    bic_images, bic_labels, transform=self.datasetloader.test_transform, return_idx=True)
                bic_loader = self.datasetloader.get_dataloader(
                    bic_dataset, True)
            #######################

            ## regular training ##
            task_best_valid_acc = 0
            for epoch in range(1, self.configs['epochs'] + 1):
                epoch_tik = time.time()

                train_info = self._train(
                    train_loader, optimizer, epoch, task_num)
                valid_info = self._eval(valid_loader, epoch, task_num)

                for key in train_info.keys():
                    info_dict = {
                        'train': train_info[key], 'eval': valid_info[key]}
                    self.summaryWriter.add_scalars(key, info_dict, epoch)
                    self.summaryWriter.flush()

                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                learning_time.update(time.time()-epoch_tik)
                lr_scheduler.step()
                if epoch in self.configs['lr_steps']:
                    print('Learning Rate: {:.6e}'.format(
                        lr_scheduler.get_last_lr()[0]))
                if task_best_valid_acc < valid_info['accuracy']:
                    task_best_valid_acc = valid_info['accuracy']
                    print("Task %d best accuracy: %.2f" %
                          (task_num, task_best_valid_acc))
                    model_dict = self.model.module.state_dict()
                    save_dict = {
                        'info': valid_info,
                        'model': model_dict,
                        # 'optim': optimizer_dict,
                    }
                    if task_num>1:
                        save_dict.update(
                            {'task{}_bias_model_{}'.format(task_num, i): self.bias_layers[i].state_dict() for i in range(task_num-1)})

                    torch.save(save_dict, os.path.join(
                        self.save_path, self.time_data, 'best_task{}_main_trained_model.pt'.format(task_num)))
                    print("Save Best Accuracy Model")
                ### epoch end ###
            h, m, s = convert_secs2time(time.time()-task_tik)
            print('Task {} Finished. [Acc] {:.2f} [Running Time] {:2d}h {:2d}m {:2d}s'.format(
                task_num, task_best_valid_acc, h, m, s))
            tasks_acc.append(task_best_valid_acc)
            #####################

            self.update_old_model()
            #######################################
            if task_num > 1:
                print("==== Start Bias Correction ====")
                bic_info = self.train_bias_correction(
                    bic_loader, valid_loader, epoch, task_num)
                after_bic_tasks_acc.append(bic_info['accuracy'])
                ## for BiC split 9:1 and then reassemble ##
                if 'cifar' in self.configs['dataset']:
                    self.datasetloader.train_data.data = np.concatenate(
                        (self.datasetloader.train_data.data, bic_images), axis=0)
                    self.datasetloader.train_data.targets = np.concatenate(
                        (self.datasetloader.train_data.targets, bic_labels), axis=0)
                else:
                    raise NotImplementedError
                print("BiC alpha:",end='')
                for i in range(task_num-1):
                    print("{:.3f} ".format(self.bias_layers[i].alpha.item()),end='')
                print("")
                print("BiC beta:",end='')
                for i in range(task_num-1):
                    print("{:.3f} ".format(self.bias_layers[i].beta.item()),end='')
                print("==== End Bias Correction ====")
            self.current_num_classes += self.task_step
            #######################################

        tok = time.time()
        h, m, s = convert_secs2time(tok-tik)
        print('Total Learning Time: {:2d}h {:2d}m {:2d}s'.format(
            h, m, s))
        str_acc = ' '.join("{:.2f}".format(x) for x in tasks_acc)
        self.logger.info("Task Accs before BiC: {}".format(str_acc))

        str_acc = ' '.join("{:.2f}".format(x) for x in after_bic_tasks_acc)
        self.logger.info("Task Accs after BiC: {}".format(str_acc))

        ############## info save #################
        df_dict = copy.deepcopy(self.configs)
        df_dict.update({'learning_time': learning_time,
                        'time': self.time_data,
                        'valid_loss': self.best_valid_loss,
                        'valid_acc': self.best_valid_accuracy,
                        'train_loss': train_info['loss'],
                        'train_acc': train_info['accuracy'],
                        'tasks_acc': tasks_acc,
                        })
        for key in df_dict.keys():
            if isinstance(df_dict[key], torch.Tensor):
                df_dict[key] = df_dict[key].view(-1).detach().cpu().tolist()
            if type(df_dict[key]) == list:
                df_dict[key] = ','.join(str(e) for e in df_dict[key])
            df_dict[key] = [df_dict[key]]
        df_cat = pd.DataFrame.from_dict(df_dict, dtype=object)
        if os.path.exists('./learning_result.csv'):
            df = pd.read_csv('./learning_result.csv',
                             index_col=0, dtype=object)
            df = pd.merge(df, df_cat, how='outer')
        else:
            df = df_cat
        df.to_csv('./learning_result.csv')
        ###############
        self.logger.info("[Best Valid Accuracy] {:.2f}".format(
            self.best_valid_accuracy))
        ##############

    def _train(self, loader, optimizer, epoch, task_num):

        tik = time.time()
        self.model.train()
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(
            len(loader),
            [batch_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
        i = 0
        end = time.time()
        for images, target, indices in loader:
            # measure data loading time
            images, target = images.to(
                self.device), target.to(self.device)
            # target_reweighted = get_one_hot(target, self.current_num_classes)
            outputs, _ = self.model(images)

            if task_num == 1:
                loss = self.criterion(outputs, target)
            else:  # after the normal learning
                cls_loss = self.criterion(outputs, target)
                with torch.no_grad():
                    score, _ = self.old_model(images)
                    score=self.bias_forward(score, task_num, self.bias_layers[task_num-1])
                kd_loss = torch.zeros(task_num)
                for t in range(task_num-1):
                    # local distillation
                    soft_target = torch.softmax(
                        score[:, self.task_step*t:self.task_step*(t+1)] / self.configs['temperature'], dim=1)
                    output_logits = (
                        outputs[:, self.task_step*t:self.task_step*(t+1)] / self.configs['temperature'])
                    kd_loss[t] = self.configs['lamb'] * \
                        self.onehot_criterion(output_logits, soft_target)
                kd_loss = kd_loss.sum()
                loss = kd_loss+cls_loss

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0]*100.0, images.size(0))
            top5.update(acc5[0]*100.0, images.size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % int(len(loader)//2) == 0:
                progress.display(i)
            i += 1

        tok = time.time()
        self.logger.info('[{:2d}/{:2d} task train] Loss: {:.4f} | top1: {:.4f} | top5: {:.4f} | time: {:.3f}'.format(
            task_num,self.configs['task_size'], losses.avg, top1.avg, top5.avg, tok-tik))
        optimizer.zero_grad(set_to_none=True)
        return {'loss': losses.avg, 'accuracy': top1.avg.item(), 'top5': top5.avg.item()}

    def _eval(self, loader, epoch, task_num, bias_correct=False):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        self.model.eval()
        end = time.time()
        i = 0
        nms_correct = 0
        all_total = 0
        with torch.no_grad():
            for images, target, indices in loader:
                # measure data loading time
                images, target = images.to(
                    self.device), target.long().to(self.device)

                # compute output
                output, feature = self.model(images)
                if bias_correct:
                    output = self.bias_forward(output, task_num,self.bias_layers[task_num-1])

                features = F.normalize(feature[-1])
                if task_num > 1 and not (self.configs['natural_inversion'] or self.configs['generative_inversion']):
                    # (nclasses,1,feature_dim)
                    class_mean_set = np.array(self.class_mean_set)
                    tensor_class_mean_set = torch.from_numpy(class_mean_set).to(
                        self.device).permute(1, 2, 0)  # (1,feature_dim,nclasses)
                    # (batch_size,feature_dim,nclasses)
                    x = features.unsqueeze(2) - tensor_class_mean_set
                    x = torch.norm(x, p=2, dim=1)  # (batch_size,nclasses)
                    x = torch.argmin(x, dim=1)  # (batch_size,)
                    nms_results = x.cpu()
                    # nms_results = torch.stack([nms_results] * images.size(0))
                    nms_correct += (nms_results == target.cpu()).sum()

                all_total += len(target)
                loss = self.criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0]*100.0, images.size(0))
                top5.update(acc5[0]*100.0, images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                i += 1
        if task_num == 1 or (self.configs['natural_inversion'] or self.configs['generative_inversion']):
            self.logger.info('[{:2d}/{:2d} task eval] [{:3d} epoch] Loss: {:.4f} | top1: {:.4f} | top5: {:.4f}'.format(
            task_num,self.configs['task_size'],epoch, losses.avg, top1.avg, top5.avg))
        else:
            self.logger.info('[{:2d}/{:2d} task eval] [{:3d} epoch] Loss: {:.4f} | top1: {:.4f} | top5: {:.4f} | NMS: {:.4f}'.format(
                task_num,self.configs['task_size'], epoch, losses.avg, top1.avg, top5.avg, 100.*nms_correct/all_total))

        return {'loss': losses.avg, 'accuracy': top1.avg.item(), 'top5': top5.avg.item()}

    def train_bias_correction(self, train_loader, valid_loader, epoch, task_num):
        bias_correction_best_acc = 0
        optimizer=torch.optim.SGD(self.bias_layers[task_num-1].parameters(), lr=self.configs['lr'], momentum=self.configs['momentum'], weight_decay=self.configs['weight_decay'])
        lr_scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,self.configs['lr_steps'], gamma=self.configs['gamma'])

        self.model.eval()
        for e in range(1,self.configs['epochs']+1):
            tik = time.time()
            batch_time = AverageMeter('Time', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')
            i = 0
            end = time.time()
            self.bias_layers[task_num-1].train()
            for images, target, indices in train_loader:
                # measure data loading time
                images, target = images.to(
                    self.device), target.to(self.device)
                # target_reweighted = get_one_hot(target, self.current_num_classes)
                with torch.no_grad():
                    outputs, _ = self.model(images)
                outputs = self.bias_forward(
                    outputs, task_num, self.bias_layers[task_num-1])
                loss = self.criterion(
                    outputs, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0]*100.0, images.size(0))
                top5.update(acc5[0]*100.0, images.size(0))
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                i += 1
            lr_scheduler.step()
            tok = time.time()
            self.logger.info('[{:2d}/{:2d} task BiC train] [{:3d} epoch] Loss: {:.4f} | top1: {:.4f} | top5: {:.4f} | time: {:.3f}'.format(
                task_num,self.configs['task_size'], e, losses.avg, top1.avg, top5.avg, tok-tik))

            self.bias_layers[task_num-1].eval()
            valid_info = self._eval(valid_loader, epoch, task_num, bias_correct=True)
            self.logger.info('[{:2d}/{:2d} task BiC valid] [{:3d} epoch] Loss: {:.4f} | top1: {:.4f} | top5: {:.4f}'.format(
                task_num,self.configs['task_size'], e, valid_info['loss'], valid_info['accuracy'], valid_info['top5']))

            if bias_correction_best_acc < valid_info['accuracy']:
                bias_correction_best_acc = valid_info['accuracy']
                bias_correction_best_top5 = valid_info['top5']
                bias_correction_best_loss= valid_info['loss']
                self.logger.info("[Task {:2d} Bias Correction Best Acc] {:.2f}".format
                                    (task_num, bias_correction_best_acc))
                model_dict = self.model.module.state_dict()
                save_dict = {
                    'info': valid_info,
                    'model': model_dict,

                    # 'optim': optimizer_dict,
                }
                save_dict.update(
                    {'task{}_bic_model_{}'.format(task_num, i): self.bias_layers[i].state_dict() for i in range(task_num - 1)})
                torch.save(save_dict, os.path.join(
                    self.save_path, self.time_data, 'best_task{}_bias_corrected_model.pt'.format(task_num)))
                print("Save Best Accuracy Model")

        return {'loss': bias_correction_best_loss, 'accuracy': bias_correction_best_acc, 'top5': bias_correction_best_top5}
