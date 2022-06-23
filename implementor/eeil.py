from data.cil_data_load import CILDatasetLoader
from data.custom_dataset import ImageDataset
from data.data_load import DatasetLoader
from implementor.baseline import Baseline
import torch
import torch.nn as nn
import time
import os
import pandas as pd
from implementor.icarl import ICARL
from utils.calc_score import AverageMeter, ProgressMeter, accuracy
from utils.logger import convert_secs2time
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from utils.onehot import get_one_hot


class EEIL(ICARL):
    def __init__(self, model, time_data, save_path, device, configs):
        super(ICARL, self).__init__(
            model, time_data, save_path, device, configs)
        # class incremental #
        self.old_model = None
        self.current_num_classes = int(
            self.configs['num_classes']//self.configs['task_size'])
        self.task_step = int(
            self.configs['num_classes']//self.configs['task_size'])
        self.exemplar_set = []
        self.class_mean_set = []



    def run(self, dataset_path):
        self.datasetloader = CILDatasetLoader(
            self.configs, dataset_path, self.device)
        train_loader, valid_loader = self.datasetloader.get_dataloader()

        ## Hyper Parameter setting ##
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.none_reduction_criterion = nn.CrossEntropyLoss(
            reduction='none').to(self.device)

        ## training ##
        tik = time.time()
        learning_time = AverageMeter('Time', ':6.3f')
        tasks_acc = []
        for task_num in range(1, self.configs['task_size']+1):
            task_tik = time.time()
            # get incremental train data
            # incremental
            self.model.eval()
            adding_classes_list = [self.task_step *
                                   (task_num-1), self.task_step*(task_num)]
            self.train_loader, self.test_loader = self.datasetloader.get_updated_dataloader(
                adding_classes_list, self.exemplar_set)

            if self.configs['task_size'] > 0:
                if 'resnet' in self.configs['model']:
                    fc = self.model.module.fc
                elif 'densenet' in self.configs['model']:
                    fc = self.model.module.linear
                else:
                    raise ValueError(
                        '{} model not supported'.format(self.configs['model']))
                weight = fc.weight.data
                bias = fc.bias.data
                in_feature = fc.in_features
                out_feature = fc.out_features

                fc = nn.Linear(in_feature, self.current_num_classes, bias=True)
                if task_num > 1:  # next task to update weight
                    if 'resnet' in self.configs['model']:
                        self.model.module.fc = fc
                        self.model.module.fc.weight.data[:out_feature] = weight
                        self.model.module.fc.bias.data[:out_feature] = bias
                    elif 'densenet' in self.configs['model']:
                        self.model.module.linear = fc
                        self.model.module.linear.weight.data[:out_feature] = weight
                        self.model.module.linear.bias.data[:out_feature] = bias
                    else:
                        raise ValueError(
                            '{} model not supported'.format(self.configs['model']))
                else:
                    if 'resnet' in self.configs['model']:
                        self.model.module.fc = fc
                    elif 'densenet' in self.configs['model']:
                        self.model.module.linear = fc
                print('{} num_classes with checksum: '.format(
                    self.current_num_classes), fc)

                self.model.train()
                self.model.to(self.device)

            ## training info ##
            self.optimizer = torch.optim.SGD(self.model.parameters(
            ), self.configs['lr'], self.configs['momentum'], weight_decay=self.configs['weight_decay'], nesterov=self.configs['nesterov'])
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, self.configs['lr_steps'], self.configs['gamma'])
            ###################
            task_best_valid_acc=0
            for epoch in range(1, self.configs['epochs'] + 1):
                epoch_tik = time.time()

                train_info = self._train(train_loader, epoch, task_num)
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
                #####################
            h, m, s = convert_secs2time(time.time()-task_tik)
            print('Task {} Finished. [Acc] {:.2f} [Running Time] {:2d}h {:2d}m {:2d}s'.format(
                task_num, task_best_valid_acc, h, m, s))
            tasks_acc.append(task_best_valid_acc)
            #####################

            ## after train- process exemplar set ##
            self.model.eval()
            ## after train- process exemplar set ##
            if task_num != self.configs['task_size']:
                print('')
                with torch.no_grad():
                    m = int(self.configs['memory_size']/self.current_num_classes)
                    self._reduce_exemplar_sets(m)  # exemplar reduce
                    # for each class
                    for class_id in range(self.task_step*(task_num-1), self.task_step*(task_num)):
                        print('\r Construct class %s exemplar set...' %
                            (class_id), end='')
                        self._construct_exemplar_set(class_id, m)

                    self.current_num_classes += self.task_step
                    self.compute_exemplar_class_mean()
                    KNN_accuracy = self._eval(
                        valid_loader, epoch, task_num)['accuracy']
                    print("NMS accuracy: "+str(KNN_accuracy))
                    filename = 'accuracy_%.2f_KNN_accuracy_%.2f_increment_%d_net.pt' % (
                        valid_info['accuracy'], KNN_accuracy, task_num*self.task_step)
                    torch.save(self.model, os.path.join(
                        self.save_path, self.time_data, filename))
                    self.old_model = torch.load(os.path.join(
                        self.save_path, self.time_data, filename))
                    self.old_model.to(self.device)
                    self.old_model.eval()
            
            #######################################
        # End of regular learning #
        valid_info=self.balance_fine_tune()

        tok = time.time()
        h,m,s=convert_secs2time(tok-tik)
        print('Total Learning Time: {:2d}h {:2d}m {:2d}s'.format(
            h,m,s))
        str_acc=' '.join("{:.2f}".format(x) for x in tasks_acc)
        print("Task Accs:",str_acc)

        ############## info save #################
        import copy

        df_dict = copy.deepcopy(self.configs)
        df_dict.update({'learning_time': learning_time,
                        'time': self.time_data,
                        'valid_loss': self.best_valid_loss,
                        'valid_acc': self.best_valid_accuracy,
                        'train_loss': train_info['loss'],
                        'train_acc': train_info['accuracy'],
                        'tasks_acc': tasks_acc,
                        }
                       )
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

    def _train(self, loader, epoch, task_num, balance_finetune=False):

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
            target_reweighted = get_one_hot(target, self.current_num_classes)
            outputs, _ = self.model(images)

            if task_num == 1:
                loss = self.onehot_criterion(outputs, target_reweighted)
            else:  # after the normal learning
                cls_loss = self.onehot_criterion(outputs, target_reweighted)
                if balance_finetune:
                    soft_target = torch.softmax(score[:, self.current_num_classes -
                                            self.task_step:self.current_num_classes]/self.configs['temperature'],dim=1)
                    output_logits = outputs[:, self.current_num_classes -
                                            self.task_step:self.current_num_classes]/self.configs['temperature']
                    kd_loss = self.onehot_criterion(output_logits,soft_target) # distillation entropy loss
                    # kd_loss = F.binary_cross_entropy_with_logits(output_logits,soft_target) # distillation entropy loss
                else:
                    score, _ = self.old_model(images)
                    kd_loss = torch.zeros(task_num)
                    for t in range(task_num-1):
                        # local distillation
                        soft_target =  torch.softmax(score [:,self.task_step*t:self.task_step*(t+1)] / self.configs['temperature'],dim=1)
                        output_logits = outputs[:,self.task_step*t:self.task_step*(t+1)] / self.configs['temperature']
                        # kd_loss[t] = F.binary_cross_entropy_with_logits(
                        #     output_logits, soft_target) * (self.configs['temperature']**2)
                        kd_loss[t] = self.onehot_criterion(output_logits, soft_target)
                        # kd_loss[t] = F.binary_cross_entropy_with_logits(output_logits, soft_target)
                    kd_loss = kd_loss.mean()
                loss = kd_loss+cls_loss

            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0]*100.0, images.size(0))
            top5.update(acc5[0]*100.0, images.size(0))
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % int(len(loader)//3) == 0:
                progress.display(i)
            i += 1

        tok = time.time()
        self.logger.info('[train] Loss: {:.4f} | top1: {:.4f} | top5: {:.4f} | time: {:.3f}'.format(
            losses.avg, top1.avg, top5.avg, tok-tik))

        return {'loss': losses.avg, 'accuracy': top1.avg.item(), 'top5': top5.avg.item()}

    def _eval(self, loader, epoch, task_num):
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

                features = F.normalize(feature[-1])
                if task_num != 1:
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
        if task_num == 1:
            self.logger.info('[eval] [{:3d} epoch] Loss: {:.4f} | top1: {:.4f} | top5: {:.4f}'.format(epoch,
                                                                                                      losses.avg, top1.avg, top5.avg))
        else:
            self.logger.info('[eval] [{:3d} epoch] Loss: {:.4f} | top1: {:.4f} | top5: {:.4f} | NMS: {:.4f}'.format(epoch,
                                                                                                                    losses.avg, top1.avg, top5.avg, 100.*nms_correct/all_total))

        return {'loss': losses.avg, 'accuracy': top1.avg.item(), 'top5': top5.avg.item()}

    def balance_fine_tune(self):
        self.update_old_model()
        optimizer = torch.optim.SGD(self.model.parameters(
        ), lr=self.configs['lr']/10.0, momentum=self.configs['momentum'], weight_decay=self.configs['weight_decay'])

        # all the classes of exemplar set should be contained in the training set
        train_loader, test_loader = self.datasetloader.get_updated_dataloader(
            (self.current_num_classes-self.task_step,self.current_num_classes), self.exemplar_set)

        bftepoch = int(self.configs['epochs']*3./4.)
        bft_lr_steps= [int(3./4.*a) for a in self.configs['lr_steps']]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, bft_lr_steps, self.configs['gamma'])
        task_best_valid_acc = 0
        print('==start fine-tuning==')
        for epoch in range(bftepoch):
            self._train(train_loader, epoch, 0, balance_finetune=True)
            lr_scheduler.step()
            valid_info=self._eval(test_loader, epoch, 0)
            if task_best_valid_acc < valid_info['accuracy']:
                task_best_valid_acc = valid_info['accuracy']
                ## save best model ##
                self.best_valid_accuracy = valid_info['accuracy']
                self.best_valid_loss = valid_info['loss']
                model_dict = self.model.module.state_dict()
                #optimizer_dict = self.optimizer.state_dict()
                save_dict = {
                    'info': valid_info,
                    'model': model_dict,
                    # 'optim': optimizer_dict,
                }
                torch.save(save_dict, os.path.join(
                    self.save_path, self.time_data, 'best_model.pt'))
                print("Save Best Accuracy Model")
            #####################
        print('Finetune finished')
        return valid_info
