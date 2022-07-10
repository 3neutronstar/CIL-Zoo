import copy
from data.cil_data_load import CILDatasetLoader
from implementor.baseline import Baseline
import torch
import torch.nn as nn
import time
import os
import pandas as pd
from utils.calc_score import AverageMeter, ProgressMeter, accuracy
from utils.logger import convert_secs2time
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from data.custom_dataset import ImageDatasetFromLoc, ImageDatasetFromData
from utils.onehot import get_one_hot


class ICARL(Baseline):
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

        if self.configs['dataset'] in ['tiny-imagenet', 'imagenet']:
            self.dataset_class = ImageDatasetFromLoc
        elif 'cifar' in self.configs['dataset']:
            self.dataset_class = ImageDatasetFromData
        else:
            raise NotImplementedError(
                'Dataset {} not supported'.format(self.configs['dataset']))

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.none_reduction_criterion = nn.CrossEntropyLoss(
            reduction='none').to(self.device)

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
        # Task Init loader #
        self.model.eval()

        # saving=True
        for task_num in range(1, self.configs['task_size']+1):
            task_tik = time.time()
            task_best_valid_acc = 0

            if self.configs['task_size'] > 0:
                self.incremental_weight(task_num)
                self.model.train()
                self.model.to(self.device)

            ## training info ##
            optimizer = torch.optim.SGD(self.model.parameters(
            ), self.configs['lr'], self.configs['momentum'], weight_decay=self.configs['weight_decay'], nesterov=self.configs['nesterov'])
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.configs['lr_steps'], self.configs['gamma'])

            ## before training setupe the dataset ##            
            self.construct_task_dataset(task_num, valid_loader)
            ###################
            
            ## regular training process ##
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
                if task_best_valid_acc < valid_info['accuracy']:
                    task_best_valid_acc = valid_info['accuracy']
                    print("Task %d best accuracy: %.2f" %
                          (task_num, task_best_valid_acc))
                    ## save best model ##
                    if task_num == self.configs['task_size']:
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
                        hour, minute, second = convert_secs2time(
                            learning_time.avg*(self.configs['epochs']-epoch))
                        print("Save Best Accuracy Model [Time Left] {:2d}h {:2d}m {:2d}s".format(
                            hour, minute, second))
                #####################
                lr_scheduler.step()
                if epoch in self.configs['lr_steps']:
                    print('Learning Rate: {:.6e}'.format(
                        lr_scheduler.get_last_lr()[0]))
                #####################
            h, m, s = convert_secs2time(time.time()-task_tik)
            print('Task {} Finished. [Acc] {:.2f} [Running Time] {:2d}h {:2d}m {:2d}s'.format(
                task_num, task_best_valid_acc, h, m, s))
            tasks_acc.append(task_best_valid_acc)
            self.update_old_model()
            self.current_num_classes += self.task_step
            #######################################
        tok = time.time()
        h, m, s = convert_secs2time(tok-tik)
        print('Total Learning Time: {:2d}h {:2d}m {:2d}s'.format(
            h, m, s))
        str_acc = ' '.join("{:.2f}".format(x) for x in tasks_acc)
        self.logger.info("Task Accs: {}".format( str_acc))

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
            target_reweighted = get_one_hot(target, self.current_num_classes)
            outputs, _ = self.model(images)

            if self.old_model == None:
                loss = F.binary_cross_entropy_with_logits(
                    outputs, target_reweighted)
            elif task_num > 1:  # second step
                old_target, _ = self.old_model(images)
                old_target = torch.sigmoid(old_target)
                old_task_size = old_target.shape[1]
                # print(old_task_size,old_target.shape,target.shape)
                target_reweighted[..., :old_task_size] = old_target
                loss = F.binary_cross_entropy_with_logits(
                    outputs, target_reweighted)

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

    def _reduce_exemplar_sets(self, m):
        print("Reducing exemplar sets!")
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            print('\rThe size of class %d examplar: %s' %
                  (index, str(len(self.exemplar_set[index]))), end='')

    def _construct_exemplar_set(self, class_id, m):
        cls_images = self.datasetloader.train_data.get_class_images(
            class_id)
        cls_dataset = self.dataset_class(
            cls_images, transform=self.datasetloader.test_transform)

        cls_dataloader = self.datasetloader.get_dataloader(
            cls_dataset, shuffle=False)
        class_mean, feature_extractor_output = self.compute_class_mean(
            cls_dataloader)
        exemplar = []
        now_class_mean = np.zeros((1, feature_extractor_output.shape[1]))

        for i in range(m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean +
                              feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            exemplar.append(cls_images[index])

        print("The size of exemplar :%s" % (str(len(exemplar))), end='')
        self.exemplar_set.append(exemplar)

    def compute_class_mean(self, cls_dataloader):
        with torch.no_grad():
            feature_extractor_outputs = []

            for datas in cls_dataloader:
                if type(datas) == tuple and len(datas) == 1:
                    images = datas[0]
                elif type(datas) == tuple and len(datas) == 2:
                    images, _ = datas
                else:
                    images = datas
                images = images.to(self.device)
                _, features = self.model(images)
                feature_extractor_outputs.append(
                    F.normalize(features[-1]).cpu())
        feature_extractor_outputs = torch.cat(
            feature_extractor_outputs, dim=0).numpy()
        class_mean = np.mean(feature_extractor_outputs,
                             axis=0, keepdims=True)  # (feature, nclasses)
        return class_mean, feature_extractor_outputs

    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        print("")
        for index in range(len(self.exemplar_set)):
            print("\r Compute the class mean of {:2d}".format(index), end='')
            exemplar = self.exemplar_set[index]

            # why? transform differently #
            exemplar_dataset = self.dataset_class(
                exemplar, transform=self.datasetloader.test_transform)
            exemplar_dataloader = self.datasetloader.get_dataloader(
                exemplar_dataset, False)
            class_mean, _ = self.compute_class_mean(exemplar_dataloader)
            self.class_mean_set.append(class_mean)
        print("")

    def update_old_model(self):
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()

    def incremental_weight(self, task_num):
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

    def fix_feature(self, model, fix=True):
        if fix:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.parameters():
                param.requires_grad = True
        if fix:
            if 'resnet' in self.configs['model']:
                model.module.fc.weight.requires_grad = fix
                model.module.fc.bias.requires_grad = fix
            elif 'densenet' in self.configs['model']:
                model.module.linear.weight.requires_grad = fix
                model.module.linear.bias.requires_grad = fix
        return model

    
    def construct_task_dataset(self,task_num, valid_loader):
        self.model.eval()
        adding_classes_list = [self.task_step *
                                (task_num-1), self.task_step*task_num]
        if task_num > 1:
            self.old_model.eval()
        ## after train- process exemplar set ##

        if task_num >1:
            self.model.eval()
            print('')
            with torch.no_grad():
                m = int(self.configs['memory_size'] /
                        self.current_num_classes)
                self._reduce_exemplar_sets(m)  # exemplar reduce
                # for each class
                for class_id in range(self.task_step*(task_num-2), self.task_step*(task_num-1)):
                    print('\r Construct class %s exemplar set...' %
                            (class_id), end='')
                    self._construct_exemplar_set(class_id, m)

                self.compute_exemplar_class_mean()
                KNN_accuracy = self._eval(
                    valid_loader, 0, task_num)['accuracy']
                self.logger.info(
                    "NMS accuracy: {}".format(str(KNN_accuracy)))


        self.datasetloader.train_data.update(
            adding_classes_list, self.exemplar_set)  # update for class incremental style #
        self.datasetloader.test_data.update(
            adding_classes_list, self.exemplar_set)  # Don't need to update loader
        self.model.train()
