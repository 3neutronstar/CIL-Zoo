from data.custom_dataset import ImageDatasetFromData
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
import torchvision.transforms as transforms
from utils.eeil_aug import data_augmentation_e2e
from utils.onehot import get_one_hot
import copy
from torchvision.utils import save_image
from torchvision.utils import make_grid


class EEIL(ICARL):
    def __init__(self, model, time_data, save_path, device, configs):
        super().__init__(
            model, time_data, save_path, device, configs)

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
        finetune_acc = []

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
            adding_classes_list = [self.task_step *
                                   (task_num-1), self.task_step*task_num]
            self.datasetloader.train_data.update(
                adding_classes_list, self.exemplar_set)  # update for class incremental style #
            self.datasetloader.test_data.update(
                adding_classes_list, self.exemplar_set)  # Don't need to update loader

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
                #####################
            h, m, s = convert_secs2time(time.time()-task_tik)
            print('Task {} Finished. [Acc] {:.2f} [Running Time] {:2d}h {:2d}m {:2d}s'.format(
                task_num, task_best_valid_acc, h, m, s))
            tasks_acc.append(task_best_valid_acc)
            #####################

            # End of regular learning: FineTuning #
            if task_num > 1:
                size_of_bft_exemplar = self.configs['memory_size']//(
                    self.current_num_classes-self.task_step)
                bft_train_dataset = self.datasetloader.train_data.get_bft_data(
                    size_of_bft_exemplar)
                if 'cifar' in self.configs['dataset']:
                    print("Len bft data: {}".format(len(bft_train_dataset[0])))
                    images, labels = data_augmentation_e2e(
                        bft_train_dataset[0], bft_train_dataset[1])
                    bft_dataset = self.dataset_class(
                        images, labels, self.datasetloader.test_transform, return_idx=True)
                    print("After EEIL Len: {}".format(len(bft_dataset)))
                elif self.configs['dataset'] in ['tiny-imagenet', 'imagenet']:
                    bft_dataset = self.dataset_class(
                        bft_train_dataset, self.datasetloader.train_transform, return_idx=True)
                    raise Warning('FineTuning is not supported for {} dataset'.format(
                        self.configs['dataset']))
                else:
                    raise NotImplementedError
                bft_train_loader = self.datasetloader.get_dataloader(
                    bft_dataset, True)
                valid_info = self.balance_fine_tune(
                    bft_train_loader, valid_loader, task_num)
                self.logger.info(
                    "[{} task] Fine-tune accuracy: {:.2f}".format(task_num, valid_info['accuracy']))
            finetune_acc.append(valid_info['accuracy'])
            self.update_old_model()
            #######################################

            ## after train- process exemplar set ##
            self.model.eval()
            print('')
            with torch.no_grad():
                m = int(self.configs['memory_size'] /
                        self.current_num_classes)
                self._reduce_exemplar_sets(m)  # exemplar reduce
                # for each class
                for class_id in range(self.task_step*(task_num-1), self.task_step*(task_num)):
                    print('\r Construct class %s exemplar set...' %
                            (class_id), end='')
                    self._construct_exemplar_set(class_id, m)

                self.compute_exemplar_class_mean()
                KNN_accuracy = self._eval(
                    valid_loader, epoch, task_num)['accuracy']
                self.logger.info(
                    "NMS accuracy: {}".format(str(KNN_accuracy)))

            self.current_num_classes += self.task_step
            #######################################

        tok = time.time()
        h, m, s = convert_secs2time(tok-tik)
        print('Total Learning Time: {:2d}h {:2d}m {:2d}s'.format(
            h, m, s))
        str_acc = ' '.join("{:.2f}".format(x) for x in tasks_acc)
        self.logger.info("Task Accs: {}".format(str_acc))
        str_acc = ' '.join("{:.2f}".format(x) for x in finetune_acc)
        self.logger.info("Finetune Accs: {}".format(str_acc))

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
                        'finetune_acc': finetune_acc
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

    def _train(self, loader, optimizer, epoch, task_num, balance_finetune=False):

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
                with torch.no_grad():
                    score, _ = self.old_model(images)
                if balance_finetune:
                    soft_target = torch.softmax(score[:, self.current_num_classes -
                                                      self.task_step:self.current_num_classes]/self.configs['temperature'], dim=1)
                    output_logits = (outputs[:, self.current_num_classes -
                                             self.task_step:self.current_num_classes]/self.configs['temperature'])
                    # distillation entropy loss
                    kd_loss = self.configs['lamb'] * \
                        self.onehot_criterion(output_logits, soft_target)
                else:
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
            if self.configs['noise_grad']:
                self._noise_grad(self.model.parameters(), epoch)
            # Page 8: "We apply L2-regularization and random noise [21] (with parameters eta = 0.3, gamma = 0.55)
            # on the gradients to minimize overfitting"
            # https://github.com/fmcp/EndToEndIncrementalLearning/blob/master/cnn_train_dag_exemplars.m#L367
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.configs['clip_grad'])
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % int(len(loader)//2) == 0:
                progress.display(i)
            i += 1

        tok = time.time()
        self.logger.info('[train] Loss: {:.4f} | top1: {:.4f} | top5: {:.4f} | time: {:.3f}'.format(
            losses.avg, top1.avg, top5.avg, tok-tik))
        optimizer.zero_grad(set_to_none=True)
        return {'loss': losses.avg, 'accuracy': top1.avg.item(), 'top5': top5.avg.item()}

    def balance_fine_tune(self, train_loader, valid_loader, task_num):
        self.old_model = copy.deepcopy(self.model)
        self.old_model.eval()
        # self.model=self.fix_feature(self.model,True)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(
        ), lr=self.configs['lr']/10.0, momentum=self.configs['momentum'], weight_decay=self.configs['weight_decay'])

        bftepoch = 30
        bft_lr_steps = [10, 20]
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, bft_lr_steps, self.configs['gamma'])
        task_best_valid_acc = 0
        print('==start fine-tuning==')
        for epoch in range(1, bftepoch+1):
            self._train(train_loader, optimizer, epoch,
                        0, balance_finetune=True)
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
            valid_info = self._eval(valid_loader, epoch, 0)
            if task_best_valid_acc < valid_info['accuracy']:
                task_best_valid_acc = valid_info['accuracy']
                ## save best model ##
                task_best_valid_loss = valid_info['loss']
                model_dict = self.model.module.state_dict()
                #optimizer_dict = self.optimizer.state_dict()
                save_dict = {
                    'info': valid_info,
                    'model': model_dict,
                    # 'optim': optimizer_dict,
                }
                torch.save(save_dict, os.path.join(
                    self.save_path, self.time_data, 'best_task{}_finetune_model.pt'.format(task_num)))
                print("Save Best Accuracy Model")
            #####################
        print('Finetune finished')
        # self.model=self.fix_feature(self.model,False)
        return {'loss': task_best_valid_loss, 'accuracy': task_best_valid_acc}

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

    def _noise_grad(self, parameters, iteration, eta=0.3, gamma=0.55):
        """Add noise to the gradients"""
        parameters = list(filter(lambda p: p.grad is not None, parameters))
        variance = eta / ((1 + iteration) ** gamma)
        for p in parameters:
            p.grad.add_(torch.randn(
                p.grad.shape, device=p.grad.device) * variance)
