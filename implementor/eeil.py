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
from utils.onehot_crossentropy import OnehotCrossEntropyLoss

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

        self.onehot_criterion=OnehotCrossEntropyLoss()

    def _train(self, loader, epoch, task_num):

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
        for images, target,indices in loader:
            # measure data loading time
            images, target = images.to(
                self.device), target.to(self.device)
            target_reweighted = get_one_hot(target, self.current_num_classes)
            outputs,_=self.model(images)

            if self.old_model == None:
                loss = self.onehot_criterion(outputs, target_reweighted)
            elif self.current_num_classes>= self.task_step*2: # after two steps
                   
                cls_loss= self.onehot_criterion(outputs, target_reweighted)
                score= self.old_model(images)
                kd_loss=torch.zeros(task_num)
                for t in range(task_num):
                    # local distillation
                    start_KD = (t) * self.configs['task_size']
                    end_KD = (t+1) * self.configs['task_size']

                    soft_target = score[:,start_KD:end_KD] / self.configs['temperature']
                    output_log = outputs[:,start_KD:end_KD] / self.configs['temperature']
                    kd_loss[t] = self.onehot_criterion(output_log, soft_target) * (self.configs['temperature']**2)
                kd_loss=kd_loss.mean()
                loss= kd_loss+cls_loss


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
        nms_correct=0
        all_total=0
        with torch.no_grad():
            for images, target, indices in loader:
                # measure data loading time
                images, target = images.to(
                    self.device), target.long().to(self.device)

                # compute output
                output,feature = self.model(images)

                features = F.normalize(feature[-1])
                if task_num!=1:
                    class_mean_set = np.array(self.class_mean_set) # (nclasses,1,feature_dim)
                    tensor_class_mean_set=torch.from_numpy(class_mean_set).to(self.device).permute(1,2,0) # (1,feature_dim,nclasses)
                    x = features.unsqueeze(2) - tensor_class_mean_set # (batch_size,feature_dim,nclasses)
                    x = torch.norm(x, p=2, dim=1) # (batch_size,nclasses)
                    x = torch.argmin(x, dim=1) # (batch_size,)
                    nms_results=x.cpu()
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
        if task_num==1:
            self.logger.info('[eval] [{:3d} epoch] Loss: {:.4f} | top1: {:.4f} | top5: {:.4f}'.format(epoch,
                losses.avg, top1.avg, top5.avg))
        else:
            self.logger.info('[eval] [{:3d} epoch] Loss: {:.4f} | top1: {:.4f} | top5: {:.4f} | NMS: {:.4f}'.format(epoch,
                losses.avg, top1.avg, top5.avg, 100.*nms_correct/all_total))

        return {'loss': losses.avg, 'accuracy': top1.avg.item(), 'top5': top5.avg.item()}
    