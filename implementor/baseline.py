import os
import time
import torch
import logging
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from data.data_load import DatasetLoader
from utils.calc_score import AverageMeter, ProgressMeter, accuracy
from utils.expected_calibration_error import batch_calibration_stats, ece_bin_metrics
from utils.logger import convert_secs2time, set_logging_defaults
import pandas as pd

from utils.onehot_crossentropy import OnehotCrossEntropyLoss

class Baseline:
    def __init__(self, model, time_data, save_path, device, configs):
        self.time_data = time_data
        self.save_path = save_path
        self.device = device
        self.configs = configs
        model = nn.DataParallel(model).to(configs['device'])
        self.model = model
        ## logger ##
        set_logging_defaults(time_data, save_path)
        self.logger = logging.getLogger('main')
        self.summaryWriter = SummaryWriter(os.path.join(
            self.save_path, time_data))
        self.best_valid_accuracy = 0.0
        self.best_valid_loss=0.0
        ############

        self.onehot_criterion = OnehotCrossEntropyLoss()
        

    def run(self,dataset_path):
        
        ## dataloader ##
        dataset_loader = DatasetLoader(dataset_path,configs=self.configs)
        train_loader, valid_loader = dataset_loader.get_dataloader()
        ################
        
        ## Hyper Parameter setting ##
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.none_reduction_criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

        optimizer = torch.optim.SGD(self.model.parameters(
        ), self.configs['lr'], self.configs['momentum'], weight_decay=self.configs['weight_decay'], nesterov=self.configs['nesterov'])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, self.configs['lr_steps'], self.configs['gamma'])
        model=self.model

        ## training ##
        tik = time.time()
        learning_time=AverageMeter('Time', ':6.3f')

        for epoch in range(1, self.configs['epochs'] + 1):
            epoch_tik=time.time()
            train_info = self._train(train_loader, model,optimizer, epoch)
            valid_info = self._eval(valid_loader, model,optimizer, epoch)

            for key in train_info.keys():
                info_dict = {'train': train_info[key], 'eval': valid_info[key]}
                self.summaryWriter.add_scalars(key, info_dict, epoch)
                self.summaryWriter.flush()

            if self.device == 'cuda':
                torch.cuda.empty_cache()
            learning_time.update(time.time()-epoch_tik)
            ## save best model ##
            if self.best_valid_accuracy < valid_info['accuracy']:
                self.best_valid_accuracy = valid_info['accuracy']
                self.best_valid_loss=valid_info['loss']
                model_dict = self.model.module.state_dict()
                #optimizer_dict = self.optimizer.state_dict()
                save_dict = {
                    'info': valid_info,
                    'model': model_dict,
                    # 'optim': optimizer_dict,
                }
                torch.save(save_dict, os.path.join(
                    self.save_path, self.time_data, 'best_model.pt'))
                hour,minute,second=convert_secs2time(learning_time.avg*(self.configs['epochs']-epoch))
                print("Save Best Accuracy Model [Time Left] {:2d}h {:2d}m {:2d}s".format(hour,minute,second))
            #####################
            lr_scheduler.step()
        tok = time.time()
        print("Learning Time: {:.4f}s".format(tok-tik))
        ##############
        import copy
        
        df_dict=copy.deepcopy(self.configs)
        df_dict.update({'learning_time':tok-tik,
        'time':self.time_data,
        'valid_loss':self.best_valid_loss,
        'valid_acc':self.best_valid_accuracy,
        'train_loss':train_info['loss'],
        'train_acc':train_info['accuracy'],
        }
        )
        for key in df_dict.keys():
            if isinstance(df_dict[key], torch.Tensor):
                df_dict[key]=df_dict[key].view(-1).detach().cpu().tolist()
            if type(df_dict[key])==list:
                df_dict[key]=','.join(str(e) for e in df_dict[key])
            df_dict[key]=[df_dict[key]]
        df_cat=pd.DataFrame.from_dict(df_dict,dtype=object)
        if os.path.exists('./learning_result.csv'):
            df=pd.read_csv('./learning_result.csv',index_col=0,dtype=object)
            
            df=pd.merge(df,df_cat,how='outer')
        else: df=df_cat
        df.to_csv('./learning_result.csv')
        ###############
        self.logger.info("[Best Valid Accuracy] {:.2f}".format(
            self.best_valid_accuracy))
        ##############

    def _train(self, loader, model, optimizer, epoch):
        tik = time.time()
        model.train()
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
        for images, target in loader:
            # measure data loading time
            images, target = images.to(
                self.device), target.to(self.device)

            # compute output
            output = model(images)
            loss = self.criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
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
        self.logger.info('[train] Loss: {:.4f} | top1: {:.4f} | top5: {:.4f} | time: {:.3f}'.format(
            losses.avg, top1.avg, top5.avg, tok-tik))

        return {'loss': losses.avg, 'accuracy': top1.avg.item(), 'top5': top5.avg.item()}

    def _eval(self, loader, model, optimizer, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        self.model.eval()
        end = time.time()
        i = 0
        with torch.no_grad():
            for images, target in loader:
                # measure data loading time
                images, target = images.to(
                    self.device), target.to(self.device)

                # compute output
                output = model(images)
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

        self.logger.info('[eval] [{:3d} epoch] Loss: {:.4f} | top1: {:.4f} | top5: {:.4f}'.format(epoch,
            losses.avg, top1.avg, top5.avg))

        return {'loss': losses.avg, 'accuracy': top1.avg.item(), 'top5': top5.avg.item()}
    