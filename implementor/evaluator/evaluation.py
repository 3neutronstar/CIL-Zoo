import torchvision
import os
import time
import torch
import torch.nn as nn
from utils.calc_score import AverageMeter, accuracy
from data.data_load import DatasetLoader
import numpy as np
import torch
from torch import nn


class Evaluation:
    def __init__(self, model, time_data, dataset_path, save_path, device, configs):
        self.save_path = save_path
        self.device = device
        self.configs = configs
        self.criterion = nn.CrossEntropyLoss()
        self.model = model
        self.dataset_path=dataset_path
        
    def load_model(self, model, name=None):
        if self.configs['file_name'] is not None:
            if name == None:
                model_dict = torch.load(os.path.join(
                    self.save_path, self.configs['file_name'], 'best_model.pt'),map_location=self.device)
            else:
                model_dict = torch.load(os.path.join(
                    self.save_path, self.configs['lpm_file_name'], 'best_lpm_model.pt'.format(name)),map_location=self.device)
            print("Model Performance: ", model_dict['info'])
            model.load_state_dict(model_dict['model'],strict=False)
            # model.load_state_dict(model_dict)
            print("Load Best Model Complete")

    def _eval(self, loader, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        self.model.eval()
        end = time.time()

        ## For Data saving ##
        ####################
        with torch.no_grad():
            for images, targets in loader:
                # measure data loading time
                images, targets = images.to(
                    self.device), targets.to(self.device)

                # compute output
                if self.configs['train_mode'] == 'snapmix':
                    output, _ = self.model(images)
                else:
                    output = self.model(images)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, targets, topk=(1, 5))
                top1.update(acc1[0]*100.0, images.size(0))
                top5.update(acc5[0]*100.0, images.size(0))
                loss = self.criterion(output, targets)
                losses.update(loss, images.size(0))

                # measure elapsed time #
                batch_time.update(time.time() - end)
                end = time.time()
                ########################
        return {'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg}

    def evaluation(self):
        datasetloader = DatasetLoader(configs=self.configs)
        train_loader, loader = datasetloader.get_dataloader()

        self.load_model(self.model)
        self.model.to(self.device)
        eval_dict = self._eval(
            loader, self.configs['epochs'])
        print(eval_dict)
        ############ T-SNE ##################
        eval_dict_additional = self._info_eval(
            loader, self.configs['epochs'])
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import seaborn as sns
        import pandas as pd
        n_components = 2
        model=TSNE(n_components=n_components)
        tsne_data=np.array(model.fit_transform(eval_dict_additional['logits'],eval_dict_additional['gt']))
        # print(tsne_data.shape,eval_dict_additional['gt'].shape)
        tsne_data=np.concatenate([tsne_data,eval_dict_additional['gt'].numpy().reshape(-1,1)],axis=1)
        df=pd.DataFrame(tsne_data,columns=['dfx','dfy','dfgt'])
        df=df[df['dfgt']<=20]
        print(df)
        sns.scatterplot(data=df,x='dfx',y='dfy',hue='dfgt',palette='Paired')
        plt.savefig(os.path.join(self.save_path,self.configs['file_name'],'save_tsne.png'))
        plt.close()

        from utils.expected_calibration_error import batch_calibration_stats,expected_calibration_err,ece_bin_metrics
        bin_count,bin_correct,bin_prob=batch_calibration_stats(eval_dict_additional['logits'],eval_dict_additional['gt'],10)
        ece=expected_calibration_err(bin_count,bin_correct,bin_prob,eval_dict_additional['gt'].size(0))
        print('{} ece: '.format(self.configs['file_name']), ece*100.0)
        prefix=self.configs['file_name']
        metrics=ece_bin_metrics(bin_count,bin_correct,bin_prob,10,prefix)
        conf=[]
        acc=[]
        for bins in range(1,11):
            conf.append(metrics[prefix+'_bin_conf_{:.2f}'.format(bins*0.1)])
            acc.append(metrics[prefix+'_bin_acc_{:.2f}'.format(bins*0.1)])
        df=pd.DataFrame({'conf':conf,'acc':acc})
        df.to_csv(os.path.join(self.save_path,'{}_conf_acc.csv'.format(self.configs['file_name'])))

        ################################################

    def _info_eval(self, loader, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')

        self.model.eval()
        end = time.time()

        ## For Data saving ##
        logits = []
        true_labels = []
        ####################
        with torch.no_grad():
            for images, targets in loader:
                # measure data loading time
                images, targets = images.to(
                    self.device), targets.to(self.device)

                # compute output
                if self.configs['train_mode'] == 'snapmix':
                    output, _ = self.model(images)
                else:
                    output = self.model(images)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, targets, topk=(1, 5))
                top1.update(acc1[0]*100.0, images.size(0))
                top5.update(acc5[0]*100.0, images.size(0))
                loss = self.criterion(output, targets)
                losses.update(loss, images.size(0))
                true_labels.append(targets.detach().clone().cpu())
                logits.append(output.detach().clone().cpu())

                # measure elapsed time #
                batch_time.update(time.time() - end)
                end = time.time()
                ########################
        logits = torch.cat(logits, dim=0)
        true_labels = torch.cat(true_labels, dim=0)
        return {'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg, 'logits':logits, 'gt':true_labels}
