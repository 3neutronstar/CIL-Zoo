import sys
sys.path.append('.')
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from utils.augmentation import *
import os
from torch.utils.data import DataLoader
from utils.calc_score import AverageMeter, accuracy

class DatasetLoader:
    def __init__(self, dataset_path=None, configs=None) -> None:
        self.configs = configs
        self.dataset_path = dataset_path

        ## Normalize Mean & Std ##
        if configs['dataset'] == 'cifar100':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif configs['dataset'] == 'cifar10':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif configs['dataset'] == 'imagenet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif configs['dataset'] == 'mini-imagenet':
            mean = (0.485, 0.456, 0.406) 
            std = (0.229, 0.224, 0.225)
        elif configs['dataset'] == 'tiny-imagenet':
            mean = [0.4802, 0.4481, 0.3975]
            std = [0.2302, 0.2265, 0.2262]
        elif configs['dataset'] in ['dogs120','cub200','cars196','caltech101','caltech256','flowers102','aircraft100','food101','caltech101','caltech256']:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        configs['mean'] = torch.tensor(
            mean, dtype=torch.float32).reshape(1, 3, 1, 1).to(self.configs['device'])
        configs['std'] = torch.tensor(
            std, dtype=torch.float32).reshape(1, 3, 1, 1).to(self.configs['device'])
        ##########################

        if configs['dataset'] == 'cifar100':
            normalize = transforms.Normalize(mean=mean, std=std)
            self.train_transform = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop((32, 32)),
                transforms.RandomHorizontalFlip(),
                # transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
            self.test_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
            if self.configs['train_mode']=='eeil':
                self.train_transform = transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop((32, 32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
        elif configs['dataset'] == 'cifar10':
            normalize = transforms.Normalize(
                mean=mean, std=std)
            self.train_transform = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop((32, 32)),
                transforms.RandomHorizontalFlip(),
                # transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
            self.test_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
        elif configs['dataset'] == 'imagenet':
            '''
            https://github.com/pytorch/examples/blob/master/imagenet/main.py
            '''
            normalize = transforms.Normalize(mean=mean,
                                                std=std)

            jittering = ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.4)
            lighting = Lighting(alphastd=0.1,
                                eigval=[0.2175, 0.0188, 0.0045],
                                eigvec=[[-0.5675, 0.7192, 0.4009],
                                        [-0.5808, -0.0045, -0.8140],
                                        [-0.5836, -0.6948, 0.4203]])
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ])
            self.test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        elif configs['dataset'] == 'imagenet':
            '''
            https://github.com/pytorch/examples/blob/master/imagenet/main.py
            '''
            normalize = transforms.Normalize(mean=mean,
                                             std=std)

            jittering = ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.4)
            lighting = Lighting(alphastd=0.1,
                                eigval=[0.2175, 0.0188, 0.0045],
                                eigvec=[[-0.5675, 0.7192, 0.4009],
                                        [-0.5808, -0.0045, -0.8140],
                                        [-0.5836, -0.6948, 0.4203]])
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ])
            self.test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

        elif configs['dataset'] == 'tiny-imagenet':
            '''
            https://github.com/alinlab/lookahead_pruning/blob/master/dataset.py
            '''
            normalize = transforms.Normalize(mean=mean,
                                             std=std)
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(64,4),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                normalize,
            ])
            self.test_transform = transforms.Compose([
                # transforms.Resize(256),transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

    def get_dataset(self):
        if self.dataset_path is None:
            if sys.platform == 'linux':
                dataset_path = '/data'
            elif sys.platform == 'win32':
                dataset_path = '..\data\dataset'
            else:
                dataset_path = '\dataset'
        else:
            dataset_path = self.dataset_path # parent directory
                
        if self.configs['dataset'] =='imagenet200':
            dataset_path = os.path.join(dataset_path, 'imagenet')
        else:
            dataset_path = os.path.join(dataset_path, self.configs['dataset'])

        if self.configs['dataset'] == 'cifar100':
            train_data = datasets.CIFAR100(root=dataset_path, train=True,
                                           download=True, transform=self.train_transform)
            test_data = datasets.CIFAR100(root=dataset_path, train=False,
                                          download=False, transform=self.test_transform)

        elif self.configs['dataset'] == 'cifar10':
            train_data = datasets.CIFAR10(root=dataset_path, train=True,
                                          download=True, transform=self.train_transform)
            test_data = datasets.CIFAR10(root=dataset_path, train=False,
                                         download=False, transform=self.test_transform)

        elif self.configs['dataset'] == 'imagenet':
            traindata_save_path = os.path.join(dataset_path, 'train')
            testdata_save_path = os.path.join(dataset_path, 'val3')
            train_data = torchvision.datasets.ImageFolder(
                root=traindata_save_path, transform=self.train_transform)
            test_data = torchvision.datasets.ImageFolder(
                root=testdata_save_path, transform=self.test_transform)

        elif self.configs['dataset'] == 'tiny-imagenet':
            traindata_save_path = os.path.join(dataset_path, 'train')
            testdata_save_path = os.path.join(dataset_path, 'val')
            train_data = torchvision.datasets.ImageFolder(
                root=traindata_save_path, transform=self.train_transform)
            test_data = torchvision.datasets.ImageFolder(
                root=testdata_save_path, transform=self.test_transform)
            
        return train_data, test_data

    def get_dataloader(self):
        train_data, test_data = self.get_dataset()
        if self.configs['device'] == 'cuda':
            pin_memory = True
            # pin_memory=False
        else:
            pin_memory = False
        train_data_loader = DataLoader(train_data, batch_size=self.configs['batch_size'],
                                       shuffle=True, pin_memory=pin_memory,
                                       num_workers=self.configs['num_workers'],
                                       )
        # if self.configs['batch_size']<=32:
        #     batch_size=64
        # else: batch_size=self.configs['batch_size']
        test_data_loader = DataLoader(test_data, batch_size=self.configs['batch_size'],
                                      shuffle=False, pin_memory=pin_memory,
                                      num_workers=self.configs['num_workers'],
                                      )

        print("Using Datasets: ", self.configs['dataset'])
        return train_data_loader, test_data_loader

if __name__=='__main__':
    configs={
        'model':'resnet50',
        'batch_size':128,
        'dataset':'imagenet',
        'mode':'eval',
        'device':'cuda',
        # 'condition':'random',
        'num_classes':1000,
        'cutout':False,
        'num_workers':3,
        'train_mode':'baseline',
        # 'train_mode':'snapmix',
        'num_classes':1000,
        'seed':1,
    }
    datasetloader=DatasetLoader(configs)
    _,eval_loader=datasetloader.get_dataloader()
    from torchvision.models import resnet50
    a=resnet50(pretrained=True)
    import time
    import torch.nn as nn
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    a.eval()
    end = time.time()
    criterion=nn.CrossEntropyLoss()
    a.cuda()
    ## For Data saving ##
    ####################
    with torch.no_grad():
        for images, targets in eval_loader:
            # measure data loading time
            images, targets = images.cuda(), targets.cuda()

            # compute output
            output = a(images)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            top1.update(acc1[0]*100.0, images.size(0))
            top5.update(acc5[0]*100.0, images.size(0))
            loss = criterion(output, targets)
            losses.update(loss, images.size(0))

            # measure elapsed time #
            batch_time.update(time.time() - end)
            end = time.time()
            ########################
    print({'loss': losses.avg, 'accuracy': top1.avg, 'top5': top5.avg})
