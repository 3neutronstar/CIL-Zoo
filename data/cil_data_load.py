from data.CIL.cifar100 import iCIFAR100
from data.CIL.imagenet import iImageNet, iTinyImageNet
from data.custom_dataset import ImageDataset
from data.data_load import DatasetLoader
import os
import torchvision
import torchvision.datasets as datasets
import sys
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from utils.augmentation import *

# class CILDatasetLoader(DatasetLoader):
#     def __init__(self, configs, dataset_path, device):
#         super(CILDatasetLoader, self).__init__(dataset_path,configs)

#     def get_dataset(self):
#         if self.dataset_path is None:
#             if sys.platform == 'linux':
#                 dataset_path = '/data'
#             elif sys.platform == 'win32':
#                 dataset_path = '..\..\data\dataset'

#             else:
#                 dataset_path = '\dataset'
#         else:
#             dataset_path = self.dataset_path # parent directory
        
#         dataset_path = os.path.join(dataset_path, self.configs['dataset'])
        
#         if self.configs['dataset'] == 'cifar100':
#             train_data = iCIFAR100(root=dataset_path, train=True,
#                                            download=True, transform=self.train_transform)
#             test_data = iCIFAR100(root=dataset_path, train=False,
#                                           download=False, transform=self.test_transform)
#         elif self.configs['dataset'] == 'imagenet':
#             traindata_save_path = os.path.join(dataset_path, 'train')
#             testdata_save_path = os.path.join(dataset_path, 'val3')
#             train_data = iImageNet(
#                 root=traindata_save_path,train=True, transform=self.train_transform)
#             test_data = iImageNet(
#                 root=testdata_save_path,train=False, transform=self.test_transform)
        
#         elif self.configs['dataset'] == 'tiny-imagenet':
#             traindata_save_path = os.path.join(dataset_path, 'train')
#             testdata_save_path = os.path.join(dataset_path, 'val')
#             train_data = iTinyImageNet(
#                 root=traindata_save_path,train=True, transform=self.train_transform)
#             test_data = iTinyImageNet(
#                 root=testdata_save_path,train=False, transform=self.test_transform)

#         return train_data, test_data

#     def get_dataloader(self):
#         self.train_data, self.test_data = self.get_dataset()
#         if self.configs['device'] == 'cuda':
#             pin_memory = True
#             # pin_memory=False
#         else:
#             pin_memory = False
#         train_data_loader = DataLoader(self.train_data, batch_size=self.configs['batch_size'],
#                                        shuffle=True, pin_memory=pin_memory,
#                                        num_workers=self.configs['num_workers'],
#                                        )
#         # if self.configs['batch_size']<=32:
#         #     batch_size=64
#         # else: batch_size=self.configs['batch_size']
#         test_data_loader = DataLoader(self.test_data, batch_size=self.configs['batch_size'],
#                                       shuffle=False, pin_memory=pin_memory,
#                                       num_workers=self.configs['num_workers'],
#                                       )

#         print("Using Datasets: ", self.configs['dataset'])
#         return train_data_loader, test_data_loader

#     def get_updated_dataloader(self,num_classes,exemplar_set=list()):
#         self.train_data.update(num_classes,exemplar_set)
#         self.test_data.update(num_classes,exemplar_set)

#         if self.configs['device'] == 'cuda':
#             pin_memory = True
#             # pin_memory=False
#         else:
#             pin_memory = False
#         train_data_loader = DataLoader(self.train_data, batch_size=self.configs['batch_size'],
#                                        shuffle=True, pin_memory=pin_memory,
#                                        num_workers=self.configs['num_workers'],
#                                        )
#         test_data_loader = DataLoader(self.test_data, batch_size=self.configs['batch_size'],
#                                       shuffle=False, pin_memory=pin_memory,
#                                       num_workers=self.configs['num_workers'],
#                                       )
#         print("Updated classes index ", num_classes)
#         return train_data_loader, test_data_loader

#     def get_class_dataloader(self,cls,transform=None,no_return_target=False):
#         cls_images=self.train_data.get_class_images(cls)

#         if transform==None:
#             transform=self.test_transform

#         dataset=ImageDataset(cls_images,transform=transform,no_return_target=no_return_target)

#         if self.configs['device'] == 'cuda':
#             pin_memory = True
#         else:
#             pin_memory = False
#         train_class_data_loader = DataLoader(dataset, batch_size=self.configs['batch_size'],
#                                       shuffle=False, pin_memory=pin_memory,
#                                       num_workers=self.configs['num_workers'],
#                                       )
#         return train_class_data_loader,cls_images
    
#     def _get_loader(self,shuffle,dataset):
#         if self.configs['device'] == 'cuda':
#             pin_memory = True
#             # pin_memory=False
#         else:
#             pin_memory = False
#         data_loader = DataLoader(dataset, batch_size=self.configs['batch_size'],
#                                        shuffle=shuffle, pin_memory=pin_memory,
#                                        num_workers=self.configs['num_workers'],
#                                        )
#         return data_loader


class CILDatasetLoader:
    def __init__(self, configs, dataset_path, device):
        self.configs = configs
        self.dataset_path = dataset_path
    
    def get_normalization_mean_std(self,dataset_name):

        ## Normalize Mean & Std ##
        if dataset_name == 'cifar100':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif dataset_name == 'cifar10':
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
        elif dataset_name == 'imagenet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        elif dataset_name == 'mini-imagenet':
            mean = (0.485, 0.456, 0.406) 
            std = (0.229, 0.224, 0.225)
        elif dataset_name == 'tiny-imagenet':
            mean = [0.4802, 0.4481, 0.3975]
            std = [0.2302, 0.2265, 0.2262]
        elif dataset_name in ['dogs120','cub200','cars196','caltech101','caltech256','flowers102','aircraft100','food101','caltech101','caltech256']:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        self.configs['mean'] = torch.tensor(
            mean, dtype=torch.float32).reshape(1, 3, 1, 1).to(self.configs['device'])
        self.configs['std'] = torch.tensor(
            std, dtype=torch.float32).reshape(1, 3, 1, 1).to(self.configs['device'])
        ##########################
        return mean, std
    
    def get_transforms(self,dataset_name,mean=None,std=None):
        if mean and std:
            normalize = transforms.Normalize(mean=mean, std=std)
        else:
            raise ValueError('mean and std must be specified')
        if dataset_name == 'cifar100':
            train_transform = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop((32, 32)),
                transforms.RandomHorizontalFlip(),
                # transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
            if self.configs['train_mode']=='eeil':
                train_transform = transforms.Compose([
                    transforms.Pad(4),
                    transforms.RandomCrop((32, 32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    ColorJitter(brightness=0.63),
                    transforms.ColorJitter(contrast=(0.2,1.8)),
                    normalize,
                ])
        elif dataset_name == 'cifar10':
            train_transform = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop((32, 32)),
                transforms.RandomHorizontalFlip(),
                # transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                # transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize,
            ])
        elif dataset_name == 'imagenet':
            '''
            https://github.com/pytorch/examples/blob/master/imagenet/main.py
            '''
            jittering = ColorJitter(brightness=0.4, contrast=0.4,
                                    saturation=0.4)
            lighting = Lighting(alphastd=0.1,
                                eigval=[0.2175, 0.0188, 0.0045],
                                eigvec=[[-0.5675, 0.7192, 0.4009],
                                        [-0.5808, -0.0045, -0.8140],
                                        [-0.5836, -0.6948, 0.4203]])
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                jittering,
                lighting,
                normalize,
            ])
            test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        elif dataset_name == 'tiny-imagenet':
            '''
            https://github.com/alinlab/lookahead_pruning/blob/master/dataset.py
            '''
            train_transform = transforms.Compose([
                transforms.RandomCrop(64,4),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                normalize,
            ])
            test_transform = transforms.Compose([
                # transforms.Resize(256),transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
        return train_transform,test_transform

    def get_dataset(self,dataset_name,train_transform=None,test_transform=None):

        if self.dataset_path is None:
            if sys.platform == 'linux':
                dataset_path = '/data'
            elif sys.platform == 'win32':
                dataset_path = '..\..\data\dataset'
            else:
                dataset_path = '\dataset'
        else:
            dataset_path = self.dataset_path # parent directory

        dataset_path=os.path.join(dataset_path,dataset_name)
        
        if dataset_name == 'cifar100':
            train_data = iCIFAR100(root=dataset_path, train=True,
                                           download=True, transform=train_transform)
            test_data = iCIFAR100(root=dataset_path, train=False,
                                          download=False, transform=test_transform)
        elif dataset_name == 'imagenet':
            traindata_save_path = os.path.join(dataset_path, 'train')
            testdata_save_path = os.path.join(dataset_path, 'val3')
            train_data = iImageNet(
                root=traindata_save_path,train=True, transform=train_transform)
            test_data = iImageNet(
                root=testdata_save_path,train=False, transform=test_transform)
        
        elif dataset_name == 'tiny-imagenet':
            traindata_save_path = os.path.join(dataset_path, 'train')
            testdata_save_path = os.path.join(dataset_path, 'val')
            train_data = iTinyImageNet(
                root=traindata_save_path,train=True, transform=train_transform)
            test_data = iTinyImageNet(
                root=testdata_save_path,train=False, transform=test_transform)
        return train_data, test_data
    
    def get_dataloader(self,dataset,shuffle=False):
        if self.configs['device'] == 'cuda':
            pin_memory = True
            # pin_memory=False
        else:
            pin_memory = False
        data_loader = DataLoader(dataset, batch_size=self.configs['batch_size'],
                                       shuffle=shuffle, pin_memory=pin_memory,
                                       num_workers=self.configs['num_workers'],
                                       )
        return data_loader

    def get_settled_dataloader(self):
        mean,std=self.get_normalization_mean_std(self.configs['dataset'])
        self.train_transform,self.test_transform=self.get_transforms(self.configs['dataset'],mean,std)
        self.train_data,self.test_data=self.get_dataset(self.configs['dataset'],self.train_transform,self.test_transform)
        self.train_loader=self.get_dataloader(self.train_data,shuffle=True)
        self.test_loader=self.get_dataloader(self.test_data,shuffle=False)
        return self.train_loader,self.test_loader