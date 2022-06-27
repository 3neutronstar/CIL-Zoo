from data.CIL.cifar100 import iCIFAR100
from data.CIL.imagenet import iImageNet, iTinyImageNet
from data.custom_dataset import ImageDataset
from data.data_load import DatasetLoader
import os
import torchvision
import torchvision.datasets as datasets
import sys
from torch.utils.data import DataLoader


class CILDatasetLoader(DatasetLoader):
    def __init__(self, configs, dataset_path, device):
        super(CILDatasetLoader, self).__init__(dataset_path,configs)

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
        
        dataset_path = os.path.join(dataset_path, self.configs['dataset'])
        
        if self.configs['dataset'] == 'cifar100':
            train_data = iCIFAR100(root=dataset_path, train=True,
                                           download=True, transform=self.train_transform)
            test_data = iCIFAR100(root=dataset_path, train=False,
                                          download=False, transform=self.test_transform)
        elif self.configs['dataset'] == 'imagenet':
            traindata_save_path = os.path.join(dataset_path, 'train')
            testdata_save_path = os.path.join(dataset_path, 'val3')
            train_data = iImageNet(
                root=traindata_save_path,train=True, transform=self.train_transform)
            test_data = iImageNet(
                root=testdata_save_path,train=False, transform=self.test_transform)
        
        elif self.configs['dataset'] == 'tiny-imagenet':
            traindata_save_path = os.path.join(dataset_path, 'train')
            testdata_save_path = os.path.join(dataset_path, 'val')
            train_data = iTinyImageNet(
                root=traindata_save_path,train=True, transform=self.train_transform)
            test_data = iTinyImageNet(
                root=testdata_save_path,train=False, transform=self.test_transform)

        return train_data, test_data

    def get_dataloader(self):
        self.train_data, self.test_data = self.get_dataset()
        if self.configs['device'] == 'cuda':
            pin_memory = True
            # pin_memory=False
        else:
            pin_memory = False
        train_data_loader = DataLoader(self.train_data, batch_size=self.configs['batch_size'],
                                       shuffle=True, pin_memory=pin_memory,
                                       num_workers=self.configs['num_workers'],
                                       )
        # if self.configs['batch_size']<=32:
        #     batch_size=64
        # else: batch_size=self.configs['batch_size']
        test_data_loader = DataLoader(self.test_data, batch_size=self.configs['batch_size'],
                                      shuffle=False, pin_memory=pin_memory,
                                      num_workers=self.configs['num_workers'],
                                      )

        print("Using Datasets: ", self.configs['dataset'])
        return train_data_loader, test_data_loader

    def get_updated_dataloader(self,num_classes,exemplar_set=list()):
        self.train_data.update(num_classes,exemplar_set)
        self.test_data.update(num_classes,exemplar_set)

        if self.configs['device'] == 'cuda':
            pin_memory = True
            # pin_memory=False
        else:
            pin_memory = False
        train_data_loader = DataLoader(self.train_data, batch_size=self.configs['batch_size'],
                                       shuffle=True, pin_memory=pin_memory,
                                       num_workers=self.configs['num_workers'],
                                       )
        test_data_loader = DataLoader(self.test_data, batch_size=self.configs['batch_size'],
                                      shuffle=False, pin_memory=pin_memory,
                                      num_workers=self.configs['num_workers'],
                                      )
        print("Updated classes index ", num_classes)
        return train_data_loader, test_data_loader

    def get_class_dataloader(self,cls,transform=None,no_return_target=False):
        cls_images=self.train_data.get_class_images(cls)

        if transform==None:
            transform=self.test_transform

        dataset=ImageDataset(cls_images,transform=transform,no_return_target=no_return_target)

        if self.configs['device'] == 'cuda':
            pin_memory = True
            # pin_memory=False
        else:
            pin_memory = False
        train_class_data_loader = DataLoader(dataset, batch_size=self.configs['batch_size'],
                                      shuffle=False, pin_memory=pin_memory,
                                      num_workers=self.configs['num_workers'],
                                      )

        return train_class_data_loader,cls_images
    
    def get_bft_dataloader(self):
        bft_train_data=self.train_data.get_bft_data()
        bft_test_data=self.test_data.get_bft_data()
        if len(bft_train_data)==2:
            bft_train_dataset=ImageDataset(bft_train_data[0],bft_train_data[1],transform=self.train_transform,return_index=True)
            bft_test_dataset=ImageDataset(bft_test_data[0],bft_test_data[1],transform=self.test_transform,return_index=True)
        else: # len ==1
            bft_train_dataset=ImageDataset(bft_train_data[0],transform=self.train_transform,return_index=True)
            bft_test_dataset=ImageDataset(bft_test_data[0],transform=self.test_transform,return_index=True)

        train_loader = self._get_loader(True, bft_train_dataset)
        test_loader = self._get_loader(False, bft_test_dataset)
        return train_loader, test_loader

    def _get_loader(self,shuffle,dataset):
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