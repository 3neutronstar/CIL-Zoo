
from torchvision import datasets
import numpy as np
from PIL import Image
import copy

from implementor.eeil import data_augmentation_e2e


class iCIFAR100(datasets.CIFAR100):
    def __init__(self, root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,eeil_aug=False):
        super(iCIFAR100, self).__init__(root,
                                        train=train,
                                        transform=transform,
                                        target_transform=target_transform,
                                        download=download)
        self.original_data = copy.deepcopy(self.data)
        self.original_labels = copy.deepcopy(self.targets)
        self.bft_data=None
        self.bft_label=None
        self.eeil_aug=eeil_aug

    def concatenate(self, datas, labels):
        con_data = datas[0]
        con_label = labels[0]
        for i in range(1, len(datas)):
            con_data = np.concatenate((con_data, datas[i]), axis=0)
            con_label = np.concatenate((con_label, labels[i]), axis=0)
        return con_data, con_label

    def update(self, classes, exemplar_set=list()):
        if self.train:  # exemplar_set
            datas, labels = [], []
            if len(exemplar_set) != 0:
                datas = [exemplar for exemplar in exemplar_set]
                length = len(datas[0])
                labels = [np.full((length), label)
                          for label in range(len(exemplar_set))]
                bft_data, bft_label = self.concatenate(datas, labels)
                if self.bft_data is None:
                    self.bft_data,self.bft_label = bft_data, bft_label
                else:
                    self.bft_data = np.concatenate((self.bft_data, bft_data), axis=0)
                    self.bft_label = np.concatenate((self.bft_label, bft_label), axis=0)

            for label in range(classes[0], classes[1]):
                data = self.original_data[np.array(self.original_labels) == label]
                datas.append(data)
                labels.append(np.full((data.shape[0]), label))
            self.data, self.targets = self.concatenate(datas, labels)
            
        else:
            datas, labels = [], []
            for label in range(classes[0], classes[1]):
                data = self.original_data[np.array(self.original_labels) == label]
                datas.append(data)
                labels.append(np.full((data.shape[0]), label))
            datas=np.concatenate(datas, axis=0)
            labels=np.concatenate(labels, axis=0)
            if classes[0]!=0:
                self.data = np.concatenate((self.data, datas), axis=0)
                self.targets = np.concatenate((self.targets, labels), axis=0)
            else:
                self.data = datas
                self.targets = labels
            # if self.bft_data is not None:
            #     self.bft_data = np.concatenate((self.bft_data, datas), axis=0)
            #     self.bft_label = np.concatenate((self.bft_label, labels), axis=0)
            # self.bft_data = self.data
            # self.bft_label= self.targets
        str_train = 'train' if self.train else 'test'
        print("The size of {} set is {}".format(str_train, self.data.shape))
        print("The size of {} label is {}".format(
            str_train, self.targets.shape))
        if self.eeil_aug:
            self.data, self.targets = data_augmentation_e2e(self.data, self.targets)
            print("The size of {} set is {}".format(str_train, self.data.shape))
            print("The size of {} label is {}".format(
                str_train, self.targets.shape))

    def __getitem__(self, index):
        img, target = Image.fromarray(self.data[index]), self.targets[index]

        if self.transform:
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.data)

    def get_class_images(self, cls):
        return self.original_data[np.array(self.original_labels) == cls]

    def get_bft_data(self):
        self.bft_datas=[]
        self.bft_labels=[]
        for index in range(len(self.bft_data)):
            img, target = Image.fromarray(self.bft_data[index]), self.bft_label[index]
            self.bft_datas.append(img)
            self.bft_labels.append(target)
        return (self.bft_datas, self.bft_labels)