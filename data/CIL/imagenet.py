
from torchvision import datasets
import numpy as np
from PIL import Image
import copy


class iImageNet(datasets.ImageFolder):
    def __init__(self, root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 ):
        super(iImageNet, self).__init__(root,
                                        transform=transform,
                                        target_transform=target_transform,
                                        )
        self.train=train
        self.original_data = copy.deepcopy(self.data)
        self.original_labels = copy.deepcopy(self.targets)
        self.data, self.targets = [],[]

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

            for label in range(classes[0], classes[1]):
                data = self.original_data[np.array(self.original_labels) == label]
                datas.append(data)
                labels.append(np.full((data.shape[0]), label).tolist())
            self.data.extend(datas)
            self.targets.extend(labels)
        else:
            datas, labels = [], []
            for label in range(classes[0], classes[1]):
                data = self.original_data[np.array(self.original_labels) == label]
                datas.append(data)
                labels.append(np.full((data.shape[0]), label))
            datas=np.concatenate(datas, axis=0)
            labels=np.concatenate(labels, axis=0)
            if classes[0]!=0:
                self.data.extend(datas)
                self.targets.extend(labels.tolist())
            else:
                self.data = datas
                self.targets = labels
        str_train = 'train' if self.train else 'test'
        print("The size of {} set is {}".format(str_train, self.data.shape))
        print("The size of {} label is {}".format(
            str_train, self.targets.shape))

    def __len__(self):
        return len(self.data)

    def get_class_images(self, cls):
        return self.original_data[np.array(self.original_labels) == cls]

class iTinyImageNet(iImageNet):
    def __init__(self, root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 ):
        super(iTinyImageNet, self).__init__(root,train=train,
                                        transform=transform,
                                        target_transform=target_transform,
                                        )