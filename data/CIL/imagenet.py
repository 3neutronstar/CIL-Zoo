
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
        self.data = copy.deepcopy(self.samples)
        self.ground_targets = []
        self.bft_data = None


    def update(self, classes, exemplar_set=list()):
        if self.train:  # exemplar_set
            datas = []
            if len(exemplar_set) != 0:
                datas = []
                # datas = [exemplar for exemplar in exemplar_set]
                for exemplar in exemplar_set:
                    datas.extend(exemplar)
                if self.bft_data is None:
                    self.bft_data = datas
                else:
                    self.bft_data=self.bft_data.extend(self.data)    
            for label in range(classes[0], classes[1]):
                for i in (np.array(self.targets) == label).nonzero()[0]:
                    datas.append(self.samples[i])
            self.data=datas
            if self.bft_data is None:
                self.bft_data = datas
        else:
            datas = []
            for label in range(classes[0], classes[1]):
                for i in (np.array(self.targets) == label).nonzero()[0]:
                    datas.append(self.samples[i])
            if classes[0]!=0:
                self.data.extend(datas)
            else:
                self.data = datas
            self.bft_data = copy.deepcopy(self.data)

        str_train = 'train' if self.train else 'test'
        print("The size of {} set is {}".format(str_train, len(self.data)))

    def __len__(self):
        return len(self.data)

    def get_class_images(self, cls):
        cls_images=[]
        for i in (np.array(self.targets)==cls).nonzero()[0]:
            cls_images.append(self.samples[i])
        return cls_images

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.data[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, index
    
    def get_bft_data(self):
        return self.bft_data

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
