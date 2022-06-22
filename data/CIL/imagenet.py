
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


    def update(self, classes, exemplar_set=list()):
        if self.train:  # exemplar_set
            datas = []
            if len(exemplar_set) != 0:
                datas = [exemplar for exemplar in exemplar_set]

            for label in range(classes[0], classes[1]):
                for i in (np.array(self.targets) == label).nonzero()[0]:
                    datas.append(self.samples[i])
            self.data=datas
        else:
            datas = []
            for label in range(classes[0], classes[1]):
                for i in (np.array(self.targets) == label).nonzero()[0]:
                    datas.append(self.samples[i])

            if classes[0]!=0:
                self.data.extend(datas)
            else:
                self.data = datas

        str_train = 'train' if self.train else 'test'
        print("The size of {} set is {}".format(str_train, len(self.data)))

    def __len__(self):
        return len(self.data)

    def get_class_images(self, cls):
        return self.samples[np.array(self.targets) == cls]

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