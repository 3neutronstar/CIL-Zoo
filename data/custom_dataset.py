from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import default_loader

class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transform=None,target_transform=None,no_return_target=False):
        self.X = images
        self.y = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader=default_loader
        self.no_return_target=no_return_target
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i]

        if isinstance(data,tuple) or isinstance(data,list):
            path, target = data
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            if self.no_return_target:
                return sample
            return sample, target
        
        if self.transform:
            data = self.transform(data)
            
        if self.y is not None:
            if self.no_return_target:
                return data
            return (data, self.y[i])
        else:
            return data
