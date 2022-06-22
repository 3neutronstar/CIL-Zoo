from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import default_loader
class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transform=None,target_transform=None):
        self.X = images
        self.y = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader=default_loader
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        data = self.X[i]

        if isinstance(data,tuple):
            path, target = data
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target
        
        if self.transform:
            data = self.transform(data)
            
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data