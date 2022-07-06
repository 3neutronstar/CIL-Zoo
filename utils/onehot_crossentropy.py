import torch
import torch.nn as nn
class OnehotCrossEntropyLoss(nn.Module):
    def __init__(self,reduction='mean'):
        super(OnehotCrossEntropyLoss,self).__init__()
        self.reduction=reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self,prediction,target):
        if self.reduction=='mean':
            return torch.mean(torch.sum(-target * self.logsoftmax(prediction), dim=1))
        elif self.reduction=='sum':
            return torch.sum(torch.sum(-target * self.logsoftmax(prediction), dim=1))
        else:
            return torch.sum(-target * self.logsoftmax(prediction), dim=1)