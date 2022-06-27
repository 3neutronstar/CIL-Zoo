import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

def conv3x3(inplanes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * Bottleneck.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * Bottleneck.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module): # Follow CutMix
    model_dict = {'resnet18': (BasicBlock,[2, 2, 2, 2]),
                    'resnet34': (BasicBlock,[3, 4, 6, 3]),
                    'resnet50': (Bottleneck,[3, 4, 6, 3]),
                    'resnet101': (Bottleneck,[3, 4, 23, 3]),
                    'resnet152': (Bottleneck,[3, 8, 36, 3]),
                    'resnet20': (BasicBlock,[3, 3, 3]),
                    'resnet32': (BasicBlock,[5, 5, 5]),
                    'resnet44': (BasicBlock,[7, 7, 7]),
                    'resnet56': (BasicBlock,[9, 9, 9]),
                    'resnet110': (BasicBlock,[18, 18, 18]),
                    'resnet1202': (BasicBlock,[200, 200, 200])
                    }
    def __init__(self, configs):
        super(ResNet, self).__init__()
        num_classes = configs['num_classes']
        block, num_blocks = self.model_dict[configs['model']]
        self.residual_len=len(num_blocks)
        if self.residual_len == 4:
            initial_channels = 64
            self.inplanes=initial_channels
            if configs['dataset'] in ['cifar10','cifar100']:
                self.maxpool=None
                stride=1
                self.avgpool = nn.AvgPool2d(4)
                self.conv1 = nn.Conv2d(3,
                                    initial_channels,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    bias=False)
            elif configs['dataset'] =='tiny-imagenet' and configs['tiny_resize'] ==False:
                self.maxpool=None
                stride=2
                self.avgpool = nn.AvgPool2d(4)
                self.conv1 = nn.Conv2d(3,
                                    initial_channels,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=1,
                                    bias=False)
            else: # size 224
                stride=2
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.avgpool = nn.AvgPool2d(7)
                self.conv1 = nn.Conv2d(
                    3, initial_channels, kernel_size=7, stride=stride, padding=3, bias=False)
                
            self.bn1 = nn.BatchNorm2d(64)
            stride_list = [1, 2, 2, 2]

            self.layer1 = self._make_layer(
                block, initial_channels, num_blocks[0], stride=stride_list[0])
            self.layer2 = self._make_layer(
                block, initial_channels*2, num_blocks[1], stride=stride_list[1])
            self.layer3 = self._make_layer(
                block, initial_channels*4, num_blocks[2], stride=stride_list[2])
            self.layer4 = self._make_layer(
                block, initial_channels*8, num_blocks[3], stride=stride_list[3])
            self.fc = nn.Linear(
                initial_channels*8*block.expansion, num_classes)
        elif self.residual_len == 3:
            initial_channels = 16
            self.inplanes=initial_channels

            self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            if configs['dataset']=='tiny-imagenet' and configs['tiny_resize'] ==False:
                self.avgpool=nn.AdaptiveMaxPool2d(1)
            else:
                self.avgpool = nn.AvgPool2d(8)
            stride_list = [1, 2, 2]
            self.layer1 = self._make_layer(
                block, initial_channels, num_blocks[0], stride=stride_list[0])
            self.layer2 = self._make_layer(
                block, initial_channels*2, num_blocks[1], stride=stride_list[1])
            self.layer3 = self._make_layer(
                block, initial_channels*4, num_blocks[2], stride=stride_list[2])
            self.fc = nn.Linear(
                initial_channels*4*block.expansion, num_classes)
            
        self.relu=nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.residual_len == 4:
            out = self.relu(self.bn1(self.conv1(x)))
            if self.maxpool:
                out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
        elif self.residual_len == 3:
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def extract_feature(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if len(self.plane_list)==4:
            out=self.maxpool(out)
        feature = []
        for residual in self.residual_layer:
            out = residual(out)
            feature.append(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out, feature

class ResNet_ft(ResNet):
    def __init__(self, configs):
        super().__init__(configs)
    
    def forward(self,x):
        out = self.relu(self.bn1(self.conv1(x)))
        feature = []
        if self.residual_len==4:
            if self.maxpool:
                out=self.maxpool(out)
            out = self.layer1(out)
            feature.append(out)
            out = self.layer2(out)
            feature.append(out)
            out = self.layer3(out)
            feature.append(out)
            out = self.layer4(out)
            feature.append(out)
            out = self.avgpool(out)

        elif self.residual_len==3:
            out = self.layer1(out)
            feature.append(out)
            out = self.layer2(out)
            feature.append(out)
            out = self.layer3(out)
            feature.append(out)
            out = self.avgpool(out)

        out = out.view(out.size(0), -1)
        feature.append(out)
        out = self.fc(out)

        return out, feature
