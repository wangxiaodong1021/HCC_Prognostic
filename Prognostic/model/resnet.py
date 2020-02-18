import torch
import torch.nn as nn
import math
from utils.DropBlock import DropBlock2D
from utils.Scheduler import LinearScheduler
from utils.downsample import Downsample
import numpy as np
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
from torchvision import models


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_prob=0., block_size=7):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=drop_prob, block_size=block_size),
            start_value=0.,
            stop_value=drop_prob,
            nr_steps=5e3
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        self.dropblock.step()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropblock(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropblock(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.dropblock(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, factor=1, num_classes=1, combinate=False, use_mean=False, require_grad=True,
                 way='train', drop_prob=[], block_size=7,use_entropy=False):
        """Constructs a ResNet model.

        Args:
            num_classes: int, since we are doing binary classification
                (tumor vs normal), num_classes is set to 1 and sigmoid instead
                of softmax is used later
            num_nodes: int, number of nodes/patches within the fully CRF
            use_crf: bool, use the CRF component or not
        """
        self.inplanes = 64
        self.num_classes = num_classes
        if not isinstance(drop_prob, list) or len(drop_prob) != 4:
            raise ValueError('dropb_prob is not valid:', drop_prob)
        self.combinate = combinate
        self.use_mean = use_mean
        self.use_entropy = use_entropy
        self.way = way
        self.eval = eval
        self.factor = factor
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, drop_prob=drop_prob[0], block_size=block_size)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, drop_prob=drop_prob[1], block_size=block_size)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, drop_prob=drop_prob[2], block_size=block_size)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, drop_prob=drop_prob[3], block_size=block_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if self.use_entropy:
            num_channel = 2
        else:
            num_channel = 1
        self.fc = nn.Linear(512 * block.expansion * num_channel, num_classes)

        for m in self.modules():
            for i in m.parameters():
                i.requires_grad = require_grad
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, drop_prob=0., block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample, drop_prob=drop_prob,
                  block_size=block_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes, drop_prob=drop_prob, block_size=block_size))

        return nn.Sequential(*layers)
    def forward(self, x):
        batch_size,num_patch,_,img_size,_ = x.shape
        x = x.view(-1,3,img_size,img_size)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.layer4(x)
        logits = x.view(batch_size, num_patch, x.size(1), x.size(-1), x.size(-1))
        if self.way == 'val':
            temp = logits.sort(1, descending=True)[0]
            x = temp[:, 0, :, :, :]
            x = x.squeeze(1)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x, logits
        x, _ = logits.max(1)
        x = x.squeeze(1)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.use_entropy:
            std = self._Std(logits.cpu()).cuda()
            x = torch.cat((x, std), 1)
        x = self.fc(x)

        return x


def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        print("Loading PreTrain resnet18 ckpt......")
        state_dict = load_state_dict_from_url(model_urls['resnet18'],
                                              progress=True)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and "fc" not in k}
        model.load_state_dict(state_dict, False)
    return model


def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print("Loading PreTrain resnet34 ckpt......")
        state_dict = load_state_dict_from_url(model_urls['resnet34'],
                                              progress=True)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and "fc" not in k}
        model.load_state_dict(state_dict, False)
    return model


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print("Loading PreTrain resnet50 ckpt......")
        state_dict = load_state_dict_from_url(model_urls['resnet50'],
                                              progress=True)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and "fc" not in k}
        model.load_state_dict(state_dict, False)
    return model


def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        print("Loading PreTrain resnet101 ckpt......")
        state_dict = load_state_dict_from_url(model_urls['resnet101'],
                                              progress=True)
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict and "fc" not in k}
        model.load_state_dict(state_dict, False)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model