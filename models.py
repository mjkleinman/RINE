#!/usr/bin/env python

# from __future__ import print_function
from __future__ import division
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.backends.cudnn as cudnn
import numpy as np

from layers import *


def kconv3x3(in_planes, out_planes, stride=1):
    return KFACConv(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, activation_fn=None, batch_norm=False)


class _KResBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(_KResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = kconv3x3(in_planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = kconv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class FCNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=10):
        super(FCNet, self).__init__()
        self.features = nn.Sequential(
            Flatten(),
            nn.Linear(32 * 32 * n_channels, 4000, bias=False),
            nn.BatchNorm1d(4000),
            nn.ReLU(),
            nn.Linear(4000, 1000),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 4000, bias=False),
            nn.BatchNorm1d(4000),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(4000, 10),
        )

    def forward(self, input):
        features = self.features(input)
        output = self.classifier(features)
        return output, torch.zeros(1).to(output.device)


class TOYFCnet(nn.Module):
    def __init__(self, n_classes=2, n_inputs=1):
        super(TOYFCnet, self).__init__()
        self.features = nn.Sequential(
            Flatten(),
            nn.Linear(n_inputs, 25, bias=True),
            nn.BatchNorm1d(25),
            nn.ReLU(),
            nn.Linear(25, 15, bias=False),
            nn.BatchNorm1d(15),
            # nn.Linear(15, 10, bias=False),
            # nn.BatchNorm1d(10),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(15, n_classes),
        )

    def forward(self, input):
        features = self.features(input)
        output = self.classifier(features)
        return output, torch.zeros(1).to(output.device)


class SimpleNet(nn.Module):
    def __init__(self, n_classes=2, n_inputs=1):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            Flatten(),
            nn.Linear(n_inputs, 25, bias=True),
            nn.BatchNorm1d(25),
            nn.ReLU(),
            nn.Linear(25, 15),
            nn.BatchNorm1d(15),
            nn.Linear(15, 10, bias=False),
            nn.BatchNorm1d(10),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            # Flatten(),
            # nn.BatchNorm1d(n_inputs),
            nn.Linear(n_inputs, n_classes),
            nn.BatchNorm1d(n_classes),
            nn.ReLU(),
            nn.Linear(n_classes, n_classes)
        )

    def forward(self, input):
        # features = self.features(input)
        output = self.classifier(input)
        return output, torch.zeros(1).to(output.device)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _ResBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(_ResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class ResNet18(nn.Module):
    def __init__(self, n_channels=3, block=_ResBlock, num_blocks=[2, 2, 2, 2], n_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(n_channels, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, n_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, torch.zeros(1).to(out.device)
