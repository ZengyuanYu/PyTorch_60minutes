#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 18-2-26
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from skimage import io,transform
import glob
import os
import numpy as np
import time


#CNN 网络
#1.batch_size*3*128*128
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),#input 3*128*128
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2) #output 16*64*64
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),#input 16*64*64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) #output 32*32*32
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2),  #intput 32*32*32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # output 64*16*16
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer6 = nn.Sequential(
            #fully 1
            nn.Linear(256*5*5, 2000),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            )
        self.fc1 = nn.Linear(2000,100)
        self.fc2 = nn.Linear(100, 6)
    def forward(self, x):
        out = self.layer1(x)
        #print('layer1', out.size())
        out = self.layer2(out)#shape(batch_size,32,7,7)
        #print('layer2', out.size())
        out = self.layer3(out)
        #print('layer3', out.size())
        out = self.layer4(out)
        #print('layer4', out.size())
        out = self.layer5(out)
        #print('layer5', out.size())

        out = out.view(out.size(0), -1)#faltten 将数据out铺展 (shape batch_size, 32*7*7)
        #out = self.fc1(out)
        out = self.layer6(out)
        #print('layer6', out.size())
        out = self.fc1(out)
        out = self.fc2(out)
        #print(out)
        #print('layer7', out.size())
        return out