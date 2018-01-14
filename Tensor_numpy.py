#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 2017/11/14
from __future__ import print_function
import torch

#创建一个5x3的矩阵
x = torch.Tensor(5, 3)
print(x)

#创建一个随机矩阵
x = torch.rand(5, 3)
print(x)

#获取规格
print(x.size())

#矩阵合并操作
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))

#输出一个张量
result = torch.Tensor(5, 3)
torch.add(x ,y, out=result)
print(result)

#就地改变，增加X到Y
y.add_(x)
print(y)
"""
    值得注意的是，在就地（in_place）操作张量的过程之中
    最后都会有固定的_符号出现
"""
#像numpy一样去战斗
print(x[:, 1])

#numpy桥，将张量转化为numpy数组，则改变其中的一个另一个也跟着改变
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
  #改变一个的值
a.add_(1)
print(a)
print(b)

#将numpy数组转化为Torch张量
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
