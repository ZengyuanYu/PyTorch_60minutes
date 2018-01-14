#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 2017/11/15
# import torch
# from torch.autograd import Variable
#
# #创建一个变量x
# x = Variable(torch.ones(2, 2), requires_grad=True)
# print(x)
#
# #对变量进行操作 y=x+2
# y = x + 2;
# print(y)
# #y是操作运算创建的，所以它有一个grad_fn
# print(y.grad_fn)
#
# #对y进行更多的操作 z=3*(x+2)^2
# z = y * y * 3
# out = z.mean() #求平均值
# print(z, out)
#
# #####求梯度
# out.backward()
# print(x.grad)
# #######
#
import torch
from torch.autograd import Variable
#设置一个torch型的变量 in 2x2
x = Variable(torch.ones(2, 2), requires_grad=True)
j = x + 2
k = j * j * 3
out = k.mean()
#使用反向传播
out.backward()
#计算微分d(out)/d(in)
print(x.grad)

#自动求导可以做更有趣的事情
x = torch.randn(3)
x = Variable(x, requires_grad=True)
print(x)
y = x * 2
while y.data.norm() < 1000: #.data.norm()代表什么？
    y = y * 2
print(y)
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
print(x.grad)