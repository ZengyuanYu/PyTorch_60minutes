#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 2017/11/15

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#制造数据，unsqueeze函数是将一维数据变为二维，linspace(-1,1,100),在-1---1之间生成一百个数据，维度为1.
x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())  #m.pow(x)为m的x次方 后面加上的是噪声

#将x,y都变为Variable类型
x, y = Variable(x), Variable(y)

# #打印出来散点图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

#定义神经网络
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output): #神经网络的一些信息
        super(Net, self).__init__() #继承一下Net里面的东西
        #接下来开始自己定义神经网络的基本参数
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)#回归问题 同一个x定义同一个y

    def forward(self, x):  #前向传递的过程
        x = F.relu(self.hidden(x)) #x = n_hidden
        x = self.predict(x) #x = n_output
        return x #相当于返回y的值

net = Net(1, 20, 1)
print(net)

#可视化实时显示定义
plt.ion()
plt.show()

optimazer = torch.optim.SGD(net.parameters(), lr=0.5) #优化神经网络的参数，并赋予学习率

#损失函数
loss_func = torch.nn.MSELoss()#MSEloss均方差

for t in range(100):
    #得到输出结果
    prediction = net(x)
    #计算误差
    loss = loss_func(prediction, y)
    #优化结果
    optimazer.zero_grad() #梯度降为零
    loss.backward()#误差进行反向传播
    optimazer.step()#以学习效率lr 优化梯度

    #可视化
    if t % 5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw = 5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0],
                 fontdict={'size' : 20, 'color': 'green'})
        plt.pause(0.1)
plt.ioff()
plt.show()