#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by XiaoYu on 2017/11/20
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#1.数据定义
x = torch.unsqueeze(torch.linspace(-1,1,200), dim=1)
#y = x.pow(1) + 0.2*torch.rand(x.size())
y = x.pow(1) + 0.2*torch.rand(x.size()) ## y = x + 1

x,y = Variable(x), Variable(y)

# #可视化
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

#2.神经网络定义
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net,self).__init__()#继承Net里面的东西
        #定义隐藏层和输出层
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

#3.处理（优化SGD，损失MSE，）
net = Net(1,20,1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.MSELoss()

for t in range(200):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
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
