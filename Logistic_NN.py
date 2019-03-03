import torch
import torch.nn.functional as F

#requirments:
"""
torch==0.4.1
"""
x = [[0.75, 0.75], [0.75, 0.25], [0.25, 0.75], [0.25, 0.25]] # Make Data
y = [0, 1, 1, 0] # 0-A 1-B

x = torch.FloatTensor(x)    # Data To Tensor
y = torch.LongTensor(y)
print(x, y)

# Make signle hidden NN Model
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.out(x)
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2)     # define the network
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.001)
loss_func = torch.nn.CrossEntropyLoss()  

for t in range(100):
    out = net(x)               
    loss = loss_func(out, y)     

    optimizer.zero_grad()   
    loss.backward()       
    optimizer.step()       

    if t % 2 == 0:
        # plot and show learning process
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        print(accuracy)

