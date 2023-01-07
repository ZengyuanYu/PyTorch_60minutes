import torch
import numpy as np
import math
import time
import matplotlib.pyplot as plt

dtype = torch.float32
device = torch.device("cpu")
# input & output
Ix = torch.linspace(-math.pi, math.pi, 2000, device=device)
Oy = torch.sin(Ix)

p = torch.tensor([1, 2, 3])
# xxx is [200, 3] 
# Ix value ^1 ^2 ^3
xx = Ix.unsqueeze(-1).pow(p)

# import pdb;pdb.set_trace()
model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)
# print(model)
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-6
start_time = time.perf_counter()
for i in range(2000):
    y_pred = model(xx)

    loss = loss_fn(y_pred, Oy)

    if i % 100 == 99:
        print(i, loss.item())

    model.zero_grad()
    
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= param.grad * learning_rate

end_time = time.perf_counter()
print(f"Total Traning Time is : {end_time - start_time}")
# You can access the first layer of `model` like accessing the first item of a list
linear_layer = model[0]
y_pred = linear_layer.bias.item() \
        + linear_layer.weight[:, 0].item() * Ix  \
        + linear_layer.weight[:, 1].item() * Ix**2 \
        + linear_layer.weight[:, 2].item() * Ix**3

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(Ix, Oy, label='orignal')  # Plot some data on the axes.
ax.plot(Ix, y_pred, label='traning')  # Plot more data on the axes...

# ax.set_xlabel('x label')  # Add an x-label to the axes.
# ax.set_ylabel('y label')  # Add a y-label to the axes.
ax.set_title("Simple Plot")  # Add a title to the axes.
ax.legend()  # Add a legend.

plt.savefig('torchnn_result.png')
# For linear layer, its parameters are stored as `weight` and `bias`.
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')


model.eval()
input  = torch.randn([2000, 3])
output = torch.randn([2000, 1])
torch.onnx.export(model, input, "model.onnx")