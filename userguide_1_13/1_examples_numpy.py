import torch
import numpy as np
import math
import time
import matplotlib.pyplot as plt
# input & output
Ix = np.linspace(-math.pi, math.pi, 2000)
Oy = np.sin(Ix)



# weight
Wa = np.random.randn()
Wb = np.random.randn()
Wc = np.random.randn()
Wd = np.random.randn()

learning_rate = 1e-6
start_time = time.perf_counter()
for i in range(20000):
    y_pred = Wa + Wb * Ix + Wc * Ix**2 + Wd * Ix**3
    loss = np.square(y_pred - Oy).sum()
    if i % 100 == 99:
        print("curent index is : {}, loss is : {}".format(i, loss))
    
    # bp
    grad_y_pred = 2 * (y_pred - Oy)
    grad_Wa = grad_y_pred.sum()
    grad_Wb = (grad_y_pred * Ix).sum()
    grad_Wc = (grad_y_pred * Ix**2).sum()
    grad_Wd = (grad_y_pred * Ix**3).sum()

    # update weight
    Wa = Wa - learning_rate * grad_Wa
    Wb = Wb - learning_rate * grad_Wb
    Wc = Wc - learning_rate * grad_Wc
    Wd = Wd - learning_rate * grad_Wd

print(f'Final : y = {Wa} + {Wb} * x + {Wc} * x^2 + {Wd} * x^3')
end_time = time.perf_counter()
print(f"Total Traning Time is : {end_time - start_time}")
y_pred = Wa + Wb * Ix + Wc * Ix**2 + Wd * Ix**3
# diff = np.square(Oy - y_pred).sum()
# print(f"Diff is : {diff}")

fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(Ix, Oy, label='orignal')  # Plot some data on the axes.
ax.plot(Ix, y_pred, label='traning')  # Plot more data on the axes...

# ax.set_xlabel('x label')  # Add an x-label to the axes.
# ax.set_ylabel('y label')  # Add a y-label to the axes.
ax.set_title("Simple Plot")  # Add a title to the axes.
ax.legend()  # Add a legend.

plt.savefig('result.png')