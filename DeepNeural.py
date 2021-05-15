"""
1. Deep Neural Network using PyTorch
2. Using non linear boundaries to seperate the data
"""


import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from sklearn import datasets



number_of_points =500
centers = [[-0.5, 0.5], [0.5, -0.5]]
x,y = datasets.make_circles(n_samples = number_of_points, random_state = 123, noise = 0.1, factor = 0.2)
x_data=torch.Tensor(x)
y_data=torch.Tensor(y.reshape(500,1))



def scatter_plot():
    plt.scatter(x[y==0, 0], x[y==0, 1])
    plt.scatter(x[y==1, 0], x[y==1, 1])



class Model(nn.Module):                                          #constructing a model using Linear class
    def __init__(self, input_size,H1, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size,H1)
        self.linear2 = nn.Linear(H1,output_size)
    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        x = torch.sigmoid(self.linear2(x))
        return x
    def predict(self, x):
        pred = self.forward(x)
        if pred >= 0.5:
            return 1
        else:
            return 0


torch.manual_seed(2)
model = Model(2, 4,1)
print(list(model.parameters()))


criterion = nn.BCELoss()                                                             #Binary Classification tasks .You just need one outpt node to classify the data into two classws .The output value should be passed through a sigmoid activation function and he range of output
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)                             #Adam optimizer involves a combination of two gradient descent methodologies: Momentum: This algorithm is used to accelerate the gradient descent algorithm by taking into consideration the 'exponentially weighted average' of the gradients. Using averages makes the algorithm converge towards the minima in a faster pace.


epochs =1000
losses = []
for i in range(epochs):
    y_pred = model.forward(x_data)
    loss = criterion(y_pred, y_data)
    print("epoch:", i, "loss", loss.item())
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


plt.plot(range(epochs), losses)
plt.ylabel("Loss")
plt.xlabel("epoch")



def plot_decision_boundary(x,y):
    x_span = np.linspace(min(x[:,0]) -0.25, max(x[:,0])+0.25)                   #linpace for evenly spaced sequence in a specified interval
    y_span = np.linspace(min(x[:,1]) -0.25, max(x[:,1])+0.25)
    xx ,yy = np.meshgrid(x_span, y_span)                             #Meshgrid function is used to create a rectangular grid out of two given one _ dimensional arrays representing thr cartesian  indexing or matrix indexing
    grid = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])               # The ravel function returns a one -dimensional array  the returned array will have the same type as that of the input array
    pred_func = model.forward(grid)
    z = pred_func.view(xx.shape).detach().numpy()                     #The view function is meant to reshape the tensor
    plt.contourf(xx, yy, z)              #Contour plots are a way to show a three dimensional surface on a two dimensional plane .. It graphs two predictor variable x,y on the y-axis ad a response variable z as a contours .these contours are sometimes called x slices or the iso response values                                 



plot_decision_boundary(x,y)



x = 0.025
y = 0.025
point = torch.Tensor([x,y])
prediction = model.predict(point)
plt.plot([x],[y], marker = 'o', markersize = 10, color = "red")
print("prdiction is", prediction)
#plot_decision_boundary(x,y)




