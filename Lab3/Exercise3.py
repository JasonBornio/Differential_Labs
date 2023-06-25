# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

#Question1---------------------

def rastrigin_function(x,n: int=2,A: int=10):
    return A * n + torch.sum(x**2 - A*torch.cos(2*torch.pi*x))

def optimise(x, _type: str="SGD",lr: float = 0.01,momen : float = 0,num_epochs: int=100):
    
    xmin, xmax, xstep = -5, 5, .2
    ymin, ymax, ystep = -5, 5, .2
    x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    z = rastrigin_function(torch.tensor([x, y])).numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contourf(x, y, z, levels=np.logspace(0, 5, 35), norm=LogNorm(), cmap=plt.cm.gray)

    p = torch.tensor([[0.0],[0.0]], requires_grad=True)
    #opt = optim.SGD([p], lr=0.01)

    path = np.empty((2,0))
    path = np.append(path, p.data.numpy(), axis=1)
    
    opt = []
    loss_data = []
    loss_points = []
    
    if _type == "SGD":
        opt = optim.SGD([x],lr=lr,momentum=momen)
    if _type == "Adam":
        opt = optim.Adam([x],lr=lr)
    if _type == "Adagrad":
        opt = optim.Adagrad([x],lr=lr)
    
    for i in range(num_epochs):
        opt.zero_grad()
        output = rastrigin_function(x, A=0.5)
        loss_data.append(output.data)
        loss_points.append(i)
        output.backward()
        opt.step()
        path = np.append(path, p.data.numpy(), axis=1)

    ax.plot(path[0], path[1], color='red', label='SGD', linewidth=2)

    ax.legend()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
        
    #print(loss_data)
    #plt.plot(loss_points,loss_data)
    #plt.title(_type)
    #plt.xlabel("iteration")
    #plt.ylabel("loss")
    #plt.figure()
    
    return "endpoint: " + str(x.data)

def question1():
    print("\nQUESTION 1::::::::::::::::\n")
    start_point = torch.tensor([5,4], requires_grad=True,dtype=torch.float)
    print("SGD-------------------")
    print(optimise(start_point))
    print("SGD + Momentum--------")
    print(optimise(start_point,momen=0.9))
    print("Adagrad---------------")
    print(optimise(start_point,_type="Adagrad"))
    print("Adam------------------")
    print(optimise(start_point,_type="Adam"))

#Question2---------------------

import pandas as pd  
from torch.utils import data

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases' + '/iris/iris.data', header=None)

df = df.sample(frac=1, random_state=0)#shuffle
df = df[df[4].isin(['Iris-virginica', 'Iris-versicolor'])]#filter
#add label indices column
mapping = {k : v for v, k in enumerate(df[4].unique())}
df[5] = (2 * df[4].map(mapping)) - 1 #labels in {−1 ,1}
#normalise data
alldata = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values, dtype=torch.float)
alldata = (alldata - alldata.mean(dim=0)) / alldata.var(dim=0)
#create datasets
targets_tr = torch.tensor(df.iloc[:75, 5].values, dtype=torch.long)
targets_va = torch.tensor(df.iloc[75:, 5].values, dtype=torch.long)
data_tr = alldata[:75]
data_va = alldata[75:]

def svm(x, w, b):
    h = (w*x).sum(1) + b
    return h

def hinge_loss(y_pred, y_true):
  yz= y_pred.t() * y_true

  return torch.max(torch.tensor(0), 1- yz)

def optimise_svm(data_in,targets_in, _type: str="SGD",lr: float = 0.01,momen : float = 0,num_epochs: int=100, decay: float=0.0001):
    
    
    w = torch.randn(1, 4, requires_grad=True) #4 weights for 4 inputs
    b = torch.randn(1, requires_grad=True)
    dataset = data.TensorDataset(data_in,targets_in)
    dataloader = data.DataLoader(dataset, batch_size=25, shuffle=True) #batch size 25
    
    if _type == "SGD":
        opt = optim.SGD([w,b],lr=lr,momentum=momen, weight_decay=decay)
    if _type == "Adam":
        opt = optim.Adam([w,b],lr=lr, weight_decay=decay)
    if _type == "Adagrad":
        opt = optim.Adagrad([w,b],lr=lr, weight_decay=decay)

    for epoch in range(num_epochs):
        for batch in dataloader:
            opt.zero_grad()
            output = svm(batch[0],w,b)
            loss = hinge_loss(output,batch[1])
            loss = torch.mean(loss)
            loss.backward()
            opt.step()
        
    predictions = svm(data_va,w,b)
    _accuracy = accuracy(predictions,targets_va)
    
    return _accuracy

def accuracy(predictions, labels):
    count = 0
    total = 0
    for prediction, label in zip(predictions, labels):
        total += 1
        if (label == -1 and prediction < 0) or (label == 1 and prediction > 0):
            count +=1
            
    return (count/total) * 100

def question2():
    print("\nQUESTION 2::::::::::::::::\n")
    
    print("ACCURACY--------------")
    print("__________________________")
    print("SGD    |LR=0.01  |ITER=100")
    accuracy = optimise_svm(data_tr,targets_tr)
    print("accuracy:" + str(accuracy) + "%")
    
    print("__________________________")
    print("SGD    |LR=0.001 |ITER=100")
    accuracy = optimise_svm(data_tr,targets_tr,lr=0.001)
    print("accuracy:" + str(accuracy) + "%")
    
    print("__________________________")
    print("SGD    |LR=0.0001|ITER=100")
    accuracy = optimise_svm(data_tr,targets_tr,lr=0.0001)
    print("accuracy:" + str(accuracy) + "%")
                 
    print("__________________________")
    print("Adam   |LR=0.01  |ITER=100")
    accuracy = optimise_svm(data_tr,targets_tr, _type="Adam")
    print("accuracy:" + str(accuracy) + "%")
    
    print("__________________________")
    print("Adam   |LR=0.001 |ITER=100")
    accuracy = optimise_svm(data_tr,targets_tr,lr=0.001, _type="Adam")
    print("accuracy:" + str(accuracy) + "%")    
    
    print("__________________________")
    print("Adam    |LR=0.0001|ITER=100")
    accuracy = optimise_svm(data_tr,targets_tr,lr=0.0001, _type="Adam")
    print("accuracy:" + str(accuracy) + "%")
    
#Main--------------------------
import numpy as np

def main():
    question1()
    #question2()
    
    #x = np.linspace(-10,10,1000)
    #y = rastrigin_function(x)
    #plt.plot(x,y)
    return 0

if __name__ == "__main__":
    main()
    
