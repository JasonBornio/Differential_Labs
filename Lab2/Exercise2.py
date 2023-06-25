# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from typing import Tuple
import torch

#Question1---------------------
#input matrix
A_matrix = torch.tensor([[0.3374,0.6005,0.1735],[3.3359,0.0492,1.8374],[2.9407,0.5301,2.2620]], requires_grad=True, dtype=torch.float)
#rank of factorisation
rnk = 2
#number of epochs
N_epochs = 1
#learning rate
l_rate = 0.01

def gd_factorise_ad(A: torch.Tensor, rank : int, num_epochs=1000, lr=0.01):
    U_init = torch.rand(A.shape[0], rank)
    V_init = torch.rand(A.shape[1], rank)
    U = torch.tensor(U_init, requires_grad=True, dtype=torch.float)
    V = torch.tensor(V_init, requires_grad=True, dtype=torch.float)
    
    
    for i in range(0, num_epochs):
        # manually dispose of the gradient (in reality it would be better to detach and zero it to reuse memory)
        U.grad=None
        V.grad=None
        # evaluate the function
        J = function(A,U,V) 
        print(J)
        # auto-compute the gradients at the previously evaluated point x
        J.backward()
        # compute the update
        print(U.grad)
        U.data = U - U.grad*lr 
        V.data = V - V.grad*lr 
        
    return U, V

def function(A, U, V):
    return torch.sum((A - U @ V.t())**2)

def reconstruction_loss(A, U, V):
    return torch.nn.functional.mse_loss(U @ V.t(), A, reduction='sum')

def question1_1():
    print("\nQUESTION 1_1::::::::::::::::\n")
    U_mat, V_mat = gd_factorise_ad(A_matrix, rnk, num_epochs=N_epochs, lr=l_rate)
    print("U_matrix--------------")
    print(U_mat.data)
    print("V_matrix--------------")
    print(V_mat.data)
    print("RECONSTRUCTION--------")
    print((U_mat @ V_mat.t()).data)
    print("MSE LOSS--------------")
    print(reconstruction_loss(A_matrix, U_mat, V_mat).data)

#------------

import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases' + '/iris/iris.data', header=None)
data = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values)
data = data - data.mean(dim=0)

def SVD_reconstrucion(A):
    SVD = torch.svd(A) #svd
    singular_values = torch.diag(SVD.S) #singular values
    #print("SINGULAR VALUES-------")
    #print(singular_values)
    singular_values[singular_values.shape[0] - 1] = 0 #set last singular value to zero
    singular_values[singular_values.shape[0] - 2] = 0 #set second-to-last singular value to zero
    #print("RANK2 SINGULAR VALUES-")
    #print(singular_values)
    reconstruction = SVD.U @ singular_values @ SVD.V.t()
    mse_loss = torch.nn.functional.mse_loss(reconstruction, A, reduction='sum')
    return reconstruction, mse_loss

def question1_2():
    print("\nQUESTION 1_2::::::::::::::::\n")
    U_mat, V_mat = gd_factorise_ad(data, rnk, num_epochs=N_epochs, lr=l_rate)
    print("U_matrix--------------")
    print(U_mat.data)
    print("V_matrix--------------")
    print(V_mat.data)
    print("RECONSTRUCTION--------")
    print((U_mat @ V_mat.t()).data)
    print("MSE LOSS--------------")
    print(reconstruction_loss(data, U_mat, V_mat).data)
    truncated_SVD, loss = SVD_reconstrucion(data)
    #print("SVD RECONSTRUCTION----")
    #print(truncated_SVD)
    print("SVD MSE LOSS----------")
    print(loss)
    
#------------

import matplotlib.pyplot as plt

def scatter(matrix, title='scatter'):
    x,y = [],[]
    for data in matrix:
        x.append(data[0])
        y.append(data[1])
    
    plt.scatter(x,y)
    plt.title(title)
    plt.xlabel('axis1')
    plt.ylabel("axis2")
    plt.legend()
    plt.show()
    plt.figure()

def question1_3():
    print("\nQUESTION 1_3::::::::::::::::\n")
    SVD = torch.svd(data) #svd
    print("SVD U_matrix----------")
    print(SVD.U)
    scatter(SVD.U, "SVD projection")
    
    U_mat, V_mat = gd_factorise_ad(data, rnk, num_epochs=N_epochs, lr=l_rate)
    print("U_matrix--------------")
    print(U_mat.data)
    scatter(U_mat.data, "matrix factorisation projection")
 
#Question2---------------------
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases' + '/iris/iris.data', header=None)
df.sample(frac=1) #shuffle

#add label indices column
mapping = {k: v for v, k in enumerate(df[4].unique())}
df[5] = df[4].map(mapping)
#normalise data
alldata = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values, dtype=torch.float)
alldata = (alldata - alldata.mean(dim=0)) / alldata.var(dim=0)
#create datasets
targets_tr = torch.tensor(df.iloc[:100, 5].values, dtype=torch.long)
targets_va = torch.tensor(df.iloc[100:, 5].values, dtype=torch.long)
data_tr = alldata[:100]
data_va = alldata[100:]

#MLP

def MLP(data_in, targets_in, num_epochs=100, lr=0.01):    
    W1_init = torch.rand(4, 12)
    W2_init = torch.rand(12, 3)
    b_init = torch.tensor([0])
    W1 = torch.tensor(W1_init, requires_grad=True, dtype=torch.float)
    W2 = torch.tensor(W2_init, requires_grad=True, dtype=torch.float)
    b1 = torch.tensor(b_init, requires_grad=True, dtype=torch.float)
    b2 = torch.tensor(b_init, requires_grad=True, dtype=torch.float)
    
    for i in range(0, num_epochs):
        # manually dispose of the gradient (in reality it would be better to detach and zero it to reuse memory)
        W1.grad=None
        W2.grad=None
        b1.grad=None
        b2.grad=None
        # evaluate the function
        logits = torch.relu(data_in @ W1 + b1) @ W2 + b2
        J = torch.nn.functional.cross_entropy(logits, targets_in)
        # auto-compute the gradients at the previously evaluated point x
        J.backward()
        # compute the update
        W1.data = W1 - W1.grad*lr 
        W2.data = W2 - W2.grad*lr 
        b1.data = b1 - b1.grad*lr 
        b2.data = b2 - b2.grad*lr 
    
    return W1,W2,b1,b2,J

def accuracy(predictions, labels):
    count = 0
    total = 0
    for prediction, label in zip(predictions, labels):
        total += 1
        if (torch.argmax(prediction) == label):
            count +=1
            
    return (count/total) * 100
        
def question2():
    print("\nQUESTION 2::::::::::::::::\n")
    weights1,weights2,bias1,bias2,loss = MLP(data_tr,targets_tr,1000,0.01) 
    logits_trained = torch.relu(data_va @ weights1 + bias1) @ weights2 + bias2
    #print("Validation Data-------")    
    #print(targets_va)
    #print("logits_trained--------")    
    #print(logits_trained.data)
    print("CROSS ENTROPY LOSS----")
    print(loss.data)
    
    print("ACCURACY--------------")
    print("________________________")
    print("LR=0.1    |ITER=100")
    weights1,weights2,bias1,bias2,loss = MLP(data_tr,targets_tr,100,0.1)
    logits_trained = torch.relu(data_tr @ weights1 + bias1) @ weights2 + bias2
    print("training  |" + str(accuracy(logits_trained, targets_tr)) +"%")
    weights1,weights2,bias1,bias2,loss = MLP(data_va,targets_va,100,0.1)
    logits_trained = torch.relu(data_va @ weights1 + bias1) @ weights2 + bias2
    print("validation|" + str(accuracy(logits_trained, targets_va))+"%")
    
    print("__________|_____________\nLR=0.01   |ITER=100")
    weights1,weights2,bias1,bias2,loss = MLP(data_tr,targets_tr,100,0.01)
    logits_trained = torch.relu(data_tr @ weights1 + bias1) @ weights2 + bias2
    print("training  |" + str(accuracy(logits_trained, targets_tr))+"%")
    weights1,weights2,bias1,bias2,loss = MLP(data_va,targets_va,100,0.01)
    logits_trained = torch.relu(data_va @ weights1 + bias1) @ weights2 + bias2
    print("validation|" + str(accuracy(logits_trained, targets_va))+"%")
    
    print("__________|_____________\nLR=0.001  |ITER=100")
    weights1,weights2,bias1,bias2,loss = MLP(data_tr,targets_tr,100,0.001)
    logits_trained = torch.relu(data_tr @ weights1 + bias1) @ weights2 + bias2
    print("training  |" + str(accuracy(logits_trained, targets_tr))+"%")
    weights1,weights2,bias1,bias2,loss = MLP(data_va,targets_va,100,0.001)
    logits_trained = torch.relu(data_va @ weights1 + bias1) @ weights2 + bias2
    print("validation|" + str(accuracy(logits_trained, targets_va))+"%")

    print("__________|_____________\nLR=0.1    |ITER=1000")
    weights1,weights2,bias1,bias2,loss = MLP(data_tr,targets_tr,1000,0.1)
    logits_trained = torch.relu(data_tr @ weights1 + bias1) @ weights2 + bias2
    print("training  |" + str(accuracy(logits_trained, targets_tr))+"%")
    weights1,weights2,bias1,bias2,loss = MLP(data_va,targets_va,1000,0.1)
    logits_trained = torch.relu(data_va @ weights1 + bias1) @ weights2 + bias2
    print("validation|" + str(accuracy(logits_trained, targets_va))+"%")

    print("__________|_____________\nLR=0.01   |ITER=1000")
    weights1,weights2,bias1,bias2,loss = MLP(data_tr,targets_tr,1000,0.01)
    logits_trained = torch.relu(data_tr @ weights1 + bias1) @ weights2 + bias2
    print("training  |" + str(accuracy(logits_trained, targets_tr))+"%")
    weights1,weights2,bias1,bias2,loss = MLP(data_va,targets_va,1000,0.01)
    logits_trained = torch.relu(data_va @ weights1 + bias1) @ weights2 + bias2
    print("validation|" + str(accuracy(logits_trained, targets_va))+"%")

    print("__________|_____________\nLR=0.001  |ITER=1000")
    weights1,weights2,bias1,bias2,loss = MLP(data_tr,targets_tr,1000,0.001)
    logits_trained = torch.relu(data_tr @ weights1 + bias1) @ weights2 + bias2
    print("training  |" + str(accuracy(logits_trained, targets_tr))+"%")
    weights1,weights2,bias1,bias2,loss = MLP(data_va,targets_va,1000,0.001)
    logits_trained = torch.relu(data_va @ weights1 + bias1) @ weights2 + bias2
    print("validation|" + str(accuracy(logits_trained, targets_va))+"%")
    return
    
#Main--------------------------

def main():
    question1_1()
    #question1_2()
    #question1_3()
    #question2()
    return 0

if __name__ == "__main__":
    main()
    
