# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#block1---------------------
import torch

# we wouldn't normally do this, but for this lab we want to work in double precision
# as we'll need the numerical accuracy later on for doing checks on our gradients:
torch.set_default_dtype(torch.float64) 

def probabiltiy(Theta, X, y):
  numerator = torch.exp(Theta.t() * X)
def softmax_regression_loss_grad(Theta, X, y):
    '''Implementation of the gradient of the softmax loss function.
    
    Theta is the matrix of parameters, with the parameters of the k-th class in the k-th column
    X contains the data vectors (one vector per row)
    y is a column vector of the targets
    '''
    # YOUR CODE HERE
    #raise NotImplementedError()
    grad = 0

    return grad

def softmax_regression_loss(Theta, X, y):
    '''Implementation of the softmax loss function.
        
    Theta is the matrix of parameters, with the parameters of the k-th class in the k-th column
    X contains the data vectors (one vector per row)
    y is a column vector of the targets
    '''
    # YOUR CODE HERE

    probability = 0

    for k in range(y.size):
      if y == k:
        for xi in X:
          probabilty += torch.log(torch.exp(Theta[k].t() @ xi))/(torch.exp(Theta.t() @ xi))

    loss = -probability


    #raise NotImplementedError()

    return loss
#block2---------------------
#block3---------------------
#block4---------------------
#block5---------------------
#block6---------------------
#block7---------------------
#block8---------------------
#block9---------------------
#block10---------------------
#block11---------------------
#block12---------------------
#block13---------------------
#block14---------------------

