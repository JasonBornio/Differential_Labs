# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#block1---------------------
import torch

# Generate some data points on a straight line perturbed with Gaussian noise
N = 1000 # number of points
theta_true = torch.Tensor([[1.5], [2.0]]) # true parameters of the line

X = torch.rand(N, 2) 
X[:, 1] = 1.0
y = X @ theta_true + 0.1 * torch.randn(N, 1) # Note that just like in numpy '@' represents matrix multiplication and A@B is equivalent to torch.mm(A, B) 
#block2---------------------
import matplotlib.pyplot as plt

plt.scatter(X[:,0].numpy(), y.numpy())
plt.show()

# direct solution using moore-penrose pseudo inverse
X_inv = torch.pinverse(X)
theta_pinv = torch.mm(X_inv, y)
print(theta_pinv)
#block3---------------------
def psuedoinverse(mat):
  mat_svd = torch.svd(mat)
  VE = torch.mm(mat_svd.V, (torch.diag(1/mat_svd.S)))
  return torch.mm(VE, mat_svd.U.t())

X_inv_svd = psuedoinverse(X)

theta_pinv_svd = torch.mm(X_inv_svd, y)
print(theta_pinv_svd)
#print(X)
#print(X_svd)
#block---------------------
assert(torch.all(torch.lt(torch.abs(torch.add(theta_pinv, -theta_pinv_svd)), 1e-6)))
#block4---------------------
def linear_regression_loss_grad(theta, X, y):
    # theta, X and y have the same shape as used previously
    # YOUR CODE HERE
    #raise NotImplementedError()

    grad = X.t() @ (X @ theta - y)




    return grad

print(linear_regression_loss_grad(torch.zeros(2,1), X, y))
#block5---------------------
alpha = 0.001
theta = torch.Tensor([[0], [0]])
for e in range(0, 200):
    gr = linear_regression_loss_grad(theta, X, y)
    theta -= alpha * gr

print(theta)

#block6---------------------
from sklearn.datasets import load_diabetes

Data = tuple(torch.Tensor(z) for z in load_diabetes(return_X_y=True)) #convert to pytorch Tensors
X, y = Data[0], Data[1]
X = torch.cat((X, torch.ones((X.shape[0], 1))), 1) # append a column of 1's to the X's
y = y.reshape(-1, 1) # reshape y into a column vector
print('X:', X.shape)
print('y:', y.shape)

# We're also going to break the data into a training set for computing the regression parameters
# and a test set to evaluate the predictive ability of those parameters
perm = torch.randperm(y.shape[0])
X_train = X[perm[0:253], :]
y_train = y[perm[0:253]]
X_test = X[perm[253:], :]
y_test = y[perm[253:]]

#block7---------------------

# compute the regression parameters in variable theta
# YOUR CODE HERE
theta = psuedoinverse(X_train) @ y_train
print(theta)

print(theta.shape)
#block8---------------------
assert(theta.shape == (11,1))

print("Theta: ", theta.t())
print("MSE of test data: ", torch.nn.functional.mse_loss(X_test @ theta, y_test))
#block9---------------------
alpha = 0.0001
theta_gd = torch.rand((X_train.shape[1], 1))
for e in range(0, 100000):
    gr = linear_regression_loss_grad(theta_gd, X_train, y_train)
    theta_gd -= alpha * gr

print("Gradient Descent Theta: ", theta_gd.t())
print("MSE of test data: ", torch.nn.functional.mse_loss(X_test @ theta_gd, y_test))
# YOUR CODE HERE
#increasing either learning rate or iterations increases accuracy, too big an increase leads to overfitting
#raise NotImplementedError()
#block10---------------------
perm = torch.argsort(y_test, dim=0)
plt.plot(y_test[perm[:,0]].numpy(), '.', label='True Prices')
plt.plot((X_test[perm[:,0]] @ theta).numpy(), '.', label='Predicted (pinv)')
plt.plot((X_test[perm[:,0]] @ theta_gd).numpy(), '.', label='Predicted (G.D.)')
plt.xlabel('Patient Number')
plt.ylabel('Quantitative Measure of Disease Progression')
plt.legend()
plt.show()
#block11---------------------


