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

def logistic_regression_loss_grad(theta, X, y):
    # YOUR CODE HERE
    #raise NotImplementedError()
    grad = X.t() @ (torch.sigmoid(X @ theta) - y)
    return grad
#block2---------------------
theta = torch.zeros(1)
X = torch.Tensor([[1]])
y = torch.Tensor([[0]])
assert(logistic_regression_loss_grad(theta, X, y) == 0.5)
print(logistic_regression_loss_grad(theta, X, y))
#block3---------------------
from sklearn.datasets import load_digits

X, y = tuple(torch.Tensor(z) for z in load_digits(return_X_y=True)) #convert to pytorch Tensors
X = torch.cat((X, torch.ones((X.shape[0], 1))), 1) # append a column of 1's to the X's
y = y.reshape(-1, 1) # reshape y into a column vector

# We're also going to break the data into a training set for computing the regression parameters
# and a test set to evaluate the predictive ability of those parameters
perm = torch.randperm(y.shape[0])
X_train = X[perm[0:260], :]
y_train = y[perm[0:260]]
X_test = X[perm[260:], :]
y_test = y[perm[260:]]
#block4---------------------
alpha = 0.001
theta_gd = torch.rand((X_train.shape[1], 1))
for e in range(0, 10):
    gr = logistic_regression_loss_grad(theta_gd, X_train, y_train)
    theta_gd -= alpha * gr
    print("Epoch:", e, " BCE of training data:", torch.nn.functional.binary_cross_entropy_with_logits(X_train @ theta_gd, y_train))

print("Gradient Descent Theta:", theta_gd.t())
print("BCE of test data:", torch.nn.functional.binary_cross_entropy_with_logits(X_test @ theta_gd, y_test))
#block5---------------------
from random import randrange

def grad_check(f, x, analytic_grad, num_checks=10, h=1e-5):
    sum_error = 0
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape]) #randomly sample value to change

        oldval = x[ix].item()
        x[ix] = oldval + h # increment by h
        fxph = f(x) # evaluate f(x + h)
        x[ix] = oldval - h # increment by h
        fxmh = f(x) # evaluate f(x - h)
        x[ix] = oldval # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic) + 1e-8)
        sum_error += rel_error
        print('numerical: %f\tanalytic: %f\trelative error: %e' % (grad_numerical, grad_analytic, rel_error))
    return sum_error / num_checks
#block6---------------------
#we'll use random parameters:
theta = torch.rand_like(theta_gd)*0.001
# and compute the analytic gradient (w.r.t the test data we loaded in this case)
grad = logistic_regression_loss_grad(theta, X_test, y_test)

# we need a function that computes the loss for a given theta (and implicitly the data)
def func(th):
    sigm = torch.sigmoid(X_test @ th)
    f = -(y_test.t() @ torch.log(sigm) + (1 - y_test.t()) @ torch.log(1 - sigm));
    return f

# and run the gradient checker
relerr = grad_check(func, theta, grad)
print("average error:", relerr)

assert relerr < 1e-5

#block7---------------------
#block8---------------------
#block9---------------------
#block10---------------------
#block11---------------------
#block12---------------------
#block13---------------------
#block14---------------------

