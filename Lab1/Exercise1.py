# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from typing import Tuple
import torch

#Question1---------------------
#input matrix
A_matrix = torch.tensor([[0.3374,0.6005,0.1735],[3.3359,0.0492,1.8374],[2.9407,0.5301,2.2620]])
#rank of factorisation
rnk = 2
#number of epochs
N_epochs = 1000
#learning rate
l_rate = 0.01

def sgd_factorise(A: torch.Tensor, rank: int, num_epochs = 1000, lr = 0.01):
    m = A.shape[0]
    n = A.shape[1]
    
    U = torch.rand(m, rank) 
    V = torch.rand(n, rank)
    
    for epoch in range(num_epochs):
        for r in range(m):
            for c in range(n):
                e = A[r][c] - U[r] @ V[c].t()
                U[r] = U[r] + lr * e * V[c]
                V[c] = V[c] + lr * e * U[r]
    
    return U, V 

def reconstruction_loss(A, U, V):
    return torch.nn.functional.mse_loss(U @ V.t(), A, reduction='sum')

def question1():
    print("\nQUESTION 1::::::::::::::::\n")
    U_mat, V_mat = sgd_factorise(A_matrix, rnk, num_epochs=N_epochs, lr=l_rate)
    print("U_matrix--------------")
    print(U_mat)
    print("V_matrix--------------")
    print(V_mat)
    print("RECONSTRUCTION--------")
    print(U_mat @ V_mat.t())
    print("MSE LOSS--------------")
    print(reconstruction_loss(A_matrix, U_mat, V_mat))
    return 
    
#Question2---------------------

def SVD_reconstrucion(A):
    SVD = torch.svd(A) #svd
    singular_values = torch.diag(SVD.S) #singular values
    singular_values[singular_values.shape[0] - 1] = 0 #set last singular value to zero
    reconstruction = SVD.U @ singular_values @ SVD.V.t()
    mse_loss = torch.nn.functional.mse_loss(reconstruction, A, reduction='sum')
    return reconstruction, mse_loss

def question2():
    print("\nQUESTION 2::::::::::::::::\n")
    truncated_SVD, loss = SVD_reconstrucion(A_matrix)
    print("SVD RECONSTRUCTION----")
    print(truncated_SVD)
    print("MSE LOSS--------------")
    print(loss)
    
#Question3---------------------

#input mask
mask = torch.tensor([[1,1,1],[0,1,1],[1,0,1]])

def sgd_factorise_masked(A: torch.Tensor, M: torch.Tensor, rank: int, num_epochs = 1000, lr = 0.01):
    m = A.shape[0]
    n = A.shape[1]
    
    U = torch.rand(m, rank) 
    V = torch.rand(n, rank)
    
    for epoch in range(num_epochs):
        for r in range(m):
            for c in range(n):
                if M[r][c] == 1:
                    e = A[r][c] - U[r] @ V[c].t()
                    U[r] = U[r] + lr * e * V[c]
                    V[c] = V[c] + lr * e * U[r]
    
    return U, V 

def question3():
    print("\nQUESTION 3::::::::::::::::\n")
    U_mat, V_mat = sgd_factorise_masked(A_matrix, mask, rnk, num_epochs=N_epochs, lr=l_rate)
    print("U_matrix--------------")
    print(U_mat)
    print("V_matrix--------------")
    print(V_mat)
    print("RECONSTRUCTION--------")
    print(U_mat @ V_mat.t())
    print("MSE LOSS--------------")
    print(reconstruction_loss(A_matrix, U_mat, V_mat))
    
#Question4---------------------

import csv

def gd_factorise_masked(A: torch.Tensor, M: torch.Tensor, rank : int, num_epochs : int=1000, lr : float=1e-5):
    U = torch.rand(A.shape[0], rank)
    V = torch.rand(A.shape[1], rank)
    
    for e in range (num_epochs) :
        err = (A - U @ V.t()) * M
        U += lr * err @ V
        V += lr * err.t() @ U
        
    return U, V

def load_titles():
    titles = []
    with open('titles.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in spamreader:
            titles.append(' '.join(row))
            
    return titles

def masked_reconstruction_loss(A, U, V,M):
    #return torch.nn.functional.mse_loss((U @ V.t())*M, A * M, reduction='sum')
    return  torch.sum((A * M - (U @ V.t())*M)**2)

def question4():
    print("\nQUESTION 4::::::::::::::::\n")
    movie_data = torch.load('ratings.pt') #load data
    movie_titles = load_titles() #load titles
    rnk = 5 #rank 5
    mask = (movie_data > 0).int()
    U_mat, V_mat = gd_factorise_masked(movie_data, mask, rnk)
    print("U_matrix--------------")
    print(U_mat)
    print("V_matrix--------------")
    print(V_mat)
    print("RECONSTRUCTION--------")
    print(U_mat @ V_mat.t())
    print("ORIGINAL_MATRIX-------")
    print(movie_data)
    print("SSE LOSS--------------")
    print(masked_reconstruction_loss(movie_data, U_mat, V_mat,mask))
    print("5th USER RATING-------")
    print((U_mat @ V_mat.t())[4][124])
    print(movie_data[4][124])
        
#Main--------------------------

def main():
    #question1()
    #question2()
    question3()
    #question4()
    return 0

if __name__ == "__main__":
    main()
    
