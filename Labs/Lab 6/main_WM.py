# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:45:25 2019

@author: sarah
"""
import Hebb_weights_matrix_algorithm as HNN
import numpy as np

x_train=np.matrix([[1,-1,-1,-1],[-1,1,-1,-1],[-1,-1,1,-1],[-1,-1,-1,1]])
#x_train=np.matrix('1,-1,-1,-1;-1,1,-1,-1;-1,-1,1,-1;-1,-1,-1,1')

y_train=np.matrix('1,-1,-1;1,-1,1;-1,1,-1;-1,1,1')

x_test=np.matrix('-1,-1,-1,-1')
y_test=np.matrix([0,0,0])

Hebb_Net_A_WM = HNN.Hebb_Network_Weights_Matrix(x_train.shape[1],y_train.shape[1])(x_train,y_train,x_test,y_test)

#ANN=HNN.Hebb_Network([0],[0,0])(x,y)
#ANN(x,y)

#y_hat=HNN.Hebb_Net.activation_feedforward(x)

