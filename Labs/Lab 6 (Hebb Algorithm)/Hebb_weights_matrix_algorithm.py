# -*- coding: utf-8 -*-
"""
  Created on Mon Apr  2 21:15:56 2019
  In this code, I implement Hebb Neural Network using alternative method: Weights_Matrix 
  the inputSize,outputSize parameters contain the number of neurons in 
  the input and output layers. 
  So, for example, if we want to create a NN object with 5 neurons in the input layer and  3 neurons in 
  the output layer, we'd do this with the code: net = Network(5,3)
  The weights in the Network object are all claculated based on thetrainig sampels 
  
@author: Sarah Osama
"""

import numpy as np
#import numpy.matlib
class Hebb_Network_Weights_Matrix:
    # The  __init__ is used to initialise newly created instance, and receives parameters
    # the class constructor takes 2 parameters which are inputSize and outputSize
    # and based on the input and output sizes the weight matrix is defined and initialized by zeros
    def __init__(self,inputSize,outputSize):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.weight_matrix = np.zeros((self.inputSize,self.outputSize))    
    
    # Define the activation function as a statistic method
    @staticmethod        
    def activation_function(net):
        if net>0:
            return 1  
        elif net<0:
            return -1
        return 0
    
    # The following function was defined to construct the weights matrix, and also this function takes 2 parameters
    def create_weight_matrix(self,x,y):
        #if we don't have the input and output sizes we use that: weight_matrix = np.zeros((x.shape[0],y.shape[1]))
        #weight_matrix = np.zeros((self.inputSize,self.outputSize))
        for i in range(len(x)):
            self.weight_matrix+=x[i].T* y[i]
            #print(x[i].T.shape, y[i].shape)

    def feedforward_activation(self,x_i):
        #y_hat = np.zeros(self.outputSize)
        y_hat = np.empty(self.outputSize)
       
        for index in range(self.outputSize):
            net=np.multiply(x_i,self.weight_matrix[:,index]).sum()
            y_hat[index]= self.activation_function(net)
            #y_hat[index]= self.activation_function(x_i.T * self.weight_matrix[:,index])
            #print("y_hat ",y_hat[index])
            #print("Weighted sum=",weighted_sum+self.biases)
        return y_hat
    
    def training_phase(self,x,y):
        #self.weight_matrix= self.create_weight_matrix(x,y)
        print("Training phase:" )
        #self.create_weight_matrix(x,y)
        for i in range(len(x)):
            y_hat=self.feedforward_activation(x[i,:]) #,self.weight_matrix
            print("x and y_hat: ",x[i,:], y_hat)
        return y_hat  
     
    def testing_phase(self,x,y):
        print("Testing phase:" )
        #print("Final weights: ",self.weight_matrix)
        print("input ",x)
        print("y ",y)
        for i in range(len(x)):
            y_hat=self.feedforward_activation(x[i,:])
            print("y_hat ", y_hat)  
            
    # The __call__ implements function call operator
    def __call__(self,x_train,y_train,x_test,y_test):
        self.create_weight_matrix(x_train,y_train)
        self.training_phase(x_train,y_train)
        self.testing_phase(x_test,y_test)