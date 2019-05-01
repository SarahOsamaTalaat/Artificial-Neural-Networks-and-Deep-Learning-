# -*- coding: utf-8 -*-
"""
  Created on Mon Apr  1 21:15:56 2019
  In this code, I implement Hebb Neural Network using alternative method: Weights_Matrix 
  the inputSize,outputSize parameters contain the number of neurons in 
  the input and output layers. 
  So, for example, if we want to create a NN object with 5 neurons in the input layer and  3 neurons in 
  the output layer, we'd do this with the code: net = Network(5,3)
  The weights in the Network object are all claculated based on thetrainig sampels 
  
@author: Sarah Osama
"""
import numpy as np

class Hebb_Network():
    # The  __init__ is used to initialise newly created instance, and receives parameters
    # the class constructor takes 2 parameters which are biases and weights 
    def __init__(self,biases=None,weights=None):
        input_units=len(weights)
        if weights is None:
            input_units=2
            self.weights = np.zeros(input_units)
        else:
            self.weights = weights
        if biases is None:
            output_units=1
            self.biases = np.zeros(output_units)
        else:
            self.biases = biases
    # Define the activation function as a statistic method   
    @staticmethod        
    def binary_activation(net):
        #print("net",net)
        if net>=0:
            return 1  
        return -1
        
    # The following function was defined to construct the weights matrix, and also this function takes 2 parameters
    def activation_feedforward(self,x_i):
            weighted_input = self.weights * x_i 
            weighted_sum = weighted_input.sum()
            y_hat = self.binary_activation(weighted_sum+self.biases)
            return y_hat

    def learning(self,x,y):
        delta_weights=x*y
        self.weights+=delta_weights
        self.biases+=y

    def training_phase(self,x,y):
        for i in range(len(x)):
            y_hat=self.activation_feedforward(x[i,:])
            self.learning(x[i,:],y[i])
            #print(x[i,:], y_hat)
            
    def testing_phase(self,x,y):
        print("Final weights: ",self.weights)
        print("Final biases: ",self.biases)
        for i in range(len(x)):
            y_hat=self.activation_feedforward(x[i,:])
            print(x[i,:], y_hat)     

    def __call__(self,x,y):
        self.training_phase(x,y)
        self.testing_phase(x,y)