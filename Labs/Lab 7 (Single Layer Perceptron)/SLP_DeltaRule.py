# -*- coding: utf-8 -*-
"""
  Created on Wed Apr 17 18:13:40 2019
  In this code, I implement Single Layer Perceptron 
  the inputSize,outputSize parameters contain the number of neurons in the input and output layers. 
  Learning_rate refers to the learning rate and epochs parameter refers to numper of epochs 
  So, for example, if we want to create a NN object with 5 neurons in the input layer,  3 neurons in 
  the output layer, learning rate =0.7 and number of epochs =100 we'd do this with the code: net = SLP(5,3,0.7,100)
  
  
@author: Sarah Osama
"""

import numpy as np
#import numpy.matlib
class SLP:
    # The  __init__ is used to initialise newly created instance, and receives parameters
    # the class constructor takes 2 parameters which are inputSize and outputSize
    # and based on the input and output sizes the weight matrix is defined and initialized by zeros
    def __init__(self,inputSize,learningRate=0.01, epochs=50):
        self.inputSize = inputSize
        #self.outputSize = outputSize    
        self.learningRate = learningRate
        self.epochs = epochs
        self.biase= 0
        self.weights = np.zeros(self.inputSize)
        #self.weights = [np.random.randn(y, x) for x, y in
        #self.weights = np.zeros(self.inputSize+1)    
        
    
    # Define the activation function as a statistic method
    @staticmethod        
    def activation_function(net):
        if net>0:
            return 1  
        return 0
    
    def feedforward_activation(self,x_i):
        #weighted_input = self.weights * x_i 
        #weighted_sum = weighted_input.sum()
        net = (self.weights * x_i).sum()+self.biase
        y_hat= self.activation_function(net) 
        #print("Y_hat ",y_hat)
        #y_hat = self.activation_function(weighted_sum+self.biases)
        return y_hat

    def learning(self,deltaTerm,o_pi):
        deltaWeights = self.learningRate * deltaTerm * o_pi
        deltaBiase = self.learningRate * deltaTerm  # here the o_pi refers to biase input which is equal 1
        self.weights += deltaWeights
        self.biase += deltaBiase
    
    #Update the network's weights and biases by applying Stochastic gradient descent using backpropagation    
    def training_phase(self,x,y):
        print("Training phase using SGD:" ) 
        wChange = True
        current_epoch = 1
        while (wChange == True) and (current_epoch <= self.epochs):            
            print("Epoch:",current_epoch) 
            wChange = False
            for i in range(len(x)):
                y_hat=self.feedforward_activation(x[i,:]) 
                deltaTerm= y[i]-y_hat
                print("Delta term:",deltaTerm) 
                if deltaTerm != 0.0:
                    self.learning(deltaTerm,x[i,:])
                    wChange = True
                print("x, y and y_hat: ",x[i,:],y[i], y_hat)
            current_epoch +=1
        return y_hat  
     
    def testing_phase(self,x,y):
        print("Testing phase:" )
        print("input ",x)
        print("y ",y)
        for i in range(len(x)):
            y_hat=self.feedforward_activation(x[i,:])
            print("y_hat ", y_hat)  


    # The __call__ implements function call operator
    def __call__(self,x_train,y_train):
        self.training_phase(x_train,y_train)
  
        
    #def __call__(self,x_train,y_train,x_test,y_test):
     #   self.training_phase(x_train,y_train)
      #  self.testing_phase(x_test,y_test)