# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:45:25 2019

@author: sarah
"""
#import G_2_Hebb_algorithm as HNN
from G_2_Hebb_algorithm import Hebb_Network
import numpy as np

x=np.full((4,2), [[1,1],[1,-1],[-1,1],[-1,-1]])
y=np.full((4,1), [[1],[1],[1],[-1]])

"""ANN=HNN.Hebb_Network(x.shape[1])
ANN.training_phase(x,y)
ANN.testing_phase(x,y)"""

ANN=Hebb_Network(x.shape[1])(x,y)