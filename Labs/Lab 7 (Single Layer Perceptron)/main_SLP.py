# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 22:48:58 2019

@author: sarah osama
"""
#import G_2_Hebb_algorithm as HNN
#from SLP_DeltaRule import SLP 
import SLP_DeltaRule as S
import numpy as np

x=np.full((4,3), [[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
y=np.full((4,1), [[0],[1],[1],[0]])

ANN=S.SLP(x.shape[1])(x,y)