# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:51:42 2019
@author: sarah
F5 to set the working directory 
ctrl+enter to run the current cell
"""
# Data Preprocessing

# Importing the libraries
import pandas #as pd

# Importing the dataset
dataset = pandas.read_csv('Dataset.csv')

# Now we have a dataset but actually we need to distinguish ... 
# the predictors matrix and the response vector

# Create a matrix to store the predictor (independent) variables
X = dataset.iloc[:, :-1].values

# Create a victore to store the response (dependent) variable
Y = dataset.iloc[:, -1].values
# Dealing with missing data by the eliminating method
# Remove the rows that have missing cells
#D = dataset.dropna()
# Remove the columns that have missing cells
#dataset.dropna(axis=1)

# Dealing with missing data by using mean, median or most frequent 
from sklearn.preprocessing import Imputer

# strategy can be "mean", "median"  or "most_frequent"
#If "mean", then replace missing values using the mean along the axis.
#If “median”, then replace missing values using the median along the axis.
#If “most_frequent”, then replace missing using the most frequent value along the axis.

# Create an object/instance from Imputer class 
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# Fit the imputer object/instance by our predictor matrix X which include 
# the missing values
imputer = imputer.fit(X[:, 2:3])
# Create and fit imputer object/instance
# imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0).fit(X[:,2:3])

# transform function is a function in the imputer class which is used to fill
# the missing values in our predictor matrix by the mean method
X[:, 2:3] = imputer.transform(X[:, 2:3])
DS = [X[:,0],X[:,1],X[:,2],X[:,3],Y]


# For find an item in matrix 
    #for row, item in enumerate(X):
    #for col, element in enumerate(item):
     #   if element=='\t?':
      #      print (row, col)

#for index, item in enumerate(arr):
#    if item > 100:
 #       return index, item

# Write in Excel
#worksheet = workbook.add_worksheet()
#row = 0
#for col, data in enumerate(DS):
 #   worksheet.write_column(row, col, data)
#workbook.close()
