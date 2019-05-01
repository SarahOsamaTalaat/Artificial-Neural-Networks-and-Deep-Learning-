# Data Preprocessing
# Importing the libraries
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Dataset_3.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
# Feature scaling: standardization
from sklearn.preprocessing import StandardScaler
standardization = StandardScaler()
X_scaled = standardization.fit_transform(X)
# Partitioning a dataset in training and testing sets

from sklearn.model_selection import LeaveOneOut

LOO = LeaveOneOut()

for train, test in LOO.split(X_scaled):
    print("%s %s" % (train, test))
    
