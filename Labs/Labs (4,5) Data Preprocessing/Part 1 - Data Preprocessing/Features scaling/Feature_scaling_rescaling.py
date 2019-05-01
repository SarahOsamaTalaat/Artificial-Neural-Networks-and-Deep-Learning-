# Data Preprocessing

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Dataset_3.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Feature scaling: rescaling
#The min-max scaling procedure is implemented in scikit-learn and can be used  
# as follows:
from sklearn.preprocessing import MinMaxScaler
# Here we scale all input features 
minMaxScaled_rescaling = MinMaxScaler()
X_scaled = minMaxScaled_rescaling.fit_transform(X)

