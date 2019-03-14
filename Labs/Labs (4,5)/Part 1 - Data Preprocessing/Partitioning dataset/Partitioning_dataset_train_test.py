# Data Preprocessing

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Dataset_3.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Feature scaling: standardization
from sklearn.preprocessing import StandardScaler
# Here we scale all input features 
standardization = StandardScaler()
X_scaled = standardization.fit_transform(X)

# Partitioning a dataset in training and testing sets
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size = 0.2, random_state = 0)

