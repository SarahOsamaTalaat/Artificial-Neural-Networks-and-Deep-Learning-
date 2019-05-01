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
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import KFold

# shuffle is used to whether to shuffle the data before splitting into batches.
kfold = KFold(n_splits=3, shuffle=False, random_state=None)

# Returns the number of splitting iterations in the cross-validator
k = kfold.get_n_splits(X) # or # k = kfold.get_n_splits([X,Y,3])
# Generate indices to split data into training and test set.
indices = kfold.split(X)
i =1
for train_index, test_index in kfold.split(X):
    print("The fold number:  ",i);i+=1
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    print("X_train:", X[train_index], "\n Y_train:",Y[train_index])
    print("X_test:", X[test_index], "\n Y_test:",Y[test_index])
        