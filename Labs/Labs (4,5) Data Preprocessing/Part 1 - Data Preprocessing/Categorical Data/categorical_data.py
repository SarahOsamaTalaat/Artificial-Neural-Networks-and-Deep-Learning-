# Data Preprocessing
# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Dataset_2.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

# Dealing with categorical data
# Mapping ordinal feature/independent variable 
# Define a new dictionary to store the sizes and its values
size_mapping = {'S':1,'M':2,'L':3,'XL':4,'2XL':5}
dataset['Size'] = dataset['Size'].map(size_mapping)
X[:,2] = dataset.iloc[:,2].values

# Encoding nominal features/independent variable 
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
Country_labelencoder = LabelEncoder()
X[:, 0] = Country_labelencoder.fit_transform(X[:, 0])
Color_labelencoder = LabelEncoder()
X[:, 1] = Color_labelencoder.fit_transform(X[:, 1])

# Encoding nominal features/independent variable 
# by using dummy encoding 
from sklearn.preprocessing import OneHotEncoder
Country_color_onehotencoder = OneHotEncoder(categorical_features = [0,1])
X = Country_color_onehotencoder.fit_transform(X).toarray()


