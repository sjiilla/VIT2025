import numpy as np
from sklearn import decomposition #PCA Package
#from sklearn.decomposition import PCA #Alternative way
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#3 features with 5 records
df1= pd.DataFrame({
      'SibSp':[1,2,3,4,5],
      'FamilySize':[2,4,6,8,10],
      'Age':[200,400,1000,800,100],      
      'Fare':[100,200,300,400,50]})


# 1. Load the dataset (using the built-in Iris dataset)
#titanic_train = pd.read_csv("C:\\Data Science\\Data\\train.csv")
#X_titanic_train = titanic_train[['Pclass', 'SibSp', 'Parch']] #X-Axis
#X_titanic_train.describe()
#X = pd.DataFrame(titanic_train, columns=['Pclass', 'SibSp', 'Parch'])

# 2. Prepare the data for PCA (Standardize the data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df1)
print(X_scaled)

# 3. Apply PCA
pca = PCA() # Initialize PCA with default parameters (no. of components = n_features)
pca.fit(X_scaled)

#titanic_train = pd.read_csv("C:\\Data Science\\Data\\train.csv")
#X_titanic_train = titanic_train[['Pclass', 'SibSp', 'Parch']] #X-Axis
#X_titanic_train.shape
# =============================================================================
# df1= pd.DataFrame({
#         'Age':[10,20,30,40,50],
#         'FamilySize':[2,4,6,8,10],
#         'SibSp':[1,2,3,4,5],        
#         'Fare':[100,200,300,400,500]}) #Age, FamilySize, Fare... Are features
# 
# =============================================================================
pca = decomposition.PCA(n_components=3) #n_components means, transform the data to n dimensions.

#find eigen values and eigen vectors of covariance matrix of df1
#.fit builds PCA model for given fetures to prinicpal components
#Equation: 
#PC1 = Age*w11+FamilySize*w12+Fare*w13.....
#PC2 = Age*w21+FamilySize*w22+Fare*w23.....
#PC3 = Age*w31+FamilySize*w32+Fare*w33.....
pca.fit(X_scaled)
#print(pca.components_)
#convert all the data points from standard basis to eigen vector basis
df1_pca = pca.transform(X_scaled)
print(df1_pca)

#variance of data along original axes
np.var(df1.SibSp) + np.var(df1.Age) + np.var(df1.FamilySize) + np.var(df1.Fare)
#variance of data along principal component axes
#show eigen values of covariance matrix in decreasing order
pca.explained_variance_

np.sum(pca.explained_variance_)

#understand how much variance captured by each principal component
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())


