# -*- coding: utf-8 -*-
"""
@author: Sreeni Jilla
"""
import os
import pydotplus #if we need to use any external .exe files.... Here we are using dot.exe

import io #For i/o operations
import pandas as pd
from sklearn import tree #For Decissin Tree
import graphviz
from sklearn.tree import export_graphviz

#Read Train Data file
titanic_train = pd.read_csv("C:\\Data Science\\Data\\train.csv")
os.chdir("C:/Data Science/Data/")
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


titanic_train.shape 
titanic_train.info() 
titanic_train.describe()

#Let's start the journey with non categorical and non missing data columns
X_titanic_train = titanic_train[['Pclass', 'SibSp', 'Parch']] #X-Axis
y_titanic_train = titanic_train['Survived'] #Y-Axis

#Build the decision tree model
dt = tree.DecisionTreeClassifier()
dt.fit(X_titanic_train, y_titanic_train)

# Visualize the decision tree - Option1
dot_data = export_graphviz(dt, out_file=None, 
                           feature_names=X_train.columns,  
                           filled=True, rounded=True,  
                           special_characters=True)  

# Visualize the decision tree
graph = graphviz.Source(dot_data)  
graph.render("decision_tree")  # Saves the tree as a file
graph.view()  # Opens the file in the default viewer

#visualize the decission tree - Option2
objStringIO = io.StringIO() 
tree.export_graphviz(dt, out_file = objStringIO, feature_names = X_titanic_train.columns)
#Use out_file = objStringIO to getvalues()
file1 = pydotplus.graph_from_dot_data(objStringIO.getvalue())
#os.chdir("D:\\Data Science\\Data\\")
os.getcwd()
file1.write_pdf("DecissionTree2.pdf")

#Predict the outcome using decision tree
#Read the Test Data
titanic_test = pd.read_csv("E:\\Data Science\\Data\\titanic_test.csv")
X_test = titanic_test[['Pclass', 'SibSp', 'Parch']]
#Use .predict method on Test data using the model which we built
titanic_test['Survived'] = dt.predict(X_test) 
os.getcwd() #To get current working directory
titanic_test.to_csv("submission_Titanic.csv", columns=['PassengerId','Survived'], index=False)



