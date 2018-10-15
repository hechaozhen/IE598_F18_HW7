# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 16:04:38 2018

@author: hecha
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier 


df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)


df.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

feature_names = [ 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df.head()
X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values  ##.iloc dataframe取列

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, 
                     stratify=y,
                     random_state=42)


##Part 1: Random forest estimators

for i in (50,100,200,300,400,500):

    random_forest = RandomForestClassifier(n_estimators= i,
    max_depth=5,
    max_features=4,
    random_state=42)


    cv_scores = cross_val_score(random_forest, X_train, y_train, cv = 10)
    print("cv score of ", i, " trees : {:.6f}".format(np.mean(cv_scores)))

##Part 2: Random forest feature importance

rf = RandomForestClassifier(n_estimators= 50,
    max_depth=5,
    max_features=4,
    random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_


# index of greatest to least feature importances
sorted_index = np.argsort(importances)[::-1]
x = range(len(importances))
# create tick labels
labels = np.array(feature_names)[sorted_index]
plt.bar(x, importances[sorted_index], tick_label=labels)
# rotate tick labels to vertical
plt.xticks(rotation=90)
plt.show()


print(" ")


print("**********************************************************************")
print("My name is Chaozhen He")
print("My NetID is: che19")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")
print("**********************************************************************")
print("  ")





