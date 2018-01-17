import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



import seaborn as sns
iris = sns.load_dataset('iris')

iris.keys()

from sklearn.model_selection import train_test_split

X = iris.drop('species',axis=1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)

from sklearn.metrics import classification_report,confusion_matrix

#Gridsearch

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)

print(grid.best_params_)

grid.best_estimator_
grid_predictions = grid.predict(X_test)

print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))


