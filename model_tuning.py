"""Tuning model parameters with GridSearchCV"""



"""****************Applying GridSearchCV for Random Forest************************"""

# import required libraries
##In this code, we first import the required libraries, read the input CSV file, and split the data into features and target. Then, we define a parameter grid that contains different values for the hyperparameters of the Random Forest algorithm. We create an instance of the Random Forest classifier and perform a grid search with 5-fold cross-validation to find the best set of parameters. Finally, we extract the best parameters and score and print them on the console.

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# read the CSV file
data = pd.read_csv('Dataset_Selected_features_SA-WA_ActivityRecognition.csv')

# split the data into features (X) and target (y)
X = data.drop('activity', axis=1)
y = data['activity']

# define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 4]
}

# create an instance of Random Forest classifier
rfc = RandomForestClassifier()

# perform a grid search with 5-fold cross-validation
grid_search = GridSearchCV(rfc, param_grid, cv=5)

# fit the grid search object to the data
grid_search.fit(X, y)

# extract the best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# print the best parameters and score
print("Best parameters: ", best_params)
print("Best score: ", best_score)

"""*********************************Applying GridSearchCV for KNN*****************************************"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Load CSV file into a pandas dataframe
data = pd.read_csv('Dataset_Selected_features_SA-WA_ActivityRecognition.csv')

# Separate the target variable from the rest of the data
X = data.drop('target_variable', axis=1)
y = data['target_variable']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid to search over
param_grid = {'n_neighbors': [3, 5, 7, 9, 11],
              'weights': ['uniform', 'distance'],
              'p': [1, 2]}

# Create a KNN classifier object
knn = KNeighborsClassifier()

# Create a GridSearchCV object to search over the parameter grid
grid_search = GridSearchCV(knn, param_grid=param_grid, cv=5)

# Fit the GridSearchCV object to the training data
grid_search.fit(X_train, y_train)

# Print the best set of parameters found by the grid search
print("Best parameters:", grid_search.best_params_)

# Print the accuracy score of the best KNN classifier
print("Best score:", grid_search.best_score_)

#We then define a parameter grid to search over, which includes the number of neighbors, the weighting scheme, and the distance metric. We create a KNN classifier object and a GridSearchCV object to search over the parameter grid. We fit the GridSearchCV object to the training data, and print the best set of parameters found by the grid search, as well as the accuracy score of the best KNN classifier.

"""************************************Applying GridSearchCV for MLP**************************"""

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score

# Load the data from the CSV file
data = pd.read_csv('Dataset_Selected_features_SA-WA_ActivityRecognition.csv')

# Split the data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid to search over
param_grid = {
    'hidden_layer_sizes': [(10,), (50,), (100,), (10, 10), (50, 50), (100, 100)],
    'activation': ['relu', 'logistic'],
    'solver': ['adam', 'sgd'],
    'learning_rate': ['constant', 'adaptive']
}

# Create an MLP classifier object
mlp = MLPClassifier(max_iter=1000)

# Use GridSearchCV to find the best set of parameters
grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best set of parameters
print("Best Parameters: ", grid_search.best_params_)

# Use the best set of parameters to train and test the model
best_mlp = MLPClassifier(**grid_search.best_params_, max_iter=1000)
best_mlp.fit(X_train, y_train)
y_pred = best_mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy of the model
print("Accuracy: {:.2f}%".format(accuracy * 100))
