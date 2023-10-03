"""Feature Selection"""

###The file ActivityRecognition_SAWA contains all the 71 features avialable in the dataset from the smartphone and smartwatch accelerometers
###  and the activity label, as obtained by processing the initial dataset files using the preprocessong_activity.py file

""" *************************Random Forest Feature Selection*************************************"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the CSV data
data = pd.read_csv('ActivityRecognition_SAWA.csv')

# Split the data into features (X) and target (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column

# Train a Random Forest classifier and rank the features based on their importance scores
clf = RandomForestClassifier()
clf.fit(X, y)
importances = clf.feature_importances_
ranked_features = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
ranked_features = ranked_features.sort_values('Importance', ascending=False)

print('Random Forest')
#print(ranked_features)
print(ranked_features[0:40])
#print(ranked_features[0:72])

"""*********************************Decision Tree Feature Selection***********************************************"""
from sklearn.tree import DecisionTreeClassifier

# Load the CSV data
data = pd.read_csv('ActivityRecognition_SAWA.csv')

# Split the data into features (X) and target (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column

# Train a Decision Tree classifier and rank the features based on their importance scores
clf = DecisionTreeClassifier()
clf.fit(X, y)
importances = clf.feature_importances_
ranked_features = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
ranked_features = ranked_features.sort_values('Importance', ascending=False)

print('Decision Tree')
#print(ranked_features)
print(ranked_features[0:40])

"""*************************Logistic Regression feature selection****************************"""


from sklearn.linear_model import LogisticRegression

# Load the CSV data
data = pd.read_csv('ActivityRecognition_SAWA.csv')

# Split the data into features (X) and target (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column

# Train a Logistic Regression classifier and rank the features based on their coefficients
clf = LogisticRegression()
clf.fit(X, y)
coefficients = clf.coef_[0]
ranked_features = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
ranked_features = ranked_features.sort_values('Coefficient', ascending=False)

print('Logistic Regression')
#print(ranked_features)
print(ranked_features[0:40])

"""*******************************Support Vector Machine Feature Selection*********************************"""


from sklearn.svm import LinearSVC

# Load the CSV data
data = pd.read_csv('ActivityRecognition_SAWA.csv')

# Split the data into features (X) and target (y)
X = data.iloc[:, :-1]  # All columns except the last one
y = data.iloc[:, -1]   # The last column

# Train a Linear Support Vector Machine classifier and rank the features based on their coefficients
clf = LinearSVC(penalty='l1', dual=False)
clf.fit(X, y)
coefficients = clf.coef_[0]
ranked_features = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
ranked_features = ranked_features.sort_values('Coefficient', ascending=False)

print('Support Vector Machine')
#print(ranked_features)
print(ranked_features[0:40])