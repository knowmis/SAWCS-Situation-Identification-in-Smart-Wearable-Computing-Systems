"""situation_identification"""

### The file contains the code of the 5 different classifiers compared for the identification of situations
### in the comprehension phase.

### The file Dataset_with_ContextSpaceFeatures_SituationIdentification.csv contains the following features:
### -activity predicted by the high level perception phase
### - context attrubute values for the context attributes included in the definition of the context space
### - the target is the label of one of the 14 situations included in S. 

"""**********************************************Decision Tree**********************************"""

# Import necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold

# Load data
df = pd.read_csv('Dataset_with_ContextSpaceFeatures_SituationIdentification.csv')

#le = LabelEncoder()
#df['Prediction'] = le.fit_transform(df['Prediction'])
#data['context_article'] = le.fit_transform(data['context_article'])

#X = df.iloc[:, :-1]
#y = df.iloc[:, -1]
# Split the data into features and target
X = df.drop('context_article_code', axis=1)
y = df['context_article_code']

# Define the decision tree model with best hyperparameters
dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=4, min_samples_split=5,criterion='gini')

# Train the model using 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
balanced_accuracy_scores = []
confusion_matrices = []

for train_index, test_index in cv.split(X, y):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model and make predictions on the test set
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    # Calculate evaluation metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
    recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))
    confusion_matrices.append(confusion_matrix(y_test, y_pred))

# Calculate mean evaluation metrics across all folds
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
mean_precision = sum(precision_scores) / len(precision_scores)
mean_recall = sum(recall_scores) / len(recall_scores)
mean_f1 = sum(f1_scores) / len(f1_scores)
mean_balanced_accuracy = sum(balanced_accuracy_scores) / len(balanced_accuracy_scores)
mean_confusion_matrix = sum(confusion_matrices) / len(confusion_matrices)
mean_confusion_matrix = mean_confusion_matrix.astype(int) # convert values to integers

# Print evaluation metrics
print(f'Accuracy: {mean_accuracy}')
print(f'Precision: {mean_precision}')
print(f'Recall: {mean_recall}')
print(f'F1 score: {mean_f1}')
print(f'Balanced accuracy: {mean_balanced_accuracy}')

# Define class names
class_names = sorted(y.unique())

# Calculate classification report
y_pred_all = dt.predict(X)
class_report = classification_report(y, y_pred_all, target_names=class_names, digits=4, output_dict=True)
print(f'Classification report:\n{class_report}')

# Print accuracy of every class
for class_name in class_names:
    class_accuracy = class_report[class_name]['precision']
    print(f'Accuracy for class {class_name}: {class_accuracy}')

# Plot confusion matrix heatmap
sns.heatmap(mean_confusion_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
import time

start_time = time.time()

# Your code for training the model goes here

end_time = time.time()

print("Computation time: ", end_time - start_time, " seconds")


"""***************************************Random Forest***********************************************"""
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier


# Load data
df = pd.read_csv('Dataset_with_ContextSpaceFeatures_SituationIdentification.csv.csv')
#le = LabelEncoder()
#df['Prediction'] = le.fit_transform(df['Prediction'])
#X = df.iloc[:, :-1]
#y = df.iloc[:, -1]
X = df.drop('context_article_code', axis=1)
y = df['context_article_code']

# Define the Random Forest model with best hyperparameters
rf = RandomForestClassifier(n_estimators=100, max_depth=500, random_state=42,min_samples_split=6,min_samples_leaf=4)

# Train the model using 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
balanced_accuracy_scores = []
confusion_matrices = []
#roc_auc_scores = []

for train_index, test_index in cv.split(X, y):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model and make predictions on the test set
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Calculate evaluation metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
    recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))
    confusion_matrices.append(confusion_matrix(y_test, y_pred))
    #fpr, tpr, _ = roc_curve(y_test, y_pred)
    #roc_auc_scores.append(auc(fpr, tpr))


# Calculate mean evaluation metrics across all folds
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
mean_precision = sum(precision_scores) / len(precision_scores)
mean_recall = sum(recall_scores) / len(recall_scores)
mean_f1 = sum(f1_scores) / len(f1_scores)
mean_balanced_accuracy = sum(balanced_accuracy_scores) / len(balanced_accuracy_scores)
mean_confusion_matrix = sum(confusion_matrices) / len(confusion_matrices)
mean_confusion_matrix = mean_confusion_matrix.astype(int) # convert values to integers
# Print evaluation metrics
print(f'Accuracy: {mean_accuracy}')
print(f'Precision: {mean_precision}')
print(f'Recall: {mean_recall}')
print(f'F1 score: {mean_f1}')
print(f'Balanced accuracy: {mean_balanced_accuracy}')

# Define class names
class_names = sorted(y.unique())

# Calculate classification report
y_pred_all = rf.predict(X)
class_report = classification_report(y, y_pred_all, target_names=class_names, digits=4, output_dict=True)
print(f'Classification report:\n{class_report}')

# Print accuracy of every class
for class_name in class_names:
    class_accuracy = class_report[class_name]['precision']
    print(f'Accuracy for class {class_name}: {class_accuracy}')

# Plot confusion matrix heatmap
sns.heatmap(mean_confusion_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

import time

start_time = time.time()

# Your code for training the model goes here

end_time = time.time()

print("Computation time: ", end_time - start_time, " seconds")

"""****************************************K Nearest Neighbour ******************************************************"""

# Import necessary libraries

from sklearn.neighbors import KNeighborsClassifier

# Load data
df = pd.read_csv('Dataset_with_ContextSpaceFeatures_SituationIdentification.csv')
#le = LabelEncoder()
#df['Prediction'] = le.fit_transform(df['Prediction'])
#X = df.iloc[:, :-1]
#y = df.iloc[:, -1]
X = df.drop('context_article_code', axis=1)
y = df['context_article_code']

# Define the K Nearest Neighbour with best hyperparameters
knn = KNeighborsClassifier(n_neighbors=7,weights='distance',p=2)

# Train the model using 5-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
balanced_accuracy_scores = []
confusion_matrices = []
#roc_auc_scores = []

for train_index, test_index in cv.split(X, y):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model and make predictions on the test set
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Calculate evaluation metrics
    accuracy_scores.append(accuracy_score(y_test, y_pred))
    precision_scores.append(precision_score(y_test, y_pred, average='weighted'))
    recall_scores.append(recall_score(y_test, y_pred, average='weighted'))
    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
    balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))
    confusion_matrices.append(confusion_matrix(y_test, y_pred))

# Calculate mean evaluation metrics across all folds
mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
mean_precision = sum(precision_scores) / len(precision_scores)
mean_recall = sum(recall_scores) / len(recall_scores)
mean_f1 = sum(f1_scores) / len(f1_scores)
mean_balanced_accuracy = sum(balanced_accuracy_scores) / len(balanced_accuracy_scores)
mean_confusion_matrix = sum(confusion_matrices) / len(confusion_matrices)
mean_confusion_matrix = mean_confusion_matrix.astype(int) # convert values to integers
# Print evaluation metrics
print(f'Accuracy: {mean_accuracy}')
print(f'Precision: {mean_precision}')
print(f'Recall: {mean_recall}')
print(f'F1 score: {mean_f1}')
print(f'Balanced accuracy: {mean_balanced_accuracy}')

# Define class names
class_names = sorted(y.unique())

# Calculate classification report
y_pred_all = knn.predict(X)
class_report = classification_report(y, y_pred_all, target_names=class_names, digits=4, output_dict=True)
print(f'Classification report:\n{class_report}')

# Print accuracy of every class
for class_name in class_names:
    class_accuracy = class_report[class_name]['precision']
    print(f'Accuracy for class {class_name}: {class_accuracy}')

# Plot confusion matrix heatmap
sns.heatmap(mean_confusion_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
import time

start_time = time.time()


end_time = time.time()

print("Computation time: ", end_time - start_time, " seconds")


"""*********************************************Artifiicial Neural Network (MLP)**************************************************"""
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
filename = 'Dataset_with_ContextSpaceFeatures_SituationIdentification.csv'

classification = pd.read_csv(filename)
situation = classification.iloc[:, -1]
classification = classification.iloc[:, 0:-1]

data_table = classification
Situation = situation
data_table['context_article_code'] = situation


train_data, test_data = train_test_split(data_table, test_size=0.2)

num_bins = 10
for i in range(len(train_data.columns) - 1):
    train_data.iloc[:, i] = pd.cut(train_data.iloc[:, i], bins=num_bins, labels=False)

mlp_classifier = MLPClassifier(max_iter=1000)

start_time = time.time()
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
scores = cross_validate(mlp_classifier, train_data.iloc[:, :-1], train_data.iloc[:, -1], scoring=scoring, cv=KFold(n_splits=5))
train_time = time.time() - start_time

print('Cross-validation time:', train_time)

print("accuracy", scores['test_accuracy'])
print("precision：", scores['test_precision_macro'])
print("recall", scores['test_recall_macro'])
print("F1 score", scores['test_f1_macro'])

print("avgAccuracy：", scores['test_accuracy'].mean())
print("avgPrecision：", scores['test_precision_macro'].mean())
print("avgRecll", scores['test_recall_macro'].mean())
print("avgF1score", scores['test_f1_macro'].mean())
print(f"Train time: {train_time} seconds")

for i in range(len(test_data.columns) - 1):
    test_data.iloc[:, i] = pd.cut(test_data.iloc[:, i], bins=num_bins, labels=False)

start_time = time.time()
predicted_labels = mlp_classifier.fit(train_data.iloc[:, :-1], train_data.iloc[:, -1]).predict(test_data.iloc[:, :-1])
test_time = time.time() - start_time

accuracy = accuracy_score(test_data.iloc[:, -1], predicted_labels)

C = confusion_matrix(test_data.iloc[:, -1], predicted_labels)

precision = np.zeros(C.shape[0])
recall = np.zeros(C.shape[0])
f1score = np.zeros(C.shape[0])

for i in range(C.shape[0]):
    tp = C[i, i]
    fp = np.sum(C[:, i]) - tp
    fn = np.sum(C[i, :]) - tp

    precision[i] = tp / (tp + fp)
    recall[i] = tp / (tp + fn)
    f1score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

avgPrecision = np.mean(precision)
avgRecall = np.mean(recall)
avgF1score = np.mean(f1score)

print(f"Accuracy: {accuracy}")
print(f"Average Precision: {avgPrecision}")
print(f"Average Recall: {avgRecall}")
print(f"Average F1-score: {avgF1score}")
print(f"Test time: {test_time} seconds")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
plt.figure


# Assuming you have the confusion matrix stored in variable C
sns.heatmap(C, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#pd.DataFrame(C).to_csv('confusion_matrix_nn_1000_sel.csv')

#the confusion matrix is in the attached file.
#0 corresponds to category SITTING
#1  corresponds to category   walking
#2  corresponds to category   standing
#3  corresponds to category   LYING_DOWN
#4  corresponds to category   running

"""*********************************************Naive Bayes **************************************************"""
import time
from sklearn.naive_bayes import CategoricalNB
import pandas as pd
import numpy as np

filename = 'Dataset_with_ContextSpaceFeatures_SituationIdentification.csv'

classification = pd.read_csv(filename)
situation = classification.iloc[:, -1]
classification = classification.iloc[:, 0:-1]

# 读取数据 Load
data_table = classification
Situation = situation
data_table['context_article_code'] = situation


train_data, test_data = train_test_split(data_table, test_size=0.2)

num_bins = 10
for i in range(len(train_data.columns) - 1):
    train_data.iloc[:, i] = pd.cut(train_data.iloc[:, i], bins=num_bins, labels=False)

nb_classifier = CategoricalNB()


start_time = time.time()
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
scores = cross_validate(nb_classifier , train_data.iloc[:, :-1], train_data.iloc[:, -1], scoring=scoring, cv=KFold(n_splits=5))
train_time = time.time() - start_time

print('Cross-validation time:', train_time)
print("accuracy", scores['test_accuracy'])
print("precision：", scores['test_precision_macro'])
print("recall", scores['test_recall_macro'])
print("F1 score", scores['test_f1_macro'])

print("avgAccuracy：", scores['test_accuracy'].mean())
print("avgPrecision：", scores['test_precision_macro'].mean())
print("avgRecll", scores['test_recall_macro'].mean())
print("avgF1score", scores['test_f1_macro'].mean())
print(f"Train time: {train_time} seconds")

for i in range(len(test_data.columns) - 1):
    test_data.iloc[:, i] = pd.cut(test_data.iloc[:, i], bins=num_bins, labels=False)

start_time = time.time()
predicted_labels = nb_classifier .fit(train_data.iloc[:, :-1], train_data.iloc[:, -1]).predict(test_data.iloc[:, :-1])
test_time = time.time() - start_time

accuracy = accuracy_score(test_data.iloc[:, -1], predicted_labels)

C = confusion_matrix(test_data.iloc[:, -1], predicted_labels)
precision = np.zeros(C.shape[0])
recall = np.zeros(C.shape[0])
f1score = np.zeros(C.shape[0])

for i in range(C.shape[0]):
    tp = C[i, i]
    fp = np.sum(C[:, i]) - tp
    fn = np.sum(C[i, :]) - tp

    precision[i] = tp / (tp + fp)
    recall[i] = tp / (tp + fn)
    f1score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])

avgPrecision = np.mean(precision)
avgRecall = np.mean(recall)
avgF1score = np.mean(f1score)

print(f"Accuracy: {accuracy}")
print(f"Average Precision: {avgPrecision}")
print(f"Average Recall: {avgRecall}")
print(f"Average F1-score: {avgF1score}")
print(f"Test time: {test_time} seconds")

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)
plt.figure

# save confusion matrix as CSV file for further analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have the confusion matrix stored in variable C
sns.heatmap(C, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

pd.DataFrame(C).to_csv('Bayesian_Situation.csv')