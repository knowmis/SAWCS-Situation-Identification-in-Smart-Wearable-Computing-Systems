"""Activity_Recognition"""
#This file implements the Activity recognition task in the High-level perception phase of the SA-WCS architecture.
#The files implements the five classifier that have been compared in the paper: Decision Tree, Random Forest, KNN, Naive Bayes and ANN

#The input file Dataset_Selected_features_SA-WA_ActivityRecognition.csv contains the 26 features selected by the Feature Selection
#process in the High-level perception Phase: 13 features are from the smartphone accelerometer, and 13 features are from the
#smartwatch accelerometer. The target label is the activity which contains the five activities:
#Lying down, sitting, standing, running, walking. The target label is the last column of the dataset.

"""********************************Decision Tree********************************************************"""
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

def evaluate_decision_tree(X, y):
    # Define the decision tree model with best hyperparameters
    dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=4, min_samples_split=5, criterion='gini')

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
        precision_scores.append(precision_score(y_test, y_pred, average='macro'))
        recall_scores.append(recall_score(y_test, y_pred, average='macro'))
        f1_scores.append(f1_score(y_test, y_pred, average='macro'))
        balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))
        confusion_matrices.append(confusion_matrix(y_test, y_pred))

    # Calculate mean evaluation metrics across all folds
    mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    mean_precision = sum(precision_scores) / len(precision_scores)
    mean_recall = sum(recall_scores) / len(recall_scores)
    mean_f1 = sum(f1_scores) / len(f1_scores)
    mean_balanced_accuracy = sum(balanced_accuracy_scores) / len(balanced_accuracy_scores)
    mean_confusion_matrix = sum(confusion_matrices) / len(confusion_matrices)
    mean_confusion_matrix = mean_confusion_matrix.astype(int)  # convert values to integers

    # Print evaluation metrics
    print(f'Decision Tree Classifier Metrics:')
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
"""*********************************************************Random Forest***************************************************"""

def evaluate_random_forest(X, y):
    # Define the random forest model with best hyperparameters
    rf = RandomForestClassifier(n_estimators=100, max_depth=500, random_state=42, min_samples_split=6, min_samples_leaf=4)

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
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

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
    mean_confusion_matrix = mean_confusion_matrix.astype(int)  # convert values to integers

    # Print evaluation metrics
    print(f'Random Forest Classifier Metrics:')
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

"""****************************************K Nearest Neighbour*********************************************"""

def evaluate_knn(X, y):
    # Define the K Nearest Neighbors model with best hyperparameters
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance', p=2)

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
    mean_confusion_matrix = mean_confusion_matrix.astype(int)  # convert values to integers

    # Define class names
    class_names = sorted(y.unique())

    # Calculate classification report
    y_pred_all = knn.predict(X)
    class_report = classification_report(y, y_pred_all, target_names=class_names, digits=4, output_dict=True)
    print(f'K Nearest Neighbors (KNN) Classifier Metrics:')
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
    sns.heatmap(mean_confusion_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
"""*********************************************Naive Bayes************************************************"""

def evaluate_naive_bayes(filename='Dataset_Selected_features_SA-WA_ActivityRecognition.csv', num_bins=10):
    # Load data
    classification = pd.read_csv(filename)
    situation = classification.iloc[:, -1]
    classification = classification.iloc[:, 0:-1]

    # Encode the target labels
    label_encoder = LabelEncoder()
    situation_encoded = label_encoder.fit_transform(situation)

    # Combine data and target
    data_table = classification.copy()
    data_table['activity'] = situation_encoded

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data_table, test_size=0.2, random_state=42)

    # Discretize continuous values into bins
    for i in range(len(train_data.columns) - 1):
        train_data.iloc[:, i] = pd.cut(train_data.iloc[:, i], bins=num_bins, labels=False)

    # Create a Categorical Naive Bayes classifier
    nb_classifier = CategoricalNB()

    # Perform 5-fold cross-validation
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    scores = cross_validate(nb_classifier, train_data.iloc[:, :-1], train_data['activity'], scoring=scoring, cv=KFold(n_splits=5))

    # Print all fold evaluation metrics
    print("accuracy", scores['test_accuracy'])
    print("precision:", scores['test_precision_weighted'])
    print("recall", scores['test_recall_weighted'])
    print("F1 score", scores['test_f1_weighted'])

    # Print average evaluation metrics
    print("avgAccuracy:", scores['test_accuracy'].mean())
    print("avgPrecision:", scores['test_precision_weighted'].mean())
    print("avgRecall:", scores['test_recall_weighted'].mean())
    print("avgF1score:", scores['test_f1_weighted'].mean())

    # Discretize continuous values in the test set
    for i in range(len(test_data.columns) - 1):
        test_data.iloc[:, i] = pd.cut(test_data.iloc[:, i], bins=num_bins, labels=False)

    # Fit the classifier, predict
    predicted_labels = nb_classifier.fit(train_data.iloc[:, :-1], train_data['activity']).predict(test_data.iloc[:, :-1])

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_data['activity'], predicted_labels)
    precision = precision_score(test_data['activity'], predicted_labels, average='weighted')
    recall = recall_score(test_data['activity'], predicted_labels, average='weighted')
    f1 = f1_score(test_data['activity'], predicted_labels, average='weighted')

    # Print results
    print(f"Accuracy: {accuracy}")
    print(f"Average Precision: {precision}")
    print(f"Average Recall: {recall}")
    print(f"Average F1-score: {f1}")

    # Create a confusion matrix
    C = confusion_matrix(test_data['activity'], predicted_labels)

    # Calculate class-wise precision, recall, and F1-score
    class_precision = precision_score(test_data['activity'], predicted_labels, average=None)
    class_recall = recall_score(test_data['activity'], predicted_labels, average=None)
    class_f1 = f1_score(test_data['activity'], predicted_labels, average=None)

    # Print class-wise metrics
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"Class {class_name}: Precision={class_precision[i]}, Recall={class_recall[i]}, F1-score={class_f1[i]}")

    # Calculate class-wise accuracy
    class_accuracy = []
    for i in range(C.shape[0]):
        tp = C[i, i]
        class_accuracy.append(tp / (tp + np.sum(C[i, :]) - tp))

    # Print class-wise accuracy
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"Accuracy for class {class_name}: {class_accuracy[i]}")

    # Plot the confusion matrix
    sns.set(font_scale=1.2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(C, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Save the confusion matrix as a CSV file
    pd.DataFrame(C, columns=label_encoder.classes_, index=label_encoder.classes_).to_csv('Bayesian_activity(13+13).csv')

    """***********************************Multi Layer Perceptron******************************************"""
def evaluate_mlp(filename='Dataset_Selected_features_SA-WA_ActivityRecognition.csv', num_bins=10):
    # Load data
    classification = pd.read_csv(filename)
    situation = classification.iloc[:, -1]
    classification = classification.iloc[:, 0:-1]

    # Encode the target labels
    label_encoder = LabelEncoder()
    situation_encoded = label_encoder.fit_transform(situation)

    # Combine data and target
    data_table = classification.copy()
    data_table['activity'] = situation_encoded

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data_table, test_size=0.2, random_state=42)

    # Discretize continuous values into bins
    for i in range(len(train_data.columns) - 1):
        train_data.iloc[:, i] = pd.cut(train_data.iloc[:, i], bins=num_bins, labels=False)

    # Create an MLP classifier
    mlp = MLPClassifier(max_iter=1000)

    # Perform 5-fold cross-validation
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    scores = cross_validate(mlp, train_data.iloc[:, :-1], train_data['activity'], scoring=scoring, cv=KFold(n_splits=5))

    # Print all fold evaluation metrics
    print("accuracy", scores['test_accuracy'])
    print("precision:", scores['test_precision_weighted'])
    print("recall", scores['test_recall_weighted'])
    print("F1 score", scores['test_f1_weighted'])

    # Print average evaluation metrics
    print("avgAccuracy:", scores['test_accuracy'].mean())
    print("avgPrecision:", scores['test_precision_weighted'].mean())
    print("avgRecall:", scores['test_recall_weighted'].mean())
    print("avgF1score:", scores['test_f1_weighted'].mean())

    # Discretize continuous values in the test set
    for i in range(len(test_data.columns) - 1):
        test_data.iloc[:, i] = pd.cut(test_data.iloc[:, i], bins=num_bins, labels=False)

    # Fit the classifier, predict
    predicted_labels = mlp.fit(train_data.iloc[:, :-1], train_data['activity']).predict(test_data.iloc[:, :-1])

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_data['activity'], predicted_labels)
    precision = precision_score(test_data['activity'], predicted_labels, average='weighted')
    recall = recall_score(test_data['activity'], predicted_labels, average='weighted')
    f1 = f1_score(test_data['activity'], predicted_labels, average='weighted')

    # Print results
    print(f"Accuracy: {accuracy}")
    print(f"Average Precision: {precision}")
    print(f"Average Recall: {recall}")
    print(f"Average F1-score: {f1}")

    # Create a confusion matrix
    C = confusion_matrix(test_data['activity'], predicted_labels)

    # Calculate class-wise precision, recall, and F1-score
    class_precision = precision_score(test_data['activity'], predicted_labels, average=None)
    class_recall = recall_score(test_data['activity'], predicted_labels, average=None)
    class_f1 = f1_score(test_data['activity'], predicted_labels, average=None)

    # Print class-wise metrics
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"Class {class_name}: Precision={class_precision[i]}, Recall={class_recall[i]}, F1-score={class_f1[i]}")

    # Calculate class-wise accuracy
    class_accuracy = []
    for i in range(C.shape[0]):
        tp = C[i, i]
        class_accuracy.append(tp / (tp + np.sum(C[i, :]) - tp))

    # Print class-wise accuracy
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"Accuracy for class {class_name}: {class_accuracy[i]}")

    # Plot the confusion matrix
    sns.set(font_scale=1.2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(C, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Save the confusion matrix as a CSV file
    pd.DataFrame(C, columns=label_encoder.classes_, index=label_encoder.classes_).to_csv('MLP_activity(13+13).csv')


"""*****************************Saving Predicted Labels by Using KNN for the comprehension phase***************************"""
#The following code is used to export and save the activity labels predicted by the KNN
#(which is the best classifier for the activity recognition task).
#The predicted label will be used by the next phase: Situation Identification in the comprehension phase,
#to identify the situation by merginf the predicted activity with the contextual information.

def evaluate_knn_activity(X, y):
    # Define the decision tree model with best hyperparameters
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance', p=2)

    # Train the model using 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    balanced_accuracy_scores = []
    confusion_matrices = []
    predicted_values = []
    # roc_auc_scores = []

    df.loc[:, 'Prediction'] = ''

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
        predicted_values.extend(y_pred)
        # Add values to the dataframe
        df.loc[test_index, 'Prediction'] = y_pred
        # fpr, tpr, _ = roc_curve(y_test, y_pred)
        # roc_auc_scores.append(auc(fpr, tpr))

        # Calculate classification report
    # y_pred_all = dt.predict(X)
    # class_report = classification_report(y, y_pred_all, target_names=class_names, digits=4, output_dict=True)
    # print(f'Classification report:\n{class_report}')

    # Calculate mean evaluation metrics across all folds
    mean_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    mean_precision = sum(precision_scores) / len(precision_scores)
    mean_recall = sum(recall_scores) / len(recall_scores)
    mean_f1 = sum(f1_scores) / len(f1_scores)
    mean_balanced_accuracy = sum(balanced_accuracy_scores) / len(balanced_accuracy_scores)
    mean_confusion_matrix = sum(confusion_matrices) / len(confusion_matrices)
    mean_confusion_matrix = mean_confusion_matrix.astype(int)  # convert values to integers
    # Add predicted values to the original dataframe
    # df['predicted_value'] = predicted_values
    # Add label back per question

    # Save the full dataset with predicted values as a CSV file
    df.to_csv('predicted_dataset_onelebeladded.csv', index=False)
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
    sns.heatmap(mean_confusion_matrix, annot=True, cmap='Blues', fmt='g', xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    # Load data
    df = pd.read_csv('Dataset_Selected_features_SA-WA_ActivityRecognition.csv')
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Evaluate Decision Tree Classifier
    evaluate_decision_tree(X, y)
    # Evaluate Random Forest Classifier
    evaluate_random_forest(X, y)
# Evaluate KNN Classifier
    evaluate_knn(X, y)
    # Call the functions to evaluate the Naive Bayes
    evaluate_naive_bayes()
    # Call the functions to evaluate the MLP
    evaluate_mlp()
    # Call the functions to evaluate the KNN and save the activities
    evaluate_knn_activity(X, y)
