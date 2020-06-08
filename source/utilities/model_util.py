# -*- coding: utf-8 -*-
"""
Utility functions for model creation

"""

# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from src.utilities import configure_util as configure


def measure_performance(y_actual, y_pred):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    cm = confusion_matrix(y_actual, y_pred)
    accuracy = accuracy_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred, average='macro')
    recall = recall_score(y_actual, y_pred, average='macro')
    f1 = f1_score(y_actual, y_pred, average='macro')
    cm = confusion_matrix(y_actual, y_pred)
    return accuracy, precision, recall, f1, cm


def measure_performance_kfold_validation(classifier, X, y):
    from sklearn.model_selection import cross_validate
    scores = cross_validate(classifier, X, y, cv=10,
                            scoring=('accuracy', 'precision_macro', 'recall_macro', 'f1_macro'),
                            return_train_score=False)
    accuracy = np.mean(scores['test_accuracy'])
    precision = np.mean(scores['test_precision_macro'])
    recall = np.mean(scores['test_recall_macro'])
    f1_score = np.mean(scores['test_f1_macro'])
    return accuracy, precision, recall, f1_score


def process_data_set(data_file):
    # Importing the dataset
    data_set = pd.read_csv(data_file)
    X = data_set.iloc[:, :-1].values
    y = data_set.iloc[:, 12].values

    # Encoding the categorical variables (Chords to predict)
    from sklearn.preprocessing.label import LabelEncoder
    label_encoder_y = LabelEncoder()
    y = label_encoder_y.fit_transform(y)
    # print(label_encoder_y.classes_)
    return X, y, label_encoder_y.classes_


def split_train_test_set(data_file):
    X, y, y_classes = process_data_set(data_file)
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    return X, X_train, X_test, y, y_train, y_test


def print_metric_stats(model_name, accuracy, precision, recall, f1_score, cm):
    print("************* " + model_name + " *************")
    print("Accuracy : " + str(accuracy))
    print("Precision : " + str(precision))
    print("Recall : " + str(recall))
    print("F1 score : " + str(f1_score))

    if cm is not None:
        plot_confusion_matrix(cm, classes=configure.get_chord_list())


# This function prints and plots the confusion matrix.
def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    # print confusion matrix
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
