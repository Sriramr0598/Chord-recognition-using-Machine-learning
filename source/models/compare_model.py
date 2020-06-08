# -*- coding: utf-8 -*-

# Importing the libraries
from src.utilities import model_util as mu
from src.models import classification_models as standard_models
from src.models import tuned_classification_models as tuned_models


# Logistic Regression
def compare_logistic_regression(X, X_train, X_test, y, y_train, y_test):
    standard_classifier = standard_models.get_logistic_regression_model()
    tuned_model = tuned_models.get_logistic_regression_model(X_train, y_train)
    tuned_classifier = tuned_model.best_estimator_
    print_tuned_classifier(tuned_classifier)

    compare_test_set_performance('Logistic Regression classifier(Standard Model)', standard_classifier,
                                 'Logistic Regression classifier(Tuned Model)', tuned_classifier, X_train, X_test,
                                 y_train, y_test)
    compare_kfold_validation('Logistic Regression classifier(Standard Model)', standard_classifier,
                             'Logistic Regression classifier(Tuned Model)', tuned_classifier, X, y)


# K-nearest Neighbors (K-NN)
def compare_knn(X, X_train, X_test, y, y_train, y_test):
    standard_classifier = standard_models.get_knn_model()
    tuned_model = tuned_models.get_knn_model(X_train, y_train)
    tuned_classifier = tuned_model.best_estimator_
    print_tuned_classifier(tuned_classifier)

    compare_test_set_performance('K-NN classifier(Standard Model)', standard_classifier, 'K-NN classifier(Tuned Model)',
                                 tuned_classifier, X_train, X_test, y_train, y_test)
    compare_kfold_validation('K-NN classifier(Standard Model)', standard_classifier, 'K-NN classifier(Tuned Model)',
                             tuned_classifier, X, y)


# SVM Classifier
def compare_svm(X, X_train, X_test, y, y_train, y_test):
    standard_classifier = standard_models.get_svm_model()
    tuned_model = tuned_models.get_svm_model(X_train, y_train)
    tuned_classifier = tuned_model.best_estimator_
    print_tuned_classifier(tuned_classifier)

    compare_test_set_performance('Linear SVM classifier(Standard Model)', standard_classifier,
                                 'Linear SVM classifier(Tuned Model)', tuned_classifier, X_train, X_test, y_train,
                                 y_test)
    compare_kfold_validation('Linear SVM classifier(Standard Model)', standard_classifier,
                             'Linear SVM classifier(Tuned Model)', tuned_classifier, X, y)


# Kernel SVM Classifier
def compare_kernel_svm(X, X_train, X_test, y, y_train, y_test):
    standard_classifier = standard_models.get_kernel_svm_model()
    tuned_model = tuned_models.get_kernel_svm_model(X_train, y_train)
    tuned_classifier = tuned_model.best_estimator_
    print_tuned_classifier(tuned_classifier)

    compare_test_set_performance('Kernel SVM classifier(Standard Model)', standard_classifier,
                                 'Kernel SVM classifier(Tuned Model)', tuned_classifier, X_train, X_test, y_train,
                                 y_test)
    compare_kfold_validation('Kernel SVM classifier(Standard Model)', standard_classifier,
                             'Kernel SVM classifier(Tuned Model)', tuned_classifier, X, y)


# Naive Bayes Classifier
def compare_naive_bayes(X, X_train, X_test, y, y_train, y_test):
    print("\n\n######### 1. Comparing result for test data set ################\n")
    # getting naive bayes classifier
    standard_classifier = standard_models.get_naive_bayes_model()
    # Fitting the classifier to training set
    standard_classifier.fit(X_train, y_train)
    # Predicting results for test data 
    y_pred = standard_classifier.predict(X_test)
    accuracy, precision, recall, f1_score, cm = mu.measure_performance(y_test, y_pred)
    mu.print_metric_stats('Naive Bayes Classifier', accuracy, precision, recall, f1_score, cm)

    # K-fold cross validation
    print("\n\n######### 2. Comparing result for k-fold validation ################\n")
    accuracy, precision, recall, f1_score = mu.measure_performance_kfold_validation(standard_classifier, X, y)
    mu.print_metric_stats('Naive Bayes Classifier', accuracy, precision, recall, f1_score, cm=None)


# Decision Tree Classifier
def compare_decision_tree(X, X_train, X_test, y, y_train, y_test):
    standard_classifier = standard_models.get_decision_tree_model()
    tuned_model = tuned_models.get_decision_tree_model(X_train, y_train)
    tuned_classifier = tuned_model.best_estimator_
    print_tuned_classifier(tuned_classifier)

    compare_test_set_performance('Decision Tree classifier(Standard Model)', standard_classifier,
                                 'Decision Tree classifier(Tuned Model)', tuned_classifier, X_train, X_test, y_train,
                                 y_test)
    compare_kfold_validation('Decision Tree classifier(Standard Model)', standard_classifier,
                             'Decision Tree classifier(Tuned Model)', tuned_classifier, X, y)


# Random Forest Classifier
def compare_random_forest(X, X_train, X_test, y, y_train, y_test):
    standard_classifier = standard_models.get_random_forest_model()
    tuned_model = tuned_models.get_random_forest_model(X_train, y_train)
    tuned_classifier = tuned_model.best_estimator_
    print_tuned_classifier(tuned_classifier)

    compare_test_set_performance('Random Forest classifier(Standard Model)', standard_classifier,
                                 'Random Forest classifier(Tuned Model)', tuned_classifier, X_train, X_test, y_train,
                                 y_test)
    compare_kfold_validation('Random Forest classifier(Standard Model)', standard_classifier,
                             'Random Forest classifier(Tuned Model)', tuned_classifier, X, y)



def compare_mlp(X, X_train, X_test, y, y_train, y_test):
    standard_classifier = standard_models.get_neural_network_model()
    tuned_model = tuned_models.get_neural_network_model(X_train, y_train)
    tuned_classifier = tuned_model.best_estimator_
    print_tuned_classifier(tuned_classifier)

    compare_test_set_performance('Neural network classifier(Standard Model)', standard_classifier,
                                 'Neural network classifier(Tuned Model)', tuned_classifier, X_train, X_test, y_train,
                                 y_test)
    compare_kfold_validation('Neural network classifier(Standard Model)', standard_classifier,
                             'Neural network classifier(Tuned Model)', tuned_classifier, X, y)



def compare_test_set_performance(classifier_label_1, classifier_1, classifier_label_2, classifier_2, X_train, X_test,
                                 y_train, y_test):
    print("\n\n######### 1. Comparing result for test data set ################\n")
    # Fitting the classifier to training set
    classifier_1.fit(X_train, y_train)
    # Predicting results for test data 
    y_pred = classifier_1.predict(X_test)
    accuracy, precision, recall, f1_score, cm = mu.measure_performance(y_test, y_pred)
    mu.print_metric_stats(classifier_label_1, accuracy, precision, recall, f1_score, cm)

    # Fitting the classifier to training set
    classifier_2.fit(X_train, y_train)
    # Predicting results for test data 
    y_pred = classifier_2.predict(X_test)
    accuracy, precision, recall, f1_score, cm = mu.measure_performance(y_test, y_pred)
    mu.print_metric_stats(classifier_label_2, accuracy, precision, recall, f1_score, cm)


def compare_kfold_validation(classifier_label_1, classifier_1, classifier_label_2, classifier_2, X, y):
    print("\n\n######### 2. Comparing result for k-fold validation ################\n")

    accuracy, precision, recall, f1_score = mu.measure_performance_kfold_validation(classifier_1, X, y)
    mu.print_metric_stats(classifier_label_1, accuracy, precision, recall, f1_score, cm=None)

    accuracy, precision, recall, f1_score = mu.measure_performance_kfold_validation(classifier_2, X, y)
    mu.print_metric_stats(classifier_label_2, accuracy, precision, recall, f1_score, cm=None)


def print_tuned_classifier(tuned_classifier):
    print("---------------------------------------------------------------")
    print("TUNED MODEL : ")
    print(tuned_classifier)
    print("---------------------------------------------------------------")
