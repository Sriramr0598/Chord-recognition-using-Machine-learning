# -*- coding: utf-8 -*-


"""
Create classification models with hyper-parameter optimization

"""


# Logistic Regression
def get_logistic_regression_model(X_train, y_train):
    # Parameters declaration
    parameters = [{'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}]
    # Create Logistic Regression model
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    # return optimized model with optimized parameters and score
    return grid_search_parameter(classifier, parameters, X_train, y_train)


# K-nearest Neighbors (K-NN)
def get_knn_model(X_train, y_train):
    # Parameters declaration
    parameters = [{
        'n_neighbors': [5, 6, 7, 8, 9, 10, 20, 50, 100],
        'p': [1, 2],
        'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'hamming', 'braycurtis']
    }]
    # Create K-NN model
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier()
    # return optimized model with optimized parameters and score
    return grid_search_parameter(classifier, parameters, X_train, y_train)


# SVM Classifier
def get_svm_model(X_train, y_train):
    # Parameters declaration
    parameters = [{'C': [1, 10, 50, 100, 1000], 'kernel': ['linear']}]
    # Create SVM Classifier
    from sklearn.svm import SVC
    classifier = SVC()
    # return optimized model with optimized parameters and score
    return grid_search_parameter(classifier, parameters, X_train, y_train)


# Kernal SVM Classifier
def get_kernel_svm_model(X_train, y_train):
    # Parameters declaration
    parameters = [{'C': [1, 10, 50, 100, 1000], 'kernel': ['linear']},
                  {'C': [1, 10, 50, 100, 1000], 'kernel': ['poly', 'rbf', 'sigmoid'],
                   'gamma': [0.5, 0.1, 0.01, 0.001, 0.0001]}]
    # Create kernel SVM Classifier
    from sklearn.svm import SVC
    classifier = SVC()
    # return optimized model with optimized parameters and score
    return grid_search_parameter(classifier, parameters, X_train, y_train)


# Naive Bayes Classifier - No tuning required


# Decision Tree Classifier
def get_decision_tree_model(X_train, y_train):
    # Parameters declaration
    parameters = [{'criterion': ['gini', 'entropy']}]
    # Create Decision Tree Classifier
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    # return optimized model with optimized parameters and score
    return grid_search_parameter(classifier, parameters, X_train, y_train)


# Random Forest Classifier
def get_random_forest_model(X_train, y_train):
    # Parameters declaration
    parameters = [{'criterion': ['gini', 'entropy'], 'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}]
    # Create Random Forest Classifier
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier()
    # return optimized model with optimized parameters and score
    return grid_search_parameter(classifier, parameters, X_train, y_train)


#MLP classifier

def get_neural_network_model(X_train, y_train):
    # Parameters declaration
    parameters = [{'solver':['lbfgs','sgd','adam'],'hidden_layer_sizes':[4,5],'activation':['logistic','tanh','relu']}]
    # Create Random Forest Classifier
    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier()
    # return optimized model with optimized parameters and score
    return grid_search_parameter(classifier, parameters, X_train, y_train)



# Use Grid Search for parameter tuning
def grid_search_parameter(classifier, parameters, X, y):
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(estimator=classifier,
                               param_grid=parameters,
                               cv=10,
                               n_jobs=-1)
    gs = grid_search.fit(X, y)
    print("Grid Search Best Score :" + str(gs.best_score_))
    return gs



