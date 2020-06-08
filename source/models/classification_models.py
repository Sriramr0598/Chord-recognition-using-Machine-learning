"""
Create standard classification models

"""


# Logistic Regression
def get_logistic_regression_model():
    # creating Logistic Regression classifier
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    return classifier


# K-nearest Neighbors (K-NN)
def get_knn_model():
    # creating K-NN classifier
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier()
    return classifier


# SVM Classifier
def get_svm_model():
    # creating SVM classifier
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear')
    return classifier


# Kernel SVM Classifier
def get_kernel_svm_model():
    # creating Kernel SVM classifier
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf')
    return classifier


# Naive Bayes Classifier
def get_naive_bayes_model():
    # creating Naive Bayes classifier
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    return classifier


# Decision Tree Classifier
def get_decision_tree_model():
    # creating Decision Tree classifier
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier()
    return classifier


# Random Forest Classifier
def get_random_forest_model():
    # creating Random Forest classifier
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier()
    return classifier

#MLP classifier

def get_neural_network_model():
    # creating Random Forest classifier
    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier()
    return classifier



