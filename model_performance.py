import sys
import os
from source.utilities import configure_util as configure
from source.models import compare_model as cm
from source.utilities import model_util as mu
from feature_extraction import extract_audio_features

target_models = ['logistic', 'knn', 'svm', 'kernel_svm', 'naive_bayes', 'decision_tree', 'random_forest']
target_chroma_type = ['stft', 'cqt', 'cens']


# compare different models with and without hyperparameter tuning
def calculate_model_performance(model, chroma_type="cqt"):
    if model not in target_models:
        print("Invalid classification model name parameter!!")
        sys.exit(-1)

    if chroma_type not in target_chroma_type:
        print("Invalid chroma type!!")
        sys.exit(-1)

    chroma_feature_csv = configure.get_chroma_dataset_file_path(chroma_type)

    if not os.path.exists(chroma_feature_csv):
        extract_audio_features(chroma_type)

    X, X_train, X_test, y, y_train, y_test = mu.split_train_test_set(chroma_feature_csv)

    if model == 'logistic':
        cm.compare_logistic_regression(X, X_train, X_test, y, y_train, y_test)
    elif model == 'knn':
        cm.compare_knn(X, X_train, X_test, y, y_train, y_test)
    elif model == 'svm':
        cm.compare_svm(X, X_train, X_test, y, y_train, y_test)
    elif model == 'kernel_svm':
        cm.compare_kernel_svm(X, X_train, X_test, y, y_train, y_test)
    elif model == 'naive_bayes':
        cm.compare_naive_bayes(X, X_train, X_test, y, y_train, y_test)
    elif model == 'decision_tree':
        cm.compare_decision_tree(X, X_train, X_test, y, y_train, y_test)
    elif model == 'random_forest':
        cm.compare_random_forest(X, X_train, X_test, y, y_train, y_test)


def main():
    args_len = len(sys.argv) - 1
    if args_len == 0:
        print("Target classification model name parameter missing!! Can't proceed with performance check")
    elif args_len == 1:
        calculate_model_performance(sys.argv[1])
    elif args_len == 2:
        calculate_model_performance(sys.argv[1], sys.argv[2])
    else:
        print("Invalid parameter count!!")


if __name__ == '__main__':
    sys.exit(main())
