import sys
import os
import joblib
import numpy as np
from source.utilities import configure_util as configure
from source.utilities import model_util as mu
from source.models import tuned_classification_models as tuned_model
from source.models import classification_models as standard_model
from feature_extraction import extract_audio_features

target_models = ['logistic', 'knn', 'svm', 'kernel_svm', 'naive_bayes', 'decision_tree', 'random_forest']
target_chroma_type = ['stft', 'cqt', 'cens']


# compare different models with and without hyperparameter tuning
def create_and_save_model(model, chroma_type="cqt"):
    if model not in target_models:
        print("Invalid classification model name parameter!!")
        sys.exit(-1)

    if chroma_type not in target_chroma_type:
        print("Invalid chroma type!!")
        sys.exit(-1)

    chroma_feature_csv = configure.get_chroma_dataset_file_path(chroma_type)
    if not os.path.exists(chroma_feature_csv):
        extract_audio_features(chroma_type)
    X, y, y_classes = mu.process_data_set(chroma_feature_csv)

    print("Starting training the model...")
    if model == 'logistic':
        classifier = tuned_model.get_logistic_regression_model(X, y)
    elif model == 'knn':
        classifier = tuned_model.get_knn_model(X, y)
    elif model == 'svm':
        classifier = tuned_model.get_svm_model(X, y)
    elif model == 'kernel_svm':
        classifier = tuned_model.get_kernel_svm_model(X, y)
    elif model == 'naive_bayes':
        classifier = standard_model.get_naive_bayes_model(X, y)
    elif model == 'decision_tree':
        classifier = tuned_model.get_decision_tree_model(X, y)
    elif model == 'random_forest':
        classifier = tuned_model.get_random_forest_model(X, y)

    save_path = configure.settings['model_save_path']
    print("Finished training the model. Saving it to " + str(save_path))

    # Saving model
    os.makedirs(save_path, exist_ok=True)
    joblib.dump(classifier, os.path.join(save_path, model + ".model"))
    print("Model saved successfully!")

    # Saving label encoder classes
    np.save(os.path.join(save_path, model + "_classes.npy"), y_classes)
    print("Label Encoder for chords saved successfully!")


def main():
    arg_count = len(sys.argv) - 1
    if arg_count == 0:
        print("Target classification model name parameter missing!! Can't proceed with model training.")
    elif arg_count == 1:
        create_and_save_model(sys.argv[1])
    elif arg_count == 2:
        create_and_save_model(sys.argv[1], sys.argv[2])
    else:
        print("Invalid parameter count!!")


if __name__ == '__main__':
    sys.exit(main())
