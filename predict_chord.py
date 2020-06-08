import sys
import joblib
import os
from source.utilities import configure_util as configure
from source.utilities import audio_util as au
from source.utilities import features_util as fu
from sklearn.preprocessing.label import LabelEncoder
import numpy as np

from model_training import create_and_save_model

target_models = ['logistic', 'knn', 'svm', 'kernel_svm', 'naive_bayes', 'decision_tree', 'random_forest']
target_chroma_type = ['stft', 'cqt', 'cens']


# record audio chord
def record_audio():
    audio_save_path = configure.get_audio_recording_file_path()
    au.record_and_save_audio(chunk_size=configure.settings["record_audio"]["chunk_size"],
                             channels=configure.settings["record_audio"]["channels"],
                             sample_rate=configure.settings["record_audio"]["sample_rate"],
                             record_seconds=configure.settings["record_audio"]["record_seconds"],
                             audio_save_file_path=audio_save_path)
    return audio_save_path


# record and identify the played chord
def predict_chord(chroma_type="cqt", model="kernel_svm"):
    # record guitar chord
    save_path = record_audio()

    # extract chroma features
    chroma = fu.extract_chroma_vector_for_audio(chroma_type, save_path)

    # load the model & predict the result
    model_save_path = configure.settings['model_save_path']
    saved_model = os.path.join(model_save_path, model + ".model")
    if not os.path.exists(saved_model):
        create_and_save_model(model, chroma_type)
    classifier = joblib.load(open(os.path.join(model_save_path, model + ".model"), 'rb'))
    pred_chord = classifier.predict([chroma])

    # load encoder
    encoder = LabelEncoder()
    saved_label_encoder = os.path.join(model_save_path, model + "_classes.npy")
    if not os.path.exists(saved_label_encoder):
        print("label encoder file doesn't exist! Aborting...")
        sys.exit(-1)

    encoder.classes_ = np.load(saved_label_encoder)
    pred_chord = encoder.inverse_transform(pred_chord)
    print("CHORD PLAYED : " + str(pred_chord[0]))


def process_user_request(args):
    # single argument (either chroma type or model)
    if len(args) == 2:
        if args[1] in target_chroma_type:
            predict_chord(chroma_type=args[1])
        elif args[1] in target_models:
            predict_chord(model=args[1])
        else:
            print("Invalid chroma type or model name")
    # chroma type and model argument present
    elif len(args) == 3:
        predict_chord(chroma_type=args[1], model=args[2])
    else:
        print("Invalid number of arguments!!")
        sys.exit(-1)


def main():
    # no argument (record and predict)
    if len(sys.argv) == 1:
        predict_chord()
    # process other request
    else:
        process_user_request(sys.argv)
    # create_and_save_model("cqt")
    # pred_chord = predict_chord("cqt")
    # print(pred_chord)
    # create_chromagram_dataset("stft")
    # create_chromagram_dataset("cqt")
    # create_chromagram_dataset("cens")
    # compare_classification_model_performance("cqt")


if __name__ == '__main__':
    sys.exit(main())
