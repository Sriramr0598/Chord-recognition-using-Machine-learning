# -*- coding: utf-8 -*-


import librosa
from src.utilities import configure_util as configure
import numpy as np
import os
import sys


def create_chromagram(chroma_type, target_chords):
    chroma_list = np.empty((0, 13))
    for chord in target_chords:
        audio_data_files = configure.get_audio_files_for_chord(chord)
        for f in audio_data_files:
            print(f)

            if not os.path.exists(f):
                print("Cannot find the audio file: " + str(f))
                sys.exit(-1)

            y, sr = librosa.core.load(f, sr=44100)
            if chroma_type is "stft":
                chroma = np.mean(librosa.feature.chroma_stft(y, sr).T, axis=0)
            elif chroma_type is "cqt":
                chroma = np.mean(librosa.feature.chroma_cqt(y, sr).T, axis=0)
            elif chroma_type is "cens":
                chroma = np.mean(librosa.feature.chroma_cens(y, sr).T, axis=0)
            else:
                print("Invalid chroma type!!!!")
                raise SystemExit

            chroma = np.append(chroma, chord)
            chroma_list = np.vstack((chroma_list, chroma))
    return chroma_list


def create_chromagram_csv_dataset(chroma_type, target_chords, save_path, csv_header):
    print("Extracting " + chroma_type + " chroma features...")
    extracted_data_set = create_chromagram(chroma_type, target_chords)
    np.savetxt(save_path, extracted_data_set, delimiter=',', fmt='%s',
               header=csv_header)
    print("features extracted in csv file : " + str(save_path))


def extract_chroma_vector_for_audio(chroma_type, audio_file):
    print("recorded audio : " + str(audio_file))
    y, sr = librosa.core.load(audio_file, sr=44100)
    if chroma_type is "stft":
        chroma = np.mean(librosa.feature.chroma_stft(y, sr).T, axis=0)
    elif chroma_type is "cqt":
        chroma = np.mean(librosa.feature.chroma_cqt(y, sr).T, axis=0)
    elif chroma_type is "cens":
        chroma = np.mean(librosa.feature.chroma_cens(y, sr).T, axis=0)
    else:
        print("Invalid chroma type!!!!")
        raise SystemExit
    return chroma
