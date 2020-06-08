# -*- coding: utf-8 -*-
import os
import yaml
import uuid

property_file = os.path.join("src/utilities/", "properties.yaml")
with open(property_file, "r") as f:
    settings = yaml.load(f)


def list_files_with_extn(directory, file_extn):
    return (directory + file for file in os.listdir(directory) if file.endswith(file_extn))


def get_audio_files_for_chord(chord):
    return list_files_with_extn(settings["chord_audio_data_set"][chord],
                                settings["chord_audio_data_set"]["file_extn"])


def get_chroma_dataset_file_path(chroma_type):
    return get_file_path(file_path=settings["chromagram_data_set"]["file_path"],
                         file_name=settings["chromagram_data_set"][chroma_type],
                         file_extn=settings["chromagram_data_set"]["file_extn"])


def get_audio_recording_file_path():
    return get_file_path(file_path=settings["record_audio"]["file_path"],
                         file_name=settings["record_audio"]["file_name"] + str(uuid.uuid4()),
                         file_extn=settings["record_audio"]["file_extn"])


def get_file_path(file_path, file_name, file_extn):
    os.makedirs(file_path, exist_ok=True)
    return os.path.join(file_path, file_name + file_extn)


def get_chord_list():
    return settings["chords"]
