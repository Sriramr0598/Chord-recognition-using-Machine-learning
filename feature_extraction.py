import sys
from source.utilities import configure_util as configure
from source.utilities import features_util as fu


# extract features from audio files and write to csv
def create_chromagram_dataset(chroma_type):
    chords = configure.get_chord_list()
    fu.create_chromagram_csv_dataset(chroma_type=chroma_type, target_chords=chords,
                                     save_path=configure.get_chroma_dataset_file_path(chroma_type),
                                     csv_header=configure.settings['chromagram_data_set']['header'])


def extract_audio_features(chroma_type):
    if chroma_type is None:
        create_chromagram_dataset('stft')
        create_chromagram_dataset('cqt')
        create_chromagram_dataset('cens')
    elif chroma_type == 'stft':
        create_chromagram_dataset('stft')
    elif chroma_type == 'cqt':
        create_chromagram_dataset('cqt')
    elif chroma_type == 'cens':
        create_chromagram_dataset('cens')
    else:
        print("Invalid chroma_type!!")


def main():
    if len(sys.argv) == 1:
        extract_audio_features(chroma_type=None)
    else:
        extract_audio_features(sys.argv[1])


if __name__ == '__main__':
    sys.exit(main())
