import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import glob

FIXED_CHROMA_LENGTH = 12
SEQUENCE_LENGTH = 2

def extract_features(file_path):
    """
    extracts the main chroma feature from an audio file.

    Args:
        file_path (str): the path to the audio file.

    Returns:
        int or None: the rounded index of the maximum chroma feature, or None if an error occurs.
    """
    try:
        y, sr = librosa.load(file_path)

        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
        mean_chroma = np.mean(chromagram, axis=1)
        if len(mean_chroma) < FIXED_CHROMA_LENGTH:
            mean_chroma = np.pad(mean_chroma, (0, FIXED_CHROMA_LENGTH - len(mean_chroma)), mode='constant')
        else:
            mean_chroma = mean_chroma[:FIXED_CHROMA_LENGTH]

        return round(np.argmax(mean_chroma))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_sequences(features, sequence_length):
    """
    creates sequences of features for training.

    Args:
        features (np.ndarray): array of extracted features.
        sequence_length (int): the length of each sequence.

    Returns:
        np.ndarray: array of sequences of features.
    """
    sequences = []

    for i in range(len(features) - sequence_length + 1):
        sequences.append(features[i:i + sequence_length])

    return np.array(sequences)

def load_audio_files(directory='audio', extensions=('wav', 'mp3')):
    """
    loads audio files from a specified directory.

    Args:
        directory (str): the directory to search for audio files.
        extensions (tuple): file extensions to include in the search.

    Returns:
        list: list of file paths that match the given extensions.
    """
    return [file for ext in extensions for file in glob.glob(f'{directory}/*.{ext}')]

def normalize_features(features):
    """
    normalizes feature values to a range of 0 to 1.

    Args:
        features (list): list of extracted features.

    Returns:
        np.ndarray: normalized feature array.
    """
    scaler = MinMaxScaler()

    return scaler.fit_transform(np.array(features).reshape(-1, 1))

def save_data(X, y, output_dir='preprocessed_data'):
    """
    saves the processed input (X) and output (y) data to files.

    Args:
        X (np.ndarray): input data array.
        y (np.ndarray): output data array.
        output_dir (str): directory where the data will be saved.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)

# load and process audio files
file_paths = load_audio_files()

features = []
for fp in file_paths:
    feature = extract_features(fp)

    if feature is not None:
        features.append(feature)

if len(features) < SEQUENCE_LENGTH:
    raise ValueError("not enough valid audio files to create sequences with the specified sequence length")

# normalize and create sequences
features = normalize_features(features)
sequences = create_sequences(features, sequence_length=SEQUENCE_LENGTH)

# prepare input (X) and output (y) data
X = sequences[:, :-1, :]  # all but the last time step
y = sequences[:, -1, :]  # the last time step

# if predicting a specific output feature
y = y[:, 0]

# ensure input shape is correct
number_of_samples = X.shape[0]
timesteps = X.shape[1]
input_dim = X.shape[2]

X = X.reshape((number_of_samples, timesteps, input_dim))

print(X)
print(y)

# save the data
save_data(X, y)
