import librosa
import numpy as np

import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def key_to_string(key):
    """
    convert a key index to its corresponding musical key string.

    Args:
        key (int): index of the detected key.

    Returns:
        str: musical key name.
    """
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    return keys[key]

def extract_key(y, sr):
    """
    Extracts the musical key of an audio signal.

    Args:
        y (numpy.ndarray): audio time series.
        sr (int): sampling rate of `y`.

    Returns:
        str: estimated musical key (e.g., 'C', 'C#', 'D', etc.).
    """
    # apply noise reduction and normalization
    y = librosa.effects.trim(y, top_db=30)
    y = librosa.util.normalize(y)

    # compute the chromagram using constant-Q transform for better pitch detection
    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)

    # apply PCA to reduce dimensionality and improve key estimation
    pca = PCA(n_components=3)
    chromagram_pca = pca.fit_transform(chromagram.T)

    # compute the mean of the chroma energy distribution
    mean_chroma = np.mean(chromagram_pca, axis=0)

    # use the Krumhansl-Schmuckler key-finding algorithm
    key_profile = np.array([
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
    ])
    key_correlation = np.corrcoef(mean_chroma, key_profile)[0, 1]
    key_estimate_index = np.argmax(key_correlation)

    return key_estimate_index

def extract_key_over_time(file_path):
    """
    Extracts the musical key of an audio file every 5 seconds.

    Args:
        file_path (str): Path to the audio file.

    Returns:
        list: List of estimated keys (as strings) for each 5-second chunk.
    """
    # load the audio file
    y, sr = librosa.load(file_path)

    # define the chunk size (5 seconds)
    chunk_size = int(5 * sr)

    # initialize an empty list to store the estimated keys
    keys_over_time = []

    # loop through the audio file in chunks
    for i in range(0, len(y), chunk_size):
        # extract the current chunk
        chunk = y[i:i + chunk_size]

        # apply the extract_key method to the chunk
        key = extract_key(chunk, sr)

        # append the estimated key to the list
        keys_over_time.append(key)
    
    # convert list of keys into musical key strings
    keys_over_time = [key_to_string(key) for key in keys_over_time]

    return keys_over_time

"""
example usage
-------------
filename = "audio/lark-call.wav"
y, sr = librosa.load(filename)
key_estimate = extract_key(y, sr)

print(f"Estimated Key: {key_estimate}")
"""
