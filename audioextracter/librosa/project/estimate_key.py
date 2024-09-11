import librosa
import numpy as np

def extract_key(y, sr):
    """
    extracts the musical key of an audio signal.

    Args:
        y (numpy.ndarray): audio time series.
        sr (int): sampling rate of `y`.

    Returns:
        str: estimated musical key (e.g., 'C', 'C#', 'D', etc.).
    """
    # compute the chromagram using constant-Q transform for better pitch detection
    chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)

    # calculate the mean of the chroma energy distribution
    mean_chroma = np.mean(chromagram, axis=1)
    
    # find the index of the maximum chroma value to estimate the key
    key_estimate_index = np.argmax(mean_chroma)
    key_string = key_to_string(key_estimate_index)

    return key_string

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

"""
example usage
-------------
filename = "audio/lark-call.wav"
y, sr = librosa.load(filename)
key_estimate = extract_key(y, sr)

print(f"Estimated Key: {key_estimate}")
"""
