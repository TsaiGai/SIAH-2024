import librosa
import numpy as np

def extract_bpm(y, sr):
    """
    extracts the BPM (beats per minute) from an audio signal.

    Args:
        y (numpy.ndarray): the audio time series.
        sr (int): the sampling rate of the audio time series.

    Returns:
        float: estimated BPM of the audio.
    """
    # preprocess the audio to improve beat tracking accuracy
    # and apply a high-pass filter to remove low-frequency noise
    y = librosa.effects.preemphasis(y)

    # use a harmonic-percussive source separation to isolate percussive elements
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # extract tempo (BPM) from the percussive variable
    tempo, _ = librosa.beat.beat_track(y=y_percussive, sr=sr)

    return tempo


# example usage
# -------------
# filename = "audio/lark-call.wav"
# y, sr = librosa.load(filename)
# bpm = extract_bpm(y, sr)

# print(f"BPM: {bpm}")
