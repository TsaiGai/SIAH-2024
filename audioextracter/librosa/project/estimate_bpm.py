import librosa
import numpy as np

def extract_bpm(y, sr):
    """
    Extracts the BPM (beats per minute) from an audio signal.

    Args:
        y (numpy.ndarray): The audio time series.
        sr (int): The sampling rate of the audio time series.

    Returns:
        float: Estimated BPM of the audio.
    """
    # apply pre-emphasis filter to boost high frequencies
    y = librosa.effects.preemphasis(y)

    # harmonic-Percussive Source Separation (HPSS)
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # compute onset envelope
    onset_env = librosa.onset.onset_strength(y=y_percussive, sr=sr)

    # estimate tempo (BPM) using beat tracking
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

    return float(tempo[0])  # return the first tempo estimate as a float

"""
example usage
-------------
filename = "audio/lark-call.wav"
y, sr = librosa.load(filename)
bpm_estimate = extract_bpm(y, sr)

print(f"Estimated BPM: {bpm_estimate}")
"""
