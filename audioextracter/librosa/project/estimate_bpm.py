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

    # noise reduction using wavelet denoising
    y = librosa.effects.denoise(y, method='wavelet')

    # use a harmonic-percussive source separation to isolate percussive elements
    y_harmonic, y_percussive = librosa.effects.hpss(y)

    # more advanced beat tracking with tightness parameter
    tempo_hpss, _ = librosa.beat.tempo(y=y_percussive, sr=sr, tightness=5)

    # multi-resolution analysis using wavelet transform
    cwt = librosa.cwt(y_percussive, sr=sr)
    tempo_cwt, _ = librosa.beat.tempo(cwt, sr=sr)

    # combine tempo estimates using a weighted average
    tempo_final = 0.5 * tempo_hpss + 0.5 * tempo_cwt

    return tempo_final