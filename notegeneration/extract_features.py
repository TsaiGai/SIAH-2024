import numpy as np
import librosa

# define the Krumhansl key profiles (major and minor)
KRUMHANSL_MAJOR = np.array([
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],  # C Major
    [2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29],  # C# Major
    [2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66],  # D Major
    [3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39],  # D# Major
    [2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19],  # E Major
    [5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52],  # F Major
    [2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09],  # F# Major
    [4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33, 4.38],  # G Major
    [4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48, 2.33],  # G# Major
    [2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23, 3.48],  # A Major
    [3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35, 2.23],  # A# Major
    [2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88, 6.35]   # B Major
])

KRUMHANSL_MINOR = np.roll(KRUMHANSL_MAJOR, shift=3, axis=1)  # rotate profiles for minor keys

KEY_NAMES_MAJOR = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KEY_NAMES_MINOR = ["Cmin", "C#min", "Dmin", "D#min", "Emin", "Fmin", "F#min", "Gmin", "G#min", "Amin", "A#min", "Bmin"]

def extract_key_features(y, sr):
    """
    Extracts features for musical key detection from an audio signal.

    Args:
        y (numpy.ndarray): Audio time series.
        sr (int): Sampling rate of `y`.

    Returns:
        dict: A dictionary containing extracted key features.
    """
    # validate inputs
    if not isinstance(y, np.ndarray):
        raise TypeError("Input audio signal must be a numpy array.")
    if not isinstance(sr, int) or sr <= 0:
        raise ValueError("Sampling rate must be a positive integer.")
    if len(y) == 0:
        raise ValueError("Input audio signal is empty.")

    try:
        # trim silence and normalize audio
        y, _ = librosa.effects.trim(y, top_db=30)
        y = librosa.util.normalize(y)

        # compute chromagram using constant-Q transform
        chromagram = librosa.feature.chroma_cqt(y=y, sr=sr)

        # compute mean and variance of chroma energy (12 values each)
        mean_chroma = np.mean(chromagram, axis=1)  
        var_chroma = np.var(chromagram, axis=1)

        # compute correlation scores for each key profile
        correlations_major = [np.corrcoef(mean_chroma, key_profile)[0, 1] for key_profile in KRUMHANSL_MAJOR]
        correlations_minor = [np.corrcoef(mean_chroma, key_profile)[0, 1] for key_profile in KRUMHANSL_MINOR]

        # flatten the feature set into a single array
        feature_array = np.hstack([
            mean_chroma,        # 12 values
            var_chroma,         # 12 values
            correlations_major, # 12 values
            correlations_minor  # 12 values
        ])

        return feature_array  # Shape: (48,)

    except Exception as e:
        raise RuntimeError(f"Error extracting key features: {e}")
