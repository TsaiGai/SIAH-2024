import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def get_dynamics(y, sr, frame_length=2048, hop_length=512):
    """
    calculate the RMS energy to measure audio dynamics.

    Args:
        y (numpy.ndarray): audio time series.
        sr (int): sampling rate of `y`.
        frame_length (int): the length of each frame for RMS calculation.
        hop_length (int): the number of samples between frames.

    Returns:
        tuple: RMS energy values and corresponding time stamps.
    """
    # calculate the RMS energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    
    return rms, times

def analyze_dynamics(rms):
    """
    analyze the dynamic range and other statistics of RMS energy.

    Args:
        rms (numpy.ndarray): RMS energy values.

    Returns:
        dict: a dictionary containing statistics related to the dynamics.
    """
    # calculate statistics for the dynamics
    dynamics_stats = {
        "mean_rms": np.mean(rms),
        "max_rms": np.max(rms),
        "min_rms": np.min(rms),
        "std_rms": np.std(rms),  # standard deviation to measure variation
        "dynamic_range": np.max(rms) - np.min(rms)  # difference between max and min RMS
    }
    
    return dynamics_stats

def plot_dynamics(y, sr, rms, times):
    """
    plot the waveform and RMS energy of an audio signal.

    Args:
        y (numpy.ndarray): audio time series.
        sr (int): sampling rate of `y`.
        rms (numpy.ndarray): RMS energy values.
        times (numpy.ndarray): corresponding time stamps for the RMS values.
    """
    # plot the waveform and RMS energy
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.title('Waveform')
    
    plt.subplot(2, 1, 2)
    plt.plot(times, rms, label='RMS Energy', color='r')
    plt.title('RMS Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS')
    plt.legend()
    plt.tight_layout()
    plt.show()

"""
example usage
-------------
filename = "audio/lark-call.wav"
y, sr = librosa.load(filename)
rms, times = get_dynamics(y, sr)
plot_dynamics(y, sr, rms, times)

extract and print statistics related to dynamics
dynamics_stats = analyze_dynamics(rms)
print(f"Mean RMS: {dynamics_stats['mean_rms']}")
print(f"Max RMS: {dynamics_stats['max_rms']}")
print(f"Min RMS: {dynamics_stats['min_rms']}")
print(f"Standard Deviation of RMS: {dynamics_stats['std_rms']}")
print(f"Dynamic Range: {dynamics_stats['dynamic_range']}")
"""
