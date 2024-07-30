import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def get_dynamics(y, sr, frame_length=2048, hop_length=512):
    # Calculate the RMS energy
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
    
    return rms, times

def plot_dynamics(y, sr, rms, times):
    # Plot the waveform
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.title('Waveform')
    
    # Plot the RMS energy
    plt.subplot(2, 1, 2)
    plt.plot(times, rms, label='RMS Energy', color='r')
    plt.title('RMS Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS')
    plt.legend()
    plt.tight_layout()
    plt.show()

# filename = "audio/lark-call.wav"
# y, sr = librosa.load(filename)
# rms, times = get_dynamics(y, sr)
# plot_dynamics(y, sr, rms, times)

# Print summary statistics for RMS energy
# print(f"Mean RMS: {np.mean(rms)}")
# print(f"Max RMS: {np.max(rms)}")
# print(f"Min RMS: {np.min(rms)}")
