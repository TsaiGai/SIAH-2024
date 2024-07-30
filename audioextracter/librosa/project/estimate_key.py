import librosa
import numpy as np

def extract_key(y, sr):
    chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
    mean_chroma = np.mean(chromagram, axis=1)
    key_estimate_index = np.argmax(mean_chroma)
    key_string = key_to_string(key_estimate_index)

    return key_string

def key_to_string(key):
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    return keys[key]

# filename = "audio/lark-call.wav"
# y, sr = librosa.load(filename)
# key_estimate = extract_key(y, sr)

# print(f"Estimated Key: {key_string}")
