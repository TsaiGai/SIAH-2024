import librosa
import numpy as np

def extract_bpm(y, sr):
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    return bpm

# filename = "audio/lark-call.wav"
# y, sr = librosa.load(filename)
# bpm = extract_bpm(y, sr)

# print(f"BPM: {bpm}")