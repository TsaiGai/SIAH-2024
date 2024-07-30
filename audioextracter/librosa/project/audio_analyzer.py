import os
import librosa
import numpy as np
from pydub import AudioSegment
import argparse

import estimate_key
import estimate_bpm
import estimate_dynamics

def load_audio(filename):
    audio_path = os.path.join('audio', filename)
    audio = AudioSegment.from_file(audio_path)
    return audio

def main(filename):
    audio_path = os.path.join('audio', filename)
    y, sr = librosa.load(audio_path)

    key = estimate_key.extract_key(y, sr)
    bpm = estimate_bpm.extract_bpm(y, sr)
    rms, _ = estimate_dynamics.get_dynamics(y, sr, frame_length=2048, hop_length=512)
    dynamic = np.mean(rms)

    print(f"Key: {key}")
    print(f"BPM: {bpm}")
    print(f"Average Dynamic: {dynamic}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process an audio file to extract key, BPM, and dynamics.")
    parser.add_argument("filename", type=str, help="The filename of the audio file to process")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.filename)
