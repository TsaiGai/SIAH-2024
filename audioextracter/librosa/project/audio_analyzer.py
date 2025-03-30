import os
import librosa
import numpy as np
from pydub import AudioSegment
import argparse

import estimate_key
import estimate_bpm
import estimate_dynamics

def load_all_audio(directory='audio'):
    audio_files = []
    for filename in os.listdir(directory):
        if filename.endswith(('.mp3', '.wav', '.ogg')):  # Add more extensions as needed
            try:
                audio_path = os.path.join(directory, filename)
                audio = AudioSegment.from_file(audio_path)
                audio.filename = filename  # Keep track of the filename
                audio_files.append(audio)

            except Exception as e:
                print(f"Error loading audio file '{filename}': {e}")

    return audio_files

def audiosegment_to_librosa(audio):
    """Convert an AudioSegment to a NumPy array and return it with its sampling rate."""
    y = np.array(audio.get_array_of_samples()).astype(np.float32)
    
    # Normalize to [-1, 1] if it's 16-bit PCM
    if audio.sample_width == 2:  # 16-bit audio
        y /= 32768.0
    elif audio.sample_width == 4:  # 32-bit audio
        y /= 2147483648.0

    # Convert stereo to mono (if necessary)
    if audio.channels > 1:
        y = y.reshape(-1, audio.channels).mean(axis=1)

    return y, audio.frame_rate

def main(audio_files):
    features = []
    for audio in audio_files:
        try:
            y, sr = audiosegment_to_librosa(audio)

            # Extract the BPM and dynamics
            key = estimate_key.extract_key(y, sr)
            bpm = estimate_bpm.extract_bpm(y, sr)
            # rms, _ = estimate_dynamics.get_dynamics(y, sr, frame_length=2048, hop_length=512)
            # dynamic = np.mean(rms)

            # Store the features in a subarray
            features.append({
                'key': key,
                'bpm': bpm,
                # 'average_dynamic': dynamic
            })

        except np.linalg.LinAlgError as e:
            print(f"Linear algebra error occurred during audio processing: {e}")
            continue

        except IndexError as e:
            print(f"Index error occurred during audio processing: {e}")
            continue

        except TypeError as e:
            print(f"Type error occurred during audio processing: {e}")
            continue

        except ValueError as e:
            print(f"Value error occurred during audio processing: {e}")
            continue

        except Exception as e:
            print(f"An error occurred during audio processing: {e}")
            continue

    return features

if __name__ == "__main__":
    audio_files = load_all_audio()
    features = main(audio_files)
    print(features)
