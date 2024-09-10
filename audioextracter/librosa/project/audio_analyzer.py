import os
import librosa
import numpy as np
from pydub import AudioSegment
import argparse

import estimate_key
import estimate_bpm
import estimate_dynamics

def load_audio(filename):
    """
    load an audio file using.

    args:
        filename (str): the name of the audio file to load.

    returns:
        AudioSegment: the loaded audio file.
    
    raises:
        FileNotFoundError: if the file does not exist.
        Exception: if the file format is unsupported or an error occurs during loading.
    """
    try:
        audio_path = os.path.join('audio', filename)

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File '{audio_path}' does not exist.")
        
        audio = AudioSegment.from_file(audio_path)

        return audio
    except FileNotFoundError as fnf_error:
        print(fnf_error)

        raise
    except Exception as e:
        print(f"Error loading audio file: {e}")
        
        raise

def main(filename):
    """
    main function to process an audio file to extract key, BPM, and dynamics.

    Args:
        filename (str): the name of the audio file to process.
    """
    try:
        audio_path = os.path.join('audio', filename)
        
        # load the audio file using librosa
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"File '{audio_path}' does not exist.")
        
        y, sr = librosa.load(audio_path)
        
        # extract audio features
        key = estimate_key.extract_key(y, sr)
        bpm = estimate_bpm.extract_bpm(y, sr)
        rms, _ = estimate_dynamics.get_dynamics(y, sr, frame_length=2048, hop_length=512)
        dynamic = np.mean(rms)
        
        # display extracted features
        print(f"Key: {key}")
        print(f"BPM: {bpm}")
        print(f"Average Dynamic: {dynamic}")

    except FileNotFoundError as fnf_error:
        print(fnf_error)
    except Exception as e:
        print(f"An error occurred during audio processing: {e}")

def parse_arguments():
    """
    parse command-line arguments for the script.

    Returns:
        Parser: parsed command-line arguments containing the filename.
    """
    parser = argparse.ArgumentParser(description="Process an audio file to extract key, BPM, and dynamics.")
    parser.add_argument("filename", type=str, help="The filename of the audio file to process")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.filename)
