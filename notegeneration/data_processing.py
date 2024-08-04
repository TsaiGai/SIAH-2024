import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import glob

FIXED_CHROMA_LENGTH = 12
SEQUENCE_LENGTH = 2

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path)

        chromagram = librosa.feature.chroma_stft(y=y, sr=sr)
        mean_chroma = np.mean(chromagram, axis=1)
        if len(mean_chroma) < FIXED_CHROMA_LENGTH:
            mean_chroma = np.pad(mean_chroma, (0, FIXED_CHROMA_LENGTH - len(mean_chroma)), mode='constant')
        else:
            mean_chroma = mean_chroma[:FIXED_CHROMA_LENGTH]
        key = np.argmax(mean_chroma)
        rounded_key = round(key)
        
        return rounded_key
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def create_sequences(features, sequence_length):
    sequences = []
    for i in range(len(features) - sequence_length + 1):
        sequences.append(features[i:i + sequence_length])
    return np.array(sequences)

# example data
file_paths = glob.glob('audio/*.wav') + glob.glob('audio/*.mp3')  # list of audio files with pattern matching

# extract features for all files
features = []
for fp in file_paths:
    feature = extract_features(fp)
    if feature is not None:
        features.append(feature)

# print(len(features))
if len(features) < SEQUENCE_LENGTH:
    raise ValueError("not enough valid audio files to create sequences with the specified sequence length")

features = np.array(features).reshape(-1, 1)

sequences = create_sequences(features, sequence_length=SEQUENCE_LENGTH)

# normalize features
# scaler = MinMaxScaler()
# features_normalized = scaler.fit_transform(features)

# # create sequences
# sequences_normalized = create_sequences(features_normalized, sequence_length=SEQUENCE_LENGTH)

# prepare input (X) and output (y) data
X = sequences[:, :-1, :]  # all but the last time step
y = sequences[:, -1, :]  # the last time step

# if predicting a specific output feature
y = y[:, 0]

# ensure input shape is correct
number_of_samples = X.shape[0]
timesteps = X.shape[1]
input_dim = X.shape[2]

X = X.reshape((number_of_samples, timesteps, input_dim))

print(X)
print(y)

# save the data
output_dir = 'preprocessed_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

np.save(os.path.join(output_dir, 'X.npy'), X)
np.save(os.path.join(output_dir, 'y.npy'), y)
