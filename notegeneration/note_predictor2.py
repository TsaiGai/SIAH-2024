import os
import numpy as np
import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from extract_features import extract_key_features

def load_all_audio(directory='audio'):
    """Load all audio files in a directory using librosa."""
    audio_data = []
    file_paths = []
    
    for filename in os.listdir(directory):
        if filename.endswith(('.mp3', '.wav', '.ogg')):
            try:
                audio_path = os.path.join(directory, filename)
                y, sr = librosa.load(audio_path, sr=None)
                audio_data.append((y, sr))  # store waveform and sample rate
                file_paths.append(audio_path)
                
            except Exception as e:
                print(f"Error loading audio file '{filename}': {e}")

    return audio_data, file_paths  # return both audio data and file paths

# load dataset
audio_data, audio_files = load_all_audio("audio")

X = []
for y_raw, sr in audio_data:
    features = extract_key_features(y_raw, sr)
    if features is not None:
        X.append(features)

X = np.array(X)
print("Final shape of X:", X.shape)

if X.ndim == 2:  # ensure it has two dimensions before reshaping
    X = X.reshape(X.shape[0], 1, X.shape[1])  # (samples, timestamps=1, features)
else:
    raise ValueError("Unexpected shape of X, check feature extraction.")


# dummy target variable (replace with actual targets)
y = np.random.rand(X.shape[0])  # assuming regression task

# define LSTM model
model = Sequential([
    LSTM(50, input_shape=(X.shape[1], X.shape[2])),
    Dense(1)  # adjust based on target variable
])

# compile model
model.compile(optimizer='adam', loss='mse')

# early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# train the model
history = model.fit(X, y, epochs=20, validation_split=0.2, batch_size=4, callbacks=[early_stopping])

# define noise leve
noise_level = 0.1

# predict future values using extracted features
n_steps = 20
last_sequence = X[-1:]  # start with last extracted feature sequence
print(last_sequence.shape)
predicted_values = []

for _ in range(n_steps):
    predicted_value = model.predict(last_sequence)

    # add stochasticity with Gaussian noise
    noisy_prediction = predicted_value[0, 0] + np.random.normal(0, noise_level)
    predicted_values.append(noisy_prediction)
    
    # update the sequence: Shift left and append predicted value
    predicted_value_reshaped = np.full((1, 1, 48), noisy_prediction)  # (1, 1, 48) with the scalar value
    last_sequence = np.append(last_sequence[:, 1:, :], predicted_value_reshaped, axis=1)

print("Predicted next values:", predicted_values)
