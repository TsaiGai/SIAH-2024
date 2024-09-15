import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# load the data
X = np.load('preprocessed_data/X.npy') # shape (samples, timestamps, features)
y = np.load('preprocessed_data/y.npy') # shape (samples,)

# ensure y has the correct shape
y = y.reshape(-1)

# define the model
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))  # input shape based on X
model.add(Dense(20))  # in this case, predicting the next 20 values

# compile the model
model.compile(optimizer='adam', loss='mse')

# add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# train the model
history = model.fit(X, y, epochs=20, validation_split=0.2, batch_size=4, callbacks=[early_stopping])

# predict the next 20 values
n_steps = 20
last_sequence = X[-1:]  # start with the last sequence in your dataset
predicted_values = []

for _ in range(n_steps):
    predicted_value = model.predict(last_sequence)
    predicted_values.append(predicted_value[0, 0])
    
    # update the sequence: remove the oldest value and append the predicted value
    # last_sequence is of shape (1, 1, 1), predicted_value is of shape (1, 1)
    predicted_value_reshaped = predicted_value.reshape(1, 1, 1)  # reshape for concatenation
    last_sequence = np.append(last_sequence[:, 1:, :], predicted_value_reshaped, axis=1)

# output
print("Previously existing sequence:", y[-n_steps:])
print("Predicted next values:", predicted_values)
