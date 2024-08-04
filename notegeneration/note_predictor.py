import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# load the data
X = np.load('preprocessed_data/X.npy')
y = np.load('preprocessed_data/y.npy')

# ensure y has the correct shape
y = y.reshape(-1)

# define the model
model = Sequential()
model.add(LSTM(50, input_shape=(1, 1)))  # only one LSTM layer is needed
model.add(Dense(1))  # predicting a single next value

# compile the model
model.compile(optimizer='adam', loss='mse')

# train the model
history = model.fit(X, y, epochs=20, validation_split=0.2, batch_size=4)

# predict the next value for a given sequence
# use the last available sequence for prediction
# last_sequence = X[-1:]  # The last sequence in your dataset
# predicted_next_value = model.predict(last_sequence)
# print("Predicted next value:", predicted_next_value)

# predict the next 10 values
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

print("Previously existing sequence:", last_sequence)
print("Predicted next values:", predicted_values)
