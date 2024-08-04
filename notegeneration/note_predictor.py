import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# HARDCODED_DIMENSION = 15

# load the data
# X = np.load('preprocessed_data/X.npy')
# y = np.load('preprocessed_data/y.npy')

# # if y is not one-hot encoded for multi-class classification
# if len(np.unique(y)) > 2:
#     y = to_categorical(y)

# # define the model
# model = Sequential()
# model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
# model.add(LSTM(128))
# # if y.shape[1] > 1:
# #     model.add(Dense(y.shape[1], activation='softmax'))
# #     loss = 'categorical_crossentropy'
# # else:
# model.add(Dense(1, activation='sigmoid'))
# loss = 'binary_crossentropy'

# model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

# # train the model
# history = model.fit(X, y, validation_split=0.2, epochs=100, batch_size=64)

# # sample input sequence from your data (e.g., the last sequence in X)
# input_sequence = X[-1]  # shape: (time_steps, features)

# # reshape input to match model's expected input shape: (1, time_steps, features)
# input_sequence = np.expand_dims(input_sequence, axis=0)  # shape: (1, time_steps, features)

# # initialize the generated sequence with the input sequence
# generated_sequence = input_sequence.copy()

# print(generated_sequence)


# number of steps to generate
# num_steps_to_generate = 10

# predicted_steps = []

# for _ in range(num_steps_to_generate):
#     # predict the next step
#     predicted_step = model.predict(input_sequence)

#     if predicted_step.ndim == 2:
#         predicted_step = np.expand_dims(predicted_step, axis=1)

#     # hardcode the shape to match (1, 1, 15)
#     hardcoded_shape = (1, 1, 15)

#     # create an array of zeros with the desired shape
#     padded_predicted_step = np.zeros(hardcoded_shape)

#     # fill the array with the predicted values
#     num_predicted_features = min(predicted_step.shape[-1], hardcoded_shape[-1])
#     padded_predicted_step[0, 0, :num_predicted_features] = predicted_step[0, 0, :num_predicted_features]

#     # append the hardcoded predicted step to the predicted_steps array
#     predicted_steps.append(padded_predicted_step)

#     # update the sequence by appending the hardcoded predicted step
#     generated_sequence = np.append(generated_sequence[:, 1:, :], padded_predicted_step, axis=1)

# # convert predicted_steps to a numpy array
# predicted_steps = np.concatenate(predicted_steps, axis=1)

# print("Final Generated Sequence:", generated_sequence)
# print("Predicted Steps:", predicted_steps)



# load the data
X = np.load('preprocessed_data/X.npy')
y = np.load('preprocessed_data/y.npy')

# print shapes to verify
print("Shape of X:", X.shape)  # Should be (34, 1, 1)
print("Shape of y:", y.shape)  # Should be (34,)

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

print("Predicted next values:", predicted_values)
