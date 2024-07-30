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

# check if y needs to be one-hot encoded for multi-class classification
if len(np.unique(y)) > 2:
    y = to_categorical(y)
    output_units = y.shape[1]
    activation = 'softmax'
    loss = 'categorical_crossentropy'
else:
    output_units = 1
    activation = 'sigmoid'
    loss = 'binary_crossentropy'

# define the model
model = Sequential()
model.add(LSTM(128, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(output_units, activation=activation))

# compile the model
model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])

# train the model
history = model.fit(X, y, validation_split=0.2, epochs=100, batch_size=64)

# sample input sequence from your data (e.g., the last sequence in X)
input_sequence = X[-1]  # shape: (time_steps, features)

# reshape input to match model's expected input shape: (1, time_steps, features)
input_sequence = np.expand_dims(input_sequence, axis=0)  # shape: (1, time_steps, features)

# initialize the generated sequence with the input sequence
generated_sequence = input_sequence.copy()

print(generated_sequence)

# predict the next 10 steps
predicted_steps = []

for _ in range(10):
    # predict the next step
    next_step = model.predict(generated_sequence)
    
    # append the predicted value to the sequence
    predicted_steps.append(next_step[0, 0])  # Adjust indexing based on your model's output shape
    
    # update the input sequence: remove the first step and add the predicted step
    next_step = next_step.reshape(1, 1, -1)
    generated_sequence = np.append(generated_sequence[:, 1:, :], next_step, axis=1)

print("Predicted next 10 steps:", predicted_steps)
