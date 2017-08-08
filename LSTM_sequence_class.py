from keras.models import Sequential
from keras.layers import LSTM, Dense, Layer
from keras.utils import plot_model
import numpy as np


data_dim = 16
timesteps = 8
num_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
# Sequential model: a linear stack of layers. We can pass a layer list to Sequential or
# use add function to create this model.
model = Sequential()
# Only the first layer of Sequential model need to pass a parameter about input shape,
# then the following layers can do automatic shape inference.
# 32 is the dimensionality of output sequence.
# While return_sequence is True, this layer returns the full sequence, rather than
# the last output in the output sequence.
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
# Dense is a regular densely-connected NN layer.
# activation is the Activation function to use.
model.add(Dense(10, activation='softmax'))

# By running compile function ,we configures the learning process.
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.random((1000, num_classes))

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.random((100, num_classes))

# Trains the model <epochs> times.
# batch_size: Number of samples per gradient update.
# x_train: input data, a Numpy list of Numpy arrays.
# y_train: labels, a Numpy array.
model.fit(x_train, y_train,
          batch_size=64, epochs=5,
          validation_data=(x_val, y_val))

model.save_weights("data.h5")

plot_model(model, to_file='model.png', show_shapes=True)

layer = model.get_layer(index=1)
print(layer.get_weights())
print(layer.get_config())
