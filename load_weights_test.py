import keras
from keras.models import Model, Sequential
from keras.layers import LSTM, Dense
from keras.utils import plot_model

data_dim = 16
timesteps = 8
num_classes = 10

model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.load_weights(filepath="data.h5")

plot_model(model, to_file='loaded_model.png', show_shapes=True)

layer = model.get_layer(index=1)
print(layer.get_weights())
print(layer.get_config())
