""" This script demonstrates the use of a convolutional LSTM network.
This network is used to predict the next frame of an artificially
generated movie which contains moving squares.
"""
from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Activation
from keras.layers.core import Reshape, Flatten
import numpy as np
import pylab as plt

from skimage.io import imread
from skimage.transform import resize

# We create a layer which take as input movies of shape
# (n_frames, width, height, channels) and returns a movie
# of identical shape.

seq = Sequential()

seq.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                   input_shape=(None, 40, 40, 1),
                   batch_input_shape=(None, 11, 40, 40, 1),
                   border_mode='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                   border_mode='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Flatten())
seq.add(Dense(128))
seq.add(Dense(1600*11))
seq.add(Reshape((11, 40, 40, 1)))

seq.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                   border_mode='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                   border_mode='same', return_sequences=True))
seq.add(BatchNormalization())
seq.add(Convolution3D(nb_filter=1, kernel_dim1=1, kernel_dim2=3,
                      kernel_dim3=3, activation='sigmoid',
                      border_mode='same', dim_ordering='tf'))
seq.compile(loss='binary_crossentropy', optimizer='adadelta')

def image_data(file):
    img = imread(file, as_grey=True)
    img = resize(img, (40, 40))
    return img

# Artificial data generation:
# Generate movies with 3 to 7 moving squares inside.
# The squares are of shape 1x1 or 2x2 pixels,
# which move linearly over time.
# For convenience we first create movies with bigger width and height (80x80)
# and at the end we select a 40x40 window.

def generate_movies(n_samples=1200, n_frames=11, dataFolder=None):
    row = 40
    col = 40
    
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, 11, row, col, 1), dtype=np.float)

    for i in range(n_samples):
        # read custom dataset
        for t in range(n_frames):
            # read and resize to 40x40
            x = image_data(dataFolder + "/" + str(i) + "_" + str(200+t) + ".png")
            shifted_movies[i, t, :, :, 0] = x
            
            # figure out how to add noise to this data
            noise_f = (-1)**np.random.randint(0, 2, size=(40,40))
            noisy_movies[i, t, :, :, 0] = np.dot(x, noise_f)
                
    return noisy_movies, shifted_movies

# Train the network
noisy_movies, shifted_movies = generate_movies(n_samples=10, dataFolder="../../../../_swarm-lfd-data/Flocking")

# original nb_epoch = 500

seq.fit(noisy_movies, shifted_movies, batch_size=10,
        nb_epoch=1, validation_split=0.05)

# save model as JSON
model = seq
modelAsJson = model.to_json()
with open("conv_lstm_model.json", "w") as jsonFile:
    jsonFile.write(modelAsJson)
# serialize weights to HDF5
model.save_weights('conv_lstm_model.h5')

'''
# Testing the network on one movie
# feed it with the first 7 positions and then
# predict the new positions
which = 1004
track = noisy_movies[which][:7, ::, ::, ::]

for j in range(16):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)
'''
