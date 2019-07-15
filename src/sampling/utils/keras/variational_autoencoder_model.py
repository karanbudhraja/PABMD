'''This script demonstrates how to build a variational autoencoder with Keras.

Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt
import glob
from skimage.io import imread
from skimage.transform import resize

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives
from keras.datasets import mnist
from keras.models import model_from_json

import math

#batch_size = 100
batch_size = 1
original_dim = 784
latent_dim = 128
intermediate_dim = 256
#nb_epoch = 50
nb_epoch = 1
epsilon_std = 0.01

x = Input(batch_shape=(batch_size, original_dim))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)


def sampling(args):
    z_mean, z_log_var = args

    # karan
    #batch_size = 100
    #latent_dim = 128
    #epsilon_std = 0.01
        
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                              std=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)


def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim * objectives.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(x, x_decoded_mean)
vae.compile(optimizer='rmsprop', loss=vae_loss)

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# load all images in folder

def image_data(file):
	img = imread(file, as_grey=True)
        img = resize(img, (28, 28))
        img = img.flatten()
        return img

def folder_data(folder):
	files = glob.glob(folder + "/*.png")
	print("Calculating descriptors. Number of images is", len(files))
	return [image_data(file) for file in files]

folder = "images/1108"
#folder = "../../../../_swarm-lfd-data/Flocking"
x_folder = folder_data(folder)
splitPoint = int(math.ceil(len(x_folder)*2.0/3.0))
x_train = x_folder[:splitPoint]
x_test = x_folder[splitPoint:]

# artificially increase size of lists
#x_train *= 100
#x_test *= 100

x_train = np.array(x_train)
x_test = np.array(x_test)

print(len(x_train))
print(len(x_test))

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

print(x_train.shape)
print(x_train.shape[1:])

vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=nb_epoch,
        batch_size=batch_size,
        validation_data=(x_test, x_test))

'''
# save model as JSON
model = vae
modelAsJson = model.to_json()
with open("variational_autoencoder_model.json", "w") as jsonFile:
    jsonFile.write(modelAsJson)
# serialize weights to HDF5
model.save_weights('variational_autoencoder_model.h5')
'''

'''
# load json and create model
json_file = open('variational_autoencoder_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("variational_autoencoder_model.h5")

# get output for specific layer
# with a Sequential model
model = loaded_model
print(len(model.layers))

image = image_data("images/1108/1108_200.png")
image = image.astype('float32') / 255.
image = image.reshape((1, np.prod(x_train.shape[1:])))

get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[3].output])

layer_output = get_layer_output([image, 0])[0]
print(len(layer_output[0]))
'''
