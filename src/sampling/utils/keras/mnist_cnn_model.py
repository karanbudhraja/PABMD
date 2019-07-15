'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import model_from_json
from skimage.io import imread
from skimage.transform import resize

# load json and create model
json_file = open('mnist_cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("mnist_cnn_model.h5")
print("Loaded model from disk")

# the data, shuffled and split between train and test sets
# input image dimensions
nb_classes = 10
img_rows, img_cols = 28, 28
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# test model
model = loaded_model
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# import image and resize
image = imread("409_200.png", as_grey=True)
image = resize(image, (28, 28))

from skimage.io import imshow

image = image.reshape(1, 1, 28, 28)
print(image.shape)
classes = model.predict_classes(image)
print(classes)

# get output for specific layer
from keras import backend as K
# with a Sequential model
print(len(model.layers))
get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[7].output])
layer_output = get_layer_output([image, 0])[0]
print(layer_output)
