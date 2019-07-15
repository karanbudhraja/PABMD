# libraries used
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.io import imread
from skimage.transform import resize
from keras import backend as K
from keras.models import model_from_json
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array

from keras.models import Sequential
from keras.layers.convolutional import Convolution3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Activation
from keras.layers.core import Reshape, Flatten

import os

# using magic numbers for now

# feature extraction initialization
def get_network(descriptorSetting, pathPrefix=None):
    # load feature extractor if needed
    net = None

    # prefix to folder location
    if(pathPrefix == None):
        pathPrefix = ""

    functionName = "get_" + descriptorSetting.lower() + "_network"
    if(functionName in globals().keys()):
        net = globals()[functionName](pathPrefix)

    return net

# feature extraction
def get_features(net, imageName, descriptorSetting):
    # initialization
    features = None

    functionName = "get_" + descriptorSetting.lower() + "_features"
    features = globals()[functionName](net, imageName)
        
    # return features
    return features

# saved keras network
def get_saved_keras_network(modelFile, weightFile):
    # load and compile saved network
    jsonFile = open(modelFile, 'r')
    modelJson = jsonFile.read()
    jsonFile.close()
    net = model_from_json(modelJson)
    net.load_weights(weightFile)
    net.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # return network
    return net

# contour information
def get_contour_features(net, imageName):
    imageData = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)

    # dilate for better contour extraction
    kernel = np.ones((5,5),np.uint8)
    dilatedImageData = cv2.dilate(imageData, kernel, iterations=4)
    #cv2.imwrite(imageName[:-4] + "_dilated.png", dilatedImageData)

    # extract contours
    ret, threshold = cv2.threshold(dilatedImageData, 127, 255, 0)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contouredImageData = np.zeros(dilatedImageData.shape)
    for i in xrange(len(contours)):
        cv2.drawContours(contouredImageData, contours, i, (255, 255, 255), 1, 8, hierarchy)                                    
    #cv2.imwrite(imageName[:-4] + "_contoured.png", contouredImageData)

    # extract largest contour area and perimeter
    areas = []
    perimeters = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        areas.append(area)
        perimeters.append(perimeter)

    # define dependent values
    maxArea = max(areas)
    maxPerimeter = max(perimeters)
    features = [maxArea, maxPerimeter]

    features = [np.sum(areas)]
    
    return features

import sys

# density information
def get_density_features(net, imageName, blocks=eval(os.environ["NUM_DEPENDENT"])):
    # read image
    image = imread(imageName, as_grey=True)
    
    # resize to a pixels power of 2
    # required for equal division when splitting
    resizeDimension = 2**int(min([np.log2(x) for x in image.shape]))
    image = resize(image, (resizeDimension, resizeDimension), mode="wrap")
    
    # polarize pixels
    threshold = np.amax(image)
    image[image < threshold] = 0
    image[image >= threshold] = 1

    # segment image
    # segments must be a power of 4
    # because each block is split into 4 sub blocks
    currentBlocksCount = 1
    blocksList = np.array([image])
    
    while(currentBlocksCount < blocks):
        # split horizontally
        subBlocksList = []
        for block in blocksList:
            subBlocks = np.hsplit(block, 2)
            subBlocksList.extend(subBlocks)
        blocksList = subBlocksList

        # split vertically
        subBlocksList = []
        for block in blocksList:
            subBlocks = np.vsplit(block, 2)
            subBlocksList.extend(subBlocks)
        blocksList = subBlocksList

        # update number of blocks
        currentBlocksCount *= 4
        
    # compute density
    # since all images are the same size, we can use count instead
    features = [x.sum() for x in blocksList]
    
    return features
    
# vgg network
def get_vgg_network(pathPrefix):
    # pretrained vgg16 network
    net = VGG16(include_top=True, weights='imagenet', input_tensor=None)
    return net
    
# vgg based features
def get_vgg_features(net, imageName):
    # preprocess image
    imageData = image.load_img(imageName, target_size=(224, 224))
    imageData = image.img_to_array(imageData)
    imageData = np.expand_dims(imageData, axis=0)
    imageData = preprocess_input(imageData)

    # final layer output
    features = net.predict(imageData)[0]
    return features

# lbp features
def get_lbp_features(net, imageName):
    # settings for LBP
    METHOD = 'uniform'
    radius = 3
    nPoints = 8 * radius
    nBins = 128
    
    # dilate for better contour extraction
    imageData = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5,5),np.uint8)
    dilatedImageData = cv2.dilate(imageData, kernel, iterations=4)
    #cv2.imwrite(imageName[:-4] + "_dilated.png", dilatedImageData)

    # compute normalized histogram distribution
    lbp = local_binary_pattern(dilatedImageData, nPoints, radius, METHOD)
    hist, binEdges = np.histogram(lbp, bins=nBins, range=(0, nBins), density=True)

    # define dependent values
    features = hist
    return features
    
# mnist network
def get_mnist_network(pathPrefix):
    # path to saved network
    modelFile = pathPrefix + "keras/mnist_cnn_model.json"
    weightFile = pathPrefix + "keras/mnist_cnn_model.h5"
    
    # return network
    net = get_saved_keras_network(modelFile, weightFile)
    return net

# mnist based convolutional features
def get_mnist_features(net, imageName):
    # output layer specification
    LAYER_NUMBER = 7
    LEARNING_PHASE = 0

    # dilate for better contour extraction
    imageData = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5,5),np.uint8)
    dilatedImageData = cv2.dilate(imageData, kernel, iterations=4)
    #cv2.imwrite(imageName[:-4] + "_dilated.png", dilatedImageData)

    imageData = resize(dilatedImageData, (28, 28))
    imageData = imageData.reshape(1, 1, 28, 28)

    # layer output
    get_layer_output = K.function([net.layers[0].input, K.learning_phase()],
                                  [net.layers[LAYER_NUMBER].output])
    layerOutput = get_layer_output([imageData, LEARNING_PHASE])[0]
    features = layerOutput[0]
    return features

# resnet network
def get_resnet50_network(pathPrefix):
    # pretrained resnet50 network
    net = ResNet50(include_top=True, weights='imagenet', input_tensor=None)
    return net

# resnet based features
def get_resnet50_features(net, imageName):
    # preprocess image
    dimensionOrdering = K.image_dim_ordering()
    mean = (103.939, 116.779, 123.68)
    imageData = load_img(imageName, target_size=(224, 224))
    imageData = img_to_array(imageData, dim_ordering=dimensionOrdering)

    if dimensionOrdering == 'th':
        imageData[0, :, :] -= mean[0]
        imageData[1, :, :] -= mean[1]
        imageData[2, :, :] -= mean[2]
        # 'RGB'->'BGR'
        imageData = imageData[::-1, :, :]
    else:
        imageData[:, :, 0] -= mean[0]
        imageData[:, :, 1] -= mean[1]
        imageData[:, :, 2] -= mean[2]
        imageData = imageData[:, :, ::-1]

    imageData = np.expand_dims(imageData, axis=0)

    # final layer output
    features = net.predict(imageData)[0]
    return features

# autoencoder network
def get_vae_network(pathPrefix):
    # path to saved network
    modelFile = pathPrefix + "keras/variational_autoencoder_model.json"
    weightFile = pathPrefix + "keras/variational_autoencoder_model.h5"

    # return network
    net = get_saved_keras_network(modelFile, weightFile)
    return net

# autoencoder based features
def get_vae_features(net, imageName):
    # output layer specification
    LAYER_NUMBER = 3
    LEARNING_PHASE = 0
    X_TRAIN_SHAPE = tuple([1,784])

    # preprocess image
    imageData = imread(imageName, as_grey=True)
    imageData = resize(imageData, (28, 28))
    imageData = imageData.flatten()
    imageData = imageData.astype('float32') / 255.
    imageData = imageData.reshape((1, np.prod(X_TRAIN_SHAPE[1:])))

    # layer output
    get_layer_output = K.function([net.layers[0].input, K.learning_phase()],
                                  [net.layers[LAYER_NUMBER].output])
    layer_output = get_layer_output([imageData, LEARNING_PHASE])[0]
    features = layer_output[0]
    return features

# conv lstm network
def get_lstm_network(pathPrefix):
    # path to saved network
    modelFile = pathPrefix + "keras/conv_lstm_model.json"
    weightFile = pathPrefix + "keras/conv_lstm_model.h5"

    #net = get_saved_keras_network_lstm(modelFile, weightFile)

    # manually specify network
    # need to do this because convLTSM layer gives error
    net = Sequential()
    net.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       input_shape=(None, 40, 40, 1),
                       batch_input_shape=(None, 11, 40, 40, 1),
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(Flatten())
    net.add(Dense(128))
    net.add(Dense(1600*11))
    net.add(Reshape((11, 40, 40, 1)))
    net.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(ConvLSTM2D(nb_filter=40, nb_row=3, nb_col=3,
                       border_mode='same', return_sequences=True))
    net.add(BatchNormalization())
    net.add(Convolution3D(nb_filter=1, kernel_dim1=1, kernel_dim2=3,
                          kernel_dim3=3, activation='sigmoid',
                          border_mode='same', dim_ordering='tf'))
    net.load_weights(weightFile)
    net.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    # return network
    return net

# convolutional lstm based features
def get_lstm_features(net, imageName):
    # output layer specification
    LAYER_NUMBER = 5
    LEARNING_PHASE = 0

    # generate movie
    rows = 40
    columns = 40
    numberOfSamples = 1
    numberOfFrames = 11
    shiftedMovie = np.zeros((numberOfSamples, numberOfFrames, rows, columns, 1), dtype=np.float)
    imageNameSplit = imageName.split("_")

    # stitch all but last
    imageSeriesName = ""
    for index in range(len(imageNameSplit)-1):
        splitItem = imageNameSplit[index]
        imageSeriesName += splitItem + "_"
    imageSeriesName = imageSeriesName[:-1]

    # image series name should be the entire thing up to underscore    
    for i in range(numberOfSamples):
        # read custom dataset
        for t in range(numberOfFrames):
            currentFrameName = imageSeriesName + "_" + str(200 + t) + ".png"
            
            # read and resize to 40x40
            imageData = imread(currentFrameName, as_grey=True)
            imageData = resize(imageData, (40, 40))    
            shiftedMovie[i, t, :, :, 0] = imageData

    # layer output
    get_layer_output = K.function([net.layers[0].input, K.learning_phase()],
                                  [net.layers[LAYER_NUMBER].output])
    layer_output = get_layer_output([shiftedMovie, LEARNING_PHASE])[0]
    features = layer_output[0]
    return features

# image search features
def get_image_match_features(net, imageName):
    # use a system call to invoke the python3 library using this image name
    os.system("python3 sampling/utils/get_image_match_features.py " + imageName)

    # use temporary file for communication
    features = np.load("image_match_features.npy")

    return features
