import numpy as np
from keras.models import Model, Sequential
from keras.layers import Lambda, Input, Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import plot_model
from keras.losses import mse
from keras import optimizers
from keras import backend as K
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor
import sys
import argparse
import os

#####################
# utility functions #
#####################

def print_errors(errorNames, errorValues):
    for errorName, error in zip(errorNames, errorValues):
        print("===")
        print(errorName)
        print("---")
        print("Mean Error (L2 Distance): " + str(np.mean(error)) + " (" + str(np.std(error)) + ")")
        print("Mean Error (Squared Error): " + str(np.mean(error**2)/numDependent) + " (" + str(np.std(error**2)/numDependent) + ")")
        print("===")

        # add file writing
        with open("nn_evaluation_" + abm + "_mse.txt", "a") as outFile:
            line = ""
            line += str(np.mean(error)) + " " + str(np.std(error)) + " "
            line += str(np.mean(error**2)/numDependent) + " " + str(np.std(error**2)/numDependent)
            line += "\n"
            outFile.write(line)

##############
# parameters #
##############

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--abm', default="civil_violence")
parser.add_argument('--numIndependent', default="3")
parser.add_argument('--numDependent', default="3")
parser.add_argument('--fmOnly')
parser.add_argument('--rmOnly')
parser.add_argument('--splitNumber', default="0")
parser.add_argument('--testCase', default="0")
parser.add_argument('--rmType', default="variational")
args = parser.parse_args()

# abm parameters
abm = args.abm
numIndependent = eval(args.numIndependent)
numDependent = eval(args.numDependent)
fmOnly = eval(args.fmOnly)
rmOnly = eval(args.rmOnly)
splitNumber = args.splitNumber
testCase = eval(args.testCase)
rmType = args.rmType

images = os.environ["IMAGES"]
descriptorSetting = os.environ["DESCRIPTOR_SETTING"]
if(bool(eval(images)) == True):
    abm += "_" + descriptorSetting.lower()

fmEpochs = 1000
fmBatchSize = 32
fmKernelInitializer = "glorot_uniform"
fmOpt = optimizers.nadam()

rmFmEpochs = 1000
rmFmBatchSize = 32
rmFmKernelInitializer = "glorot_uniform"
rmFmOpt = optimizers.nadam()

########
# data #
########

# get data
data = np.genfromtxt("../data/domaindata/cross_validation/" + abm + "_split_" + splitNumber + "_train.txt", skip_header=1, invalid_raise=False)
data = data[~np.isnan(data).any(axis=1)]
XTrain = data[:,:numIndependent]
YTrain = data[:,-numDependent:]
data = np.genfromtxt("../data/domaindata/cross_validation/" + abm + "_split_" + splitNumber + "_test.txt", skip_header=0, invalid_raise=False)
data = data[~np.isnan(data).any(axis=1)]
XTest = data[:,:numIndependent]
YTest = data[:,-numDependent:]

# scaling x helps
# scaling y does not
# so we use linear activation on y later
xScaler = StandardScaler()
xScaler.fit(XTrain)
XTrain = xScaler.transform(XTrain)
XTest = xScaler.transform(XTest)

#########
# model #
#########

def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    # reparameterization trick
    # instead of sampling from Q(z|X), sample eps = N(0,I)
    # z = z_mean + sqrt(var)*eps
    
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def get_fm_model(parameters):

    # get parameters
    [numIndependent, numDependent] = parameters
    
    # create model
    fmInputs = Input(shape=(numIndependent,), name='fm_input')
    fmX1 = Dense(2*numIndependent, activation='relu', kernel_initializer=fmKernelInitializer, name='fm_1')(fmInputs)
    fmX2 = Dense(numIndependent**2, activation='relu', kernel_initializer=fmKernelInitializer, name='fm_2')(fmX1)
    fmX3 = Dense(numDependent**2, activation='relu', kernel_initializer=fmKernelInitializer, name='fm_3')(fmX2)    
    fmX4 = Dense(2*numDependent, activation='relu', kernel_initializer=fmKernelInitializer, name='fm_4')(fmX3)
    fmOutputs = Dense(numDependent, activation='sigmoid', kernel_initializer=fmKernelInitializer, name='fm_output')(fmX4)
    fm = Model(fmInputs, fmOutputs, name='fm')

    return fm

def get_rm_model_mlp(parameters):

    # get parameters
    [numIndependent, numDependent] = parameters
    
    # create model
    rmInputs = Input(shape=(numDependent,), name='rm_input')

    # ours
    rmX1 = Dense(2*numDependent, activation='relu', kernel_initializer=rmFmKernelInitializer, name='rm_1')(rmInputs)
    rmX2 = Dense(numDependent**2, activation='relu', kernel_initializer=rmFmKernelInitializer, name='rm_2')(rmX1)
    rmX3 = Dense(numIndependent**2, activation='relu', kernel_initializer=rmFmKernelInitializer, name='rm_3')(rmX2)    
    rmX4 = Dense(2*numIndependent, activation='relu', kernel_initializer=rmFmKernelInitializer, name='rm_4')(rmX3)
    rmOutputs = Dense(numIndependent, activation='linear', kernel_initializer=rmFmKernelInitializer, name='rm_output')(rmX4)

    # duka, claim 1e-2
    #rmX1 = Dense(100, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmInputs)
    #rmOutputs = Dense(numIndependent, activation='linear', kernel_initializer=rmFmKernelInitializer)(rmX1)

    # nagata, claim 1e-2
    #rmX1 = Dense(20, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmInputs)
    #rmX2 = Dense(20, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmX1)
    #rmOutputs = Dense(numIndependent, activation='linear', kernel_initializer=rmFmKernelInitializer)(rmX2)

    # jha, claim 1e-3
    #rmX1 = Dense(4, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmInputs)
    #rmX2 = Dense(4, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmX1)
    #rmOutputs = Dense(numIndependent, activation='linear', kernel_initializer=rmFmKernelInitializer)(rmX2)

    # daya, claim 1e-2
    #rmX1 = Dense(100, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmInputs)
    #rmX2 = Dense(100, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmX1)
    #rmOutputs = Dense(numIndependent, activation='linear', kernel_initializer=rmFmKernelInitializer)(rmX2)

    # almusawi, claim 1e-8
    #rmX1 = Dense(20, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmInputs)
    #rmX2 = Dense(20, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmX1)
    #rmX3 = Dense(20, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmX2)
    #rmX4 = Dense(20, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmX3)
    #rmX5 = Dense(20, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmX4)
    #rmX6 = Dense(20, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmX5)
    #rmX7 = Dense(20, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmX6)
    #rmX8 = Dense(20, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmX7)
    #rmX9 = Dense(20, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmX8)
    #rmX10 = Dense(20, activation='relu', kernel_initializer=rmFmKernelInitializer)(rmX9)
    #rmOutputs = Dense(numIndependent, activation='linear', kernel_initializer=rmFmKernelInitializer)(rmX10)
    
    rm = Model(rmInputs, rmOutputs, name='rm')

    return [rm, rmInputs]

def get_rm_model_variational(parameters):

    # get parameters
    [numIndependent, numDependent] = parameters
    
    # create model
    rmInputs = Input(shape=(numDependent,), name='rm_input')
    rmX1 = Dense(2*numDependent, activation='relu', kernel_initializer=rmFmKernelInitializer, name='rm_1')(rmInputs)
    rmX2 = Dense(numDependent**2, activation='relu', kernel_initializer=rmFmKernelInitializer, name='rm_2')(rmX1)

    latentDimension = numDependent**2
    zMean = Dense(latentDimension, name='zMean', activation='relu', kernel_initializer=rmFmKernelInitializer)(rmX2)
    zLogVar = Dense(latentDimension, name='zLogVar', activation='relu', kernel_initializer=rmFmKernelInitializer)(rmX2)
    z = Lambda(sampling, output_shape=(latentDimension,), name='z')([zMean, zLogVar])

    rmX3 = Dense(numIndependent**2, activation='relu', kernel_initializer=rmFmKernelInitializer, name='rm_3')(z)    
    rmX4 = Dense(2*numIndependent, activation='relu', kernel_initializer=rmFmKernelInitializer, name='rm_4')(rmX3)
    rmOutputs = Dense(numIndependent, activation='linear', kernel_initializer=rmFmKernelInitializer, name='rm_output')(rmX4)
    rm = Model(rmInputs, rmOutputs, name='rm')

    return [rm, rmInputs, zMean, zLogVar]

def get_fm_model_compiled(parameters):

    # get and compile model
    fm = get_fm_model(parameters)
    fm.summary()
    plot_model(fm, to_file='fm_independent.png', show_shapes=True)
    fm.compile(loss="mean_squared_error", optimizer=fmOpt)

    return fm

def rm_fm_loss_mlp(parameters):
    def loss(y_true, y_pred):
        # reconstruction loss
        reconstructionLoss = mse(y_pred, y_true)
        
        # total loss
        totalLoss = K.mean(reconstructionLoss)
        
        # return loss value
        return totalLoss

    # return function
    return loss

def rm_fm_loss_variational(zParameters):
    def loss(y_true, y_pred):
        # loss mixing constant
        # made to be the order of reconstruction mse loss
        # used to scale down kl divergence loss
        #ALPHA = 0.0001
        ALPHA = 0.0
        #ALPHA = 0.5
        
        # kl loss
        [zMean, zLogVar] = zParameters
        klLoss = 1 + zLogVar - K.square(zMean) - K.exp(zLogVar)
        klLoss = K.sum(klLoss, axis=-1)
        klLoss *= -0.5

        # reconstruction loss
        reconstructionLoss = mse(y_pred, y_true)

        alpha = K.variable(value=ALPHA)

        # print loss components
        #klLoss = K.print_tensor(K.mean(klLoss), message="########################### kl loss = ")
        #reconstructionLoss = K.print_tensor(K.mean(reconstructionLoss), message="########################### reconstruction loss = ")
        
        # total loss
        totalLoss = K.mean((1-alpha)*reconstructionLoss*numDependent + klLoss*alpha)

        # return loss value
        return totalLoss

    # return function
    return loss

def get_rm_fm_model_compiled(parameters):

    # get parameters
    [numIndependent, numDependent, fmWeights] = parameters
    
    # create model
    # the network has a modular design
    # the fm and rm 2 models that share weights
    
    # slps to alps
    rmModelFunction = globals()["get_rm_model_" + rmType]
    modelParameters = rmModelFunction([numIndependent, numDependent])
    rm = modelParameters[0]
    rmInputs = modelParameters[1]
    lossParameters = modelParameters[2:]

    rm.summary()
    plot_model(rm, to_file='rm.png', show_shapes=True)
    
    # alps to slps
    fm = get_fm_model([numIndependent, numDependent])
    fm.summary()
    plot_model(fm, to_file='fm.png', show_shapes=True)
    
    # combine
    rmOutputs = rm(rmInputs)
    fmOutputs = fm(rmOutputs)
    model = Model(rmInputs, fmOutputs, name='model')
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)

    # freeze fm layers
    # set weights based on trained fm
    for layer in model.layers:
        if(layer.name == "fm"):
            layer.trainable = False
            layer.set_weights(fmWeights)
                
    # define loss and optimizer
    rmFmLossFunction = globals()["rm_fm_loss_" + rmType]
    model.compile(loss=rmFmLossFunction(lossParameters), optimizer=rmFmOpt)
    
    return model 

# fix randomness for consistency
np.random.seed(0)

if(rmOnly == True):
    # direct mapping from slps to alps

    # rm
    parameters = [numIndependent, numDependent]
    rmModelFunction = globals()["get_rm_model_" + rmType]
    rmModel = rmModelFunction(parameters)[0]
    rmModel.summary()
    plot_model(rmModel, to_file='rm_independent.png', show_shapes=True)
    rmModel.compile(loss="mean_squared_error", optimizer=rmFmOpt)
    rmModel.fit(YTrain, XTrain, epochs=fmEpochs, batch_size=fmBatchSize, verbose=1)

    # test rm
    XPredicted = rmModel.predict(YTrain).reshape(XTrain.shape)
    rmTrainError = np.linalg.norm(XTrain-XPredicted, axis=1)
    XPredicted = rmModel.predict(YTest).reshape(XTest.shape)
    rmTestError = np.linalg.norm(XTest-XPredicted, axis=1)
    print_errors(["rm train", "rm test"], [rmTrainError, rmTestError])
    
    # do not execute further
    sys.exit()

# fm
parameters = [numIndependent, numDependent]
fmModel = get_fm_model_compiled(parameters)
fmModel.fit(XTrain, YTrain, epochs=fmEpochs, batch_size=fmBatchSize, verbose=1)

# test fm
YPredicted = fmModel.predict(XTrain).reshape(YTrain.shape)
fmTrainError = np.linalg.norm(YTrain-YPredicted, axis=1)
YPredicted = fmModel.predict(XTest).reshape(YTest.shape)
fmTestError = np.linalg.norm(YTest-YPredicted, axis=1)

if(fmOnly == True):
    # print error and exit if only running fm
    print_errors(["fm train", "fm test"], [fmTrainError, fmTestError])

    # knn for comparison
    kNN = KNeighborsRegressor()
    kNN.fit(XTrain, YTrain)
    YPredicted = kNN.predict(XTrain).reshape(YTrain.shape)
    knnTrainError = np.linalg.norm(YTrain-YPredicted, axis=1)
    YPredicted = kNN.predict(XTest).reshape(YTest.shape)
    knnTestError = np.linalg.norm(YTest-YPredicted, axis=1)
    print_errors(["knn train", "knn test"], [knnTrainError, knnTestError])

    # do not execute further
    sys.exit()
    
# get fm weights
fmWeights = fmModel.get_weights()

# rm + fm
parameters = [numIndependent, numDependent, fmWeights]
rmModel = get_rm_fm_model_compiled(parameters)
rmModel.fit(YTrain, YTrain, epochs=rmFmEpochs, batch_size=rmFmBatchSize, verbose=1)

# test rm + fm
YPredicted = rmModel.predict(YTrain).reshape(YTrain.shape)
rmFmTrainError = np.linalg.norm(YTrain-YPredicted, axis=1)
YPredicted = rmModel.predict(YTest).reshape(YTest.shape)
rmFmTestError = np.linalg.norm(YTest-YPredicted, axis=1)

# print error
print_errors(["fm train", "fm test", "rm+fm train", "rm+fm test"], [fmTrainError, fmTestError, rmFmTrainError, rmFmTestError])

# do not proceed further if not running test case
if(testCase == True):
#for _ in range(1000):
    #
    # test case: get alps
    #
    
    # output index results in same output for 0 and 1
    rmOutput = Model(inputs=rmModel.input, outputs=rmModel.get_layer("rm").get_output_at(1))
    suggestedAlps = xScaler.inverse_transform(rmOutput.predict(YTest))
    suggestedAlps = [tuple(x) for x in suggestedAlps.tolist()[:eval(os.environ["TESTS_PER_FOLD"])]]

    # scale alps
    scaleDataFileName = os.environ["SCALE_DATA_FILE"]

    # this value is equal to abm without the image descriptor if any
    _abm = abm
    if(bool(eval(images)) == True):
        # remove the image descriptor from the abm string
        _abm = _abm.replace("_" + descriptorSetting.lower(), "")
        
    scaleDataFileNamePrefix = scaleDataFileName.split(".")[0]
    scaleDataFileNameExtension = scaleDataFileName.split(".")[1]
    scaleDataFileName = scaleDataFileNamePrefix + "_" + _abm
    if(bool(eval(images)) == True):
        scaleDataFileName += "_" + descriptorSetting.lower()
    scaleDataFileName += "." + scaleDataFileNameExtension

    with open("sampling/" + _abm + "/" + scaleDataFileName, "r") as scaleDataFile:
        # read in order of writing in map.py
        maxs = eval(scaleDataFile.readline().strip("\n"))
        mins = eval(scaleDataFile.readline().strip("\n"))
        
        independentMaxs = maxs[:numIndependent]
        independentMins = mins[:numIndependent]
        independentValuesMax = {i: independentMaxs[i] for i in range(len(independentMaxs))}
        independentValuesMin = {i: independentMins[i] for i in range(len(independentMins))}

    # print for suggested_alps.txt
    with open("suggested_alps.txt", "w") as outFile:
    #with open("suggested_alps.txt", "a") as outFile:
        for independentValues in suggestedAlps:
            independentValues = [tuple([independentValuesMin[index] + independentValues[index]*(independentValuesMax[index] - independentValuesMin[index]) for index in range(numIndependent)])]
            outFile.write(str(independentValues) + "\n")
