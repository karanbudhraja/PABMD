import numpy as np
from keras.models import Model, Sequential
from keras.layers import Lambda, Input, Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import plot_model
from keras.losses import mse
from keras import backend as K
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

##############
# parameters #
##############

# abm parameters
abm = "flocking_contour"
numIndependent = 3
numDependent = 2

fmEpochs = 100
fmBatchSize = 32
rmEpochs = 100
rmBatchSize = 32

########
# data #
########

# get data
data = np.genfromtxt("../../data/domaindata/cross_validation/" + abm + "_split_0_train.txt", skip_header=1)
XTrain = data[:,:numIndependent]
YTrain = data[:,numIndependent:]
data = np.genfromtxt("../../data/domaindata/cross_validation/" + abm + "_split_0_test.txt", skip_header=0)
XTest = data[:,:numIndependent]
YTest = data[:,numIndependent:]

############
# fm model #
# X -> Y   #
############

def get_fm_model(parameters):

    # get parameters
    [numIndependent, numDependent] = parameters
    
    # create model
    model = Sequential()
    model.add(Dense(2*numIndependent, input_dim=numIndependent, activation="relu"))
    model.add(Dense(2*numDependent, input_dim=numIndependent, activation="relu"))
    model.add(Dense(numDependent, activation="sigmoid"))

    # compile model
    model.compile(loss="mean_squared_error", optimizer="adam")

    return model

def train_fm_model():
    
    # evaluation
    estimators = []
    estimators.append(('preprocessing', StandardScaler()))
    estimators.append(('network', KerasRegressor(build_fn=get_fm_model, parameters=[numIndependent, numDependent], epochs=fmEpochs, batch_size=fmBatchSize, verbose=1)))
    pipeline = Pipeline(estimators)
    pipeline.fit(XTrain, YTrain)

    '''
    YPredicted = pipeline.predict(XTest)
    error = np.linalg.norm(YTest-YPredicted, axis=1)
    
    print("")
    print("Mean Error (L2 Distance): " + str(np.mean(error)) + " (" + str(np.std(error)) + ")")
    
    # compare with knn
    from sklearn.neighbors import KNeighborsRegressor 
    r = KNeighborsRegressor() 
    r.fit(XTrain, YTrain) 
    YPredicted = r.predict(XTest) 
    error = np.linalg.norm(YTest-YPredicted, axis=1) 
    print ("")
    print ("Mean Error (L2 Distance): " + str(np.mean(error)) + " (" + str(np.std(error)) + ")")
    '''
    
    return pipeline
    
############
# rm model #
# Y -> X   #
############
    
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

def get_rm_model(parameters):

    # get parameters
    [numIndependent, numDependent, fmPipeline] = parameters
    
    # create model
    # the vae has a modular design
    # the encoder, decoder and vae are 3 models that share weights
    latentDimension = 2*numIndependent
    
    # encoder
    inputs = Input(shape=(numDependent,), name='encoder_input')
    x = Dense(2*numDependent, activation='relu')(inputs)
    zMean = Dense(latentDimension, name='zMean')(x)
    zLogVar = Dense(latentDimension, name='zLogVar')(x)
    z = Lambda(sampling, output_shape=(latentDimension,), name='z')([zMean, zLogVar])
    encoder = Model(inputs, [zMean, zLogVar, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
    
    # decoder + fm
    latentInputs = Input(shape=(latentDimension,), name='zSampling')
    x = Dense(2*numIndependent, activation='relu')(latentInputs)
    outputs = Dense(numIndependent, activation='sigmoid')(x)
    fmX1 = Dense(2*numIndependent, activation='relu')(outputs)
    fmX2 = Dense(2*numDependent, activation='relu')(fmX1)
    fmOutputs = Dense(numDependent, activation='sigmoid')(fmX2)
    decoderFm = Model(latentInputs, fmOutputs, name='decoder')
    decoderFm.summary()
    plot_model(decoderFm, to_file='vae_mlp_decoderFm.png', show_shapes=True)

    # now combine
    outputs = decoderFm(encoder(inputs)[2])
    model = Model(inputs, outputs, name='vae_mlp')
    
    # compile model with custom loss
    # the loss must talk about the outputs
    # kl divergence loss + fm reconstruction loss
    klLoss = 1 + zLogVar - K.square(zMean) - K.exp(zLogVar)
    klLoss = K.sum(klLoss, axis=-1)
    klLoss *= -0.5
    #vaeLoss = K.mean(klLoss + mse(inputs, outputs))
    vaeLoss = K.mean(mse(inputs, outputs))

    #+ rm_fm_loss(inputs, outputs)
    model.add_loss(vaeLoss)
    model.compile(optimizer="adam")
    model.summary()
    plot_model(model, to_file='vae_mlp.png', show_shapes=True)
    
    return model 

def train_rm_model(fmPipeline):

    # evaluation
    estimators = []
    estimators.append(('preprocessing', StandardScaler()))
    estimators.append(('network', KerasRegressor(build_fn=get_rm_model, parameters=[numIndependent, numDependent, fmPipeline], epochs=rmEpochs, batch_size=rmBatchSize, verbose=1)))
    pipeline = Pipeline(estimators)
    pipeline.fit(YTrain)
    
    #pipeline.fit(XTrain, YTrain)
    #YPredicted = pipeline.predict(XTest)
    #error = np.linalg.norm(YTest-YPredicted, axis=1)

def main():

    fmPipeline = train_fm_model()
    train_rm_model(fmPipeline)
    
main()
