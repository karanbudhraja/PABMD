from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import numpy as np

def evaluate_regressor(regressor):

    #abm = "forward_kinematics"
    #numIndependent = 2
    #numDependent = 2
    abm = "brouwer_sin"
    numIndependent = 1
    numDependent = 1
    
    # get data
    # only use one y colum for now
    data = np.genfromtxt("../../data/domaindata/cross_validation/" + abm + "_split_0_train.txt", skip_header=1)
    XTrain = data[:,:numIndependent]
    #YTrain = data[:,numIndependent:-1].flatten()
    YTrain = data[:,numIndependent:].flatten()
    data = np.genfromtxt("../../data/domaindata/cross_validation/" + abm + "_split_0_test.txt", skip_header=0)
    XTest = data[:,:numIndependent]
    #YTest = data[:,numIndependent:-1].flatten()
    YTest = data[:,numIndependent:].flatten()

    regressor.fit(XTrain, YTrain)
    YPredicted = regressor.predict(XTest)    
    mse = np.mean((YPredicted-YTest)**2)
    print mse
    
# train method
evaluate_regressor(KNeighborsRegressor())
evaluate_regressor(GaussianProcessRegressor())
evaluate_regressor(RandomForestRegressor())
evaluate_regressor(SVR(kernel="poly"))
