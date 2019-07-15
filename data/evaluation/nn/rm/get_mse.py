import numpy as np
import sys

abm = sys.argv[1]
data = np.genfromtxt("nn_evaluation_" + abm + "_mse.txt")

# split into test and train data
trainData = data[::2]
testData = data[1::2]

# get mse in each
trainDataMSE = trainData[:,2]
testDataMSE = testData[:,2]

# print summary
print np.mean(trainDataMSE)
print np.std(trainDataMSE)
print ""
print np.mean(testDataMSE)
print np.std(testDataMSE)
