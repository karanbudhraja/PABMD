import numpy as np
import sys

abm = sys.argv[1]
data = np.genfromtxt("nn_evaluation_" + abm + "_mse.txt")

# split into test and train data
fmTrainData = data[::4]
fmTestData = data[1::4]
rmFmTrainData = data[2::4]
rmFmTestData = data[3::4]

# get mse in each
fmTrainDataMSE = fmTrainData[:,2]
fmTestDataMSE = fmTestData[:,2]
rmFmTrainDataMSE = rmFmTrainData[:,2]
rmFmTestDataMSE = rmFmTestData[:,2]

# print summary
print np.mean(fmTrainDataMSE)
print np.std(fmTrainDataMSE)
print ""
print np.mean(fmTestDataMSE)
print np.std(fmTestDataMSE)
print "#####"
print np.mean(rmFmTrainDataMSE)
print np.std(rmFmTrainDataMSE)
print ""
print np.mean(rmFmTestDataMSE)
print np.std(rmFmTestDataMSE)
