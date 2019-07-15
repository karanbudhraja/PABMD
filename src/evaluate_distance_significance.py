import scipy.stats
import numpy as np
import sys
import os

abm = os.environ["ABM"]

predictionFileName = "evaluation_distance_" + abm + "_cross_validation.txt"
randomBaselineFileName = "evaluation_distance_" + abm + "_cross_validation_random_baseline.txt"

with open(predictionFileName, "r") as inFile:
    lines = inFile.readlines()
    predictionData = [eval(x) for x in lines]

with open(randomBaselineFileName, "r") as inFile:
    lines = inFile.readlines()
    randomBaselineData = [eval(x) for x in lines]

predictionData = np.array(predictionData)
randomBaselineData = np.array(randomBaselineData)
statistic, pValue = scipy.stats.ttest_ind(predictionData, randomBaselineData)

print >> sys.stderr, "abm: " + abm
print >> sys.stderr, "prediction distance mean: " + str(np.mean(predictionData))
print >> sys.stderr, "random baseline distance mean: " + str(np.mean(randomBaselineData))
print >> sys.stderr, "statistic: " + str(statistic)
print >> sys.stderr, "pValue:" + str(pValue)
