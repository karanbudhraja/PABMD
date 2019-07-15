import numpy as np
import sys

abm = sys.argv[1]

predictionFileName = "evaluation_distance_" + abm + "_cross_validation.txt"

with open(predictionFileName, "r") as inFile:
    lines = inFile.readlines()
    predictionData = [eval(x) for x in lines]

predictionData = np.array(predictionData)

print >> sys.stderr, "abm: " + abm
print >> sys.stderr, "prediction distance mean: " + str(np.mean(predictionData))
print >> sys.stderr, "prediction distance std: " + str(np.std(predictionData))

