import os
from sklearn.model_selection import KFold
import numpy as np
import sys

# number of folds
K = 10

# get abm
abm = sys.argv[1]

# create destination directory if not existing already
os.system("mkdir -p cross_validation")

with open("processed_" + abm + ".txt") as inFile:
    lines = inFile.readlines()

    # get header
    header = lines[0]

    # data
    data = lines[1:]
    
    kf = KFold(n_splits=K, shuffle=True)
    splitNumber = 0
    for trainIndexes, testIndexes in kf.split(data):
        trainData = [data[x] for x in trainIndexes]
        testData = [data[x] for x in testIndexes]

        # create files for this split
        with open("cross_validation/" + abm + "_split_" + str(splitNumber) + "_train.txt", "w") as outFile:
            outFile.write(header)
            for item in trainData:
                outFile.write(item)
        with open("cross_validation/" + abm + "_split_" + str(splitNumber) + "_test.txt", "w") as outFile:
            for item in testData:
                outFile.write(item)

        splitNumber += 1
