import numpy as np

with open("../src/suggested_alps_fm_prediction_error.txt", "r") as inFile:
    errors = [eval(x) for x in inFile.readlines()]

    print np.mean(errors)
    print np.std(errors)
