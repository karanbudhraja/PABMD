import os
import numpy as np

# get queried slps
with open("queried_slps.txt", "r") as inFile:
    queriedSlps = eval(inFile.readlines()[0].replace("\n", "").strip())

# get predicted slps
# PREDICTION_NAME must be set
abm = os.environ["ABM"]
predictionName = os.environ["PREDICTION_NAME"]
predictionsFileName = "../data/predictions/" + abm + "/" + predictionName + "/predicted_alps.txt"
with open(predictionsFileName, "r") as inFile:
    prediction = inFile.readlines()[0].replace("\n", "").strip()
    predictedSlps = eval("(" + prediction.split(") (")[1])

# get distance between them
distance = np.linalg.norm(np.array(predictedSlps) - np.array(queriedSlps))
#print(distance)


# KARAN CHANGING TEMPORARILY TO GET MSE OUTPUT
print(distance**2/len(predictedSlps))
