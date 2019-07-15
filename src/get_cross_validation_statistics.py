import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.cluster import KMeans

abm = sys.argv[1]
removeOutliers = eval(sys.argv[2])
datasetSize = sys.argv[3]

#predictionFileName = "evaluation_distance_" + abm + "_cross_validation.txt"
#predictionFileName = "neural_network_" + abm + "_cross_validation" + "_" + datasetSize + ".txt"
predictionFileName = "neural_network_" + abm + "_cross_validation" + "" + ".txt"

with open(predictionFileName, "r") as inFile:
    lines = inFile.readlines()
    predictionData = [eval(x) for x in lines]
predictionData = np.array(predictionData)

#
# clustering to remove outliers
#

labels = KMeans(n_clusters=2).fit_predict(predictionData.reshape(-1,1))
label1Count = sum(labels)
label0Count = len(labels)-sum(labels)
print "cluster sizes: " + str((label1Count, label0Count))
if label1Count > label0Count:
    outlierLabel = 0
else:
    outlierLabel = 1

if removeOutliers == 1:
    predictionData = [predictionData[index] for index in range(len(predictionData)) if labels[index] != outlierLabel]

#
# compute statistics
#

print >> sys.stderr, "abm: " + abm
print >> sys.stderr, "prediction distance mean: " + str(np.mean(predictionData))
print >> sys.stderr, "prediction distance std: " + str(np.std(predictionData))
print >> sys.stderr, "prediction distance median: " + str(np.median(predictionData))

plt.hist(np.log(predictionData))
plt.xlabel("log(Output Difference)")
plt.ylabel("Number of Points in each value range")
plt.savefig("output_difference")
plt.close()



