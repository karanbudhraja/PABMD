import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
import sys

#
# scale information
#
scaleDataFileName = os.environ["SCALE_DATA_FILE"]
abm = os.environ["ABM"]
images = os.environ["IMAGES"]
descriptorSetting = os.environ["DESCRIPTOR_SETTING"]
numDependent = eval(os.environ['NUM_DEPENDENT'])

configurationFileName = os.environ["CONFIGURATION_FILE"]
configuration = open(configurationFileName).read()
execd = {}
exec(configuration, execd)
numIndependent = execd['NUM_INDEPENDENT']

scaleDataFileNamePrefix = scaleDataFileName.split(".")[0]
scaleDataFileNameExtension = scaleDataFileName.split(".")[1]
scaleDataFileName = scaleDataFileNamePrefix + "_" + abm
if(bool(eval(images)) == True):
    scaleDataFileName += "_" + descriptorSetting.lower()
scaleDataFileName += "." + scaleDataFileNameExtension

with open("sampling/" + abm + "/" + scaleDataFileName, "r") as scaleDataFile:
    # read in order of writing in map.py
    maxs = eval(scaleDataFile.readline().strip("\n"))
    mins = eval(scaleDataFile.readline().strip("\n"))    

    dependentMaxs = maxs[-numDependent:]
    dependentMins = mins[-numDependent:]
    dependentValuesMax = {i: dependentMaxs[i] for i in range(len(dependentMaxs))}
    dependentValuesMin = {i: dependentMins[i] for i in range(len(dependentMins))}

    independentMaxs = maxs[:numIndependent]
    independentMins = mins[:numIndependent]
    independentValuesMax = {i: independentMaxs[i] for i in range(len(independentMaxs))}
    independentValuesMin = {i: independentMins[i] for i in range(len(independentMins))}

#
# plot slps
#

# read from file
with open("suggested_slps.txt", "r") as inFile:
    lines = inFile.readlines()
    suggestedSlps = [list(eval(line.strip())[0]) for line in lines]

for suggestionIndex in range(len(suggestedSlps)):
    dependentValues = suggestedSlps[suggestionIndex]
    dependentValues = [(dependentValues[index] - dependentValuesMin[index])/(dependentValuesMax[index] - dependentValuesMin[index]) for index in range(numDependent)]
    suggestedSlps[suggestionIndex] = dependentValues
suggestedSlps = np.array(suggestedSlps)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
slpPoints = zip(*suggestedSlps)
[xs, ys, zs] = slpPoints
ax.scatter(xs, ys, zs, c="b", marker="o")
ax.set_xlabel("Quiescent Citizens")
ax.set_ylabel("Actively Rebeling Citizens")
ax.set_zlabel("Jailed Citizens")
plt.savefig("suggested_slps")
plt.close()

#
# plot alps
#

# read from file
with open("suggested_alps.txt", "r") as inFile:
    lines = inFile.readlines()
    suggestedAlps = [list(eval(line.strip())[0]) for line in lines]

for suggestionIndex in range(len(suggestedAlps)):
    independentValues = suggestedAlps[suggestionIndex]
    independentValues = [(independentValues[index] - independentValuesMin[index])/(independentValuesMax[index] - independentValuesMin[index]) for index in range(numIndependent)]
    suggestedAlps[suggestionIndex] = independentValues
suggestedAlps = np.array(suggestedAlps)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
alpPoints = zip(*suggestedAlps)
[xs, ys, zs] = alpPoints
ax.scatter(xs, ys, zs, c="b", marker="o")
ax.set_xlabel("Citizen Density")
ax.set_ylabel("Cop Density")
ax.set_zlabel("Legitimacy")
plt.savefig("suggested_alps")
plt.close()
