import os
import sys
import numpy as np

scaleDataFileName = os.environ["SCALE_DATA_FILE"]
abm = os.environ["ABM"]
images = os.environ["IMAGES"]
descriptorSetting = os.environ["DESCRIPTOR_SETTING"]

# read configuration
configurationFileName = os.environ["CONFIGURATION_FILE"]
configuration = open(configurationFileName).read()
execd = {}
exec(configuration, execd)
numIndependent = execd['NUM_INDEPENDENT']

line = sys.argv[1:]
demonstrationAlps = tuple([eval(x) for x in line[:numIndependent]])

# edit scale data file name based on descriptor setting
scaleDataFileNamePrefix = scaleDataFileName.split(".")[0]
scaleDataFileNameExtension = scaleDataFileName.split(".")[1]
scaleDataFileName = scaleDataFileNamePrefix + "_" + abm
if(bool(eval(images)) == True):
    scaleDataFileName += "_" + descriptorSetting.lower()
scaleDataFileName += "." + scaleDataFileNameExtension

# scale demonstration data
# compute distances
distances = []
with open("sampling/" + abm + "/" + scaleDataFileName, "r") as scaleDataFile:
    # read in order of writing in map.py
    maxs = eval(scaleDataFile.readline().strip("\n"))
    mins = eval(scaleDataFile.readline().strip("\n"))
    
    independentMaxs = maxs[:numIndependent]
    independentMins = mins[:numIndependent]
    independentValuesMax = {i: independentMaxs[i] for i in range(len(independentMaxs))}
    independentValuesMin = {i: independentMins[i] for i in range(len(independentMins))}

with open("suggested_alps.txt", "r") as inFile:
    for line in inFile:
        independentValues = eval(line)[0]
        independentValues = tuple((independentValues[index] - independentValuesMin[index])/(independentValuesMax[index] - independentValuesMin[index]) if (independentValuesMax[index] > independentValuesMin[index]) else independentValuesMax[index] for index in range(len(independentValues)))
        distance = np.linalg.norm(np.array(independentValues)-np.array(demonstrationAlps))
        distances.append(distance)

print min(distances)
