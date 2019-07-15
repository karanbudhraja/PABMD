import numpy as np
import os

# read from file
with open("suggested_slps.txt", "r") as inFile:
    lines = inFile.readlines()
    suggestedSlps = [list(eval(line.strip())[0]) for line in lines]

# scale
scaleDataFileName = os.environ["SCALE_DATA_FILE"]
abm = os.environ["ABM"]
images = os.environ["IMAGES"]
descriptorSetting = os.environ["DESCRIPTOR_SETTING"]
numDependent = eval(os.environ['NUM_DEPENDENT'])

scaleDataFileName = os.environ["SCALE_DATA_FILE"]
images = os.environ["IMAGES"]
descriptorSetting = os.environ["DESCRIPTOR_SETTING"]
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

for suggestionIndex in range(len(suggestedSlps)):
    dependentValues = suggestedSlps[suggestionIndex]
    dependentValues = [(dependentValues[index] - dependentValuesMin[index])/(dependentValuesMax[index] - dependentValuesMin[index]) for index in range(numDependent)]
    suggestedSlps[suggestionIndex] = dependentValues
    
suggestedSlps = np.array(suggestedSlps)

# add image descriptor to abm if neccessary
_abm = abm
if(bool(eval(images)) == True):
    _abm += "_" + descriptorSetting.lower()

# get target slps
# we only need first 10 for cross validation
data = np.genfromtxt("../data/domaindata/cross_validation/" + _abm + "_split_" + os.environ["SPLIT_NUMBER"] + "_test.txt", skip_header=0, invalid_raise=False)
targetSlps = data[:eval(os.environ["TESTS_PER_FOLD"]),-numDependent:]
error = np.linalg.norm(suggestedSlps-targetSlps, axis=1).tolist()

# write to file
with open("neural_network_" + abm + "_cross_validation.txt", "a") as outFile:
    for item in error:
        outFile.write(str(item) + "\n")

