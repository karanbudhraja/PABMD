import os
import sampling.utils.featurize_image as fi
from sampling.utils.process_folder_images import process_folder
import sys
import inspect
import glob
import numpy as np

currentDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentDir = os.path.dirname(currentDir)
sys.path.insert(0,parentDir)

import amf.misc as misc

def normalized_distance(_a, _b):
    _a = np.array(_a)
    _b = np.array(_b)

    b = _b.astype(int)
    a = _a.astype(int)
    norm_diff = np.linalg.norm(b - a)
    norm1 = np.linalg.norm(b)
    norm2 = np.linalg.norm(a)
    return norm_diff / (norm1 + norm2)

class FilterConfiguration(object):
    def __init__(self):
        # get environment variables
        self.abm = os.environ["ABM"]
        self.demonstrationFolder = os.environ["DEMONSTRATION_FOLDER"]
        self.descriptorSetting = os.environ["DESCRIPTOR_SETTING"]
        self.numDependent = eval(os.environ["NUM_DEPENDENT"])
        self.dataFileName = os.environ["DATA_FILE"]
        self.scaleDataFileName = os.environ["SCALE_DATA_FILE"]
        self.images = eval(os.environ["IMAGES"])

        # read configuration
        configurationFileName = os.environ["CONFIGURATION_FILE"]
        configuration = open(configurationFileName).read()
        execd = {}
        exec(configuration, execd)
        self.numIndependent = execd['NUM_INDEPENDENT']
        
# filter selected configurations based on closeness to demonstrations
def select_demonstration_configuratons(filterConfiguration, scaleData):
    # load network if needed
    net = fi.get_network(filterConfiguration.descriptorSetting)

    # for normalization
    if(scaleData is not None):
        [dependentValuesMax, dependentValuesMin] = scaleData
    
    # process directories in sorted order
    sortedPaths = []
    for dirName, subdirList, fileList in os.walk(filterConfiguration.demonstrationFolder):
        for sortedSubdirName in sorted(subdirList):
            sortedPaths.append(dirName + "/" + sortedSubdirName)

    # get slps for demonstrations
    demonstrationSlpsList = []
    demonstrationNameList = []
    for sortedPath in sortedPaths:
        for dirName, subdirList, fileList in os.walk(sortedPath):
            # only use non empty subdirectories
            if(len(fileList) > 0):
                # traverse all files in this directory
                if(bool(filterConfiguration.images) == True):
                    # read dependent values as images
                    dependentValues = [tuple(fi.get_features(net, dirName + "/" + fileName, filterConfiguration.descriptorSetting)) for fileName in fileList]
                else:
                    # read dependent values as text
                    dependentValues = [tuple([eval(x) for x in open(dirName + "/" + fileName).readlines()[0].replace("\n", "").split(" ")]) for fileName in fileList]

                dependentValues = tuple(misc.col_average(dependentValues))
                demonstrationSlpsList.append(dependentValues)
                demonstrationName = dirName.split("/")[-1].strip(" ")
                demonstrationNameList.append(demonstrationName)

    minDistanceAlpsList = []
    minDistanceSlpsList = []
    for demonstrationIndex in range(len(demonstrationSlpsList)):
        demonstrationSlps = demonstrationSlpsList[demonstrationIndex]

        # normalize demonstration slps
        if(scaleData is not None):
            demonstrationSlps = tuple((demonstrationSlps[index] - dependentValuesMin[index])/(dependentValuesMax[index] - dependentValuesMin[index]) if (dependentValuesMax[index] > dependentValuesMin[index]) else dependentValuesMax[index] for index in range(len(demonstrationSlps)))
        
        distanceList = []
        
        with open("suggested_slps.txt", "r") as suggestedSlpsFile:
            lines = suggestedSlpsFile.readlines()
            
            for suggestionIndex in range(len(lines)):
                # get slp corresponding to this demonstration
                line = lines[suggestionIndex]
                suggestedSlps = eval(line)[demonstrationIndex]

                if(bool(filterConfiguration.images == True)):
                    if(eval(os.environ["IMAGE_FEATURIZATION_HOMOGENEOUS"]) == 0):
                        # this is needed if the two image featurization stages are different
                        folderNames = glob.glob("sampling/" + filterConfiguration.abm + "/images_*")
                        suggestionFolder = folderNames[suggestionIndex]
                        suggestedSlps = process_folder(suggestionFolder)
                    
                # normalize suggested slps
                if(scaleData is not None):
                    suggestedSlps = tuple((suggestedSlps[index] - dependentValuesMin[index])/(dependentValuesMax[index] - dependentValuesMin[index]) if (dependentValuesMax[index] > dependentValuesMin[index]) else dependentValuesMax[index] for index in range(len(suggestedSlps)))
                
                # compute distance
                if(os.environ["DISTANCE_METHOD"] == "NORMALIZED"):
                    distance = normalized_distance(suggestedSlps, demonstrationSlps)
                else:
                    # default option
                    distance = misc.distance(suggestedSlps, demonstrationSlps)
                distanceList.append((distance, suggestionIndex, suggestedSlps))

        # select entry corresponding to smallest distance
        minDistanceIndex = min(distanceList)[1]
        minDistanceSlps = min(distanceList)[2]

        # read the suggested alps corresponding to that index
        with open("suggested_alps.txt", "r") as suggestedAlpsFile:
            minDistanceAlps = eval(suggestedAlpsFile.readlines()[minDistanceIndex])[demonstrationIndex]
            
        # write to file
        directory = "../data/predictions/" + filterConfiguration.abm + "/" + demonstrationNameList[demonstrationIndex] 
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(directory + "/predicted_alps.txt", "w") as predictionFile:
            predictionFile.write(str(minDistanceAlps) + " " + str(minDistanceSlps))

        if(bool(filterConfiguration.images == True)):
            # if images used, copy images
            # get all image simulation folders
            folderNames = glob.glob("sampling/" + filterConfiguration.abm + "/images_*")
            minFolder = folderNames[minDistanceIndex]
            os.system("cp -r " + minFolder + " " + directory)
                        
# filter configurations with command line parameters
def main():
    filterConfiguration = FilterConfiguration()

    # edit scale data file name based on descriptor setting
    scaleDataFileNamePrefix = filterConfiguration.scaleDataFileName.split(".")[0]
    scaleDataFileNameExtension = filterConfiguration.scaleDataFileName.split(".")[1]
    scaleDataFileName = scaleDataFileNamePrefix + "_" + filterConfiguration.abm
    if(bool(filterConfiguration.images) == True):
        scaleDataFileName += "_" + filterConfiguration.descriptorSetting.lower()
    scaleDataFileName += "." + scaleDataFileNameExtension

    # get scale information
    fileName = "sampling/" + filterConfiguration.abm + "/" + scaleDataFileName

    # scale data assumed not present by default
    scaleData = None

    if(os.path.exists(fileName) == True):
        with open(fileName, "r") as scaleDataFile:
            # read in order of writing in map.py
            maxs = eval(scaleDataFile.readline().strip("\n"))
            mins = eval(scaleDataFile.readline().strip("\n"))

            dependentMaxs = maxs[filterConfiguration.numIndependent:]
            dependentMins = mins[filterConfiguration.numIndependent:]
            dependentValuesMax = {i: dependentMaxs[i] for i in range(len(dependentMaxs))}
            dependentValuesMin = {i: dependentMins[i] for i in range(len(dependentMins))}
            scaleData = [dependentValuesMax, dependentValuesMin]

    select_demonstration_configuratons(filterConfiguration, scaleData)
                                
# execute main
main()
