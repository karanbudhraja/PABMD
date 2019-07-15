import os
import sampling.utils.featurize_image as fi
import sys
import glob
import amf.misc
import matplotlib.pyplot as plt
from sklearn.neighbors import LSHForest
import numpy as np
import cPickle
import random

def initialize_featurization():
    # edit scale data file name based on descriptor setting
    scaleDataFileNamePrefix = os.environ["SCALE_DATA_FILE"].split(".")[0]
    scaleDataFileNameExtension = os.environ["SCALE_DATA_FILE"].split(".")[1]
    scaleDataFileName = scaleDataFileNamePrefix + "_" + os.environ["ABM"]
    if(bool(eval(os.environ["IMAGES"])) == True):
        scaleDataFileName += "_" + os.environ["DESCRIPTOR_SETTING"].lower()
    scaleDataFileName += "." + scaleDataFileNameExtension

    # get scale information
    with open("sampling/" + os.environ["ABM"] + "/" + scaleDataFileName, "r") as scaleDataFile:
        # read in order of writing in map.py
        maxs = eval(scaleDataFile.readline().strip("\n"))
        mins = eval(scaleDataFile.readline().strip("\n"))

        dependentMaxs = maxs[-eval(os.environ["NUM_DEPENDENT"]):]
        dependentMins = mins[-eval(os.environ["NUM_DEPENDENT"]):]
        dependentValuesMax = {i: dependentMaxs[i] for i in range(len(dependentMaxs))}
        dependentValuesMin = {i: dependentMins[i] for i in range(len(dependentMins))}
        scaleData = [dependentValuesMax, dependentValuesMin]

    
    # load feature extractor if needed
    net = fi.get_network(os.environ["DESCRIPTOR_SETTING"])

    return [net, scaleData]

def get_dependent_values_image(net, scaleData, imageName):
        # define dependent values
        dependentValuesImage = fi.get_features(net, imageName, os.environ["DESCRIPTOR_SETTING"])
        # normalize
        [dependentValuesMax, dependentValuesMin] = scaleData
        dependentValuesImage = tuple((dependentValuesImage[index] - dependentValuesMin[index])/(dependentValuesMax[index] - dependentValuesMin[index]) if (dependentValuesMax[index] > dependentValuesMin[index]) else dependentValuesMax[index] for index in range(len(dependentValuesImage)))

        return dependentValuesImage

def get_dependent_values_folder(net, scaleData, folderName, downsample=1):
    # initialize collection of dependent values
    imageNamesFolder = []
    dependentValuesFolder = []
    
    # expect everything in this folder to be an image
    # read sketches data
    allFileNames = glob.glob(folderName + "/*.*")
    allFileNames = random.sample(allFileNames, int(len(allFileNames)/downsample))
    for imageName in sorted(allFileNames):
        # featurize image
        print >> sys.stderr, ("now processing " + imageName)        
        dependentValuesImage = get_dependent_values_image(net, scaleData, imageName)

        # append to list
        imageNamesFolder.append(imageName)
        dependentValuesFolder.append(dependentValuesImage)

    return [imageNamesFolder, dependentValuesFolder]

def do_matching_lsh(net, scaleData, imageNamesSketchList, dependentValuesSketchList):
    lshModelFileName = "../../_swarm-lfd-data/" + os.environ["ABM"] + "/lsh_forest_model" + "_" + os.environ["DESCRIPTOR_SETTING"].lower() + ".p"

    if(os.path.exists(lshModelFileName) == False):
        # no model found in directory
        # train a locality sensitive hashing model on the simulation data
        [imageNamesSimulationList, dependentValuesSimulationList] = get_dependent_values_folder(net, scaleData, "../../_swarm-lfd-data/" + os.environ["ABM"] + "/images")
        lshForest = LSHForest()
        lshForest.fit(dependentValuesSimulationList)

        # use disk to cache results
        with open(lshModelFileName, "wb") as outFile:
            cPickle.dump(lshForest, outFile)
        with open("image_names_simulation.p", "wb") as outFile:
            cPickle.dump(imageNamesSimulationList, outFile)
    else:
        # use existing model
        with open(lshModelFileName, "rb") as inFile:
            lshForest = cPickle.load(inFile)
        with open("../../_swarm-lfd-data/" + os.environ["ABM"] + "/image_names_simulation.p", "rb") as inFile:
            imageNamesSimulationList = cPickle.load(inFile)

    # now match
    print >> sys.stderr, "\n"
    print >> sys.stderr, "now matching"
    matches = []
    for imageNameSketch, dependentValuesSketch in zip(imageNamesSketchList, dependentValuesSketchList):
        minDistance = float('Inf')
        minDistanceImage = None
        distances = []

        # use lsh to get neighbors
        [distances, indices] = lshForest.kneighbors(np.array(dependentValuesSketch).reshape(1, -1))

        # use indices to get closest images
        # read image name and image feature files
        # then only compare with the few retrieved
        # this is only useful if LSH features are different from features used in the loop
        for index in indices[0]:
            imageNameSimulation = imageNamesSimulationList[index]
            dependentValuesSimulation = get_dependent_values_image(net, scaleData, imageNameSimulation)

            # compute distance
            distance = amf.misc.distance(dependentValuesSketch, dependentValuesSimulation)
            if(distance < minDistance):
                minDistance = distance
                minDistanceImage = imageNameSimulation

        # pick minimum distance
        matches.append((imageNameSketch, minDistanceImage))

    # clear memory
    del lshForest

    return matches

def do_matching(net, scaleData, imageNamesSketchList, dependentValuesSketchList):

    # get raw features
    # [imageNamesSimulationList, dependentValuesSimulationList] = get_dependent_values_folder(net, scaleData, "../../_swarm-lfd-data/" + os.environ["ABM"] + "/images", downsample=11000)
    [imageNamesSimulationList, dependentValuesSimulationList] = get_dependent_values_folder(net, scaleData, "../../_swarm-lfd-data/" + os.environ["ABM"] + "/images_1000")

    # now match
    print >> sys.stderr, "\n"
    print >> sys.stderr, "now matching"
    matches = []
    for imageNameSketch, dependentValuesSketch in zip(imageNamesSketchList, dependentValuesSketchList):
        minDistance = float('Inf')
        minDistanceImage = None
        distances = []

        for imageNameSimulation, dependentValuesSimulation in zip(imageNamesSimulationList, dependentValuesSimulationList):
            # compute distance
            distance = amf.misc.distance(dependentValuesSketch, dependentValuesSimulation)
            if(distance < minDistance):
                minDistance = distance
                minDistanceImage = imageNameSimulation

        # pick minimum distance
        matches.append((imageNameSketch, minDistanceImage))

    return matches

def main():
    # get sketches folder
    sketchesFolder = sys.argv[1]
    
    # initialize and retrieve features
    [net, scaleData] = initialize_featurization()
    [imageNamesSketchList, dependentValuesSketchList] = get_dependent_values_folder(net, scaleData, sketchesFolder)

    # matches = do_matching_lsh(net, scaleData, imageNamesSketchList, dependentValuesSketchList)
    matches = do_matching(net, scaleData, imageNamesSketchList, dependentValuesSketchList)

    # print matches
    # save matches
    simulationSaveFolder = sys.argv[2]
    print >> sys.stderr, "\n"
    print >> sys.stderr, "now printing matches:"
    print >> sys.stderr, "\n"
    for x, y in matches:
        print >> sys.stderr, x
        print >> sys.stderr, y
        print >> sys.stderr, "\n"

        os.system("cp " + y + " " + simulationSaveFolder)
                
main()
