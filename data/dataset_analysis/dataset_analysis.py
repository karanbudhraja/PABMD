import matplotlib.pyplot as plt
import numpy
import sys

def read_dataset(fileName):
    dataset = []
    
    with open(fileName + "/slp_dataset.txt", "r") as slpDatasetFile:
        for line in slpDatasetFile:
            point = eval(line)
            dataset.append(point)

    return dataset

def polarize_matrix(matrix, threshold=0):
    # polarize matrix values
    [rows, columns] = matrix.shape

    for row in range(rows):
        for column in range(columns):
            if(matrix[row][column] > threshold):
                matrix[row][column] = 1
            else:
                matrix[row][column] = 0

def analyze_datasets(goodDatasetFolders=[], badDatasetFolders=[], figureSuffix = ""):
    # determine comparison method for quality of datasets

    # demonstration specific data
    demonstrationID = sys.argv[1]
    
    # get all points in good dataset
    goodDataset = []
    for folder in goodDatasetFolders:
        goodDataset += read_dataset(demonstrationID + "/" + "predictions_" + folder)

    # get all points in bad dataset
    badDataset = []
    for folder in badDatasetFolders:
        badDataset += read_dataset(demonstrationID + "/" + "predictions_" + folder)

    # define plot ranges
    # each value represents a threshold for a bucket
    dim1Step = eval(sys.argv[2])
    dim2Step = eval(sys.argv[3])
    dim1ThresholdValues = numpy.arange(0, 1, dim1Step)
    dim2ThresholdValues = numpy.arange(0, 1, dim2Step)

    # now compute matrix based on discrete values
    goodDensityMap = numpy.zeros((len(dim1ThresholdValues), len(dim2ThresholdValues)))
    badDensityMap = numpy.zeros((len(dim1ThresholdValues), len(dim2ThresholdValues)))

    # for good datasets
    for point in goodDataset:
        # determine bucket for this point
        [dim1Value, dim2Value] = point

        # will be an integer value between 0 and len(dim1ThresholdValues)-1
        # will be an integer value between 0 and len(dim2ThresholdValues)-1
        dim1Bucket = int(dim1Value/dim1Step)
        dim2Bucket = int(dim2Value/dim2Step)

        # account for upper limits
        # use capping
        if(dim1Bucket == len(dim1ThresholdValues)):
            dim1Bucket = len(dim1ThresholdValues) - 1
        if(dim2Bucket == len(dim2ThresholdValues)):
            dim2Bucket = len(dim2ThresholdValues) - 1

        # add count to matrix
        goodDensityMap[dim1Bucket][dim2Bucket] += 1

    # for bad datasets
    for point in badDataset:
        # determine bucket for this point
        [dim1Value, dim2Value] = point

        # will be an integer value between 0 and len(dim1ThresholdValues) - 1
        # will be an integer value between 0 and len(dim2ThresholdValues) - 1
        dim1Bucket = int(dim1Value/dim1Step) - 1
        dim2Bucket = int(dim2Value/dim2Step) - 1

        # account for upper limits
        # use capping
        if(dim1Bucket == len(dim1ThresholdValues)):
            dim1Bucket = len(dim1ThresholdValues) - 1
        if(dim2Bucket == len(dim2ThresholdValues)):
            dim2Bucket = len(dim2ThresholdValues) - 1
        
        # add count to matrix
        badDensityMap[dim1Bucket][dim2Bucket] += 1

    # compute difference
    # and normalize
    goodDifferenceDensityMap = goodDensityMap - badDensityMap
    badDifferenceDensityMap = badDensityMap - goodDensityMap

    polarize_matrix(goodDifferenceDensityMap, 0)
    polarize_matrix(badDifferenceDensityMap, 0)

    # splice matrices if needed
    # to zoom in on finer sections
    #goodDifferenceDensityMap = goodDifferenceDensityMap[:10, :10]
    #badDifferenceDensityMap = badDifferenceDensityMap[:10, :10]
    #dim1ThresholdValues = dim1ThresholdValues[:10]
    #dim2ThresholdValues = dim2ThresholdValues[:10]
    
    plt.figure()
    plt.imshow(goodDifferenceDensityMap, interpolation='nearest', cmap=plt.cm.gray)
    plt.ylabel("Maximum Area")
    plt.xlabel("Maximum Perimeter")
    #plt.title("Good Difference Density Map")
    plt.yticks(range(len(dim1ThresholdValues)), dim1ThresholdValues)
    plt.xticks(range(len(dim2ThresholdValues)), dim2ThresholdValues)
    plt.savefig(demonstrationID + "/" + "good_difference_density_map" + figureSuffix + ".png")
    plt.close("all")
    
    plt.figure()
    plt.imshow(badDifferenceDensityMap, interpolation='nearest', cmap=plt.cm.gray)
    plt.ylabel("Maximum Area")
    plt.xlabel("Maximum Perimeter")
    #plt.title("Bad Difference Density Map")
    plt.yticks(range(len(dim1ThresholdValues)), dim1ThresholdValues)
    plt.xticks(range(len(dim2ThresholdValues)), dim2ThresholdValues)
    plt.savefig(demonstrationID + "/" + "bad_difference_density_map" + figureSuffix + ".png")
    plt.close("all")
    
def main():
    analyze_datasets(["best", "second_best", "third_best"], ["worst", "second_worst", "third_worst"], "_all")
    analyze_datasets(["best"], ["worst"], "_1_1")
    analyze_datasets(["best"], ["second_worst"], "_1_2")
    analyze_datasets(["best"], ["third_worst"], "_1_3")
    analyze_datasets(["second_best"], ["worst"], "_2_1")
    analyze_datasets(["second_best"], ["second_worst"], "_2_2")
    analyze_datasets(["second_best"], ["third_worst"], "_2_3")
    analyze_datasets(["third_best"], ["worst"], "_3_1")
    analyze_datasets(["third_best"], ["second_worst"], "_3_2")
    analyze_datasets(["third_best"], ["third_worst"], "_3_3")
    
main()
