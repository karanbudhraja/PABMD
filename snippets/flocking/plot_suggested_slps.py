import matplotlib.pyplot as plt
import numpy as np

#demonstrations = ["a", "b", "c"]
#demonstrations = ["_a", "_b", "_c"]
#demonstrations = ["a_filtered", "b_filtered", "c_filtered"]
#demonstrations = ["_a"]
#demonstrations = ["a"]
demonstrations = ["a_filtered"]

#folderName = "error_weighted"
folderName = "min_error"

numberOfBins = 10

with open("../../src/sampling/flocking/scale_data_flocking_contour.txt", "r") as inFile:
    allLines = inFile.readlines()
    maxValueArea = eval(allLines[0])[-2]
    maxValuePerimeter = eval(allLines[0])[-1]
    minValueArea = eval(allLines[1])[-2]
    minValuePerimeter = eval(allLines[1])[-1]

for index, item in enumerate(demonstrations):
    allPointsX = []
    allPointsY = []
    errors = []

    # scatter
    
    plt.figure()
    with open(folderName + "/suggested_slps_" + item + ".txt", "r") as inFile:
        for line in inFile:
            point = eval(line)[0]
            plt.scatter(point[0], point[1], c="b", marker="o", alpha=0.3)
            allPointsX.append(point[0])
            allPointsY.append(point[1])

            # compute error
            with open("demonstration_slps.txt", "r") as inFile:
                demonstration = list(eval(inFile.readlines()[index]))
                demonstration[0] = (demonstration[0]-minValueArea)/(maxValueArea-minValueArea)
                demonstration[1] = (demonstration[1]-minValuePerimeter)/(maxValuePerimeter-minValuePerimeter)

                _point = [x for x in point]
                _point[0] = (_point[0]-minValueArea)/(maxValueArea-minValueArea)
                _point[1] = (_point[1]-minValuePerimeter)/(maxValuePerimeter-minValuePerimeter)
                
                error = (_point[0]-demonstration[0])**2 + (_point[1]-demonstration[1])**2
                errors.append(error)
            
        plt.scatter(point[0], point[1], c="b", marker="o", alpha=0.3, label="Suggested SLPs")

    with open("demonstration_slps.txt", "r") as inFile:
        point = eval(inFile.readlines()[index])
        plt.scatter(point[0], point[1], c="r", marker="s", alpha=1.0, label="Demonstration SLPs")

    plt.scatter(np.mean(allPointsX), np.mean(allPointsY), c="g", marker="s", alpha=1.0, label="Mean SLPs")

    print np.mean(errors)
    print np.std(errors)
    
    ax = plt.gca()

    ticks = ax.get_xticks()
    minValue = minValueArea
    maxValue = maxValueArea
    scaledTicks = [(value-minValue)/(maxValue-minValue) for value in ticks]
    scaledTicks = [int(value*100)/100.0 for value in scaledTicks]
    ax.set_xticklabels(scaledTicks)
    
    ticks = ax.get_yticks()
    minValue = minValuePerimeter
    maxValue = maxValuePerimeter
    scaledTicks = [(value-minValue)/(maxValue-minValue) for value in ticks]
    scaledTicks = [int(value*100)/100.0 for value in scaledTicks]
    ax.set_yticklabels(scaledTicks)
        
    plt.xlabel("Maximum Area")
    plt.ylabel("Maximum Perimeter")
    plt.legend()
    plt.savefig(folderName + "/suggested_slps_" + item)

    # histogram

    plt.figure()
    allPointsX = [(x-minValueArea)/(maxValueArea-minValueArea) for x in allPointsX]
    allPointsY = [(y-minValuePerimeter)/(maxValuePerimeter-minValuePerimeter) for y in allPointsY]    
    histogram, xEdges, yEdges = np.histogram2d(allPointsX, allPointsY, bins=numberOfBins, range=[[0,1],[0,1]])
    plt.imshow(histogram, interpolation='nearest', origin='low')
    plt.ylabel("Maximum Area")
    plt.xlabel("Maximum Perimeter")
    plt.colorbar(label="Number of Points")
    ax = plt.gca()

    # works for normalized values
    ticks = ax.get_xticks()
    ticks = [value/numberOfBins for value in ticks]
    ax.set_xticklabels(ticks)
    ticks = ax.get_yticks()
    ticks = [value/numberOfBins for value in ticks]
    ax.set_yticklabels(ticks)

    plt.savefig(folderName + "/histogram_suggested_slps_" + item)
