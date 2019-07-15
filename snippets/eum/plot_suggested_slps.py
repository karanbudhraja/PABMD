import matplotlib.pyplot as plt
import numpy as np

#demonstrations = ["a", "b", "c"]
#demonstrations = ["_a", "_b", "_c"]
demonstrations = ["a"]
#demonstrations = ["_a"]
#demonstrations = ["a_filtered"]

folderName = "error_weighted"
#folderName = "min_error"

numberOfBins = 10

with open("../../src/sampling/eum/scale_data_eum.txt", "r") as inFile:
    allLines = inFile.readlines()
    maxValueMean = eval(allLines[0])[-2]
    maxValueMedian = eval(allLines[0])[-1]
    minValueMean = eval(allLines[1])[-2]
    minValueMedian = eval(allLines[1])[-1]

for index, item in enumerate(demonstrations):
    allPointsX = []
    allPointsY = []
    errors = []
    
    # scatter
    
    plt.figure()
    with open(folderName + "/suggested_slps_" + item + ".txt", "r") as inFile:
        for line in inFile:
            try:
                point = eval(line)[0]
                plt.scatter(point[0], point[1], c="b", marker="o", alpha=0.3)
                allPointsX.append(point[0])
                allPointsY.append(point[1])

                # compute error
                with open("demonstration_slps.txt", "r") as inFile:
                    demonstration = eval(inFile.readlines()[index])
                    error = (point[0]-demonstration[0])**2 + (point[1]-demonstration[1])**2
                    errors.append(error)
                    
            except:
                continue
            
        plt.scatter(point[0], point[1], c="b", marker="o", alpha=0.3, label="Suggested SLPs")

    with open("demonstration_slps.txt", "r") as inFile:
        point = eval(inFile.readlines()[index])
        plt.scatter(point[0], point[1], c="r", marker="s", alpha=1.0, label="Demonstration SLPs")
        
    plt.scatter(np.mean(allPointsX), np.mean(allPointsY), c="g", marker="s", alpha=1.0, label="Mean SLPs")

    print np.mean(errors)
    print np.std(errors)
    
    ax = plt.gca()

    ticks = ax.get_xticks()
    minValue = minValueMean
    maxValue = maxValueMean
    scaledTicks = [(value-minValue)/(maxValue-minValue) for value in ticks]
    scaledTicks = [int(value*100)/100.0 for value in scaledTicks]
    ax.set_xticklabels(scaledTicks)
    
    ticks = ax.get_yticks()
    minValue = minValueMedian
    maxValue = maxValueMedian
    scaledTicks = [(value-minValue)/(maxValue-minValue) for value in ticks]
    scaledTicks = [int(value*100)/100.0 for value in scaledTicks]
    ax.set_yticklabels(scaledTicks)

    plt.ylabel("Median")
    plt.xlabel("Mean")
    plt.legend()
    plt.savefig(folderName + "/suggested_slps_" + item)

    # histogram
    
    plt.figure()
    histogram, xEdges, yEdges = np.histogram2d(allPointsX, allPointsY, bins=numberOfBins)
    plt.imshow(histogram, interpolation='nearest', origin='low')
    plt.xlabel("Mean Final Position")
    plt.ylabel("Median Final Position")
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
