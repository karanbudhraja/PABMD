import matplotlib.pyplot as plt
import numpy as np

#demonstrations = ["a", "b", "c"]
#demonstrations = ["_a", "_b", "_c"]
demonstrations = ["a"]
#demonstrations = ["_a"]
#demonstrations = ["a_filtered"]

folderName = "error_weighted"
#folderName = "min_error"

with open("../../src/sampling/turbulence/scale_data_turbulence_density.txt", "r") as inFile:
    allLines = inFile.readlines()
    maxValueDensity = eval(allLines[0])[-1]
    minValueDensity = eval(allLines[1])[-1]

for index, item in enumerate(demonstrations):
    allPointsX = []
    allPointsY = []

    # scatter
    
    plt.figure()
    with open(folderName + "/suggested_slps_" + item + ".txt", "r") as inFile:
        for line in inFile:
            point = eval(line)[0]

            # scale
            #point = tuple([(point[0]-minValueDensity)/(maxValueDensity-minValueDensity)])

            plt.scatter(point[0], point[0], c="b", marker="o", alpha=0.3)
            allPointsX.append(point[0])
            allPointsY.append(point[0])
            
        plt.scatter(point[0], point[0], c="b", marker="o", alpha=0.3, label="Suggested SLPs")

    with open("demonstration_slps.txt", "r") as inFile:
        point = eval(inFile.readlines()[index])

        # scale
        #point = tuple([(point[0]-minValueDensity)/(maxValueDensity-minValueDensity)])
            
        plt.scatter(point[0], point[0], c="r", marker="s", alpha=1.0, label="Demonstration SLPs")

    plt.scatter(np.mean(allPointsX), np.mean(allPointsY), c="g", marker="s", alpha=1.0, label="Mean SLPs")
    plt.xlabel("Density")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(folderName + "/suggested_slps_" + item)

    # histogram
    
    plt.figure()
    plt.hist(allPointsX, bins=50, range=(0,50))
    plt.xlabel("Density")
    plt.ylabel("Number of Points")
    plt.gca().set_ylim([0,100])
    plt.savefig(folderName + "/histogram_suggested_slps_" + item)
