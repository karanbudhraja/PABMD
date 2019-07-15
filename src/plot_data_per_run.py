import numpy as np
import matplotlib.pyplot as plt

def plot_confidence(lines, index, marker, markerSize=None, label=None):
    # compute plot data
    data = np.array([x[index] for x in lines])
    average = np.median(data, axis=0)

    # compute upper and lower ranks
    # index starts from zero so we take floor
    # formula from here: http://www.ucl.ac.uk/ich/short-courses-events/about-stats-courses/stats-rm/Chapter_8_Content/confidence_interval_single_median
    n = data.shape[0]
    lowerRank = int(np.floor((n/2) - (1.96 * np.sqrt(n))/2))
    upperRank = int(np.floor(1 + (n/2) + (1.96 * np.sqrt(n))/2))

    upperBound = []
    lowerBound = []
    for index in range(data.shape[1]):
        items = sorted(list(data[:, index]), reverse=True)
        upperBound.append(items[lowerRank])
        lowerBound.append(items[upperRank])
    
    # plot
    if(markerSize is not None):
        plot = plt.plot(average, marker=marker, markersize=markerSize, label=label)
    else:
        plot = plt.plot(average, marker=marker, label=label)
    plt.fill_between(range(average.shape[0]), upperBound, lowerBound, color=plot[-1].get_color(), alpha=.5)
        
inFileNames = []
inFileNames.append("plot_data_per_run_a.txt")
inFileNames.append("plot_data_per_run_b.txt")
inFileNames.append("plot_data_per_run_c.txt")
inFileNames.append("plot_data_per_run_a_od.txt")
inFileNames.append("plot_data_per_run_b_od.txt")
inFileNames.append("plot_data_per_run_c_od.txt")

for inFileName in inFileNames:
    outFileName = inFileName.replace(".txt", ".png")

    with open(inFileName, "r") as inFile:
        lines = inFile.readlines()
        lines = [eval(x.strip("\n")) for x in lines]

    # data = [lenCommonSimplexes, lenAllConfigurationsCommon, errorMu, errorSuggested]
    plt.figure()
    plot_confidence(lines, 0, marker="*", markerSize=10, label="Simplexes Used")
    plot_confidence(lines, 1, marker="s", label="Configurations Used")
    plot_confidence(lines, 2, marker="o", label="Mean FM Error (All Possible ALPs)")
    plot_confidence(lines, 3, marker="D", label="FM Error (Selected ALPs)")
    plt.legend(loc="best")

    if("_od" in inFileName):
        plt.ylabel("Ratio of Measure With vs. Without Pruning + Outlier Detection")
    else:
        plt.ylabel("Ratio of Measure With vs. Without Pruning")

    plt.xlabel("Minimum Number of Used Simplex Edges")
    [low, high] = plt.gca().get_ylim()
    plt.gca().set_ylim([0,1.1*high])
    plt.gca().set_xlim([0,13])
    plt.savefig(outFileName)
