import matplotlib.pyplot as plt
import glob
import numpy as np

#methods = ["knn_random_subset", "knn_fixed_subset"]
methods = ["gp_random_subset", "gp_fixed_subset"]
labels = ["Random Subset", "Fixed Subset"]
inputMarkers = ["^", "s"]
outputMarkers = ["o", "x"]
inputColors = ["k", "r"]
outputColors = ["b", "g"]
compositePlot = plt.figure()

for method, inputMarker, outputMarker, inputColor, outputColor, label in zip(methods, inputMarkers, outputMarkers, inputColors, outputColors, labels):
    plt.figure()
    outputDifferences = []
    inputDifferences = []
    parameterValues = []
    
    for fileName in glob.glob("feedback_difference_" + method + "/*output*.txt"):
        with open(fileName) as inFile:
            difference = inFile.readlines()
            difference = [eval(x) for x in difference]
            outputDifferences.append(difference)
    for fileName in glob.glob("feedback_difference_" + method + "/*input*.txt"):
        with open(fileName) as inFile:
            difference = inFile.readlines()
            difference = [eval(x) for x in difference]
            inputDifferences.append(difference)
    for fileName in glob.glob("feedback_difference_" + method + "/*parameter*.txt"):
        with open(fileName) as inFile:
            difference = inFile.readlines()
            difference = [eval(x) for x in difference]
            parameterValues.append(difference)

    print method

    if("gp_fixed_subset" in method):
        outputDifferences = [x for x in outputDifferences if len(x) == 99]
    
    outputDifferences = np.array(outputDifferences)
    print outputDifferences.shape





    
    
    meanOutputDifferences = np.mean(outputDifferences, axis=0)
    medianOutputDifferences = np.median(outputDifferences, axis=0)
    #inputDifferences = np.array(inputDifferences)
    #meanInputDifferences = np.mean(inputDifferences, axis=0)
    #medianInputDifferences = np.median(inputDifferences, axis=0)

    parameterValues = np.array(parameterValues)
    medianParameterValues = np.median(parameterValues, axis=0)/100
    medianParameterValues = medianParameterValues[1:]


    print medianOutputDifferences.shape
    print medianParameterValues.shape
    
    #print medianParameterValues

    #continue
    
    
    x = range(1, len(meanOutputDifferences) + 1)
    plt.plot(x, meanOutputDifferences, label=("Output (mean)"), marker=outputMarker)
    plt.plot(x, medianOutputDifferences, label=("Output (median)"), marker=outputMarker)
    #plt.plot(x, meanInputDifferences, label=("Input (mean)"), marker=inputMarker)
    #plt.plot(x, medianInputDifferences, label=("Input (median)"), marker=inputMarker)
    plt.legend(loc="best")
    plt.title(label)
    plt.xlabel("Iteration")
    plt.ylabel("Difference (SLP space)")
    plt.savefig("feedback_difference_average_" + method + ".png")

    print "method: " + method
    print "mean (output): " + str(meanOutputDifferences[-5:])
    print "median (output): " + str(medianOutputDifferences[-5:])
    #print "mean (input): " + str(meanInputDifferences[-5:])
    #print "median (input): " + str(medianInputDifferences[-5:])

    plt.figure(compositePlot.number)
    plt.plot(x, medianOutputDifferences, label=(label + " (Output Difference)"), marker=outputMarker, c=outputColor)
    #plt.plot(x, medianInputDifferences, label=(label + " (Input)"), marker=inputMarker, c=inputColor)
    plt.plot(x, medianParameterValues, label=(label + " (Regression Parameter)"), marker=inputMarker, c=inputColor)

plt.legend(loc="best")
#plt.legend(loc="upper right")
plt.xlabel("Iteration")
#plt.xlabel("-log(" + r"$\alpha$" + ")")
#plt.xlabel("k")
#plt.ylabel("Difference (SLP space)")
plt.ylabel("Value")
plt.savefig("feedback_difference_average_composite.png")
