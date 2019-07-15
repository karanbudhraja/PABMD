import matplotlib.pyplot as plt
import numpy as np

pointSelectionMethods = ["error_weighted", "min_error"]
demonstrations = ["1", "2", "3"]

# plot fm prediction error for various demonstrations
for pointSelectionMethod in pointSelectionMethods:
    # plot unguided and guided data
    plt.figure()

    print(pointSelectionMethod)
    
    for demonstration in demonstrations:
        with open(pointSelectionMethod + "/" + demonstration + "/unguided/fm_prediction_error.txt") as unguidedDataFile:
            unguidedData = [eval(x) for x in unguidedDataFile.readlines()]
        with open(pointSelectionMethod + "/" + demonstration + "/guided/fm_prediction_error.txt") as guidedDataFile:
            guidedData = [eval(x) for x in guidedDataFile.readlines()]

        mean = np.mean(unguidedData)
        sigma = np.std(unguidedData)
        plot1 = plt.errorbar(eval(demonstration), mean, sigma, fmt='s', color='b', markersize=20, elinewidth=2)
        print(mean, sigma)
        
        mean = np.mean(guidedData)
        sigma = np.std(guidedData)
        plot2 = plt.errorbar(eval(demonstration), mean, sigma, fmt='s', color='g', markersize=20, elinewidth=2)
        print(mean, sigma)
                
    plt.gca().set_ylim(bottom=-0.01)
    plt.legend([plot1, plot2], ["Unguided Data", "Guided Data"], loc="best")
    plt.xticks(range(len(demonstrations)+2), [" ", "a", "b", "c", " "])
    plt.xlabel("Demonstration (Demo)")
    plt.ylabel("FM Prediction Error")
    plt.savefig(pointSelectionMethod + ".png")


