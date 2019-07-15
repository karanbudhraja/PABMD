import sys
import numpy as np
import matplotlib.pyplot as plt

EVALUATION_PREFIX = "evaluation_distance_"
EVALUATION_SUFFIX = "_cross_validation.txt"
EVALUATION_SUFFIX_RANDOM = "_cross_validation_random_baseline.txt"

# get abm
abm = sys.argv[1]

# get data
randomData = np.genfromtxt("random/" + EVALUATION_PREFIX + abm + EVALUATION_SUFFIX_RANDOM)
withSamplingDatasetSelectionPruningData = np.genfromtxt("with_sampling_datasetselection_pruning/" + EVALUATION_PREFIX + abm + EVALUATION_SUFFIX)
withoutSamplingData = np.genfromtxt("without_sampling/" + EVALUATION_PREFIX + abm + EVALUATION_SUFFIX)
withoutDatasetSelectionData = np.genfromtxt("without_datasetselection/" + EVALUATION_PREFIX + abm + EVALUATION_SUFFIX)
withoutPruningData = np.genfromtxt("without_pruning/" + EVALUATION_PREFIX + abm + EVALUATION_SUFFIX)
allPlusOutlierDetection = np.genfromtxt("../all_with_od/" + EVALUATION_PREFIX + abm + EVALUATION_SUFFIX)
dataMarkers = ["Random", "+All+OD", "+All", "-Sampling", "-Dataset", "-Pruning"]

# compile
# replace zeros with minimum non zero value
allData = [randomData, allPlusOutlierDetection, withSamplingDatasetSelectionPruningData, withoutSamplingData, withoutDatasetSelectionData, withoutPruningData]
allData = list(allData)
for index in range(len(allData)):
    np.place(allData[index], allData[index]==0, np.min(allData[index][np.nonzero(allData[index])]))
allData = list(np.log(allData))

quartile1, medians, quartile3 = np.percentile(allData, [25, 50, 75], axis=1)
indices = np.arange(1, len(medians) + 1)

# plot data
plots = plt.violinplot(allData, showmedians=False, showextrema=False)

# set colors
palette = ['r','b']
colors = [palette[0]] + [palette[1]]*5
for plot, color in zip(plots['bodies'], colors):
    plot.set_color(color)
plots['bodies'][np.argmin(medians)].set_color('g')

# set legend
plt.plot([], c=palette[0], label="Random Baseline")
plt.plot([], c='g', label="Best Configuration")
plt.plot([], c=palette[1], label="Other Configuration")
legend = plt.legend(loc="best")
for item in legend.legendHandles:
    item.set_linewidth(5)
    item.set_alpha(0.4)

# plot medians
plt.scatter(indices, medians, marker='o', color='w', s=30, zorder=3)
plt.vlines(indices, quartile1, quartile3, color='k', linestyle='-', lw=5)

# set labels
plt.ylabel("log (Output Difference)")
plt.xlabel("Configuration")
plt.xticks(indices, dataMarkers, rotation='horizontal')
plt.savefig("evaluation_" + abm)
