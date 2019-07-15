import matplotlib.pyplot as plt
import numpy as np
import os

abm = os.environ["ABM"]
descriptorSetting = os.environ["DESCRIPTOR_SETTING"]
numDependent = eval(os.environ["NUM_DEPENDENT"])

dataFileName = "../data/domaindata/processed_" + abm
if(eval(os.environ["IMAGES"]) == 1):
   dataFileName += "_" + descriptorSetting.lower()
dataFileName += ".txt"

data = np.genfromtxt(dataFileName, skip_header=1, invalid_raise=False)
dependentData = data[:,-numDependent:]

for index in range(numDependent):
    plotData = dependentData[:,index]
    plotData = plotData[~np.isnan(plotData)]   
    plt.hist(plotData, alpha=1.0/numDependent, label="SLP " + str(index+1))    

plt.xlabel("SLP Value")
plt.ylabel("Number of Instances")
plt.legend(loc="best")
plt.savefig(abm + "_dataset_slps_distribution")
