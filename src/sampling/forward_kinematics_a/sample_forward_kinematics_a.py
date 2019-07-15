import numpy as np

INDEPENDENT_VARIABLES = 1
DEPENDENT_VARIABLES = 2

# processed data
# drop the second column
processedData = np.genfromtxt("../../../data/domaindata/processed_" + "forward_kinematics" + ".txt", skip_header=1)
aData = np.delete(processedData, [1], axis=1)
np.savetxt("../../../data/domaindata/processed_" + "forward_kinematics_a" + ".txt", aData, fmt="%s", header="i"*INDEPENDENT_VARIABLES+"d"*DEPENDENT_VARIABLES, comments="")

# scale data
with open("../" + "forward_kinematics" + "/scale_data_forward_kinematics.txt", "r") as inFile:
    lines = inFile.readlines()
lines = [eval(x) for x in lines]
scaleData = np.asarray(lines)
aData = np.delete(scaleData, [1], axis=1)
with open("scale_data_forward_kinematics_a.txt", "w") as outFile:
    outFile.write(str(aData.tolist()[0]) + "\n" + str(aData.tolist()[1]))
