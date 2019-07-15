import numpy as np
from sklearn.decomposition import KernelPCA
import cPickle

numDependent = 50

# read input
with open("processed_turbulence_vgg.txt", "r") as inFile:
    lines = inFile.readlines()[1:]
    lines = [[eval(x) for x in line.split(" ")] for line in lines]

lines = np.array(lines)
alps = lines[:,:2]
slps = lines[:,2:]
kpca = KernelPCA(n_components=numDependent, kernel="rbf")
model = kpca.fit(slps)
slpsKpca = model.transform(slps)
xKpca = np.concatenate((alps, slpsKpca), axis=1)

rows, columns = xKpca.shape

# write output
with open("processed_turbulence_vgg_reduced.txt", "w") as outFile:
    outFile.write("i"*2 + "d"*numDependent)

    for rowIndex in range(rows):
        string = "\n"
        for columnIndex in range(columns):
            string += str(xKpca[rowIndex][columnIndex]) + " "
        string = string[:-1]
        outFile.write(string)

# save transform
cPickle.dump(model, open("../../src/sampling/utils/transforms/turbulence_reduced_model.p", "wb"))
