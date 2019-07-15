import matplotlib.pyplot as plt
import numpy as np

numIndependent = 2
numDependent = 2

abm = "forward_kinematics"
#abm = "aids"
#abm = "eum"

data = np.genfromtxt("../../domaindata/cross_validation/" + abm + "_split_0_train.txt", skip_header=1)
X = data[:,:numIndependent]
Y = data[:,numIndependent:]
#plt.scatter(Y[:,0], Y[:,1])
#plt.show()
#plt.hist(Y)
#plt.show()

# now try transforming the inputs
Y0 = Y[:,0]
Y1 = Y[:,1]

Z0 = (Y0-Y1)**2
Z1 = 1-Y0**2

plt.scatter(Z0, Z1)
plt.show()
Z = np.vstack((Z0, Z1))
plt.hist(Z.T)
plt.show()

# now concatenate X and Z matrices
#Z = np.vstack((Z0, Z1))
#transformedData = np.hstack((X, Z.T))
#np.savetxt("forward_kinematics_tranformed" + ".txt", transformedData, fmt="%s", header="i"*numIndependent+"d"*numDependent, comments="")
