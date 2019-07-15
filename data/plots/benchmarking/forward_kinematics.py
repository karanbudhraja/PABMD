import matplotlib.pyplot as plt
import numpy as np

numIndependent = 2
numDependent = 2

data = np.genfromtxt("../../domaindata/processed_" + "forward_kinematics" + ".txt", skip_header=1)
X = data[:,:numIndependent]
Y = data[:,numIndependent:]
plt.scatter(Y[:,0], Y[:,1])
plt.show()
