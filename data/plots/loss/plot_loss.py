import numpy as np
import matplotlib.pyplot as plt
import sys

alpha = sys.argv[1]
samples = eval(sys.argv[2])

def plot_fill(parameters):
    [meanValue, stdValue, color, label, linestyle] = parameters
    upperBound = meanValue + stdValue
    lowerBound = meanValue - stdValue
    x = range(len(meanValue))
    plt.plot(x, meanValue, linewidth=2, label=label, color=color, linestyle=linestyle)
    plt.fill_between(x, upperBound, lowerBound, facecolor=color, alpha=0.125) 

klLoss = np.genfromtxt(alpha + "/kl_loss.txt").reshape(10,-1)
reconstructionLoss = np.genfromtxt(alpha + "/reconstruction_loss.txt").reshape(10,-1)

# replace zeros with minimum
np.place(klLoss, klLoss==0, np.min(klLoss[np.nonzero(klLoss)]))
np.place(reconstructionLoss, reconstructionLoss==0, np.min(reconstructionLoss[np.nonzero(reconstructionLoss)]))

totalLoss = eval(alpha)*klLoss + (1-eval(alpha))*reconstructionLoss

klLoss = np.log(klLoss)
reconstructionLoss = np.log(reconstructionLoss)
totalLoss = np.log(totalLoss)

plot_fill([np.mean(klLoss, axis=0)[:samples], np.std(klLoss, axis=0)[:samples], "blue", "KL Divergence Loss", "-."])
plot_fill([np.mean(reconstructionLoss, axis=0)[:samples], np.std(reconstructionLoss, axis=0)[:samples], "red", "Reconstruction Loss", "--"])
plot_fill([np.mean(totalLoss, axis=0)[:samples], np.std(totalLoss, axis=0)[:samples], "black", "Total Loss", "-"])

plt.xlabel("Evaluation Number")
plt.ylabel("log(Loss Value)")
plt.legend(loc="best")
plt.savefig("variational_loss_" + alpha.replace(".",""))
plt.close()
