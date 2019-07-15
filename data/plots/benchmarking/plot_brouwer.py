import numpy as np
import matplotlib.pyplot as plt

dataSize = []
mseMeanStd = []
dataSize.append(100)
mseMeanStd.append((0.00399207990978, 0.0102862721674))
dataSize.append(250)
mseMeanStd.append((0.0022746470068, 0.00842019577855))
dataSize.append(500)
mseMeanStd.append((0.000437167936693, 0.00118155366598))
dataSize.append(1000)
mseMeanStd.append((0.000248135706607, 0.000683272654568))
dataSize.append(2500)
mseMeanStd.append((0.000112395675771, 0.000332818798969))
dataSize.append(5000)
mseMeanStd.append((5.03160147066e-05, 0.000160085705792))                    
dataSize.append(10000)
mseMeanStd.append((3.18084884214e-05, 9.3349292989e-05))

mseMean = np.array([x for (x, y) in mseMeanStd])
mseStd = np.array([y for (x, y) in mseMeanStd])
BASELINE = np.array(0.077)

plt.hlines(BASELINE, 0, len(mseMean)-1, linewidths=5, label="R. K. Brouwer (2004)")
plt.errorbar(x=list(range(len(mseMean))), y=mseMean, yerr=mseStd, linewidth=2, capsize=10, marker="s", markersize=10, label="AMF+")

plt.xlim([-1,7])
plt.xticks(list(range(len(dataSize))), dataSize + [""])
#plt.ylabel("log (MSE)")
plt.ylabel("MSE")
plt.xlabel("Dataset Size")
plt.legend(loc="best")
plt.savefig("brouwer_sin")
plt.show()
