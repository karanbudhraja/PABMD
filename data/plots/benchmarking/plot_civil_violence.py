import numpy as np
import matplotlib.pyplot as plt

dataSize = []
nnMeanStd = []
amfMeanStd = []
nnVariationalMeanStd = []

dataSize.append(100)
nnMeanStd.append((0.15198525007584848, 0.16513708289410955))
amfMeanStd.append((0.097739180100765, 0.160733969955631))
nnVariationalMeanStd.append((0.1394490311154111, 0.11509733167882559))
dataSize.append(250)
nnMeanStd.append((0.08879823340818987, 0.09061124803265028))
amfMeanStd.append((0.09328328761129923, 0.17908926720933419))
nnVariationalMeanStd.append((0.136850051380765, 0.12799416231386207))
dataSize.append(500)
nnMeanStd.append((0.09549226762938177, 0.12214399813935982))
amfMeanStd.append((0.10053660377125745, 0.1922613405039135))
nnVariationalMeanStd.append((0.09103654148487, 0.09337872291356436))
dataSize.append(1000)
nnMeanStd.append((0.045308563742089886, 0.0440220936994438))
amfMeanStd.append((0.07495233807375593, 0.14934520969665221))
nnVariationalMeanStd.append((0.06919772964384271, 0.053904901152214475))
dataSize.append(2500)
nnMeanStd.append((0.0369461729113718, 0.031177819683831178))
amfMeanStd.append((0.05861502828016272, 0.12582774087080834))
nnVariationalMeanStd.append((0.06775043439477779, 0.050913199676669164))
dataSize.append(5000)
nnMeanStd.append((0.038535772461978104, 0.035072298245167034))
amfMeanStd.append((0.06995707607784174, 0.1745052931584609))
nnVariationalMeanStd.append((0.0590976141745957, 0.03145616942376669))
dataSize.append(10000)
nnMeanStd.append((0.030982929294207297, 0.0247511139027531))
amfMeanStd.append((0.056992901387702286, 0.12134186753477924))
nnVariationalMeanStd.append((0.05823105513736669, 0.03860283215191619))

nnMean = np.array([x for (x, y) in nnMeanStd])
nnStd = np.array([y for (x, y) in nnMeanStd])
amfMean = np.array([x for (x, y) in amfMeanStd])
amfStd = np.array([y for (x, y) in amfMeanStd])
nnVariationalMean = np.array([x for (x, y) in nnVariationalMeanStd])
nnVariationalStd = np.array([y for (x, y) in nnVariationalMeanStd])

#nnMean = np.log(nnMean)
#nnStd = np.log(nnStd)
#amfMean = np.log(amfMean)
#amfStd = np.log(amfStd)
#nnVariationalMean = np.log(nnVariationalMean)
#nnVariationalStd = np.log(nnVariationalStd)

def plot_fill(parameters):
    [meanValue, stdValue, color, marker, label, linestyle] = parameters
    upperBound = meanValue + stdValue
    lowerBound = meanValue - stdValue
    x = range(len(dataSize))
    plt.plot(x, meanValue, linewidth=2, marker=marker, label=label, color=color, linestyle=linestyle)
    plt.fill_between(x, upperBound, lowerBound, facecolor=color, alpha=0.125) 

plot_fill([amfMean, amfStd, "blue", "o", "AMF+", "-"])
plot_fill([nnMean, nnStd, "red", "s", "RM+FM MLP", "--"])
#plot_fill([nnVariationalMean, nnVariationalStd, "green", "^", "Variational Network", "-."])

plt.xlim([-1,7])
plt.xticks(list(range(len(dataSize))), dataSize + [""])
#plt.ylabel("log (Output Difference)")
plt.ylabel("Output Difference")
plt.xlabel("Dataset Size")
plt.legend(loc="best")
plt.savefig("civil_violence")
plt.show()
