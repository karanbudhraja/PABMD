import sys
import matplotlib.pyplot as plt
import numpy as np

with open("feedback_output_difference.txt") as inFile:
    difference = inFile.readlines()
    difference = [eval(x) for x in difference]
    print >> sys.stderr, np.sum(difference)
plt.figure()
plt.plot(difference, label="difference")
plt.legend()
plt.savefig("feedback_output_difference.png")

with open("feedback_input_difference.txt") as inFile:
    difference = inFile.readlines()
    difference = [eval(x) for x in difference]
    print >> sys.stderr, np.sum(difference)
plt.figure()
plt.plot(difference, label="difference")
plt.legend()
plt.savefig("feedback_input_difference.png")
