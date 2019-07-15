import os
import sys

# get demonstration from input line
numDependent = eval(os.environ["NUM_DEPENDENT"])
line = sys.argv[1:]
demonstration = tuple([eval(x) for x in line[-numDependent:]])

# write to queried slps
with open("queried_slps.txt", "w") as outFile:
    outFile.write(str(demonstration))
