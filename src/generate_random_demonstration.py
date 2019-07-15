import os
import random

# generate a random demonstration
numDependent = eval(os.environ["NUM_DEPENDENT"])
demonstration = tuple([random.random() for _ in range(numDependent)])

# write to queried slps
with open("queried_slps.txt", "w") as outFile:
    outFile.write(str(demonstration))
