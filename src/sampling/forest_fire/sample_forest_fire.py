from model import ForestFire
import random
import numpy as np

# only one ALP
# density (0,1)

# constant ALPs
HEIGHT = 100
WIDTH = 100

# constants
# max_iters in the example is specified as 1000
# so we run for 1000 steps
INDEPENDENT_VARIABLES = 1
DEPENDENT_VARIABLES = 1
RUNS = 10000
SAMPLES = 10
TIME_LAPSE = 200

def main_forest_fire(density=0.65):
    # instantiate and run model
    model = ForestFire(height=HEIGHT, width=WIDTH, density=density)

    # run time lapse
    for i in range(TIME_LAPSE):
        model.step()

    # collect data for next steps
    dependentValues = []

    for i in range(SAMPLES):
        # step
        model.step()

    # read data
    data = model.datacollector.get_model_vars_dataframe()
    burnedTrees = np.array(list(data.BurnedOut)[-SAMPLES-1:])
    fineTrees = np.array(list(data.Fine)[-SAMPLES-1:])
    initialTrees = burnedTrees + fineTrees
    dependentValues.append(list(burnedTrees/initialTrees))

    # print line corresponding to this execution
    line = str(density)
    for dependentValue in dependentValues:
        line += " " + str(dependentValue).replace("[", "").replace("]", "").replace(" ", "")
    print(line)
    
def main():
    # fixed randomness
    random.seed(0)
    
    # header string
    print("i"*INDEPENDENT_VARIABLES + "d"*DEPENDENT_VARIABLES)

    for run in range(RUNS):

        # sample random ALPs
        # cop density + citizen density should be less than 1 
        density = random.random()
                
        # run model using those ALPs
        main_forest_fire(density)

main()
