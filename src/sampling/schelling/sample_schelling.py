from model import SchellingModel
import random

# only use three of the several ALPs
# density (0,1)
# minority_pc (0,1)

# use one model-level SLP
# happy

# constant ALPs
HEIGHT = 20
WIDTH = 20
HOMOPHILY = 4

# constants
# max_iters in the example is specified as 1000
# so we run for 1000 steps
INDEPENDENT_VARIABLES = 2
DEPENDENT_VARIABLES = 1
RUNS = 10000
SAMPLES = 10
TIME_LAPSE = 1000

def main_schelling(density, minorityPc):
    # instantiate and run model
    model = SchellingModel(
        height=HEIGHT,
        width=WIDTH,
        density=density,
        minority_pc=minorityPc,
        homophily=HOMOPHILY)

    # run time lapse
    for i in range(TIME_LAPSE):
        try:
            # step
            model.step()
        except:
            # no empty cells
            pass
        
    # collect data for next steps
    dependentValues = []

    for i in range(SAMPLES):
        try:
            # step
            model.step()
        except:
            # saturated
            # no empty cells
            pass

    # read data
    data = model.datacollector.get_model_vars_dataframe()    
    dependentValues.append(list(data.happy)[-SAMPLES-1:])

    # print line corresponding to this execution
    line = str(density) + " " + str(minorityPc)
    for dependentValue in dependentValues:
        line += " " + str(dependentValue).replace("[", "").replace("]", "").replace(" ", "")
    print(line)

    
def main():
    # header string
    print("i"*INDEPENDENT_VARIABLES + "d"*DEPENDENT_VARIABLES)

    for run in range(RUNS):
        # sample random ALPs
        density = random.random() 
        minorityPc = random.random() 
        
        # run model using those ALPs
        main_schelling(density, minorityPc)

main()
