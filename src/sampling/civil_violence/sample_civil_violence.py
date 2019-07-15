from model import CivilViolenceModel
import random

# only use three of the several ALPs
# citizen_density (0,1)
# cop_density (0,1)
# legitimacy (0,1)

# use three model-level SLPs
# Quiescent
# Active
# Jailed

# constant ALPs
HEIGHT = 40
WIDTH = 40
CITIZEN_VISION = 7
COP_VISION = 7
MAX_JAIL_TERM = 1000

# constants
# max_iters in the example is specified as 1000
# so we run for 1000 steps
INDEPENDENT_VARIABLES = 3
DEPENDENT_VARIABLES = 3
RUNS = 10000
SAMPLES = 10
TIME_LAPSE = 1000

def main_civil_violence(citizenDensity, copDensity, legitimacy):
    # instantiate and run model
    model = CivilViolenceModel(
        height=HEIGHT,
        width=WIDTH,
        citizen_vision=CITIZEN_VISION,
        cop_vision=COP_VISION,
        max_jail_term=MAX_JAIL_TERM,
        citizen_density=citizenDensity,
        cop_density=copDensity,
        legitimacy=legitimacy)

    # run time lapse
    for i in range(TIME_LAPSE):
        model.step()

    # collect data for next steps
    dependentValues = []

    for i in range(SAMPLES):
        # step
        model.step()

    # read data
    data = model.dc.get_model_vars_dataframe()    
    dependentValues.append(list(data.Quiescent)[-SAMPLES-1:])
    dependentValues.append(list(data.Active)[-SAMPLES-1:])
    dependentValues.append(list(data.Jailed)[-SAMPLES-1:])

    # print line corresponding to this execution
    line = str(citizenDensity) + " " + str(copDensity) + " " + str(legitimacy)
    for dependentValue in dependentValues:
        line += " " + str(dependentValue).replace("[", "").replace("]", "").replace(" ", "")
    print(line)

    
def main():
    # header string
    print("i"*INDEPENDENT_VARIABLES + "d"*DEPENDENT_VARIABLES)

    for run in range(RUNS):
        # sample random ALPs
        # cop density + citizen density should be less than 1
        citizenDensity = random.random() 
        copDensity = (1-citizenDensity)*random.random() 
        legitimacy = random.random()
        
        # run model using those ALPs
        main_civil_violence(citizenDensity, copDensity, legitimacy)

main()
