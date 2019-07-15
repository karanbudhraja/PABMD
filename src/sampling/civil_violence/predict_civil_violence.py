from model import CivilViolenceModel
import random
import sys
import numpy as np
import os

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
    dependentValues.append(np.mean(list(data.Quiescent)[-SAMPLES-1:]))
    dependentValues.append(np.mean(list(data.Active)[-SAMPLES-1:]))
    dependentValues.append(np.mean(list(data.Jailed)[-SAMPLES-1:]))

    return tuple(dependentValues)
    
def main():
    # simulate using specific ALPs
    alpConfigurationList = eval(sys.argv[1])

    # store corresponding slps
    slpConfigurationList = []
    
    for alpConfiguration in alpConfigurationList:
        [citizenDensity, copDensity, legitimacy] = alpConfiguration

        # cop density + citizen density should be less than 1
        if(citizenDensity + copDensity > 1):
            # invalid parameters
            # set to infinity so distance is infinite
            # this value will be considered as a bad suggestion when filtering
            dependentValues = [float("Inf")]*eval(os.environ["NUM_DEPENDENT"])
        else:
            # run model using those ALPs
            dependentValues = main_civil_violence(citizenDensity, copDensity, legitimacy)
        slpConfigurationList.append(dependentValues)

    print(slpConfigurationList)
        
main()
