import sys
import numpy as np

# constants
INDEPENDENT_VARIABLES = 1
DEPENDENT_VARIABLES = 1
RUNS = 10000
    
def main():
    # simulate using specific ALPs
    alpConfigurationList = eval(sys.argv[1])

    # store corresponding slps
    slpConfigurationList = []
    
    for alpConfiguration in alpConfigurationList:
        [x] = alpConfiguration
        
        # run model using those ALPs
        dependentValues = tuple([x + 0.5*np.sin(2*np.pi*x)])
        slpConfigurationList.append(dependentValues)

    print(slpConfigurationList)
        
main()
