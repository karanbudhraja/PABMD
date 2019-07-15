import sys
import numpy as np

# constants
INDEPENDENT_VARIABLES = 2
DEPENDENT_VARIABLES = 2
    
def main():
    # simulate using specific ALPs
    alpConfigurationList = eval(sys.argv[1])

    # store corresponding slps
    slpConfigurationList = []
    
    for alpConfiguration in alpConfigurationList:
        [a, b] = alpConfiguration
        
        # run model using those ALPs
        p1 = np.cos(a+b) + np.sin(a)*np.sin(b)
        p2 = np.sin(a+b) + np.cos(a)*np.sin(b)
        dependentValues = tuple([p1, p2])
        slpConfigurationList.append(dependentValues)

    print(slpConfigurationList)
        
main()
