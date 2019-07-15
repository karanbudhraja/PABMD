import numpy as np

# no need for seeding randomization because this is a deterministic function
INDEPENDENT_VARIABLES = 1
DEPENDENT_VARIABLES = 1
#RUNS = 10000
#RUNS = 5000
#RUNS = 2500
#RUNS = 1000
#RUNS = 500
#RUNS = 250
RUNS = 100

def main():
    # get alps
    x = np.random.rand(1,RUNS)

    # compute slps
    y = x + 0.5*np.sin(2*np.pi*x)
    
    # write to output
    print("i"*INDEPENDENT_VARIABLES + "d"*DEPENDENT_VARIABLES)

    for run in range(RUNS):
        # alp data
        line = ""
        line += str(x.flatten()[run]) + " "

        # slp data
        line += (str(y.flatten()[run]) + ",")*11
        line = line[:-1]

        # compiled output
        print line

main()
