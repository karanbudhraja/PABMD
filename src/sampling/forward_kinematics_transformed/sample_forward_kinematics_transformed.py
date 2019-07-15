import numpy as np

# no need for seeding randomization because this is a deterministic function
# problem set a has 2000 + 2000 data points
# sampled 1000 times for averaging error
INDEPENDENT_VARIABLES = 2
DEPENDENT_VARIABLES = 2
RUNS = 2000

def main():
    # get alps
    # in the range -2*pi to 2*pi
    a = -2*np.pi + np.random.rand(1,RUNS)*4*np.pi
    b = -2*np.pi + np.random.rand(1,RUNS)*4*np.pi

    # compute slps
    # assume r1 = r2 = 1
    p1 = np.cos(a+b) + np.sin(a)*np.sin(b)
    p2 = np.sin(a+b) + np.cos(a)*np.sin(b)
    z1 = (p1-p2)**2
    z2 = (1-p1)**2
    
    # write to output
    print("i"*INDEPENDENT_VARIABLES + "d"*DEPENDENT_VARIABLES)

    for run in range(RUNS):
        # alp data
        line = ""
        line += str(a.flatten()[run]) + " "
        line += str(b.flatten()[run]) + " "

        # slp data
        line += (str(z1.flatten()[run]) + ",")*11
        line = line[:-1] + " "
        line += (str(z2.flatten()[run]) + ",")*11
        line = line[:-1]

        # compiled output
        print line

main()
