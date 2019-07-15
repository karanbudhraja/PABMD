from model import SchellingModel
import random
import multiprocessing as mp

# only use three of the several ALPs
# density (0,1)
# minority_pc (0,1)

# use one model-level SLP
# happy

# parallel processing constants
BATCH_SIZE = 8

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

# process single run data
def process_run(q, run, density, minorityPc):
    try:
        line = main_eum(density, minorityPc)
    except:
        line = ""
        
    # add newline for clean output
    line = "\n" + line
    
    # write to queue
    q.put(line)
    
def main_schelling(density, minorityPc):
    # instantiate and run model
    model = SchellingModel(
        height=HEIGHT,
        width=WIDTH,
        density=density,
        minority_pc=minorityPc,
        homophily=HOMOPHILY)

    for i in range(SAMPLES):
        try:
            # step
            model.step()
        except:
            # saturated
            # no empty cells
            pass

    # collect data for next steps
    dependentValues = []

    for i in range(SAMPLES):
        # step
        model.step()

    # read data
    data = model.datacollector.get_model_vars_dataframe()    
    dependentValues.append(list(data.happy)[-SAMPLES-1:])

    # print line corresponding to this execution
    line = str(density) + " " + str(minorityPc)
    for dependentValue in dependentValues:
        line += " " + str(dependentValue).replace("[", "").replace("]", "").replace(" ", "")

    return line

    
def main():
    # header string
    print("i"*INDEPENDENT_VARIABLES + "d"*DEPENDENT_VARIABLES)

    densityValues = []
    minorityPcValues = []
    
    for run in range(RUNS):
        # sample random ALPs
        density = random.random() 
        minorityPc = random.random() 

        densityValues.append(density)
        minorityPcValues.append(minorityPc)

    # setup a list of processes that we want to run
    q = mp.Queue()
    processes = [mp.Process(target=process_run, args=(q, run, densityValues[run], minorityPcValues[run])) for run in range(RUNS)]
    batchSize = BATCH_SIZE
    batches = [processes[i:i+batchSize] for i in range(0, len(processes), batchSize)]

    outFile = open("output.txt", "a")
    
    for batch in batches:
        # run processes
        for p in batch:
            p.start()
        for p in batch:
            line = q.get()
            outFile.write(line)
        # exit the completed processes
        for p in batch:
            p.join()

    outFile.close()

main()
