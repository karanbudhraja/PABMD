import pandas as pd
import numpy as np
import sys
import multiprocessing as mp

from negotiation_model import *
from bdm_agent import *

# parallel processing constants
BATCH_SIZE = 8

# seed this for fixed environment
# for better replication of results
random.seed(0)

# constants
INDEPENDENT_VARIABLES = 2
DEPENDENT_VARIABLES = 2
RUNS = 10000
SAMPLES = 10
TIME_LAPSE = 200

# process single run data
def process_run(q, run, qValue, tValue, bookData, salienceValues):
    try:
        line = main_eum(qValue, tValue, bookData, salienceValues)
    except:
        line = ""
        
    # add newline for clean output
    line = "\n" + line
    
    # write to queue
    q.put(line)

def get_salience_values(timeLapse, numberOfAgents):
    salienceValues = np.zeros((timeLapse, numberOfAgents))

    for stepNumber in range(timeLapse):
        for agentNumber in range(numberOfAgents):
            salienceValues[stepNumber][agentNumber] = random.random()

    return salienceValues

# Defining the model objects
class BDMActor(NegotiationActor):
    DecisionClass = BDM_Agent

class NegotiationModel_(NegotiationModel):
    # Variables for median caching
    median_computed_last = -1
    median = -1
    
    def find_median(self):
        if self.median_computed_last != self.schedule.steps:
            self.median = super().find_median()
            self.median_computed_last = self.schedule.steps
        return self.median

class ModelOutput:
    def __init__(self, model):
        '''
        Store data from model run.
        '''
        self.agent_vars = model.datacollector.get_agent_vars_dataframe()
        self.model_vars = model.datacollector.get_model_vars_dataframe()
        self.log = model.log

def load_data():
    # Load data 
    bookData = pd.read_csv("BDM_ColdWar.csv")
    bookData.Position = (bookData.Position + 100)/200

    return bookData

def main_eum(qValue, tValue, bookData, salienceValues):
    # define agents
    agents = []

    for i, row in bookData.iterrows():
        newAgent = BDMActor(row.Country, row.Capability, row.Position, 1)
        newAgent.decision_model.Q = qValue
        newAgent.decision_model.T = tValue
        newAgent.salience = salienceValues[i]
        agents.append(newAgent)

    # instantiate model
    model = NegotiationModel_(agents)

    # run model
    for stepNumber in range(TIME_LAPSE):
        agentNumber = 0
        for agent in model.agents:
            #agent.salience = random.random()
            agent.salience = salienceValues[stepNumber][agentNumber]
            agentNumber += 1            
        model.step()

    # collect data for next steps
    dependentValues = []
    modelOutput = ModelOutput(model)
    dependentValues.append(list(modelOutput.model_vars["Median"][-SAMPLES-1:]))
    dependentValues.append(list(modelOutput.model_vars["Mean"][-SAMPLES-1:]))
    
    # print line corresponding to this execution
    line = str(qValue) + " " + str(tValue)
    for dependentValue in dependentValues:
        line += " " + str(dependentValue).replace("[", "").replace("]", "").replace(" ", "")

    return line

def main():
    # header string
    print("i"*INDEPENDENT_VARIABLES + "d"*DEPENDENT_VARIABLES)

    # read data
    bookData = load_data()

    # get salience values
    [rows, columns] = bookData.shape
    salienceValues = get_salience_values(TIME_LAPSE, rows)

    qValues = []
    tValues = []
    
    for run in range(RUNS):
        # sample random ALPs
        qValue = random.random()
        tValue = random.random()

        qValues.append(qValue)
        tValues.append(tValue)
        
    # setup a list of processes that we want to run
    q = mp.Queue()
    processes = [mp.Process(target=process_run, args=(q, run, qValues[run], tValues[run], bookData, salienceValues)) for run in range(RUNS)]
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
