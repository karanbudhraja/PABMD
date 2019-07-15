#imports
import MultiNEAT as NEAT
import random
import qlearning

# constants
RANDOM_SEED = 100

AGENT_POPULATION_SIZE = 2

MINIMUM_FITNESS = 0
NEAT_GENERATIONS = 25
OUTPUT_COUNT = 1
POPULATION_SIZE = 3
BIAS_DEFAULT_VALUE = 1.0

GRID_WORLD_SIZE = 3
INITIAL_AGENT_STATE = [0,0]
GOAL_AGENT_STATE = [GRID_WORLD_SIZE-1, GRID_WORLD_SIZE-1]
GOAL_REWARD = GRID_WORLD_SIZE ** 3
Q_LEARNING_ITERATIONS = 100

# generate population of genomes
def generate_population(populationSize, inputCount, outputCount):

    # set parameters
    params = NEAT.Parameters()
    params.PopulationSize = populationSize

    # create genome to specify constraints
    #genome = NEAT.Genome(0, inputCount, 0, outputCount, False, NEAT.ActivationFunction.UNSIGNED_SIGMOID, NEAT.ActivationFunction.UNSIGNED_SIGMOID, 0, params)
    genome = NEAT.Genome(0, inputCount, 0, outputCount, False, NEAT.ActivationFunction.UNSIGNED_STEP, NEAT.ActivationFunction.UNSIGNED_STEP, 0, params)

    # create a population of genomes
    pop = NEAT.Population(genome, params, True, 1.0)
    
    return pop

# evaluate a genome in grid world
def evaluate_grid_world(genome, grid, agentActions, sampleActions):

    gridWorldSize = len(grid)

    # generate reward matrix based on this genome
    values = []

    for row in range(gridWorldSize):
        # iterate over rows and columns
        rowValues = []

        for column in range(gridWorldSize):
            # evaluate the genome for this particular input
            # add bias at end
            inputState = grid[row][column].features
            inputState.append(BIAS_DEFAULT_VALUE)

            outputState = getOutput(genome, inputState)
            rowValues.append(outputState[0])

        values.append(rowValues)

    # now we have the rewards
    # generate the set of actions as per these rewards
    agentBestActions = qlearning.values_to_actions(values, agentActions, gridWorldSize)

    # now evaluate test actions with best actions
    fitness = get_fitness_grid_world(agentBestActions, sampleActions)

    return fitness

# compare two reward functions
# maybe we could just see how many are same in direction (hamming distance) or take cos of directions (1, 0, 0, -1) and take sum
def get_fitness_grid_world(agentBestActions, sampleActions):

    # evaluate based on similarity or distance
    similarity = 0

    for key in sampleActions:
        # for each grid coordinate
        bestAction = sampleActions[key]
        testAction = agentBestActions[key]

        if((key[0] == GOAL_AGENT_STATE[0]) and (key[1] == GOAL_AGENT_STATE[1])):
            # similarity at the goal state is irrelevant
            similarity += 1
        else:
            similarity += qlearning.get_action_similarity(bestAction, testAction)

    return similarity

# getoutput of a network corresponding to a genome for given input
def getOutput(genome, inputState):

    # this creates a neural network (phenotype) from the genome
    net = NEAT.NeuralNetwork()
    genome.BuildPhenotype(net)

    # let's input just one pattern to the net, activate it once and get the output
    net.Input(inputState)
    net.Activate()
    output = net.Output()

    return output

# run iterations of neat
def runNeat(pop, neatGenerations, inputCount, grid, agentActions, sampleActions):
    
    # the best genome ever found based on fitness value
    bestGenome = "null"

    # run the algorithm for a number of generations
    for generation in range(neatGenerations):
        # retrieve a list of all genomes in the population
        genomeList = NEAT.GetGenomeList(pop)

        # calculate maximum fitness for this generation
        maxFitness = MINIMUM_FITNESS

        # apply the evaluation function to all genomes
        for genome in genomeList:
            # testing grid world
            fitness = evaluate_grid_world(genome, grid, agentActions, sampleActions)

            genome.SetFitness(fitness)
        
            if(fitness > maxFitness):
                maxFitness = fitness
                bestGenome = genome

        # print statistics
        print("best fitness for generation " + str(generation) + ": " + str(maxFitness))

        # advance to the next generation
        pop.Epoch()

    return bestGenome

def main():

    # seed randomization
    #random.seed(RANDOM_SEED)

    # variables corresponding to all sample policies generated
    sampleActionsCollection = []
    populationCollection = []

    for agent in range(AGENT_POPULATION_SIZE):
        # generate grid world sample
        [grid, agentActions, sampleActions] = qlearning.get_sample_actions(GRID_WORLD_SIZE, INITIAL_AGENT_STATE, GOAL_AGENT_STATE, GOAL_REWARD, Q_LEARNING_ITERATIONS)
        sampleActionsCollection.append(sampleActions)

    #print(sampleActionsCollection)
    #return
    
    # network and neat characteristics
    # input count includes 1 bias node
    initialState = grid[INITIAL_AGENT_STATE[0]][INITIAL_AGENT_STATE[1]]
    numberOfFeatures = len(initialState.features)
    inputCount = numberOfFeatures + 1
    outputCount = OUTPUT_COUNT
    populationSize = POPULATION_SIZE

    for agent in range(AGENT_POPULATION_SIZE):
        # generate population of genomes
        pop = generate_population(populationSize, inputCount, outputCount)
        populationCollection.append(pop)

    # execute neat
    neatGenerations = NEAT_GENERATIONS
    bestGenome = runNeat(pop, neatGenerations, inputCount, grid, agentActions, sampleActions)

    # test the best genome
    #inputState = [0.0, 0.4, 0.5, BIAS_DEFAULT_VALUE]
    #output = getOutput(bestGenome, inputState)
    #print("test output: " + str(output[0]))

main()
