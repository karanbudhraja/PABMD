import os
import neat
import visualize
import sys
import numpy as np

##############
# parameters #
##############

# abm parameters
abm = sys.argv[1]
numIndependent = eval(sys.argv[2])
numDependent = eval(sys.argv[3])

########
# data #
########

# get data
data = np.genfromtxt("../../../data/domaindata/cross_validation/" + abm + "_split_0_train.txt", skip_header=1, invalid_raise=False)
XTrain = data[:,:numIndependent]
YTrain = data[:,-numDependent:]
data = np.genfromtxt("../../../data/domaindata/cross_validation/" + abm + "_split_0_test.txt", skip_header=0, invalid_raise=False)
XTest = data[:,:numIndependent]
YTest = data[:,-numDependent:]

###########
# network #
###########

# 2-input XOR inputs and expected outputs.
#xor_inputs = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
#xor_outputs = [   (0.0,),     (1.0,),     (1.0,),     (0.0,)]

xor_inputs = XTrain
xor_outputs = YTrain

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)
            genome.fitness -= (output[0] - xo[0]) ** 2

def run(config_file):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    # Run for up to 300 generations.
    winner = p.run(eval_genomes, 300)

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))

    # karan: break here for now
    sys.exit(0)
        
    node_names = {-1:'A', -2: 'B', 0:'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
