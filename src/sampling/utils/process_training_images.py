# libraries used
import sys
import os
import featurize_image as fi
import multiprocessing as mp
import glob

# parallel processing constants
BATCH_SIZE = 8

# process single run data
def process_run(q, run, runConfiguration, line):

    # unpack run configuration
    [inFolder, dataFile, numDependent, net, descriptorSetting] = runConfiguration

    # initialize collection of dependent values
    depValuesStrings = [""]*numDependent

    for imageName in glob.glob(inFolder + "/" + str(run) + "_*.*"):
        # read image
        print >> sys.stderr, ("now processing " + imageName)
        
        # define dependent values
        depValues = fi.get_features(net, imageName, descriptorSetting)

        for index in range(len(depValues)):
            depValuesStrings[index] += str(depValues[index]) + ","

    # add value data to line
    for depValuesString in depValuesStrings:
        line += " " + depValuesString[:-1]

    # remove extra "," at end
    line = line + "\n"

    # write to queue
    q.put(line)

# read images and extract features as dependent variables
def main():

    # get environment variables
    abm = os.environ["ABM"]
    descriptorSetting = os.environ["DESCRIPTOR_SETTING"]
    numDependent = eval(os.environ["NUM_DEPENDENT"])
    dataFileName = os.environ["DATA_FILE"]

    dataFile = open(abm + ".txt", "r")
    tempFile = open("temp.txt", "w")

    # set up image reading
    inFolder = abm + "/images"

    # load feature extractor if needed
    net = fi.get_network(descriptorSetting)

    # read data
    lines = dataFile.readlines()

    # get number of runs
    runs = len(lines)
    
    # process runs
    runConfiguration = [inFolder, dataFile, numDependent, net, descriptorSetting]

    # setup a list of processes that we want to run
    q = mp.Queue()
    processes = [mp.Process(target=process_run, args=(q, run, runConfiguration, lines[run].strip(" \n"))) for run in range(runs)] 
    batchSize = BATCH_SIZE
    batches = [processes[i:i+batchSize] for i in range(0, len(processes), batchSize)]

    for batch in batches:
        # run processes
        for p in batch:
            p.start()
        for p in batch:
            line = q.get()
            tempFile.write(line)
        # exit the completed processes
        for p in batch:
            p.join()

    # close files and rename
    dataFile.close()
    tempFile.close()

    os.remove(abm + ".txt")
    os.rename("temp.txt", abm + ".txt")

if __name__ == "__main__":
    main()
