import sys
import featurize_image as fi
import glob
import os
import numpy as np

# process a single folder
def process_folder(folderName):
    # get environment variables
    descriptorSetting = os.environ["DESCRIPTOR_SETTING"]
    numDependent = eval(os.environ["NUM_DEPENDENT"])

    # load feature extractor if needed
    net = fi.get_network(descriptorSetting)
    
    # initialize collection of dependent values
    dependentValuesFolder = []

    # expect everything in this folder to be an image
    for imageName in glob.glob(folderName + "/*.*"):
        # read image
        print >> sys.stderr, ("now processing " + imageName)
        
        # define dependent values
        dependentValuesImage = fi.get_features(net, imageName, descriptorSetting)

        # append to list
        dependentValuesFolder.append(dependentValuesImage)

    dependentValuesFolder = np.mean(dependentValuesFolder, axis=0)

    return tuple(dependentValuesFolder)

def main():
    folderName = sys.argv[1]
    dependentValuesFolder = process_folder(folderName)

    print dependentValuesFolder
    
if __name__ == "__main__":
   # stuff only to run when not called via 'import' here
   main()
