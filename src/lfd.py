# import libraries
import matplotlib
matplotlib.use('Agg')

import sys
import amf
import amf.misc as misc

from sklearn import svm
from sklearn.cluster import KMeans
import scipy.io as sio

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import os
import time
import random

import cPickle
import gzip
import numpy as np
from sets import Set

import sampling.utils.featurize_image as fi
import sampling.utils.outlier_detection as od

# constants
PLOT_COLOR = {0: "r", 1: "b", 2: "g", 3: "c", 4: "m", 5: "y", 6: "k"}
PLOT_SHAPE = {0: "o", 1: "^", 2: "s", 3: "d", 4: "+", 5: "v", 6: "*"}
PLOT_DEMONSTRATIONS = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g"}

RUNS_PER_SLPS = 1

IMAGE_FEATURE_PATH_PREFIX = "sampling/"

# experiment class
class Experiment(object):

   # initialize experiment parameters based on file
   def __init__(self):
      # read configuration
      configurationFileName = os.environ["CONFIGURATION_FILE"]
      configuration = open(configurationFileName).read()
      execd = {}
      exec(configuration, execd)

      self.trainingSize = execd['TRAINING_SIZE']
      self.validationSize = execd['VALIDATION_SIZE']
      self.rmGranularity = execd['RM_GRANULARITY']
      self.numIndependent = execd['NUM_INDEPENDENT']
      self.regression = execd['REGRESSION']

      # changes for feedback
      if(eval(os.environ["FEEDBACK"]) == 1):
         # randomness causes too much noise for feedback
         self.validationSize = 0

         # modify regression parameter
         if(os.path.isfile("regression_parameter.txt") == True):
            with open("regression_parameter.txt", "r") as inFile:
               regressionParameter = inFile.readlines()[0]
            #self.regression = self.regression.replace("()", "(alpha=np.exp((-1*" + regressionParameter + ")))")
            self.regression = self.regression.replace("()", "(n_neighbors=int(np.floor(" + regressionParameter + ")))")
      
      # get environment variables
      self.abm = os.environ["ABM"]
      self.demonstrationFolder = os.environ["DEMONSTRATION_FOLDER"]
      self.descriptorSetting = os.environ["DESCRIPTOR_SETTING"]
      self.numDependent = eval(os.environ["NUM_DEPENDENT"])
      self.dataFileName = os.environ["DATA_FILE"]
      self.scaleDataFileName = os.environ["SCALE_DATA_FILE"]
      self.images = eval(os.environ["IMAGES"])
      self.writeToDisk = eval(os.environ["WRITE_TO_DISK"])

      # hack for crossvalidation
      if(eval(os.environ["EVALUATION_CROSS_VALIDATION"]) == 1):
         self.dataFileName = "../data/domaindata/cross_validation/" + os.environ["SPLIT_TRAIN_FILE_NAME"]
         
      # reverse mapping configuration: ranges of agent level parameters
      self.rmConfiguration = tuple()
      for i in range(self.numIndependent):
         self.rmConfiguration += ((0.0, 1.0), )
      self.rmConfiguration = list(self.rmConfiguration)

      # for plotting
      self.rmQueryCount = 0
      
      # mapping models
      self.fm = None
      self.rm = None
      
   # retrieve reverse mapping
   def get_mappings(self, slpsList, saveData=False):
      # parse slps list
      demonstrationDescriptors = [depValues for depValues, plotColor, plotShape in slpsList]
      
      # load data
      print >> sys.stderr, "->Loading data set " + self.dataFileName
      fullData = amf.data.load(self.dataFileName)
      
      # forward mapping using largest training size
      print >> sys.stderr, "->Trainining FM"
      trainingSize = self.trainingSize
      dataset, validation = amf.data.random_subsets(fullData, [trainingSize, self.validationSize], demonstrationDescriptors)
      fm = amf.ForwardMapping(self.regression, self.numDependent)
      fm.train(dataset)
      
      if(saveData == True):
         # gather ALPs and SLPs
         # plot them in space to observe distribution
         alpPoints = [item[0] for item in dataset]
         slpPoints = [item[1] for item in dataset]
         c = "r"
         m = "o"
         
         # save ALP dataset
         with open("alp_dataset_" + str(int(time.time())) + ".txt", "w") as alpDatasetFile:
            for alpPoint in alpPoints:
               alpDatasetFile.write(str(alpPoint) + "\n")
               
         # save SLP dataset
         with open("slp_dataset_" + str(int(time.time())) + ".txt", "w") as slpDatasetFile:
            for slpPoint in slpPoints:
               slpDatasetFile.write(str(slpPoint) + "\n")

         # plot ALPs
         fig = plt.figure()
         ax = fig.add_subplot(111, projection='3d')
         plotName = "alp_points_" + str(int(time.time())) + ".png"
         alpPoints = zip(*alpPoints)
         [xs, ys, zs] = alpPoints
         ax.scatter(xs, ys, zs, c=c, marker=m)
         ax.set_xlabel("Max-align-turn")
         ax.set_ylabel("Max-cohere-turn")
         ax.set_zlabel("Max-separate-turn")
         plt.savefig(plotName)
         plt.close(fig)
         
         fig = plt.figure()
         plotName = "alp_points_dim1_" + str(int(time.time())) + ".png"
         plt.plot(xs, xs, c+m)
         plt.xlabel("ALP Dimension 1")
         plt.savefig(plotName)
         plt.close(fig)
         
         fig = plt.figure()
         plotName = "alp_points_dim2_" + str(int(time.time())) + ".png"
         plt.plot(ys, ys, c+m)
         plt.xlabel("ALP Dimension 2")
         plt.savefig(plotName)
         plt.close(fig)
         
         fig = plt.figure()
         plotName = "alp_points_dim3_" + str(int(time.time())) + ".png"
         plt.plot(zs, zs, c+m)
         plt.xlabel("ALP Dimension 3")
         plt.savefig(plotName)
         plt.close(fig)
      
         # plot SLPs
         fig = plt.figure()
         plotName = "slp_points_" + str(int(time.time())) + ".png"
         slpPoints = zip(*slpPoints)
         [xs, ys] = slpPoints
         plt.plot(xs, ys, c+m)
         plt.xlabel("SLP Dimension 1")
         plt.ylabel("SLP Dimension 2")
         plt.savefig(plotName)
         plt.close(fig)
         
         fig = plt.figure()
         plotName = "slp_points_dim1_" + str(int(time.time())) + ".png"
         plt.plot(xs, xs, c+m)
         plt.xlabel("SLP Dimension 1")
         plt.savefig(plotName)
         plt.close(fig)
         
         fig = plt.figure()
         plotName = "slp_points_dim2_" + str(int(time.time())) + ".png"
         plt.plot(ys, ys, c+m)
         plt.xlabel("SLP Dimension 2")
         plt.savefig(plotName)
         plt.close(fig)

      # reverse mapping using largest granularity
      print >> sys.stderr, "->Trainining RM"
      granularity = self.rmGranularity
      rm = amf.ReverseMapping(fm, self.rmConfiguration, granularity, slpsList)

      return [fm, rm]

   # query reverse mapping for configurations
   def query_rm(self, fm, rm, slps, ax, c, m, plotConfigurations=False):
      intersections = rm.all_intersections(slps)
      allConfigurations = []

      #
      # filter from all query results
      # 
      
      simplexNumbers = []
      edgeNumbers = []
      _allConfigurations = []

      # check for special case
      # if wiggle happened, then only one configuration is returned
      if(len(intersections[0]) == 1):
         # configuration corresponds to simplex number 0 and edge number 0
         # an intersection is stored as [simplexNumber][edgeNumber][configuration]
         if(type(intersections[0][0][0]) is tuple):
            # no simplex number is returned in this case
            wiggledConfiguration = intersections[0][0][0]
         else:
            # different formats of return
            # in this case, a default simplex and edge number is attached
            # not sure why
            wiggledConfiguration = intersections[0][0][1][0][1]
            
         allConfigurations = [list(wiggledConfiguration)]            
         return allConfigurations
      
      for intersection in intersections:
         currentEdgeNumbers = {}
         _configurations = {}
         simplexNumbers.append([simplexNumber for (simplexNumber, configurations) in intersection])
         
         for simplexNumber, configurations in intersection:
            currentEdgeNumbers[simplexNumber] = [edgeNumber for (edgeNumber, configuration) in configurations]
            _configurations[simplexNumber] = {}
            for edgeNumber, configuration in configurations:
               allConfigurations.append(configuration)
               _configurations[simplexNumber][edgeNumber] = configuration
               
         edgeNumbers.append(currentEdgeNumbers)
         _allConfigurations.append(_configurations)
         
      # get common simplexes
      simplexNumbers = [Set(numbers) for numbers in simplexNumbers]
      commonSimplexes = simplexNumbers[0]
      for index in range(1, len(simplexNumbers)):
         currentSimplexes = simplexNumbers[index]
         commonSimplexes = commonSimplexes.intersection(currentSimplexes)

      # get common edges
      commonEdges = {}
      for key in commonSimplexes:
         currentCommonEdges = Set(edgeNumbers[0][key])
         for index in range(1, len(edgeNumbers)):
            currentEdges = Set(edgeNumbers[index][key])
            currentCommonEdges = currentCommonEdges.intersection(currentEdges)
         commonEdges[key] = currentCommonEdges
      overlapCount = [len(values) for values in list(commonEdges.values())]
      
      # filter common edges
      #overlapAmount = [len(commonEdges[key]) for key in commonEdges]
      #print >> sys.stderr, overlapAmount
      #plt.figure()
      #plt.hist(overlapAmount)
      #plt.savefig("overlapAmount_before_filtering.png")

      if(len(overlapCount) > 0):
         #plt.figure()
         #plt.hist(overlapCount)
         #plt.show()
         #sys.exit()

         overlapMean = np.mean(overlapCount)
         commonEdges = {key : commonEdges[key] for key in commonEdges if len(commonEdges[key]) >= overlapMean}
         commonSimplexes = list(commonEdges.keys())

         #overlapAmount = [len(commonEdges[key]) for key in commonEdges]
         #print >> sys.stderr, overlapAmount
         #plt.figure()
         #plt.hist(overlapAmount)
         #plt.savefig("overlapAmount_after_filtering.png")
         #sys.exit()
         
         # now we need to cycle through the configurations again
         # only pick ones corresponding to filtered edges
         # TODO: we may want to combine configurations across SLP intersections for optimality. can we?
         allConfigurationsCommon = []
         for commonSimplex in commonSimplexes:
            currentCommonEdges = commonEdges[commonSimplex]
            for commonEdge in currentCommonEdges:
               for alpIndex in range(len(simplexNumbers)):
                  currentConfiguration = _allConfigurations[alpIndex][commonSimplex][commonEdge]
                  #print >> sys.stderr, currentConfiguration
                  allConfigurationsCommon.append(currentConfiguration)

         # filter configurations based on common simplexes
         if(eval(os.environ["CONFIGURATIONS_PRUNING"]) == 1):
            allConfigurations = allConfigurationsCommon
      else:
         # no overlapping edges found
         # pruning cannot be used in this case
         pass

      # outlier detection code
      # outlier detection based on hubness
      # approximate labeling based on minority clustering

      if(eval(os.environ["CONFIGURATIONS_OUTLIER_DETECTION"]) == 1):
         # outlier detection only valid for a dataset of minimum size
         # meanshift clustering sometimes fails if it somehow finds nans
         try:
            # do outlier detection
            [odPoints, labels] = od.get_approximate_labels(allConfigurations)
            [odPoints, labels] = od.do_lpod(odPoints, labels)
         
            allConfigurationsOd = []
            for configurationIndex in range(len(allConfigurations)):
               currentConfiguration = odPoints[configurationIndex]         
               if(labels[configurationIndex] == 0):
                  # not an outlier
                  allConfigurationsOd.append(list(currentConfiguration))
         
            allConfigurations = allConfigurationsOd
         except:
            # outlier detection had an error
            # it is not used in this case
            pass
            
      # add solution to scatter plot
      if(plotConfigurations == True):
         print >> sys.stderr, "->Plotting configurations"         
         allXs = []
         allYs = []
         allZs = []
         allErrors = []
         configurationsToPlot = []
         
         for configuration in allConfigurations:
            error = misc.list_sub(slps, fm.predict(configuration))
            error = sum([x**2 for x in error])
            configurationsToPlot.append((error, configuration))
         configurationsToPlot.sort()   

         # filter configurations here
         # checking threshold effect
         allConfigurationsToPlot = [x for x in configurationsToPlot]
         histogramData = [error for error, configuration in allConfigurationsToPlot]
         mu = np.mean(histogramData)
         sigma = np.std(histogramData)
         #configurationsToPlot = [x for x in configurationsToPlot if (x[0] < (mu + 0.1*sigma) and x[0] > (mu - 0.1*sigma))]
            
         for error, configuration in configurationsToPlot:
            xs, ys, zs = configuration
            allXs.append(xs)
            allYs.append(ys)
            allZs.append(zs)
            allErrors.append(error)

         odPointsAllXs = [x[0] for x in odPoints]
         odPointsAllYs = [x[1] for x in odPoints]
         odPointsAllZs = [x[2] for x in odPoints]
         
         # plot point errors in 3d space 
         plot = ax.scatter(allXs, allYs, allZs, c=allErrors, marker=m, cmap="cool")
         plt.colorbar(plot, label="FM Prediction Error")
         ax.set_xlabel("Max-align-turn")
         ax.set_ylabel("Max-cohere-turn")
         ax.set_zlabel("Max-separate-turn")
         plt.savefig("solution_space.png")

         # plot point labels in 3d space
         fig = plt.figure()
         ax = fig.add_subplot(111, projection='3d')
         cmapClusters = matplotlib.cm.get_cmap("cool", 2)
         plot = ax.scatter(odPointsAllXs, odPointsAllYs, odPointsAllZs, c=labels, marker=m, cmap=cmapClusters)
         plt.colorbar(plot, label="Cluster").set_ticks([0, 1])
         ax.set_xlabel("Max-align-turn")
         ax.set_ylabel("Max-cohere-turn")
         ax.set_zlabel("Max-separate-turn")
         plt.savefig("solution_space_clusters.png")
         
         # error histogram
         plt.figure()
         plt.hist(histogramData)
         plt.xlabel("FM Prediction Error")
         plt.title("mu = " + str(mu)  + ", sigma = " + str(sigma))
         plt.savefig("error_histogram")
            
         projectDimensions = True
         if(projectDimensions == True):
            # plotting for single demonstration SLP configuration

            # plot projections with error
            plt.figure()
            plt.scatter(allXs, allYs, c=allErrors, marker=m, cmap="cool")
            plt.colorbar(label="FM Prediction Error")
            plt.gca().set_xlabel("Max-align-turn")
            plt.gca().set_ylabel("Max-cohere-turn")
            plt.savefig("solution_space_xy.png")
            plt.figure()
            plt.scatter(allXs, allZs, c=allErrors, marker=m, cmap="cool")
            plt.colorbar(label="FM Prediction Error")
            plt.gca().set_xlabel("Max-align-turn")
            plt.gca().set_ylabel("Max-separate-turn")
            plt.savefig("solution_space_xz.png")
            plt.figure()
            plt.scatter(allYs, allZs, c=allErrors, marker=m, cmap="cool")
            plt.colorbar(label="FM Prediction Error")
            plt.gca().set_xlabel("Max-cohere-turn")
            plt.gca().set_ylabel("Max-separate-turn")
            plt.savefig("solution_space_yz.png")

            # plot projections with labels
            plt.figure()
            plt.scatter(odPointsAllXs, odPointsAllYs, c=labels, marker=m, cmap=cmapClusters)
            plt.colorbar(label="Cluster").set_ticks([0, 1])
            plt.gca().set_xlabel("Max-align-turn")
            plt.gca().set_ylabel("Max-cohere-turn")
            plt.savefig("solution_space_xy_clusters.png")
            plt.figure()
            plt.scatter(odPointsAllXs, odPointsAllZs, c=labels, marker=m, cmap=cmapClusters)
            plt.colorbar(label="Cluster").set_ticks([0, 1])
            plt.gca().set_xlabel("Max-align-turn")
            plt.gca().set_ylabel("Max-separate-turn")
            plt.savefig("solution_space_xz_clusters.png")
            plt.figure()
            plt.scatter(odPointsAllYs, odPointsAllZs, c=labels, marker=m, cmap=cmapClusters)
            plt.colorbar(label="Cluster").set_ticks([0, 1])
            plt.gca().set_xlabel("Max-cohere-turn")
            plt.gca().set_ylabel("Max-separate-turn")
            plt.savefig("solution_space_yz_clusters.png")

         # clear memory
         del allXs, allYs, allZs, allErrors, configurationsToPlot, allConfigurationsToPlot

      return allConfigurations

   # format configurations for classifiaction
   def format_data_xy(self, allConfigurationsList):
      X = []
      Y = []
      for index in range(len(allConfigurationsList)):
         X += allConfigurationsList[index]
         Y += [index]*len(allConfigurationsList[index])

      return [X, Y]

   # compute fm errors and select configuration for minimum
   def get_min_error_configurations(self, fm, allConfigurationsList, slps):
      minErrorConfigurations = []
      for allConfigurations in allConfigurationsList:
         # error[0] contains error value
         # error[1] contains configuration
         errors = [(misc.list_sub(slps, fm.predict(configuration)), configuration) for configuration in allConfigurations]
         errors = [(sum([x**2 for x in error[0]]), error[1]) for error in errors]
         errors.sort()
         minErrorConfigurations.append(errors[0][1])

      return minErrorConfigurations

   # compute average configuration
   def get_mean_configurations(self, allConfigurationsList):
      meanConfigurations = []
      for allConfigurations in allConfigurationsList:
         meanConfiguration = tuple(misc.col_average(allConfigurations))
         meanConfigurations.append(meanConfiguration)

      return meanConfigurations

   def process_error_weighted_configuration(self, configuration, fm, slps, errorWeightedConfiguration, normalizationFactor, zeroErrorConfigurations):
      prediction = fm.predict(configuration)
      error = 0

      for index in range(len(slps)):
         error += (slps[index] - prediction[index])**2
         
      if(error > 0):
         if(zeroErrorConfigurations == 0):
            # no zero error configuration found yet
            normalizationFactor += 1/error
            for index in range(len(errorWeightedConfiguration)):
               errorWeightedConfiguration[index] += configuration[index]/error
      else:
         # error is zero
         if(zeroErrorConfigurations == 0):
            # first time we found a zero error configuration
            for index in range(len(errorWeightedConfiguration)):
               # wipe old data
               errorWeightedConfiguration[index] = 0
               zeroErrorConfigurations += 1
               
               # add new data
               for index in range(len(errorWeightedConfiguration)):
                  errorWeightedConfiguration[index] += configuration[index]

      return [errorWeightedConfiguration, normalizationFactor, zeroErrorConfigurations]
                     
   # compute error weighted average configuration
   def get_error_weighted_configurations(self, fm, allConfigurationsList, slps):
      errorWeightedConfigurations = []
      listIndex = -1
            
      for allConfigurations in allConfigurationsList:
         errorWeightedConfiguration = [0]*self.numIndependent
         normalizationFactor = 0
         zeroErrorConfigurations = 0

         if(self.writeToDisk == 1):
            # read from disk
            listIndex += 1
            with gzip.open("configuration_" + str(listIndex+1) + ".pgz", 'rb') as configurationFile:
               while 1:
                  try:
                     # read configuration
                     configuration = cPickle.load(configurationFile)[listIndex]
                     [errorWeightedConfiguration, normalizationFactor, zeroErrorConfigurations] = self.process_error_weighted_configuration(configuration, fm, slps, errorWeightedConfiguration, normalizationFactor, zeroErrorConfigurations)
                  except EOFError:
                     break
         else:
            # read from ram
            for configuration in allConfigurations:
               [errorWeightedConfiguration, normalizationFactor, zeroErrorConfigurations] = self.process_error_weighted_configuration(configuration, fm, slps, errorWeightedConfiguration, normalizationFactor, zeroErrorConfigurations)
 
         # nomalization

         ################################
         ### DEBUG PRINTS
         ################################
         #print >> sys.stderr, normalizationFactor
         #print >> sys.stderr, zeroErrorConfigurations
         #print >> sys.stderr, errorWeightedConfiguration
         
         for index in range(len(errorWeightedConfiguration)):
            if(zeroErrorConfigurations == 0):
               errorWeightedConfiguration[index] /= float(normalizationFactor)
            else:
               # we found some zero error configurations
               # we will only consider those configurations when computing a representative point
               errorWeightedConfiguration[index] /= float(zeroErrorConfigurations)
               
         errorWeightedConfigurations.append(tuple(errorWeightedConfiguration))

      return errorWeightedConfigurations

   # compute classifier based optimal configuration
   def get_svm_configurations(self, allConfigurationsList):
      # train svm classifier on configurations
      classifier = svm.SVC(probability=True)
      [X, Y] = self.format_data_xy(allConfigurationsList)
      classifier.fit(X, Y)

      # select configurations which maximize probability
      svmConfigurations = []
      for index in range(len(allConfigurationsList)):
         # compute score for each probability and select maximum
         configurationProbabilities = classifier.predict_proba(allConfigurationsList[index])
         configurationProbabilities = [(configurationProbabilities[i][index], allConfigurationsList[index][i]) for i in range(len(configurationProbabilities))]
         configurationProbabilities.sort(reverse=True)
         svmConfiguration = tuple(configurationProbabilities[0][1])
         svmConfigurations.append(svmConfiguration)

      return svmConfigurations

   # compute based on k means clustering
   def get_kmeans_configurations(self, allConfigurationsList):
      # train svm classifier on configurations
      nClusters = len(allConfigurationsList)
      kmeans = KMeans(init='k-means++', n_clusters=nClusters)
      [X, Y] = self.format_data_xy(allConfigurationsList)
      kmeans.fit(X, Y)

      # select centroids
      centroids = kmeans.cluster_centers_
      
      return centroids

   # run experiment
   def run(self, slpsList, plotConfigurations=False, saveError=False):
      [fm, rm] = self.get_mappings(slpsList)

      # append to model list
      self.fm = fm
      self.rm = rm

      # plot solution space
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')

      print >> sys.stderr, "->Querying RM for configurations" 

      startTime = time.time()
      allConfigurationsList = []
      proxyPlots = []
      labels = []
      for slps,c,m in slpsList:
         allConfigurations = self.query_rm(fm, rm, slps, ax, c, m, plotConfigurations)
         allConfigurationsList.append(allConfigurations)

         proxyPlot = matplotlib.lines.Line2D([0],[0], linestyle="none", c=c, marker=m)
         proxyPlots.append(proxyPlot)
         labels.append("ALPs " + PLOT_DEMONSTRATIONS[self.rmQueryCount])
         ax.legend(proxyPlots, labels, numpoints = 1)
         self.rmQueryCount += 1
      stopTime = time.time()

      plt.close(fig)

      # how do we pick one?
      print >> sys.stderr, "configurations retrieved: " + str([len(allConfigurations) for allConfigurations in allConfigurationsList]) + " [in " + str(stopTime-startTime) + " seconds]"
      print >> sys.stderr, "->Selecting Configurations"

      # downsample by num dep
      # required to make higher dimension feature vectors tractable
      # can be removed later when computation is manageable
      downsampledAllConfigurationsList = []
      for allConfigurations in allConfigurationsList:
         if(len(allConfigurations) > 1):
            # minimum of 1
            numberOfSamples = max(1, len(allConfigurations)/(self.numDependent**self.numIndependent))
            downsampledAllConfigurationsList.append(random.sample(allConfigurations, numberOfSamples))
         else:
            downsampledAllConfigurationsList.append([allConfigurations[0]])
      allConfigurationsList = downsampledAllConfigurationsList

      # check forward mapping error for each configuratoin returned
      # check location in point cloud
      startTime = time.time()
      #configurationList = self.get_mean_configurations(allConfigurationsList)
      #configurationList = self.get_min_error_configurations(fm, allConfigurationsList, slps)
      configurationList = self.get_error_weighted_configurations(fm, allConfigurationsList, slps)
      #configurationList = self.get_svm_configurations(allConfigurationsList)
      #configurationList = self.get_kmeans_configurations(allConfigurationsList)
      stopTime = time.time()

      # experiment complete
      print >> sys.stderr, "->Done" + " [in " + str(stopTime-startTime) + " seconds]"

      # compute error for each configuration
      # print error
      demonstrationIndex = 0
      for slps,c,m in slpsList:
         configuration = configurationList[demonstrationIndex]
         prediction = fm.predict(configuration)
         error = 0
         for index in range(len(slps)):
            error += (slps[index] - prediction[index])**2
         if(saveError == True):
            with open("suggested_alps_fm_prediction_error.txt", "a") as predictionErrorFile:
               predictionErrorFile.write(str(error) + "\n")
         demonstrationIndex += 1
         
      return [allConfigurationsList, configurationList]

# learning from visual demonstrations
def lfd(plotConfigurations=False, saveError=False):
   # expects configuration file as input
   experiment = Experiment()

   # load network if needed
   net = fi.get_network(experiment.descriptorSetting, IMAGE_FEATURE_PATH_PREFIX)

   # traverse all demonstration data
   plotIndex = 0
   slpsList = []

   # process directories in sorted order
   sortedPaths = []
   for dirName, subdirList, fileList in os.walk(experiment.demonstrationFolder):
      for sortedSubdirName in sorted(subdirList):
         sortedPaths.append(dirName + "/" + sortedSubdirName)

   # get slps for demonstrations
   for sortedPath in sortedPaths:
      for dirName, subdirList, fileList in os.walk(sortedPath):
         # only use non empty subdirectories
         if(len(fileList) > 0):
            # traverse all files in this directory
            if(bool(experiment.images) == True):
               # read dependent values as images
               dependentValues = [tuple(fi.get_features(net, dirName + "/" + fileName, experiment.descriptorSetting)) for fileName in fileList]
            else:
               # read dependent values as text
               dependentValues = [tuple([eval(x) for x in open(dirName + "/" + fileName).readlines()[0].replace("\n", "").split(" ")]) for fileName in fileList]
               
            dependentValues = tuple(misc.col_average(dependentValues))
            #print >> sys.stderr, "dependent values: " + str(dependentValues)
            
            # edit scale data file name based on descriptor setting
            scaleDataFileNamePrefix = experiment.scaleDataFileName.split(".")[0]
            scaleDataFileNameExtension = experiment.scaleDataFileName.split(".")[1]
            scaleDataFileName = scaleDataFileNamePrefix + "_" + experiment.abm
            if(bool(experiment.images) == True):
               scaleDataFileName += "_" + experiment.descriptorSetting.lower()
            scaleDataFileName += "." + scaleDataFileNameExtension
            
            # scale demonstration data
            with open("sampling/" + experiment.abm + "/" + scaleDataFileName, "r") as scaleDataFile:
               # read in order of writing in map.py
               maxs = eval(scaleDataFile.readline().strip("\n"))
               mins = eval(scaleDataFile.readline().strip("\n"))
                  
               dependentMaxs = maxs[experiment.numIndependent:]
               dependentMins = mins[experiment.numIndependent:]
               dependentValuesMax = {i: dependentMaxs[i] for i in range(len(dependentMaxs))}
               dependentValuesMin = {i: dependentMins[i] for i in range(len(dependentMins))}
               dependentValues = tuple((dependentValues[index] - dependentValuesMin[index])/(dependentValuesMax[index] - dependentValuesMin[index]) if (dependentValuesMax[index] > dependentValuesMin[index]) else dependentValuesMax[index] for index in range(len(dependentValues)))

               # reduce dependent values if neccessary
               if(eval(os.environ["REDUCED"]) == 1):
                  model = cPickle.load(open("sampling/utils/transforms/" + experiment.abm +"_reduced_model.p", "rb")) 
                  dependentValues = model.transform(np.array(dependentValues).reshape(1, -1))
                  dependentValues = tuple(dependentValues[0])

               # hacky way to do this
               # assuming feedback is off during evaluation
               # overwrite dependent values from queried_slps.txt
               if((eval(os.environ["EVALUATION_RANDOM"]) == 1) or (eval(os.environ["EVALUATION_CROSS_VALIDATION"]) == 1)):
                  with open("queried_slps.txt", "r") as inFile:
                     dependentValues = eval(inFile.readlines()[0].replace("\n","").strip())
                  
               # if feedback is on
               # use feedback to modify queried slps
               if((eval(os.environ["FEEDBACK"]) == 1) and (os.path.isfile("queried_slps.txt") == True)):
                  with open("queried_slps.txt", "r") as inFile:
                     queriedSlps = eval(inFile.readlines()[0])
                  slpsList.append((queriedSlps, PLOT_COLOR[plotIndex], PLOT_SHAPE[plotIndex]))
               else:
                  slpsList.append((dependentValues, PLOT_COLOR[plotIndex], PLOT_SHAPE[plotIndex]))

               plotIndex += 1

               # default value for queried slps
               # same as current demonstration slps
               queriedSlps = dependentValues

               #
               # feedback code
               #
               
               # random initialization if feedback is not yet applicable
               if(eval(os.environ["FEEDBACK"]) == 1):
                  with open("feedback_iteration.txt", "r") as inFile:
                     iteration = eval(inFile.readlines()[0])
                     if(iteration <= 2):
                        with open("regression_parameter.txt", "w") as outFile:
                           regressionParameterLowerLimit = eval(os.environ["REGRESSION_PARAMETER_LOWER_LIMIT"])
                           regressionParameterUpperLimit = eval(os.environ["REGRESSION_PARAMETER_UPPER_LIMIT"])
                           regressionParameter = regressionParameterLowerLimit + (regressionParameterUpperLimit-regressionParameterLowerLimit)*random.random()
                           outFile.write(str(regressionParameter) + "\n")
                           
               # TODO: currently only works for a single slp suggestion
               # check if file exists
               if((eval(os.environ["FEEDBACK"]) == 1) and (os.path.isfile("queried_slps.txt") == True)):
                  # use feedback to modify queried slps
                  with open("queried_slps.txt", "r") as inFile:
                     queriedSlps = eval(inFile.readlines()[0])

                  # check if file exists
                  if(os.path.isfile("suggested_slps.txt") == True):
                     with open("suggested_slps.txt", "r") as inFile:
                        suggestedSlps = eval(inFile.readlines()[0])[0]

                        # check consistency of suggested slps with current configuration
                        if(len(suggestedSlps) == len(dependentValues)):
                           #
                           # fm dataset optimization
                           #
                           
                           # scale suggested slps
                           # TODO: make this a common function since its used everywhere
                           suggestedSlps = tuple((suggestedSlps[index] - dependentValuesMin[index])/(dependentValuesMax[index] - dependentValuesMin[index]) if (dependentValuesMax[index] > dependentValuesMin[index]) else dependentValuesMax[index] for index in range(len(suggestedSlps)))

                           # the differential accounts for the distance from the demonstration slps 
                           # demonstration slps are currently stored as dependentValues
                           with open("feedback_iteration.txt", "r") as inFile:
                              iteration = inFile.readlines()[0].strip("\n")
                           alpha = eval(os.environ["LEARNING_RATE_FM_DATASET"])
                           alpha /= eval(iteration)                           
                           queriedSlps = tuple([queriedSlps[index] + alpha*(dependentValues[index] - suggestedSlps[index]) for index in range(len(dependentValues))])
                           
                           # write data
                           with open("feedback_input_difference.txt", "a") as outFile:
                              outFile.write(str(np.abs(dependentValues[0] - queriedSlps[0])) + "\n")
                           with open("feedback_output_difference.txt", "a") as outFile:
                              outFile.write(str(dependentValues[0] - suggestedSlps[0]) + "\n")

                           #
                           # regression parameter optimization initialization
                           #

                           # if just one iteration so far, move in a random direction
                           with open("feedback_iteration.txt", "r") as inFile:
                              iteration = eval(inFile.readlines()[0])
                              if(iteration > 2):
                                 # now we do feedback stuff
                                 with open("feedback_regression_parameter.txt", "r") as feedbackRegressionParameterFile:
                                    with open("feedback_output_difference.txt", "r") as feedbackOutputDifferenceFile:
                                       regressionParameterLines = feedbackRegressionParameterFile.readlines()
                                       outputDifferenceLines = feedbackOutputDifferenceFile.readlines()
                                       currentRegressionParameter = eval(regressionParameterLines[-1])
                                       previousRegressionParameter = eval(regressionParameterLines[-2])
                                       regressionParameterChange = currentRegressionParameter - previousRegressionParameter
                                       currentOutputDifference = eval(outputDifferenceLines[-1])
                                       previousOutputDifference = eval(outputDifferenceLines[-2])
                                       outputDifferenceChange = currentOutputDifference - previousOutputDifference
                                       
                                       alpha = eval(os.environ["LEARNING_RATE_REGRESSION_PARAMETER"])
                                       alpha /= iteration

                                       # if output difference is negative, we move in direction to change
                                       nextRegressionParameter = currentRegressionParameter - regressionParameterChange*outputDifferenceChange*alpha
                                       with open("regression_parameter.txt", "w") as outFile:
                                          outFile.write(str(nextRegressionParameter) + "\n")
                              
               # TODO: currently assumes only a single demonstration at a time. fix for multiple demonstrations
               with open("queried_slps.txt", "w") as outFile:
                  outFile.write(str(queriedSlps) + "\n")

   configurationListPerRun = []   
   for run in range(RUNS_PER_SLPS):
      # run and query slps
      [allConfigurationsList, configurationList] = experiment.run(slpsList, plotConfigurations=plotConfigurations, saveError=saveError)

      # scale configurations back from normalized space
      independentMaxs = maxs[:experiment.numIndependent]
      independentMins = mins[:experiment.numIndependent]
      independentValuesMax = {i: independentMaxs[i] for i in range(len(independentMaxs))}
      independentValuesMin = {i: independentMins[i] for i in range(len(independentMins))}
      normalizedConfigurationList = []
      for configuration in configurationList:
         normalizedConfigurationList.append(tuple([independentValuesMin[cIndex] + (independentValuesMax[cIndex]-independentValuesMin[cIndex])*configuration[cIndex] for cIndex in range(len(configuration))]))
      configurationList = tuple(normalizedConfigurationList)
      configurationListPerRun.append(configurationList)
      
      # return configurations
      print >> sys.stderr, "configurationList: " + str(configurationList)

   # compute average over runs
   averageConfigurationList = []
   for slpsIndex in range(len(slpsList)):
      slpsConfigurations = tuple([configurationListOneRun[slpsIndex] for configurationListOneRun in configurationListPerRun])
      averageConfigurationList.append(tuple(misc.col_average(slpsConfigurations)))

   print >> sys.stderr, "averageConfigurationList: " + str(averageConfigurationList)
   sys.stdout.write(str(averageConfigurationList))

   if("smt" in experiment.regression):
      # circumvent excessive prints from smt
      with open("temp_average_configuration_list.txt", "w") as outFile:
         outFile.write(str(averageConfigurationList) + "\n")

# main function
def main():

   if(eval(os.environ["RANDOM_BASELINE"]) == 1):
      #
      # return dummy values
      #
      
      configurationFileName = os.environ["CONFIGURATION_FILE"]
      configuration = open(configurationFileName).read()
      execd = {}
      exec(configuration, execd)
      numIndependent = execd['NUM_INDEPENDENT']
      randomIndependentValues = tuple([random.random() for _ in range(numIndependent)])

      # apply scaling
      abm = os.environ["ABM"]
      scaleDataFileName = os.environ["SCALE_DATA_FILE"]
      descriptorSetting = os.environ["DESCRIPTOR_SETTING"]
      images = eval(os.environ["IMAGES"])
      
      # edit scale data file name based on descriptor setting
      scaleDataFileNamePrefix = scaleDataFileName.split(".")[0]
      scaleDataFileNameExtension = scaleDataFileName.split(".")[1]
      scaleDataFileName = scaleDataFileNamePrefix + "_" + abm
      if(bool(images) == True):
         scaleDataFileName += "_" + descriptorSetting.lower()
      scaleDataFileName += "." + scaleDataFileNameExtension

      with open("sampling/" + abm + "/" + scaleDataFileName, "r") as scaleDataFile:
         maxs = eval(scaleDataFile.readline().strip("\n"))
         mins = eval(scaleDataFile.readline().strip("\n"))            

      independentMaxs = maxs[:numIndependent]
      independentMins = mins[:numIndependent]
      independentValuesMax = {i: independentMaxs[i] for i in range(len(independentMaxs))}
      independentValuesMin = {i: independentMins[i] for i in range(len(independentMins))}
      randomIndependentValues = tuple([independentValuesMin[index] + randomIndependentValues[index]*(independentValuesMax[index]-independentValuesMin[index]) for index in range(len(randomIndependentValues))])
      randomConfigurationList = [randomIndependentValues]
      sys.stdout.write(str(randomConfigurationList))
   else:
      # use framework
      lfd(plotConfigurations=False, saveError=True)

# execute main
main()
