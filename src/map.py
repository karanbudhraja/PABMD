# This script takes each row of a data set and processes it for use with AMF
import sys
import math
import os

no_entry = "NOENTRY"
SCALE_DATA_FILE = "scale_data.txt"

def map_abm(tokens):
   """
   The raw data set is a combination of floats followed by a combination of lists and looks like this:
   FLOAT FLOAT ... LIST LIST ...
   Non-lists are independent variables
   Lists are dependent variables
   It takes the average of the list and returns it with the independent variables.
   """

   tokenAvg = []

   for token in tokens:
      if(type(token) == list):
         avg = sum(token)/len(token)
      else:
         avg = token

      tokenAvg.append(avg)

   return tuple(tokenAvg)

def linear_scale(dataset):
   """Linearly scales an entire data set to a range of 0-1"""

   maxs = list(dataset[0])
   mins = list(dataset[0])

   for datarow in dataset:
      for idx, item in enumerate(datarow):
         if item == no_entry: continue

         maxs[idx] = max(maxs[idx], item)
         mins[idx] = min(mins[idx], item)

   scaleDataFileName = SCALE_DATA_FILE
   abm = os.environ["ABM"]
   descriptorSetting = os.environ["DESCRIPTOR_SETTING"]

   # edit scale data file name based on descriptor setting
   scaleDataFileNamePrefix = scaleDataFileName.split(".")[0]
   scaleDataFileNameExtension = scaleDataFileName.split(".")[1]
   scaleDataFileName = scaleDataFileNamePrefix + "_" + abm
   if(len(descriptorSetting) > 0):
      # valid descriptor
      scaleDataFileName += "_" + descriptorSetting.lower()
   scaleDataFileName += "." + scaleDataFileNameExtension
         
   with open(scaleDataFileName, "w") as scaleDataFile:
      scaleDataFile.write(str(maxs))
      scaleDataFile.write("\n")
      scaleDataFile.write(str(mins))

   return [ [ (no_entry if item == no_entry else maxs[idx] if maxs[idx] == mins[idx] else (item - mins[idx]) / (maxs[idx] - mins[idx])) for idx, item in enumerate(datarow) ] for datarow in dataset ]

def parse(data_file_path):
   """
   Applies 'map_function' to each row (as a string) of a data set.
   'map_function' should takes a list of tokens as the only parameter and return a string as the only parameter.
   Prints out the new data set.
   """

   return [ [ float(token) if token.find(',') == -1 else [ float(item) for item in token.split(',') ] for token in line.split() ] for line in open(data_file_path).xreadlines() ]

def dump(dataset):
   print "\n".join( " ".join(str(item) for item in row) for row in dataset)

def process_abm():
   parsed = parse(sys.argv[1])
   averaged = map(map_abm, parsed)
   scaled = linear_scale(averaged)
   dump(scaled)

# average over system level paramters in raw data file
# raw data file is provided as argument
process_abm()
