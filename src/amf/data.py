# data.py contains everything that has to do with dealing with the actual
# data and manipulating data sets.

import random
import copy
import sys
import misc
import os

IND = 5
DEP = 6

def make_vartype(vartype_str):
   """
   Takes a string that labels columns in a data set.
   vartype_string should either have 'i' or 'd' in it, e.g., "iidid"

   i - independent variable
   d - dependent variable
   """
   # replace characters i and d with IND and DEP symbols.
   return tuple( (IND if c == 'i' else DEP) for c in vartype_str.strip() if not c.isspace())

def load(file_name):
   """
   Loads a data set from a file.

   The data set should be space delimited to distinguish columns, and newline
   delimited to distinguish rows.

   The first line should contain something that is to be passed into make_vartype.

   """

   file_h = open(file_name)

   vartype = make_vartype(file_h.readline().strip())

   dataset = []
   # go through each line and parse it
   for line in file_h.xreadlines():
      line = line.strip()

      # ignore empty lines
      if len(line) == 0:
         continue

      sline = line.split()
      if len(sline) != len(vartype):
         sys.stderr.write("Ignored line with wrong number of tokens: " + line + "\n")
         continue

      # split the data into two groups, depending on if they are independent or dependent variables.
      ind_list = []
      dep_list = []
      for idx, item in enumerate(sline):
         (ind_list if vartype[idx] == IND else dep_list).append(float(item) if item != 'NOENTRY' else 'NOENTRY')
   
      dataset.append( ( tuple(ind_list), tuple(dep_list) ) )

   dataset.sort()

   return tuple(dataset)

def random_subsets(data_set, n_list, demonstrationDescriptors=[]):
   """
   Returns a random subsets of the dataset with sizes specified in the n list.
   There will be no duplicates between subsets.
   """

   if sum(n_list) > len(data_set):
      raise IndexError, "Requested size of subsets, from set of size %d, are too large: %s (total size %d)" % (len (data_set), repr(n_list), sum(n_list))

   # generate a list of the indexes, then shuffle them. I will use these to randomly select from the data set.
   idxs = range(len(data_set))

   # this is where the randomness lies
   #random.seed(0)

   # idxs contain indexes of all elements in dataset
   # we should get the top sum(n_list) values
   # then use those to form random subsets of length specified by n_list

   # check number of demonstration descriptors
   demonstrationCount = len(demonstrationDescriptors)

   # obtain and shuffle indexes for randomness
   # divides equally per demonstration slp
   # works better when a single slp is specified

   # dataset selection used if set to true
   if(eval(os.environ["DATASET_SELECTION"]) == 1):
      useClosestIdx = True
   else:
      useClosestIdx = False

   #if(type(depData) == tuple):
   if(useClosestIdx == True):
      # we need to divide the dataset to cover all of these
      # edit last item to account for correctness of sum
      # shuffle for random allotment of extra points
      nListSubLengths = [sum(n_list)/demonstrationCount]

      nListSubLengths[-1] += sum(n_list)%demonstrationCount
      random.shuffle(nListSubLengths)

      # indexes corresponding to good points
      goodIdxs = []
      
      # for each sublength
      # corresponding to each demonstration
      for index in range(len(nListSubLengths)):
         # get distances of all descriptors in dataset
         # with respect to the input descriptor
         distances = [(misc.distance(demonstrationDescriptors[index], data_set[idx][1]), idx) for idx in idxs]
         distances.sort()

         # extract first few as needed
         nListSubLength = nListSubLengths[index]
         goodDataPoints = distances[:nListSubLength]

         # append to list of all points
         goodIdxs += [idx for distance, idx in goodDataPoints]

      random.shuffle(goodIdxs)
   else:
      # point sampling has already been done from the full dataset
      # now we are dealing with individual dep value correspondences for a given ind configuration
      random.shuffle(idxs)

   # now form subsets based on sequence of indexes
   subsets = []
   cur_offset = 0
   for subset_size in n_list:
      #if(type(depData) == tuple):
      if(useClosestIdx == True):
         # we are selecting from the full dataset
         # this dataset does not contain individual slp correspondence
         subsets.append( tuple(data_set[idx] for idx in goodIdxs[cur_offset:cur_offset + subset_size]) )
      else:
         # point sampling has already been done from the full dataset
         # now we are dealing with individual dep value correspondences for a given ind configuration
         subsets.append( tuple(data_set[idx] for idx in idxs[cur_offset:cur_offset + subset_size]) )

      cur_offset += subset_size

   return subsets

def split_dep(data_set):
   """
   Split the data set into one-data-set-per-dependent variable.
   """

   num_dep = len(data_set[0][1])

   sets = [ [] for idx in range(num_dep) ]

   for ind, dep in data_set:
      # for each dependent variable here...
      for d_idx in range(num_dep):
         sets[d_idx].append((ind, dep[d_idx]))
         
   return sets






