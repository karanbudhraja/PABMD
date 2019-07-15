# fm.py contains the ForwardMapping frontend.

import data
import sys
import numpy as np
import sklearn

# This thing is the Forward Mapping
class ForwardMapping(object):

   # pass in the regression algorithm that we will be using, along with any initialization parameters
   def __init__(self, regression_class, num_dependent):
      """
      Initialized a forward mapping.
      Pass in the regression class to be used, the initial parameters for initlializing this reg class.
      Also, pass the the number of dependent variables the data set will have.
      """

      self._mappers = [eval(regression_class) for x in range(num_dependent)]

   def train(self, training_set):
      # remove data entries which contain nan
      training_set = [x for x in training_set if not np.isnan(x[0]).any()]
      training_set = [x for x in training_set if not np.isnan(x[1]).any()]

      training_sets = data.split_dep(training_set)

      for mapper, tset in zip(self._mappers, training_sets):
         X = np.array([list(x[0]) for x in tset])
         Y = np.array([x[1] for x in tset])
         mapper.fit(X, Y)
         
   def predict(self, configuration):
      prediction = [mapper.predict(np.array(list(configuration)).reshape(1, -1)) for mapper in self._mappers]
      prediction = tuple([x[0] for x in prediction])
      
      return prediction
      
