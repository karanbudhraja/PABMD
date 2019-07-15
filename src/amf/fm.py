# fm.py contains the ForwardMapping frontend.

import data
import sys
import numpy as np
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from smt.surrogate_models import RBF, KRG, KPLS, KPLSK
import os

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

         if("smt" in str(mapper)):
            # using surrogate modeling toolbox
            X = np.array([x[0] for x in X])
            VThreshold = 0.001

            XSafe = [X[0]]
            YSafe = [Y[0]]
            for index in range(1, len(X)):
              currentX = X[index]
              currentY = Y[index]
              XSafetyCheck = XSafe + [currentX]
              YSafetyCheck = YSafe + [currentY]

              # we need to only collect those points that do not case matrix inversion problems
              # check using svd
              configurationFileName = os.environ["CONFIGURATION_FILE"]
              configuration = open(configurationFileName).read()
              execd = {}
              exec(configuration, execd)
              numIndependent = execd['NUM_INDEPENDENT']
              numDependent = eval(os.environ["NUM_DEPENDENT"])
              XSafetyCheck = np.array(XSafetyCheck).reshape(numIndependent, -1)
              YSafetyCheck = np.array(YSafetyCheck).reshape(numDependent, -1)
              XYSafetyCheck = np.concatenate((XSafetyCheck, YSafetyCheck), axis=0)
              U, s, V = np.linalg.svd(XYSafetyCheck, full_matrices=True)
              V = np.absolute(V)

              if(np.min(V) > VThreshold):
                 XSafe.append(currentX)
                 YSafe.append(currentY)
               
            # convert to numpy array
            # train model
            XSafe = np.array(XSafe)
            YSafe = np.array(YSafe)
            mapper.set_training_values(XSafe, YSafe)
            mapper.train()
         else:
            mapper.fit(X, Y)

   def predict(self, configuration):
      prediction = []
      for mapper in self._mappers:
         if("smt" in str(mapper)):
            mapperPrediction = mapper.predict_values(np.array(list(configuration)).reshape(1, -1))
            mapperPrediction = [x[0] for x in mapperPrediction]
         else:
            mapperPrediction = mapper.predict(np.array(list(configuration)).reshape(1, -1))            
         prediction.append(mapperPrediction)
         
      prediction = tuple([x[0] for x in prediction])
      
      return prediction
