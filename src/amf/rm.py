# rm.py contains the ReverseMapping

import misc
import numpy
import sys
import multiprocessing as mp
import time
import itertools
import random
import os
import gzip
import cPickle

# parallel processing constants
BATCH_SIZE = 8

# hack to avoid rounding error
def _round(n):
   return numpy.round(n, 5)

class RMSimplex(object):
   # Simplexes are kind of immutable...

   # The LAST variable of each corner vector is the dependent variable
   #   (this is important)

   def __init__(self, corners, slpsList):
      """
      A simplex.
      corners should have the form ((ind0, ind1, ...), (dep1, dep2, dep3))
      """
      # We already know which corners are adjanced to which
      #   (each corner shares an edge with every other corner)

      self._corners = tuple(corners)
      self._inds = tuple( corner[0] for corner in corners )
      self._deps = tuple( corner[1] for corner in corners )
      self._min = misc.col_min(self._deps)
      self._max = misc.col_max(self._deps)      
      self._edges = []

      # initialize edges only if needed
      # only if this simplex contains the query slp
      # otherwise the slp has no utility
      isContains = False
      
      for querySlps,c,m in slpsList:
         for index in range(len(querySlps)):
            isContains = isContains or self.contains(index, querySlps[index])
      
      if(isContains == True):
         for cidx in range(len(self._corners)):
            corner = self._corners[cidx]
            for oidx in range(cidx + 1, len(self._corners)):
               other_corner = self._corners[oidx]
               self._edges.append((corner, other_corner))
      self._edges = tuple(self._edges)

   @property
   def corners(self):
      return self._corners

   @property
   def edges(self):
      return self._edges

   def contains(self, slp_num, value):
      """Determines if the value for slp_num idx lives inside this simplex."""
      return ((value >= self._min[slp_num]) and (value <= self._max[slp_num]))

   def intersections(self, slp_num, value):
      """Returns the approximate places that the value plane intersects with this simplex"""
      inters = []
      for edgeNumber, (scorner, bcorner) in enumerate(self.edges):
         # maintain order of scorner <= bcorner
         if(scorner[1][slp_num] > bcorner[1][slp_num]):
            tempCorner = bcorner
            bcorner = scorner
            scorner = tempCorner

         # check to see that this value is inside these edges
         if not ((value[slp_num] >= scorner[1][slp_num]) and (value[slp_num] <= bcorner[1][slp_num])):
            continue

         # this is the distance between the 2 corner values
         distance = bcorner[1][slp_num] - scorner[1][slp_num]

         if(distance == 0):
            # return any end point value
            inter = tuple( (1.0)*s for s, b in zip(scorner[0], bcorner[0]) )
         else:
            # this is the distance from 'value' from the smaller corner
            # if value is close to smaller, this ratio will be close to zero
            # if value is closer to bigger, this ratio will be close to one
            ratio = (value[slp_num] - scorner[1][slp_num]) / distance
            inter = tuple( (1.0-ratio)*s + ratio * b for s, b in zip(scorner[0], bcorner[0]) )
         
         inters.append([edgeNumber, inter])

      return tuple(inters)

   def __repr__(self):
      return 'simplex:' + repr(self._edges)

class ReverseMapping(object):
   def __init__(self, trained_fm, ranges, granularity, slpsList):

      self.fm = trained_fm
      self.granularity = granularity
      self.ranges = ranges
      self.steps = tuple( (ma - mi) / float(self.granularity) for mi, ma in self.ranges )
      self.slpsList = slpsList

      # disk write stuff
      self.writeToDisk = eval(os.environ["WRITE_TO_DISK"])
      self.allIntersectionsCalls = 0
      
      # the dimensionality of the configuration space
      self.dim = len(ranges)

      # the dimensionality of the SLP space
      self.slp_dim = None

      self.__bin__ = misc.binary(self.dim)

      self.knots = {}
      self.simplexes = []
      print >> sys.stderr, 'building knots'
      startTime = time.time()
      self.build_knots()
      stopTime = time.time()
      print >> sys.stderr, 'done building knots, now building simplexes' + " [" + str(len(self.knots))  + " in " + str(stopTime-startTime) + " seconds]"
      startTime = time.time() 
      self.build_simplexes()
      stopTime = time.time()
      print >> sys.stderr, 'done building simplexes' + " [in " + str(stopTime-startTime) + " seconds]"

   def build_knots(self):
      self.build_knots_inds()
      self.build_knots_deps()

   def build_knots_inds(self):
      values = [list(numpy.arange(self.ranges[curDim][0], self.ranges[curDim][1], (self.ranges[curDim][1] - self.ranges[curDim][0])/self.granularity)) for curDim in range(self.dim)]
      indexes = [range(len(values[i])) for i in range(len(values))]
      valuesPoints = [x for x in itertools.product(*values)]
      indexesPoints = [x for x in itertools.product(*indexes)]

      # sort items for correspondence
      valuesPoints.sort()
      indexesPoints.sort()

      # assign key-value pairs
      self.knots = dict(zip(indexesPoints, valuesPoints))

   def process_build_knots_deps(self, q, new_idx):
      new_vector = self.knots[new_idx]
      predicted = self.fm.predict(new_vector)
      q.put([new_idx, predicted])

   def build_knots_deps_parallel(self):
      # setup a list of processes that we want to run
      q = mp.Queue()
      processes = [mp.Process(target=self.process_build_knots_deps, args=(q, new_idx)) for new_idx in self.knots] 
      batchSize = BATCH_SIZE
      batches = [processes[i:i+batchSize] for i in range(0, len(processes), batchSize)]

      for batch in batches:
         # run processes
         for p in batch:
            p.start()            
         for p in batch:
            [new_idx, predicted] = q.get()
            self.knots[new_idx] = predicted
         for p in batch:
            p.join()            

      # use last predicted point in dependent parameter space for length
      # we can use any key for this
      self.slp_dim = len(self.knots.values()[0])

   def build_knots_deps(self):
      # predict for each point in independent parameter space
      for new_idx in self.knots:
         new_vector = self.knots[new_idx]
         predicted = self.fm.predict(new_vector)
         self.knots[new_idx] = predicted
         
      # use last predicted point in dependent parameter space for length
      self.slp_dim = len(predicted)
      
   def build_simplexes(self, col=None):
      for key in sorted(self.knots.keys()):
         # check to see if this root is on the fringe and thus not actually a root
         for idx in key:
            if idx == (self.granularity - 1):
               break
         else:
            # extract simplexes
            # only keep those simplexes which are actually useful
            sims = self.simplexes_from_cube(key)
            sims = [x for x in sims if(len(x._edges) > 0)]

            if(self.writeToDisk == 1):
               # save to compressed file
               with gzip.open("sims.pgz", 'ab') as simsFile:
                  cPickle.dump(sims, simsFile, -1)
            else:
               # save to ram
               self.simplexes.extend(sims)

      '''
      # karan: testing downsampling
      sampleAmount = 100000

      if(len(self.simplexes) > sampleAmount):
         # downsample
         # for computational tractability
         sampledSimplexes = dict(random.sample(self.simplexes.items(), sampleAmount))
         self.simplexes = sampledSimplexes
      '''
   
   def wiggle(self, root):
      """ Returns the closest point to this root that is an actual root. """
      distance, knot, knotKey = min((misc.distance(root, self.knots[knotKey]), self.knots[knotKey], knotKey) for knotKey in self.knots.keys())
      print >> sys.stderr, "BADNESS: Wiggling point %s to knot %s" % (repr(root), repr(knot))

      return [knot, knotKey]

   def cube_at_root(self, root):
      """ Returns the members of this cube with this point as its root. """
      corners = []

      for perm in self.__bin__:
         corner = []
         for idx in xrange(self.dim):
            corner.append(perm[idx] + root[idx])
         corner = tuple(corner)
         val = self.knots[corner]
         corners.append((self.translate_idx(corner), val))

      return tuple(corners)

   def translate_idx(self, vec_idx):
      """ Translates a index to a configuration"""
      corner = []
      for idx in xrange(self.dim):
         corner.append(vec_idx[idx] * self.steps[idx] + self.ranges[idx][0])

      return tuple(corner)

   def simplexes_from_cube(self, root):
      cube_corners = self.cube_at_root(root)
      # cube_at_root always has the first right corner root as [0]
      # at [-1] we have the other right corner root

      # special case for 1 dimensional configuration space
      if self.dim == 1:
         return (RMSimplex(cube_corners, self.slpsList),)
      else:
         return RMSimplex(cube_corners[:-1], self.slpsList), RMSimplex(cube_corners[1:], self.slpsList)

   def process_intersections(self, q, simplex, param_num, value):
      out = []
      inter = simplex.intersections(param_num, value)
      if len(inter) > 0:
         out.append(inter)

      q.put(out)
      
   def intersections_parallel(self, param_num, value):
      out = []

      # setup a list of processes that we want to run
      q = mp.Queue()
      processes = [mp.Process(target=self.process_intersections, args=(q, simplex, param_num, value)) for simplex in self.simplexes] 
      batchSize = BATCH_SIZE
      batches = [processes[i:i+batchSize] for i in range(0, len(processes), batchSize)]

      for batch in batches:
         # run processes
         for p in batch:
            p.start()            
         for p in batch:
            out += q.get()
         for p in batch:
            p.join()            

      return tuple(out)
      
   def intersections(self, param_num, value):
      out = []

      if(self.writeToDisk == 1):
         # read from disk
         with gzip.open("sims.pgz", 'rb') as simsFile:
            with gzip.open("configuration_" + str(self.allIntersectionsCalls) + ".pgz", 'ab') as configurationFile:
               while 1:
                  try:
                     sims = cPickle.load(simsFile)
                     for simplex in sims:
                        inter = simplex.intersections(param_num, value)
                        if len(inter) > 0:
                           for configurations in inter:
                              for configuration in configurations:
                                 cPickle.dump(inter, configurationFile, -1)
	          except EOFError:
                     break
      else:
         # read from ram
         for simplexNumber, simplex in enumerate(self.simplexes):
            inter = simplex.intersections(param_num, value)
            if len(inter) > 0:
               out.append([simplexNumber, inter])

      return tuple(out)

   def process_all_intersections(self, q, d, value):
      q.put(self.intersections(d, value))
   
   def all_intersections_parallel(self, value):
      allIntersections = []
      
      # setup a list of processes that we want to run
      q = mp.Queue()
      processes = [mp.Process(target=self.process_all_intersections, args=(q, d, value)) for d in range(self.slp_dim)] 
      batchSize = BATCH_SIZE
      batches = [processes[i:i+batchSize] for i in range(0, len(processes), batchSize)]

      for batch in batches:
         # run processes
         for p in batch:
            p.start()            
         for p in batch:
            allIntersections.append(q.get())
         for p in batch:
            p.join()            

      if(len(allIntersections[0]) > 0):
         return tuple(allIntersections)
      else:
         # empty intersection set. need to wiggle
         print >> sys.stderr, "BADNESS: PARALLELISM NOT USEFUL BECAUSE OF WIGGLE!"
         [wiggledValue, wiggledKey] = self.wiggle(value)
         return [[[tuple([x/float(self.granularity) for x in wiggledKey])]]]
      
   def all_intersections(self, value):
      # increment call counter
      self.allIntersectionsCalls += 1
      
      """ Finds intersections with each provided SLP """
      allIntersections = [self.intersections(d, value) for d in range(self.slp_dim)]

      if(len(allIntersections[0]) > 0):
         return tuple(allIntersections)
      else:
         # empty intersection set. need to wiggle
         [wiggledValue, wiggledKey] = self.wiggle(value)
         return [[[tuple([x/float(self.granularity) for x in wiggledKey])]]]
         
   def distance_to(self, param_num, config, value):
      # i'm going to cheat for now and just pick the closest intersecting corner
      ints = self.intersections(param_num, value)

      dists = []
      for simplex in ints:
         for i in simplex:
            dists.append( (misc.distance(i, config)) )
   
      if len(dists) == 0:
         print >> sys.stderr, "BADNESS: no intersections found for %s" % repr(value)
         dist = None
      else:
         dist = min(dists)

      return dist
