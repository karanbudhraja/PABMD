import numpy as np
import scipy.io as sio
from sets import Set
from ltfatpy.sigproc.thresh import thresh
from sklearn.cluster import KMeans, MeanShift
from collections import Counter

import sys

def lpod(trainData, testData, trainLabels, testLabels, lNum, tol, lambdaValue):
    # this is a python port of the original matlab code provided by the authors
    # of the 2017 paper titled "Efficient Outlier Detection for High-Dimensional Data"
    
    # supress invalid division warning
    # this happens in the matlab code too so we're ok
    np.seterr(divide='ignore', invalid='ignore')
    
    # size of x and y
    # check for size mismatch
    [rtrainData,ctrainData] = trainData.shape
    [rtestData,ctestData] = testData.shape
    assert rtrainData == rtestData

    # allocate memory to the maximum size
    # T is the score matrix of trainData
    # P is the loading matrix of trainData
    # U is the score matrix of testData
    # Q is the loading matrix of testData
    # B is the matrix of regression coefficient
    # W is the weight matrix of trainData                           
    n = max(ctrainData, ctestData)
    T = np.zeros((rtrainData, n))                                  
    P = np.zeros((ctrainData, n))       
    U = np.zeros((rtestData, n))                                 
    Q = np.zeros((ctestData, n))       
    B = np.zeros((n, n))            
    W = P                                                      
    k = 0

    # iteration loop if residual is larger than specfied
    iteration = 1

    while(iteration < lNum and np.linalg.norm(testData) > tol and k < n):
        # choose the column of x has the largest square of sum as t
        # choose the column of y has the largest square of sum as u
        tidx = np.argmax(sum(np.multiply(trainData, trainData)))
        uidx = np.argmax(sum(np.multiply(testData, testData)))
        t1 = trainData[:, tidx]
        u = testData[:, uidx]
        t = np.zeros((rtrainData, 1))

        # iteration for outer modeling until convergence
        while (np.linalg.norm(t1 - t) > tol):
            # computing the weight vector of trainData
            # computing the score vector of trainData
            # obtaining q by soft-threshold
            w = np.matmul(trainData.transpose(), u)                  
            w = w / np.linalg.norm(w)
            t = t1
            t1 = np.matmul(trainData, w)                                     
            q, dummy = thresh(np.matmul(testData.transpose(), t1), lambdaValue, thresh_type="soft")
            
            # computing the loading vector of testData according to t
            # computing the score vector of testData
            q = q / np.linalg.norm(q)
            u = np.matmul(testData, q)                                       
        
        # update p based on t
        # computing the loading vecor of trainData
        t = t1
        p = np.matmul(trainData.transpose(), t) / np.matmul(t.transpose(), t)
        pnorm = np.linalg.norm(p)
        p = p / pnorm                                              
        t = t * pnorm
        w = w * pnorm

        # regression and residuals
        t = t.reshape((-1, 1))
        p = p.reshape((-1, 1))
        u = u.reshape((-1, 1))
        q = q.reshape((-1, 1))

        b = np.matmul(u.transpose(), t) / np.matmul(t.transpose(), t)
        trainData = trainData - np.matmul(t, p.transpose())
        testData = testData - (b * np.matmul(t, q.transpose()))

        # save iteration results to outputs:
        k += 1
        T[:, k] = list(t)
        P[:, k] = list(p)
        U[:, k] = list(u)
        Q[:, k] = list(q)
        W[:, k] = list(w)
        B[k, k] = b
        
        iteration += 1

    # end of iterations
    T = T[:, :k]
    P = P[:, :k]
    U = U[:, :k]
    Q = Q[:, :k]
    W = W[:, :k]
    B = B[:k, :k]

    # prediction stage:
    # predicting real-valued matrix for test_data
    # transforming outputs into the corresponding label matrix
    # 0.5 is the default threshold value
    outputs = np.matmul(np.matmul(np.matmul(P, B), Q.transpose()).transpose(), trainLabels)
    [numClass, numTesting] = outputs.shape
    predictionLabels = np.zeros((numClass, numTesting))
    for i in range(numTesting):
        for j in range(numClass):
            if(outputs[j,i] >= 0.5):                                  
                predictionLabels[j,i] = 1
            else:
                predictionLabels[j,i] = 0

    return predictionLabels

def do_lpod(X, Y):
    # outlier detection
    Y = Y.astype(float).transpose()
    allDataLength = len(Y)
    trainDataFraction = 0.7
    trainDataLength = np.floor(trainDataFraction * allDataLength)
    
    # create logical index vector
    # randomize order to select random elements
    trainIndexes = np.random.choice(int(allDataLength), int(trainDataLength), replace=False)
    testIndexes = np.setdiff1d(range(allDataLength), trainIndexes)
    
    trainData = X[trainIndexes, :]
    trainLabels = Y[trainIndexes]
    testData = X[testIndexes, :]
    testLabels = Y[testIndexes]
    
    # hyper parameters
    # get predictions
    lNum = 50
    tol = 1e-10
    lambdaValue = 0.1
    predictionLabels = lpod(trainData.transpose(), testData.transpose(), trainLabels, testLabels, lNum, tol, lambdaValue)
    lpodX = np.concatenate((trainData, testData), axis=0)
    lpodY = np.concatenate((trainLabels, testLabels), axis=0)

    return [lpodX, lpodY]

def get_approximate_labels(allConfigurations):
    # cluster points
    # outlier is decided as the minority cluster
    # count population of each cluster
    # outliers are labeled as 1
    # inliers are labeled as 0
    # force 2 clusters as backup in case meanshift finds a single cluster
    X = np.array(allConfigurations)
    meanShift = MeanShift(bin_seeding=True)
    meanShift.fit(X)
    nClusters = 2
    kmeans = KMeans(init='k-means++', n_clusters=nClusters)
    kmeans.fit(X)
    labelCounts = Counter(meanShift.labels_)

    if(len(labelCounts.keys()) > 1):
        # multiple clusters found by meanshift
        labels = meanShift.labels_
    else:
        # single cluster found by meanshift
        # use kmeans clusters instead
        labels = kmeans.labels_
        
    labelCounts = Counter(labels)
    minLabel = min(labelCounts, key=labelCounts.get)
    labels = ['1' if x == minLabel else '0' for x in labels]
    labels = [eval(x) for x in labels]
    labels = np.array(labels, ndmin=2)
    
    return [X, labels]
