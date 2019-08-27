from __future__ import division  # floating point division
import math
import numpy as np

'''
Modified from the CMPUT466 in UAlberta Assignment barebones code
https://marthawhite.github.io/mlcourse/schedule.html
'''

####### Main load functions
def load_breast_cancer(trainsize, testsize):
    """ A blogging dataset """
    # if trainsize + testsize < 5000:
    #     filename = 'datasets/blogData_train_small.csv'
    # else:
    #     filename = 'datasets/blogData_train.csv'
    filename = 'breast-cancer-wisconsin.data.txt'
    dataset = loadcsv(filename)
    print(type(dataset), type(trainsize), type(testsize))
    trainset, testset = splitdataset(dataset,trainsize, testsize, featureoffset = 1)
    return trainset,testset

def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset

def splitdataset(dataset, trainsize, testsize, testdataset=None, featureoffset=None, outputfirst=None):
    """
    Splits the dataset into a train and test split
    If there is a separate testfile, it can be specified in testfile
    If a subset of features is desired, this can be specifed with featureinds; defaults to all
    Assumes output variable is the last variable
    """
    print(dataset)
    # Generate random indices without replacement, to make train and test sets disjoint
    randindices = np.random.choice(dataset.shape[0],trainsize+testsize, replace=False)
    # randindices = range(trainsize + testsize)
    featureend = dataset.shape[1]-1
    print('data shape 1 ',dataset.shape[1])
    outputlocation = featureend

    if featureoffset is None:
        featureoffset = 0
    if outputfirst is not None:
        featureoffset = featureoffset + 1
        featureend = featureend + 1
        outputlocation = 0

    Xtrain = dataset[randindices[0:trainsize],featureoffset:featureend]
    ytrain = dataset[randindices[0:trainsize],outputlocation]
    # print(Xtrain, ytrain)
    Xtest = dataset[randindices[trainsize:trainsize+testsize],featureoffset:featureend]
    ytest = dataset[randindices[trainsize:trainsize+testsize],outputlocation]

    if testdataset is not None:
        Xtest = dataset[:,featureoffset:featureend]
        ytest = dataset[:,outputlocation]

    # Normalize features, with maximum value in training set
    # as realistically, this would be the only possibility
    for ii in range(Xtrain.shape[1]):
        maxval = np.max(np.abs(Xtrain[:,ii]))
        if maxval > 0:
            Xtrain[:,ii] = np.divide(Xtrain[:,ii], maxval)
            Xtest[:,ii] = np.divide(Xtest[:,ii], maxval)

    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
    Xtrain = Xtrain.T
    Xtest = Xtest.T
    return ((Xtrain,ytrain), (Xtest,ytest))
