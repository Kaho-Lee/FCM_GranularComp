from __future__ import division  # floating point division
# import csv
import random
import math
import numpy as np
import matplotlib.pyplot as plt

import load_data as dtl
# import regressionalgorithms as algs
# import utilities as utils
if __name__ == '__main__':
    trainsize = 2
    testsize = 1
    trainset, testset = dtl.load_breast_cancer(trainsize,testsize)
    print(trainset)
    print(testset)
