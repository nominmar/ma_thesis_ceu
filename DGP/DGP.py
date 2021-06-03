import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def dgp2(n, var_e):
    np.random.seed()
    p=[1./2, 1./2] #probability of treatment assignment
    d = np.random.choice([0, 1], size=(n,1), p=p) #tr
    x1 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    x2 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    x3 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    x4 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))    
    e = np.random.normal(loc = 0.0, scale = var_e, size = (n,1))
    ind = np.zeros((n,1))
    ind[np.where(x1 >= 0)] = 1
    y = np.add(np.add(np.add(np.add(np.add(d*-1.5, np.multiply(3*d, ind)),x2),x3),x4),e)
    X = np.concatenate((x1,x2,x3,x4), axis=1)
    return X, y, d

def dgp4(n, var_e):
    np.random.seed()
    #treatment (randomly assigned)
    d = np.random.choice([0, 1], size=(n,1), p=[1./2, 1./2])
    #covariates (relevate to T.E.)
    x1 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    x2 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    #covariates (relevant to outcome)
    x3 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    x4 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    x5 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    #covariates (irrelevant)
    x6 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    #error term
    e = np.random.normal(loc = 0.0, scale = var_e, size = (n,1))
    #indicator functions
    ind1 = np.zeros((n,1))
    ind2 = np.zeros((n,1))
    ind3 = np.zeros((n,1))
    ind1[np.where(x1 >= 0)] = 1
    ind2[np.where(x2 >= 0)] = 1
    ind3[np.where((x1 >= 0) & (x2 >= 0))] = 1
    y = np.add(np.add(np.add(np.add(np.add(d*-3, np.multiply(np.add(np.add(4*ind1, ind2), ind3),d)),x3),x4),x5),e)
    X = np.concatenate((x1,x2,x3,x4,x5,x6), axis=1)
    return X, y, d

def dgp8(n, var_e):
    np.random.seed()
    #treatment (randomly assigned)
    d = np.random.choice([0, 1], size=(n,1), p=[1./2, 1./2])
    #covariates (relevate to T.E.)
    x1 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    x2 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    x3 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))

    #covariates (relevant to outcome)
    x4 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    x5 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    #covariates (irrelevant)
    x6 = np.random.normal(loc = 0.0, scale = 1.0, size = (n,1))
    #error term
    e = np.random.normal(loc = 0.0, scale = var_e, size = (n,1))

    #indicator functions
    ind1 = np.zeros((n,1))
    ind2 = np.zeros((n,1))
    ind3 = np.zeros((n,1))
    ind1[np.where(x1 >= 0)] = 1
    ind2[np.where(x2 >= 0)] = 1
    ind3[np.where(x3 >= 0)] = 1

    y = np.add(np.add(np.add(np.add(d*-5, np.multiply(np.add(np.add(6*ind1, 2.5*ind2),1.5*ind3),d)),x4),x5),e)
    X = np.concatenate((x1,x2,x3,x4,x5,x6), axis=1)
    return X, y, d