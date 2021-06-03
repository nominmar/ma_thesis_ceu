import pandas as pd
import numpy as np
from random import random

from CTL2.causal_tree_learn import CausalTree
from DGP.DGP import dgp2
from DGP.DGP import dgp4
from DGP.DGP import dgp8

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse


def mc_ate2(n, tsize, var_e, reps, est_size):
    ATE1 = np.ones(reps)*1.5
    ATE2 = np.ones(reps)*-1.5
    #predicted ATE
    ate1 = []
    ate2 = []
    tauhat = np.zeros((tsize, 1))
    #test sample
    x_test, y_test, treat_test = dgp2(tsize,var_e)
    #MC iterations
    for i in range(reps):
        #train + est sample
        x_train, y_train, treat_train = dgp2(n, var_e)
        #true CATE
        tau = np.where((x_test[:,0] >= 0),1.5, -1.5)
        #fit the tree
        ctl = CausalTree(honest=True, weight=0.0, split_size=0.0, max_depth = 3) #which type of tree to call
        ctl.fit(x_train, y_train, treat_train, est_size = est_size) #select est size when fitting
        ctl.prune()
        ctl_predict = ctl.predict(x_test)
        #predicted ATE
        ate1 = np.append(ate1, np.mean(ctl_predict[np.where(x_test[:,0] >= 0)]))
        ate2 = np.append(ate2, np.mean(ctl_predict[np.where(x_test[:,0] < 0)]))
        #predicted ITE
        tauhat = np.append(tauhat, ctl_predict.reshape(-1,1), axis = 1)
    #CATE MSE's
    mean_ate = np.array([np.mean(ate1), np.mean(ate2)])    
    mse_ate = np.array([mse(ATE1, ate1), mse(ATE2, ate2)])
    bias_ate = np.array([np.sum(np.subtract(ATE1,ate1))/reps, np.sum(np.subtract(ATE2,ate2))/reps])
    var_ate = np.array([np.var(ate1), np.var(ate2)])

    #TOTAL MSE's
    ind_var = np.var(tauhat[:, 1:], axis = 1)
    ind_mean = np.mean(tauhat, axis = 1)

    total_var = np.sum(ind_var)/tsize
    total_bias = np.sum(np.square(np.subtract(ind_mean, tau)))/tsize
    total_mse = total_var + total_bias

    results = np.concatenate([np.array([est_size]), mean_ate, bias_ate, var_ate, mse_ate, np.array([total_bias]), np.array([total_var]), np.array([total_mse])])
    return results

def mc_ate4(n, tsize, var_e, reps, est_size):
    #true ATE
    ATE1 = np.ones(reps)*3
    ATE2 = np.ones(reps)*1
    ATE3 = np.ones(reps)*-2
    ATE4 = np.ones(reps)*-3
    #predicted ate
    ate1 = []
    ate2 = []
    ate3 = []
    ate4 = []
    #ITE dummy
    tauhat = np.zeros((tsize, 1))
    #true ITE
    x_test, y_test, treat_test = dgp4(tsize, var_e)
    for i in range(reps):
        x_train, y_train, treat_train = dgp4(n, var_e)
        tau = np.where(((x_test[:,0] >= 0) & (x_test[:,1] >= 0)),3, 
                       np.where(((x_test[:,0] >= 0) & (x_test[:,1] < 0)),1, 
                                np.where(((x_test[:,0] < 0) & (x_test[:,1] >= 0)),-2,-3)))
        ctl = CausalTree(honest=True, weight=0.0, split_size=0.0, max_depth = 3) #which type of tree to call
        ctl.fit(x_train, y_train, treat_train, est_size = est_size) #select est size when fitting
        ctl.prune()
        ctl_predict = ctl.predict(x_test)
        #predicted ATE
        ate1 = np.append(ate1, np.mean(ctl_predict[np.where((x_test[:,0] >= 0) & (x_test[:,1] >= 0))]))
        ate2 = np.append(ate2, np.mean(ctl_predict[np.where((x_test[:,0] >= 0) & (x_test[:,1] < 0))]))
        ate3 = np.append(ate3, np.mean(ctl_predict[np.where((x_test[:,0] < 0) & (x_test[:,1] >= 0))]))
        ate4 = np.append(ate4, np.mean(ctl_predict[np.where((x_test[:,0] < 0) & (x_test[:,1] < 0))]))
        #predicted ITE
        tauhat = np.append(tauhat, ctl_predict.reshape(-1,1), axis = 1)

    mean_ate = np.array([np.mean(ate1), np.mean(ate2), np.mean(ate3), np.mean(ate4)])    
    mse_ate = np.array([mse(ATE1, ate1), mse(ATE2, ate2),mse(ATE3, ate3), mse(ATE4, ate4)])
    bias_ate = np.array([np.sum(np.subtract(ATE1,ate1))/reps, np.sum(np.subtract(ATE2,ate2))/reps, np.sum(np.subtract(ATE3,ate3))/reps, np.sum(np.subtract(ATE4,ate4))/reps])
    var_ate = np.array([np.var(ate1), np.var(ate2), np.var(ate3), np.var(ate4)])

    #TOTAL MSE's
    ind_var = np.var(tauhat[:, 1:], axis = 1)
    ind_mean = np.mean(tauhat, axis = 1)

    total_var = np.sum(ind_var)/tsize
    total_bias = np.sum(np.square(np.subtract(ind_mean, tau)))/tsize
    total_mse = total_var + total_bias

    results = np.concatenate([np.array([est_size]), mean_ate, bias_ate, var_ate, mse_ate, np.array([total_bias]), np.array([total_var]), np.array([total_mse])])
    return results

def mc_ate8(n, tsize, var_e, reps, est_size):
    #true ATE
    ATE1 = np.ones(reps)*5
    ATE2 = np.ones(reps)*-1
    ATE3 = np.ones(reps)*2.5
    ATE4 = np.ones(reps)*-3.5
    ATE5 = np.ones(reps)*3.5
    ATE6 = np.ones(reps)*-2.5
    ATE7 = np.ones(reps)*1
    ATE8 = np.ones(reps)*-5
    #predicted ate
    ate1 = []
    ate2 = []
    ate3 = []
    ate4 = []
    ate5 = []
    ate6 = []
    ate7 = []
    ate8 = []
    #ITE dummy
    tauhat = np.zeros((tsize, 1))
    #true ITE
    x_test, y_test, treat_test = dgp8(tsize, var_e)
    for i in range(reps):
        x_train, y_train, treat_train = dgp8(n, var_e)
        tau = np.where(((x_test[:,0] >= 0) & (x_test[:,1] >= 0) & (x_test[:,2] >= 0)),5, 
                           np.where(((x_test[:,0] < 0) & (x_test[:,1] >= 0) & (x_test[:,2] >= 0)), -1, 
                                    np.where(((x_test[:,0] >= 0) & (x_test[:,1] < 0) & (x_test[:,2] >= 0)), 2.5, 
                                             np.where(((x_test[:,0] < 0) & (x_test[:,1] < 0) & (x_test[:,2] >= 0)), -3.5, 
                                                      np.where(((x_test[:,0] >= 0) & (x_test[:,1] >= 0) & (x_test[:,2] < 0)), 3.5, 
                                                               np.where(((x_test[:,0] < 0) & (x_test[:,1] >= 0) & (x_test[:,2] < 0)), -2.5, 
                                                                        np.where(((x_test[:,0] >= 0) & (x_test[:,1] < 0) & (x_test[:,2] < 0)), 1, -5)))))))
        ctl = CausalTree(honest=True, weight=0.0, split_size=0.0, max_depth = 3) #which type of tree to call
        ctl.fit(x_train, y_train, treat_train, est_size = est_size) #select est size when fitting
        ctl.prune()
        ctl_predict = ctl.predict(x_test)
        #predicted ATE
        ate1 = np.append(ate1, np.mean(ctl_predict[np.where((x_test[:,0] >= 0) & (x_test[:,1] >= 0) & (x_test[:,2] >= 0))]))
        ate2 = np.append(ate2, np.mean(ctl_predict[np.where((x_test[:,0] < 0) & (x_test[:,1] >= 0) & (x_test[:,2] >= 0))]))
        ate3 = np.append(ate3, np.mean(ctl_predict[np.where((x_test[:,0] >= 0) & (x_test[:,1] < 0) & (x_test[:,2] >= 0))]))
        ate4 = np.append(ate4, np.mean(ctl_predict[np.where((x_test[:,0] < 0) & (x_test[:,1] < 0) & (x_test[:,2] >= 0))]))
        ate5 = np.append(ate5, np.mean(ctl_predict[np.where((x_test[:,0] >= 0) & (x_test[:,1] >= 0) & (x_test[:,2] < 0))]))
        ate6 = np.append(ate6, np.mean(ctl_predict[np.where((x_test[:,0] < 0) & (x_test[:,1] >= 0) & (x_test[:,2] < 0))]))
        ate7 = np.append(ate7, np.mean(ctl_predict[np.where((x_test[:,0] >= 0) & (x_test[:,1] < 0) & (x_test[:,2] < 0))]))
        ate8 = np.append(ate8, np.mean(ctl_predict[np.where((x_test[:,0] < 0) & (x_test[:,1] < 0) & (x_test[:,2] < 0))]))
        #predicted ITE
        tauhat = np.append(tauhat, ctl_predict.reshape(-1,1), axis = 1)

    #ATE MSE's
    mean_ate = np.array([np.mean(ate1), np.mean(ate2), np.mean(ate3), np.mean(ate4), np.mean(ate5), np.mean(ate6), np.mean(ate7), np.mean(ate8)])    
    mse_ate = np.array([mse(ATE1, ate1), mse(ATE2, ate2),mse(ATE3, ate3), mse(ATE4, ate4),mse(ATE5, ate5), mse(ATE6, ate6), mse(ATE7, ate7), mse(ATE8, ate8)])
    bias_ate = np.array([np.sum(np.subtract(ATE1,ate1))/reps, np.sum(np.subtract(ATE2,ate2))/reps, np.sum(np.subtract(ATE3,ate3))/reps, np.sum(np.subtract(ATE4,ate4))/reps,
                         np.sum(np.subtract(ATE5,ate5))/reps, np.sum(np.subtract(ATE6,ate6))/reps, np.sum(np.subtract(ATE7,ate7))/reps, np.sum(np.subtract(ATE8,ate8))/reps])
    var_ate = np.array([np.var(ate1), np.var(ate2), np.var(ate3), np.var(ate4), np.var(ate5), np.var(ate6), np.var(ate7), np.var(ate8)])

    #TOTAL MSE's
    ind_var = np.var(tauhat[:, 1:], axis = 1)
    ind_mean = np.mean(tauhat, axis = 1)

    total_var = np.sum(ind_var)/tsize
    total_bias = np.sum(np.square(np.subtract(ind_mean, tau)))/tsize
    total_mse = total_var + total_bias

    results = np.concatenate([np.array([est_size]), mean_ate, bias_ate, var_ate, mse_ate, np.array([total_bias]), np.array([total_var]), np.array([total_mse])])
    return results
