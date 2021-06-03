import pandas as pd
from CTL2.causal_tree_learn import CausalTree
from DGP.dgp2 import dgp2
from DGP.dgp1 import dgp1
from DGP.dgp3 import dgp3
from sklearn.model_selection import train_test_split
import numpy as np
from random import random

from sklearn.metrics import mean_squared_error as mse


def MC_MSE_1(x_train, x_test, y_train, y_test, treat_train, treat_test, nomin_test):

    # adaptive CT (Athey and Imbens, PNAS 2016)
    ctl = CausalTree(honest=True, weight=0.0, split_size=0.0) #which type of tree to call
    ctl.fit(x_train, y_train, treat_train, nomin_test = nomin_test) #select est size when fitting
    ctl.prune()
    ctl_predict = ctl.predict(x_test)
    x1 = x_test[:,0]
    ate1 = np.mean(ctl_predict[np.where(x1 > 0)])
    ate0 = np.mean(ctl_predict[np.where(x1 < 0)])
    
    return ate1, ate0, ctl_predict