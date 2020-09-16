import susi
from susi.SOMPlots import *

import pandas as pd
import numpy as np
np.random.seed(12)

from sklearn.preprocessing import StandardScaler


class SOM_ClusterModel():

    def __init__(self, n_rows, n_columns, targets, seed):

        self.n_rows_ = n_rows
        self.n_columns_ = n_columns
        self.targets_ = targets
        self.seed_ = seed
        self.som = susi.SOMClustering(n_rows=n_rows, n_columns=n_columns, random_state=seed)
        self.training = np.concatenate((np.repeat(targets[0], 1000, axis=0), np.repeat(targets[1], 1000, axis=0)), axis=0)
        self.w1_ = None
        self.w2_ = None
        self.bmu1_ = None
        self.bmu2_ = None
        self.som_array_ = None

    def fit(self):
        self.som.fit(self.training)
        print('SOM fitted!')

    def getTargetsWeights(self):


        try:
            self.som_array_ = self.som.unsuper_som_
            self.bmu1_ = self.som.get_bmu(self.targets_[0], self.som_array_)
            self.bmu2_ = self.som.get_bmu(self.targets_[1], self.som_array_)
            self.w1_ = self.som_array_[self.bmu1_[0], self.bmu1_[1], :]
            self.w2_ = self.som_array_[self.bmu2_[0], self.bmu2_[1], :]
        except:
            raise Exception(f'fitted: {self.som.fitted_}')





    def getUtility(self, s):
        """Compute an utility for the node s in according to the distance on the SOM
        between its weights neuron prediction and the weights of the two trained cluster."""

        self.getTargetsWeights()
        prediction = self.som.transform(s)
        ws = self.som_array_[prediction[0][0], prediction[0][1], :]  # weight of the neuron which node s is mapped in the SOM
        d1 = np.linalg.norm(ws - self.w1_)
        d2 = np.linalg.norm(ws - self.w2_)
        if d1 > d2:
            # associated to the cluster for which the utility \in [1, 2]
            u = 1 + np.exp(-d2)
            return u
        else:
            # associated to the cluster for which the utility \in [0, 1]
            u = 1 - np.exp(-d1)
            return u


if __name__ == '__main__':

    x1 = np.zeros([1, 7])
    x2 = np.ones([1, 7])
    targets = (x1, x2)
    som = SOM_ClusterModel(n_rows=30, n_columns=30, targets=targets, seed=12)
    som.fit()

    s = np.random.random((1, 7))
    u = som.getUtility(s)

    df = pd.read_csv('txsData/txs_small.csv', header=0)

    X = df.drop(['id', 'timestamp', 'usdAvg', 'usd_in', 'usd_out', 'date', 'time'], axis=1).to_numpy()

    X = StandardScaler().fit_transform(X)
    U = [som.getUtility(s.reshape((1,7))) for s in X]





