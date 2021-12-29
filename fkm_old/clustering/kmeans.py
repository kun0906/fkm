import collections

import sklearn.cluster
import numpy as np

from fkm.clustering.metrics import metric_scores
from fkm.utils import distance


class KMeans:

    def __init__(self, n_clusters = 5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, clients_train):
        X = []
        y = []
        for client_train in clients_train:
            X.extend(client_train.X)
            y.extend(client_train.y)

        self.km = sklearn.cluster.KMeans(n_clusters= self.n_clusters, random_state= self.random_state)
        self.km.fit(X)


    def test(self, clients_test):
        X = []
        y = []
        for client_test in clients_test:
            X.extend(client_test.X)
            y.extend(client_test.y)

        y_true = []
        y_pred = []
        for _x, _y in zip(X, y):
            _label = np.argmin([distance(_x, c) for c in self.km.cluster_centers_])
            y_true.append(_y)
            y_pred.append(_label)

        # reassign cluster labels
        original_centroids = [(1, 1), (5, 5), (5, 10), (10, 10), (10, 5)]
        print(original_centroids, self.km.cluster_centers_)
        for i, sc in enumerate(self.km.cluster_centers_):
            idx = np.argmin([distance(oc, sc) for oc in original_centroids])
            for j in range(len(y_pred)):
                if y_pred[j] == i:
                    y_pred[j] = str(idx)
        y_pred = [int(v) for v in y_pred]

        self.scores = metric_scores(X, y_pred, y_true)
        print(f'Scores: {self.scores}')

