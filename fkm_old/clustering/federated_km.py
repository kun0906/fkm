import collections
import random

import sklearn.metrics
from sklearn.cluster import KMeans
from sklearn.cluster._k_means_common import row_norms
from sklearn.cluster import kmeans_plusplus
import numpy as np

from fkm.clustering.metrics import metric_scores
from fkm.utils import distance


class ClientParams:
    def __init__(self, centroids, n_points, average_dist):
        self.centroids = centroids
        self.n_points = n_points
        self.average_dist = average_dist


def compute_client_params(X, centroids):
    within_dists = []
    n_points = []
    clusters = [[]] * len(centroids)
    labels = []
    new_centroids = []
    for x in X:
        i = np.argmin([distance(x, c) for c in centroids])
        clusters[i].append((x, distance(x, centroids[i])))
        labels.append(i)

    for i, vs in enumerate(clusters):
        ni = len(vs)
        n_points.append(ni)
        xs, _ = zip(*vs)
        ci = np.mean(xs, axis=0)
        new_centroids.append(ci)
        with_d = sum([distance(x, ci) for x in xs]) / ni
        within_dists.append(with_d)

    return np.asarray(new_centroids), np.asarray(labels), np.asarray(n_points), np.asarray(within_dists)


class FKM:

    def __init__(self, n_clusters=5, n_rounds=100, frac_of_clients= 0.9, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.alpha = 10e-3
        self.beta = 103-3
        self.frac_of_clients = frac_of_clients
        self.n_rounds = n_rounds
        self.n_client_round = 5
        self.tol = 10e-4

    def client_execute(self, i_round, X, server_centroids):
        if i_round == 0:
            # precompute squared norms of data points
            x_squared_norms = row_norms(X, squared=True)
            _client_centroids, _ = kmeans_plusplus(X, n_clusters=self.n_clusters,
                                                   x_squared_norms=x_squared_norms,
                                                   random_state=self.random_state, n_local_trials=None)
            # assign X to its corresponding cluster and compute_client_params.
            new_client_centroids, labels, n_points, within_dists = compute_client_params(X, _client_centroids)
        else:
            # assign X to its corresponding cluster and compute_client_params.
            # new_client_centroids, labels, n_points, within_dists = compute_client_params(X, server_centroids)
            _server_centroids = np.copy(server_centroids)
            for _ in range(self.n_client_round):
                new_client_centroids, labels, n_points, within_dists = compute_client_params(X, _server_centroids)
                _server_centroids = _server_centroids + self.beta * (new_client_centroids - _server_centroids)
            # print(new_client_centroids-server_centroids)

        return new_client_centroids, labels, n_points, within_dists

    def fit(self, clients_train):
        n_clients = len(clients_train)
        server_centroids = None
        for i_round in range(self.n_rounds):
            clients_centroids = []
            clients_n_points = []
            clients_within_dists = []
            ####################################################################################################
            # For clients
            # for i in range(n_clients):    # use all clients
            # default is uniform sample from the given list
            indices_clients = np.random.choice(range(n_clients), size=int(n_clients * self.frac_of_clients), replace=False)
            print(f'Indices of clients who participate the {i_round}th round training: {indices_clients}')
            for i in indices_clients:
                # each client obtains K centroids of its data using K-Means++
                X = clients_train[i].X
                _, dim = X.shape
                new_client_centroids, labels, n_points, within_dists = self.client_execute(i_round, X, server_centroids)
                clients_centroids.append(new_client_centroids)
                clients_n_points.append(n_points)
                clients_within_dists.append(within_dists)
                # data = zip((_client_centroids, n_points, within_dists))
            # upload to the server
            clients_params = (np.asarray(clients_centroids), clients_n_points, clients_within_dists)

            ####################################################################################################
            # For the sever
            # upload clients' params (KxM, i.e., n_clusters*n_clients) to a server
            # and the sever will aggregate all the clients' params
            # Compute the difference between old sever_centroids and server_centroids_t (obtained from clients'params)
            if i_round == 0:
                # precompute squared norms of data points
                KM = clients_params[0].reshape(-1, dim) # K * M centroids in total
                x_squared_norms = row_norms(KM, squared=True)
                server_centroids_t, _ = kmeans_plusplus(KM, n_clusters=self.n_clusters,
                                                        x_squared_norms=x_squared_norms,
                                                        random_state=self.random_state, n_local_trials=None)
                new_server_centroids = server_centroids_t
            else:
                lambda_t = np.sum(np.asarray(clients_n_points), axis=0)
                server_centroids_ = []
                for _i in range(len(clients_n_points)):
                    a = np.repeat(clients_n_points[_i][:, np.newaxis], dim, axis=1)
                    b = clients_centroids[_i]
                    server_centroids_.append(a*b)   # element_wise product of two matrices
                server_centroids_ = np.repeat((1/lambda_t)[:, np.newaxis], dim, axis=1) * np.sum(np.asarray(server_centroids_), axis=0)
                # new_server_centroids = current_server_centorids + learning_rate * diff
                new_server_centroids = server_centroids + self.alpha * (server_centroids_ - server_centroids)
                # check if the model is convergent
                diff2 = sum(distance(a, b) for a, b in zip(new_server_centroids, server_centroids))
                print(i_round, diff2, server_centroids)
                if diff2 < self.tol:
                    print(f'Training phase stops in advance because the model is convergent at {i_round}th round.')
                    break
            # in the next round, all the clients will the new_centroids.
            server_centroids = new_server_centroids

        # get the final server's centroids
        self.server_centroids = new_server_centroids

    def test(self, clients_test):

        labels = [[] for _ in range(len(clients_test))]
        y_true = []
        y_pred = []
        X = []
        for i_client, client_test in enumerate(clients_test):
            Xi, yi = client_test.X, client_test.y
            ci = client_test.centroids
            n_points = client_test.n_points
            for _x, _y in zip(Xi, yi):
                _label = np.argmin([distance(_x, c) for c in self.server_centroids])
                labels[i_client].append((_y, _label))
                y_true.append(_y)
                y_pred.append(_label)
                X.append(_x)

        # reassign cluster labels
        original_centroids = [(1,1), (5,5), (5, 10), (10, 10), (10,5)]
        print(original_centroids, self.server_centroids)
        for i, sc in enumerate(self.server_centroids):
            idx = np.argmin([distance(oc, sc) for oc in original_centroids])
            for j in range(len(y_pred)):
                if y_pred[j] == i:
                    y_pred[j] = str(idx)
        y_pred = [int(v) for v in y_pred]

        # n_points = []
        # centroids = []
        # for vs in labels:
        #     n_points.append(len(vs))

        self.scores = metric_scores(X, y_pred, y_true)
        print(f'Scores: {self.scores}')




