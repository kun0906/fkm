

from pprint import pprint

import numpy as np
from sklearn.cluster import MiniBatchKMeans

class KMeans(MiniBatchKMeans):
    def __init__(
            self,
            n_clusters,
            # init_centroids='random',
            server_init_method='random',
            client_init_method=None,
            true_centroids=None,
            max_iter=300,
            tol=1e-4,
            distance_metric='euclidean',
            reassign_min=None,
            reassign_after=None,
            verbose=False,
            random_state=42,
            sample_fraction=0.5,
            epochs_per_round = 0,
            learning_rate = 0,
            adaptive_lr = 0,
            epoch_lr = 0,
            momentum = 0,
            batch_size = 1024,
            params = {},
    ):
        self.n_clusters = n_clusters
        self.server_init_method = server_init_method
        self.client_init_method = client_init_method
        self.init_centroids = server_init_method
        self.true_centroids = true_centroids
        self.max_iter = max_iter
        self.tol = tol
        self.distance_metric = distance_metric
        if distance_metric != 'euclidean':
            raise NotImplementedError
        self.verbose = verbose
        self.reassign_min = reassign_min
        self.reassign_after = reassign_after
        self.random_state = random_state
        self.params = params
        self.batch_size = batch_size

    def fit(self, X_dict, y_dict, splits, record_at=None):
        self.is_train_finished = False
        X = X_dict['train']
        self.history = []
        mbk = MiniBatchKMeans(
            init="k-means++",
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            n_init=10,
            max_no_improvement=10,
            verbose=1,
        )
        mbk.fit(X)

        # TODO: The rest of them is for debugging.
        self.initial_centroids = mbk.cluster_centers_
        self.centroids = mbk.cluster_centers_
        iteration = 0
        scores = 0
        self.training_iterations = 0

        centroids_diff = {}
        for split in splits:
            self.true_centroids[split] = self.centroids
            centroids_diff[split] = self.centroids - self.true_centroids[split]
        centroids_update = self.centroids
        self.history.append({'iteration': iteration, 'centroids': self.centroids, 'scores': scores,
                             'centroids_update': centroids_update, 'centroids_diff': centroids_diff})
        return self

    def predict(self, x):
        # before predicting, check if you already preprocessed x (e.g., std).
        # memory efficient
        sq_dist = np.zeros((x.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            sq_dist[:, i] = np.sum(np.square(x - self.centroids[i, :]), axis=1)
        labels = np.argmin(sq_dist, axis=1)
        return labels

