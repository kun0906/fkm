import numpy as np
from sklearn.cluster import kmeans_plusplus

from federated_kmeans_kmeans_pp.clustering.utils_func import randomly_init_centroid, record_state
from federated_kmeans_kmeans_pp.clustering.utils_stats import plot_progress


class KMeans:
    def __init__(
            self,
            n_clusters,
            init_centroids='random',
            max_iter=300,
            tol=1e-4,
            distance_metric='euclidean',
            reassign_min=None,
            reassign_after=None,
            verbose=False,
            random_state=42,
    ):
        self.n_clusters = n_clusters
        self.init_centroids = init_centroids
        self.max_iter = max_iter
        self.tol = tol
        self.distance_metric = distance_metric
        if distance_metric != 'euclidean':
            raise NotImplementedError
        self.verbose = verbose
        self.reassign_min = reassign_min
        self.reassign_after = reassign_after
        self.random_state = random_state

    def do_init_centroids(self, X=None):
        if isinstance(self.init_centroids, str):
            if self.init_centroids == 'random':
                # # assumes data is in range 0-1
                # centroids = np.random.rand(self.n_clusters, self.dim)
                # for dummy data
                centroids = randomly_init_centroid(0, self.n_clusters + 1, self.dim, self.n_clusters, self.random_state)
            elif self.init_centroids == 'kmeans++':
                centroids, _ = kmeans_plusplus(X, n_clusters=self.n_clusters, random_state=self.random_state)
            else:
                raise NotImplementedError
        elif self.init_centroids.shape == (self.n_clusters, self.dim):
            centroids = self.init_centroids
        else:
            raise NotImplementedError
        return centroids

    def fit(self, X, record_at=None):
        self.n_points, self.dim = X.shape
        centroids = self.do_init_centroids(X)
        self.initial_centroids = centroids
        print(f'initialization method: {self.init_centroids}, centers: {centroids}')
        # if in 5 consecutive times, the difference between new and old centroids is less than self.tol,
        # then the model converges and we stop the training.
        self.n_consecutive = 0
        means_record = []
        stds_record = []
        to_reassign = np.zeros(self.n_clusters)
        self.training_iterations = self.max_iter
        for iteration in range(1, 1 + self.max_iter):
            # compute distances
            # computationally efficient
            # differences = np.expand_dims(x, axis=1) - np.expand_dims(centroids, axis=0)
            # sq_dist = np.sum(np.square(differences), axis=2)
            # memory efficient
            sq_dist = np.zeros((self.n_points, self.n_clusters))
            for i in range(self.n_clusters):
                sq_dist[:, i] = np.sum(np.square(X - centroids[i, :]), axis=1)

            labels = np.argmin(sq_dist, axis=1)
            # update centroids
            centroid_updates = np.zeros((self.n_clusters, self.dim))
            for i in range(self.n_clusters):
                mask = np.equal(labels, i)
                size = np.sum(mask)
                if size > 0:
                    update = np.sum(X[mask] - centroids[i], axis=0)
                    centroid_updates[i, :] = update / size
                if self.reassign_min is not None:
                    if size < X.shape[0] * self.reassign_min:
                        to_reassign[i] += 1
                    else:
                        to_reassign[i] = 0

            # np.sum(np.square(centroids - (centroids + centroid_updates)), axis=1)
            print(iteration, centroid_updates, centroids)
            if np.sum(np.square(centroid_updates)) < self.tol:
                if self.n_consecutive >= 5:
                    self.training_iterations = iteration
                    # training finishes in advance
                    break
                else:
                    self.n_consecutive += 1
            else:
                self.n_consecutive = 0

            centroids = centroids + centroid_updates

            changed = np.any(np.absolute(centroid_updates) > self.tol)
            for i, num_no_change in enumerate(to_reassign):
                if num_no_change >= self.reassign_after:
                    centroids[i] = randomly_init_centroid(0, self.n_clusters + 1, self.dim, 1, self.random_state)
                    to_reassign[i] = 0
                    changed = True

            if record_at is not None and iteration in record_at:
                means, stds = record_state(centroids, X)
                means_record.append(means)
                stds_record.append(stds)
            # if not changed:
            #     break

        if record_at is not None:
            #  NOTE: only for dummy data
            plot_progress(means_record, stds_record, record_at)

        # print(sq_dist.shape)
        # print(labels.shape)
        # print(centroids.shape)
        self.cluster_centers_ = centroids
        self.labels_ = labels

        print(f'Training result: centers: {centroids}')
        return centroids, labels

    def predict(self, x):
        # memory efficient
        sq_dist = np.zeros((x.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            sq_dist[:, i] = np.sum(np.square(x - self.cluster_centers_[i, :]), axis=1)
        labels = np.argmin(sq_dist, axis=1)
        return labels
