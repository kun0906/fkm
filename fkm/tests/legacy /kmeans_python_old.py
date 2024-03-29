import numpy as np
import random

from sklearn.cluster import kmeans_plusplus

from fkm.cluster._greedy_init import greedily_initialize
from utils_stats import plot_progress


def randomly_init_centroid(min_value, max_value, n_dims, repeats=1):
    if repeats == 1:
        return min_value + (max_value - min_value) * np.random.rand(n_dims) # uniform [0, 1)
    else:
        return min_value + (max_value - min_value) * np.random.rand(repeats, n_dims)


class KMeans:
    def __init__(
        self,
        n_clusters,
        init_centroids='random',
        max_iter=100,
        tol=0.0001,
        distance_metric='euclidean',
        seed=None,
        reassign_min=None,
        reassign_after=None,
        verbose=False,
    ):
        self.n_clusters = n_clusters
        self.seed = seed
        self.init_centroids = init_centroids
        self.max_iter = max_iter
        self.tol = tol
        self.distance_metric = distance_metric
        if distance_metric != 'euclidean':
            raise NotImplementedError
        self.verbose = verbose
        self.reassign_min = reassign_min
        self.reassign_after = reassign_after

    def do_init_centroids(self):
        if isinstance(self.init_centroids, str):
            if self.init_centroids == 'random':
                # # assumes data is in range 0-1
                # centroids = np.random.rand(self.n_clusters, self.n_dims)
                # for dummy data
                centroids = randomly_init_centroid(0, self.n_clusters+1, self.n_dims, self.n_clusters)
            else:
                raise NotImplementedError
        elif self.init_centroids.shape == (self.n_clusters, self.n_dims):
            centroids = self.init_centroids
        else:
            raise NotImplementedError
        return centroids

    def fit(self, X, record_at=None):
        x = X
        self.n_dims = x.shape[1]
        # centroids = self.do_init_centroids()
        centroids, _ = kmeans_plusplus(x, n_clusters=self.n_clusters, random_state=self.seed)
        # print(X, centroids)
        means_record = []
        stds_record = []
        to_reassign = np.zeros(self.n_clusters)
        for iteration in range(1, 1+self.max_iter):
            # compute distances
            # computationally efficient
            # differences = np.expand_dims(x, axis=1) - np.expand_dims(centroids, axis=0)
            # sq_dist = np.sum(np.square(differences), axis=2)
            # memory efficient
            sq_dist = np.zeros((x.shape[0], self.n_clusters))
            for i in range(self.n_clusters): # each x to each centroid square distance
                sq_dist[:, i] = np.sum(np.square(x - centroids[i, :]), axis=1)

            labels = np.argmin(sq_dist, axis=1)
            # update centroids
            centroid_updates = np.zeros((self.n_clusters, self.n_dims))

            for i in range(self.n_clusters):
                mask = np.equal(labels, i)
                size = np.sum(mask) # get the number of points that are assigned to cluster i
                if size > 0:
                    update = np.sum(x[mask] - centroids[i], axis=0) # get the sum distance (no square distance) between new xs to old centroid i.
                    centroid_updates[i, :] = update / size  # average distance
                if self.reassign_min is not None:
                    if size < x.shape[0] * self.reassign_min:
                        to_reassign[i] += 1
                    else:
                        to_reassign[i] = 0

            centroids = centroids + centroid_updates  #
            changed = np.any(np.absolute(centroid_updates) > self.tol)

            for i, num_no_change in enumerate(to_reassign):
                if num_no_change >= self.reassign_after:
                    centroids[i] = randomly_init_centroid(0, self.n_clusters+1, self.n_dims, 1)
                    to_reassign[i] = 0
                    changed = True

            if record_at is not None and iteration in record_at:
                means, stds = record_state(centroids, x)
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
        return centroids, labels

    def predict(self, x):
        # memory efficient
        sq_dist = np.zeros((x.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            sq_dist[:, i] = np.sum(np.square(x - self.cluster_centers_[i, :]), axis=1)
        labels = np.argmin(sq_dist, axis=1)
        return labels


def init_kmeans_python(n_clusters, init_centroids='random', batch_size=None, seed=None, iterations=100, verbose=False):
    # init kmeanscluster_centers_
    if batch_size is not None:
        raise NotImplementedError
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            init_centroids=init_centroids,
            seed=seed,
            max_iter=iterations,
            verbose=verbose,
        )
    return kmeans


def compute_step_for_client(client_data, centroids):
    # compute distances
    # computationally efficient
    differences = np.expand_dims(client_data, axis=1) - np.expand_dims(centroids, axis=0)
    sq_dist = np.sum(np.square(differences), axis=2)

    # memory efficient
    # sq_dist = np.zeros((client_data.shape[0], self.n_clusters))
    # for i in range(self.n_clusters):
    #     sq_dist[:, i] = np.sum(np.square(client_data - centroids[i, :]), axis=1)

    # assign to cluster
    labels = np.argmin(sq_dist, axis=1)

    # update centroids
    centroid_updates = np.zeros_like(centroids)
    counts = np.zeros(centroids.shape[0])
    for i in range(centroids.shape[0]):
        mask = np.equal(labels, i)
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            centroid_updates[i, :] = np.sum(client_data[mask] - centroids[i], axis=0)
    return centroid_updates, counts

def compute_client_params(X, centroids):
    differences = np.expand_dims(X, axis=1) - np.expand_dims(centroids, axis=0)
    sq_dist = np.sum(np.square(differences), axis=2)

    # memory efficient
    # sq_dist = np.zeros((client_data.shape[0], self.n_clusters))
    # for i in range(self.n_clusters):
    #     sq_dist[:, i] = np.sum(np.square(client_data - centroids[i, :]), axis=1)

    # assign to cluster
    labels = np.argmin(sq_dist, axis=1)
    cluster_avg_dists = []
    cluster_sizes = []
    for i in range(centroids.shape[0]):
        mask = np.equal(labels, i)
        ni = np.sum(mask)
        if ni:
            avg_dist = np.sqrt(np.sum(np.square(X[mask] - centroids[i]))) / ni
        else:
            avg_dist = 0.0
        cluster_sizes.append(ni)
        cluster_avg_dists.append(avg_dist)

    return centroids, cluster_sizes, cluster_avg_dists

def update_server_centroids(KM_centroids, KM_ns, KM_ds, centroids, alpha):
    differences = np.expand_dims(KM_centroids, axis=1) - np.expand_dims(centroids, axis=0)
    sq_dist = np.sum(np.square(differences), axis=2)

    # memory efficient
    # sq_dist = np.zeros((client_data.shape[0], self.n_clusters))
    # for i in range(self.n_clusters):
    #     sq_dist[:, i] = np.sum(np.square(client_data - centroids[i, :]), axis=1)

    # assign to cluster
    labels = np.argmin(sq_dist, axis=1)
    n_clusters, n_dim = centroids.shape
    new_centroids = np.zeros((n_clusters, n_dim))
    # for i in range(centroids.shape[0]):
    #     mask = np.equal(labels, i)
    #     ni = np.sum(mask)
    #     if ni:
    #         diff = np.sum(KM_centroids[mask]-centroids[i]) / ni
    #     else:
    #         diff = 0
    #     new_centroids[i] = centroids[i] + alpha * diff

    # weighted update
    for i in range(centroids.shape[0]):
        mask = np.equal(labels, i)
        wi = KM_ns[mask]
        # ni = np.sum(mask)
        ni = np.sum(wi)
        if ni:
            diff = np.sum(wi[:, np.newaxis]*(KM_centroids[mask] - centroids[i]), axis=0) / ni
        else:
            diff = 0
        new_centroids[i] = centroids[i] + alpha * diff


    return new_centroids

class KMeansFederated(KMeans):
    def __init__(
            self,
            n_clusters,
            init_centroids='random',
            max_iter=100,
            tol=0.0001,
            distance_metric='euclidean',
            seed=None,
            reassign_min=None,
            reassign_after=None,
            verbose=False,
            batch_size=None,
            sample_fraction=1.0,
            epochs_per_round=1,
            learning_rate=None,
            max_no_change=None,
            adaptive_lr=None,
            momentum=None,
            epoch_lr=1,
    ):
        super().__init__(
            n_clusters=n_clusters,
            init_centroids=init_centroids,
            max_iter=max_iter,
            tol=tol,
            distance_metric=distance_metric,
            seed=seed,
            reassign_min=reassign_min,
            reassign_after=reassign_after,
            verbose=verbose
        )
        self.use_client_init_centroids = True 
        self.batch_size = batch_size
        self.sample_fraction = sample_fraction
        self.epochs = epochs_per_round
        self.lr = learning_rate
        self.adaptive_lr = adaptive_lr
        self.max_no_change = max_no_change
        self.momentum_rate = momentum
        self.epoch_lr = epoch_lr

    def do_federated_round_single_step(self, clients_in_round, centroids):
        # print(len(clients_in_round))
        # print(clients_in_round[0].shape)
        updates_sum = np.zeros((self.n_clusters, self.n_dims))
        counts = np.zeros(self.n_clusters)
        for client_data in clients_in_round:
            client_updates_sum, client_counts = compute_step_for_client(
                client_data=client_data,
                centroids=centroids
            )
            # if self.epoch_lr is not None:
            #     client_updates_sum = self.epoch_lr * client_updates_sum
            updates_sum += client_updates_sum
            counts += client_counts
            if self.verbose:
                print("client_counts: {}; client_updates: {}".format(client_counts, client_updates_sum))
        return updates_sum, counts

    def do_federated_round(self, clients_in_round, centroids):
        updates_sum = np.zeros((self.n_clusters, self.n_dims))
        counts = np.zeros(self.n_clusters)
        KM_params = {'centroids': [], 'cluster_sizes': [], 'avg_dists': []}
        for client_data in clients_in_round:
            if self.use_client_init_centroids: # added by kun
                # Calculate seeds from kmeans++
                client_centroids, indices = kmeans_plusplus(client_data, n_clusters=self.n_clusters, random_state=self.seed)
                centroids = client_centroids
            else:
                client_centroids = centroids
            for e in range(self.epochs):
                client_updates_sum, client_counts = compute_step_for_client(
                    client_data=client_data,
                    centroids=client_centroids
                )
                interim_updates = client_updates_sum / np.expand_dims(np.maximum(client_counts, np.ones_like(client_counts)), axis=1)
                if self.epoch_lr is not None:
                    interim_updates = self.epoch_lr * interim_updates
                client_centroids = client_centroids + interim_updates
            updates_sum += (client_centroids - centroids) * np.expand_dims(client_counts, axis=1)
            counts += client_counts
            # compute client's params (such as centroids, cluster_sizes, and avg_dists)
            client_centroids, client_cluster_sizes, client_avg_dists = compute_client_params(client_data, client_centroids)
            KM_params['centroids'].extend(list(client_centroids))
            KM_params['cluster_sizes'].extend(list(client_cluster_sizes))
            KM_params['avg_dists'].extend(list(client_avg_dists))
            if self.verbose:
                print("client_counts: {}; client_updates_sum: {}".format(client_counts, client_updates_sum))
        self.use_client_init_centroids = False

        KM_params['centroids'] = np.asarray(KM_params['centroids'])
        KM_params['cluster_sizes'] = np.asarray(KM_params['cluster_sizes'])
        KM_params['avg_dists'] = np.asarray(KM_params['avg_dists'])

        return updates_sum, counts, KM_params

    def fit(self, X, record_at=None):
        x = X
        self.num_clients = len(x)
        self.n_dims = x[0].shape[1]
        clients_per_round = max(1, int(self.sample_fraction * self.num_clients))
        # centroids = self.do_init_centroids()
        centroids = np.zeros((self.n_clusters, self.n_dims))

        not_changed = 0
        overall_counts = np.zeros(self.n_clusters)
        momentum = np.zeros_like(centroids)
        means_record = []
        stds_record = []
        to_reassign = np.zeros(self.n_clusters)

        # while changed and round < self.max_iter:
        for iteration in range(1, 1+self.max_iter):
            clients_in_round = random.sample(x, clients_per_round)
            if self.verbose:
                print("round: {}".format(iteration))

            # updates_sum, counts = self.do_federated_round_single_step(
            #     clients_in_round=clients_in_round,
            #     centroids=centroids,
            # )
            updates_sum, counts, KM_params = self.do_federated_round(
                clients_in_round=clients_in_round,
                centroids=centroids,
            )
            if iteration == 1:  # for the first round, the server will select K centroids from K*M client's centroids.
                KM_centroids, KM_ns, KM_ds = KM_params['centroids'], KM_params['cluster_sizes'], KM_params['avg_dists']
                # 1. ignore n_points per each centroid
                # centroids, indices = kmeans_plusplus(KM_centroids, n_clusters=self.n_clusters, random_state=self.seed)

                # 2. consider about cluster size: can't do this in the following way because the size of new_KM_centroids will be all clients'data
                # new_KM_centroids = []
                # for c, s in zip(KM_centroids, centroid_size):
                #     new_KM_centroids.extend([c] * int(s))
                # KM_centroids = np.asarray(new_KM_centroids)
                # centroids, indices = kmeans_plusplus(KM_centroids, n_clusters=self.n_clusters, random_state=self.seed)

                # 3. weighted KMeans++
                # centroids, indices = weighted_kmeans_plusplus(KM_centroids, n_clusters=self.n_clusters, random_state=self.seed)

                # 4. Greedy initialization
                centroids, indices = greedily_initialize(KM_centroids, KM_ns, KM_ds, self.n_clusters)
                continue
            # update server's params
            # overall_counts += counts
            # updates = updates_sum / np.expand_dims(np.maximum(counts, np.ones_like(counts)), axis=1)
            #
            # if self.adaptive_lr:
            #     rel_counts = counts / np.maximum(overall_counts, np.ones_like(overall_counts))
            #     update_weights = np.minimum(self.adaptive_lr, rel_counts)
            #     updates = updates * np.expand_dims(update_weights, axis=1)
            #
            # if self.lr is not None:
            #     updates = self.lr * updates
            #
            # if self.momentum_rate is not None:
            #     momentum = self.momentum_rate * momentum + (1 - self.momentum_rate) * updates
            #     updates = momentum
            # centroids = centroids + updates

            centroids = update_server_centroids(KM_centroids, KM_ns, KM_ds, centroids, alpha=self.lr)
            # changed = np.any(np.absolute(updates) > self.tol)

            if self.reassign_min is not None:
                for i in range(self.n_clusters):
                    if counts[i] < (sum(counts) * self.reassign_min):
                        to_reassign[i] += 1
                    else:
                        to_reassign[i] = 0
                    if to_reassign[i] >= self.reassign_after:
                        centroids[i] = randomly_init_centroid(0, self.n_clusters + 1, self.n_dims, 1)
                        momentum[i] = np.zeros(self.n_dims)
                        to_reassign[i] = 0
                        changed = True

            if self.max_no_change is not None:
                not_changed += 1
                if changed:
                    not_changed = 0
                if not_changed > self.max_no_change:
                    break

            if record_at is not None and iteration in record_at:
                means, stds = record_state(centroids, np.concatenate(x, axis=0))
                means_record.append(means)
                stds_record.append(stds)
        if record_at is not None:
            #  NOTE: only for dummy data
            plot_progress(means_record, stds_record, record_at)

        self.cluster_centers_ = centroids
        return centroids, overall_counts


def record_state(centroids, x):
    # note: assumes 1D data!!
    assert centroids.shape[1] == 1
    differences = np.expand_dims(x, axis=1) - np.expand_dims(centroids, axis=0)
    sq_dist = np.sum(np.square(differences), axis=2)
    labels = np.argmin(sq_dist, axis=1)
    stds = np.zeros(centroids.shape[0])
    for i in range(centroids.shape[0]):
        mask = np.equal(labels, i)
        counts = np.sum(mask)
        if counts > 0:
            stds[i] = np.std(x[mask])
    return centroids[:, 0], stds


def test_kmeans_python():
    x = np.array([[0.1, 0.2], [0.1, 0.4], [0.1, 0.6], [1.0, 0.2], [1.0, 0.1], [1.0, 0.0]])
    # kmeans = KMeans(n_clusters=3)
    kmeans = init_kmeans_python(n_clusters=3)
    centroids, labels = kmeans.fit(X=x)
    # print(x)
    # print(x.shape)
    print(kmeans.labels_)
    print(kmeans.predict([[0, 0], [1.2, 0.3]]))
    print(kmeans.cluster_centers_)


def test_federated(unbalanced=True):
    if unbalanced:
        x = [
            np.array([[0.1, 0.2], [0.1, 0.4]]),
            np.array([[0.1, 0.6]]),
            np.array([[1.0, 0.2], [1.0, 0.1]]),
            np.array([[1.0, 0.0]])
        ]
    else:
        x = [
            np.array([[0.1, 0.2]]),
            np.array([[0.1, 0.4]]),
            np.array([[0.1, 0.6]]),
            np.array([[1.0, 0.2]]),
            np.array([[1.0, 0.1]]),
            np.array([[1.0, 0.0]])
        ]
        # x = [np.array([[0.1, 0.2], [0.1, 0.4], [0.1, 0.6], [1.0, 0.2], [1.0, 0.1], [1.0, 0.0]])]
    print([d.shape for d in x])

    kmeans = KMeansFederated(
        n_clusters=2,
        sample_fraction=0,
        verbose=True,
        learning_rate=5,
        adaptive_lr=0.1,
        max_iter=100,
        # momentum=0.8,
    )

    centroids, overall_counts = kmeans.fit(X=x)

    print(kmeans.predict(np.array([[0, 0], [1.2, 0.3]])))
    print(kmeans.cluster_centers_)


if __name__ == "__main__":
    # test_kmeans_python()
    test_federated(
        unbalanced=False
    )
