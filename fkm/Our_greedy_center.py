"""
    Our proposal:
        1. For clients, we randomly select initialized centroids or use Kmeans++ to select initialized centroids.
        2. For server, we propose a greedy way to obtain the initialized centroids.

        # PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 Our_greedy_initialization.py -n '00' > a.txt &
     PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 Our_greedy_initialization.py --dataset 'FEMNIST' \
                --data_details '1client_multiwriters_1digit' --algorithm 'Centralized_random'
"""
# Email: Kun.bj@outlook.com
import argparse
import traceback

import numpy as np
from sklearn.cluster import kmeans_plusplus

from fkm import _main
from fkm.clustering.greedy_initialization import greedily_initialize, distance_sq
from fkm.clustering.my_kmeans import KMeans
from fkm.experiment_cases import get_experiment_params
from fkm.utils.utils_func import random_initialize_centroids
from fkm.utils.utils_stats import evaluate2, plot_progress


def compute_step_for_client(client_data, centroids):
    # compute distances
    # computationally efficient
    differences = np.expand_dims(client_data, axis=1) - np.expand_dims(centroids, axis=0)
    sq_dist = np.sum(np.square(differences), axis=2)  # n x k

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
            # avg_dist = np.sqrt(np.sum(np.square(X[mask] - centroids[i]))) / ni
            avg_dist = np.sum(np.square(X[mask] - centroids[i]))/ ni
        else:
            avg_dist = 0.0
        cluster_sizes.append(ni)
        cluster_avg_dists.append(avg_dist)

    # add the center of input data X
    center = np.mean(X, axis=0).reshape(1, -1)
    centroids = np.concatenate([centroids, center], axis=0)
    cluster_sizes.append(len(X))
    cluster_avg_dists.append(np.sum(np.square(X - center))/ len(X))
    return centroids, cluster_sizes, cluster_avg_dists


def update_server_centroids(KM_centroids, KM_ns, KM_ds, centroids, alpha):
    differences = np.expand_dims(KM_centroids, axis=1) - np.expand_dims(centroids, axis=0)
    sq_dist = np.sum(np.square(differences), axis=2)

    # assign to cluster
    labels = np.argmin(sq_dist, axis=1)
    n_clusters, n_dim = centroids.shape
    new_centroids = np.zeros((n_clusters, n_dim))

    # weighted update
    for i in range(centroids.shape[0]):
        mask = np.equal(labels, i)
        wi = KM_ns[mask]
        # ni = np.sum(mask)
        ni = np.sum(wi)
        if ni:
            diff = np.sum(wi[:, np.newaxis] * (KM_centroids[mask] - centroids[i]), axis=0) / ni
        else:
            diff = 0
        new_centroids[i] = centroids[i] + alpha * diff

    return new_centroids


class KMeansFederated(KMeans):
    def __init__(
            self,
            n_clusters,
            server_init_centroids='greedy',
            client_init_centroids='random',
            true_centroids=None,
            max_iter=300,
            tol=1e-4,
            distance_metric='euclidean',
            random_state=None,
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
            epoch_lr=1.0,
            params=None
    ):
        super().__init__(
            n_clusters=n_clusters,
            init_centroids=None,
            max_iter=max_iter,
            tol=tol,
            distance_metric=distance_metric,
            random_state=random_state,
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
        self.server_init_centroids = server_init_centroids
        self.client_init_centroids = client_init_centroids
        self.true_centroids = true_centroids
        self.random_state = random_state
        self.params = params

    def do_federated_round(self, clients_in_round, centroids, iteration):
        updates_sum = np.zeros((self.n_clusters, self.dim))
        counts = np.zeros(self.n_clusters)
        KM_params = {'centroids': [], 'cluster_sizes': [], 'avg_dists': []}
        client_centroids_avg = np.zeros((self.n_clusters, self.dim))
        for i, client_data in enumerate(clients_in_round):
            if self.use_client_init_centroids:
                # added by kun
                # self.init_centroids = 'kmeans++'
                # self.init_centroids = ''
                if self.client_init_centroids == 'kmeans++':
                    # Calculate seeds from kmeans++
                    client_centroids, indices = kmeans_plusplus(client_data, n_clusters=self.n_clusters,
                                                                random_state=self.random_state)
                elif self.client_init_centroids == 'true':  # true centroids
                    client_centroids = self.true_centroids['train']
                else:
                    # client_centroids = randomly_init_centroid(0, self.n_clusters + 1, self.dim, self.n_clusters,
                    #                                           self.random_state)
                    client_centroids = random_initialize_centroids(client_data, self.n_clusters,
                                                                   self.random_state)
            else:
                client_centroids = centroids
            if iteration == 0:
                client_centroids_avg += client_centroids
                updates_sum = np.zeros((self.n_clusters, self.dim))
                counts = np.zeros(self.n_clusters)
                client_updates_sum = np.zeros((self.n_clusters, self.dim))
                client_counts = np.zeros(self.n_clusters)
                interim_updates = np.zeros((self.n_clusters, self.dim))
                print(f'Client_{i}, initial_centroids: {client_centroids}')
            else:
                client_counts = None
                for e in range(self.epochs):
                    client_updates_sum, client_counts = compute_step_for_client(
                        client_data=client_data,
                        centroids=client_centroids
                    )
                    interim_updates = client_updates_sum / np.expand_dims(
                        np.maximum(client_counts, np.ones_like(client_counts)), axis=1)
                    if self.epoch_lr is not None:
                        interim_updates = self.epoch_lr * interim_updates
                    client_centroids = client_centroids + interim_updates
                    if self.verbose > 10:
                        print(f'\t~client_{i}, epoch_{e}, client_centroids: {client_centroids}, '
                              f'client_counts: {client_counts}, given centroids: {centroids}')

                client_updates_sum = (client_centroids - centroids) * np.expand_dims(client_counts, axis=1)
                updates_sum += client_updates_sum
                counts += client_counts

            if iteration == 0:
                # compute client's params (such as centroids, cluster_sizes, and avg_dists)
                client_centroids, client_cluster_sizes, client_avg_dists = compute_client_params(client_data,
                                                                                                 client_centroids)
                # client_clusters_sizes == client_counts when iteration > 0
                KM_params['centroids'].extend(list(client_centroids))
                KM_params['cluster_sizes'].extend(list(client_cluster_sizes))
                KM_params['avg_dists'].extend(list(client_avg_dists))

            if self.verbose > 5 and iteration > 0:
                print("client_{}, client_counts: {}; client_updates_sum: {}, each_update: {}".format(i, client_counts,
                                                                                                     client_updates_sum,
                                                                                                     (client_centroids
                                                                                                      - centroids)))
        self.use_client_init_centroids = False
        if self.verbose > 5:
            print(f'total update_sum: {updates_sum} and counts: {counts}')

        if iteration == 0:
            KM_params['centroids'] = np.asarray(KM_params['centroids'])
            KM_params['cluster_sizes'] = np.asarray(KM_params['cluster_sizes'])
            KM_params['avg_dists'] = np.asarray(KM_params['avg_dists'])

        return updates_sum, counts, KM_params

    def fit(self, X_dict, y_dict, splits=None, record_at=None):
        X = X_dict['train']
        self.num_clients = len(X)
        self.dim = X[0].shape[1]

        # clients_per_round = max(1, int(self.sample_fraction * self.num_clients))
        clients_per_round = self.num_clients
        # centroids = self.do_init_centroids()
        centroids = np.zeros((self.n_clusters, self.dim))
        n_consecutive = 0
        not_changed = 0
        overall_counts = np.zeros(self.n_clusters)
        momentum = np.zeros_like(centroids)
        means_record = []
        stds_record = []
        to_reassign = np.zeros(self.n_clusters)
        self.history = []
        self.training_iterations = self.max_iter
        for iteration in range(0, self.max_iter):
            r = np.random.RandomState(iteration * max(1, self.random_state))
            indices = r.choice(range(self.num_clients), size=clients_per_round,
                               replace=False)  # without replacement and random
            clients_in_round = [X[j] for j in indices]
            # print(clients_in_round)
            if self.verbose > 5:
                print("round: {}".format(iteration))

            # updates_sum, counts = self.do_federated_round_single_step(
            #     clients_in_round=clients_in_round,
            #     centroids=centroids,
            # )
            updates_sum, counts, KM_params = self.do_federated_round(
                clients_in_round=clients_in_round,
                centroids=centroids, iteration=iteration,
            )
            if iteration == 0:  # for the first round, the server will select K centroids from K*M client's centroids.
                KM_centroids, KM_ns, KM_ds = KM_params['centroids'], KM_params['cluster_sizes'], KM_params['avg_dists']
                # 1. ignore n_points per each centroid
                # centroids, indices = kmeans_plusplus(KM_centroids, n_clusters=self.n_clusters, random_state=self.random_state)

                # 2. consider about cluster size: can't do this in the following way because the size of new_KM_centroids will be all clients'data
                # new_KM_centroids = []
                # for c, s in zip(KM_centroids, centroid_size):
                #     new_KM_centroids.extend([c] * int(s))
                # KM_centroids = np.asarray(new_KM_centroids)
                # centroids, indices = kmeans_plusplus(KM_centroids, n_clusters=self.n_clusters, random_state=self.random_state)

                # 3. weighted KMeans++
                # centroids, indices = weighted_kmeans_plusplus(KM_centroids, n_clusters=self.n_clusters, random_state=self.random_state)

                # 4. Greedy initialization
                centroids, indices = greedily_initialize(KM_centroids, KM_ns, KM_ds, self.n_clusters)
                self.initial_centroids = centroids
                print(f'initial_centroids: {centroids}')
                # testing after each iteration
                self.cluster_centers_ = self.initial_centroids
                scores = evaluate2(
                    kmeans=self,
                    x=X_dict, y=y_dict,
                    splits=splits,
                    federated=True,
                    verbose=False,
                )
                centroids_diff = {}
                for split in splits:
                    centroids_diff[split] = centroids - self.true_centroids[split]
                centroids_update = np.zeros((self.n_clusters, self.dim))
                self.history.append({'iteration': iteration, 'centroids': centroids, 'scores': scores,
                                     'centroids_update': centroids_update, 'centroids_diff': centroids_diff})
                continue
            # update server's params
            overall_counts += counts
            centroids_update = updates_sum / np.expand_dims(np.maximum(counts, np.ones_like(counts)), axis=1)

            # if self.adaptive_lr:
            #     rel_counts = counts / np.maximum(overall_counts, np.ones_like(overall_counts))
            #     update_weights = np.minimum(self.adaptive_lr, rel_counts)
            #     centroids_update = centroids_update * np.expand_dims(update_weights, axis=1)
            #
            # if self.lr is not None:
            #     centroids_update = self.lr * centroids_update
            #
            # if self.momentum_rate is not None:
            #     momentum = self.momentum_rate * momentum + (1 - self.momentum_rate) * centroids_update
            #     centroids_update = momentum

            # np.sum(np.square(centroids - (centroids + centroids_update)), axis=1)
            if np.sum(np.square(centroids_update)) < self.tol:
                if n_consecutive >= 5:
                    self.training_iterations = iteration
                    # training finishes in advance
                    break
                else:
                    n_consecutive += 1
            else:
                n_consecutive = 0

            centroids = centroids + centroids_update
            if self.verbose > 5:
                print(f'server\'s centroids_update: {centroids_update} and n_points per cluster: {counts}')
                print(f'new centroids: {centroids}')
            # Need to be revised?
            # centroids = update_server_centroids(KM_centroids, KM_ns, KM_ds, centroids, alpha=self.lr)
            changed = np.any(np.absolute(centroids_update) > self.tol)

            # if self.reassign_min is not None:
            #     for i in range(self.n_clusters):
            #         if counts[i] < (sum(counts) * self.reassign_min):
            #             to_reassign[i] += 1
            #         else:
            #             to_reassign[i] = 0
            #         if to_reassign[i] >= self.reassign_after:
            #             centroids[i] = randomly_init_centroid(0, self.n_clusters + 1, self.dim, 1)
            #             momentum[i] = np.zeros(self.dim)
            #             to_reassign[i] = 0
            #             changed = True
            #
            # if self.max_no_change is not None:
            #     not_changed += 1
            #     if changed:
            #         not_changed = 0
            #     if not_changed > self.max_no_change:
            #         break
            #
            # if record_at is not None and iteration in record_at:
            #     means, stds = record_state(centroids, np.concatenate(X, axis=0))
            #     means_record.append(means)
            #     stds_record.append(stds)

            # testing after each iteration
            self.cluster_centers_ = centroids
            scores = evaluate2(
                kmeans=self,
                x=X_dict, y=y_dict,
                splits=splits,
                federated=True,
                verbose=False,
            )
            centroids_diff = {}
            for split in splits:
                centroids_diff[split] = centroids - self.true_centroids[split]
            self.history.append({'iteration': iteration, 'centroids': centroids, 'scores': scores,
                                 'centroids_update': centroids_update, 'centroids_diff': centroids_diff})
        if record_at is not None:
            #  NOTE: only for dummy data
            plot_progress(means_record, stds_record, record_at)

        self.cluster_centers_ = centroids

        return centroids, overall_counts


if __name__ == "__main__":
    print(__file__)
    parser = argparse.ArgumentParser(description='Description of your program')
    # parser.add_argument('-p', '--py_name', help='python file name', required=True)
    parser.add_argument('-S', '--dataset', help='dataset', default='2GAUSSIANS')
    parser.add_argument('-T', '--data_details', help='data details', default='n1_5000-sigma1_2+n2_5000-sigma2_2:ratio_0.0:diff_sigma_n')
    parser.add_argument('-M', '--algorithm', help='algorithm', default='Federated-Server_greedy-Client_kmeans++')
    # args = vars(parser.parse_args())
    args = parser.parse_args()
    print(args)
    p3 = __file__.split('/')[-1].split('.')[0]
    params = get_experiment_params(p0=args.dataset, p1=args.data_details, p2=args.algorithm, p3=p3)
    print(params)
    try:
        _main.run_clustering_federated(
            params,
            KMeansFederated,
            verbose=15,
        )
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()
