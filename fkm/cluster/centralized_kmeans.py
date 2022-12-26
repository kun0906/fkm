from pprint import pprint

import numpy as np
from sklearn.cluster import kmeans_plusplus
from sklearn.preprocessing import StandardScaler

from fkm.utils.utils_func import random_initialize_centroids, timer
from fkm.utils.utils_stats import plot_progress, evaluate2


class KMeans:
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

    def do_init_centroids(self, X=None):
        if isinstance(self.init_centroids, str):
            if self.init_centroids == 'random':
                # # assumes data is in range 0-1
                # centroids = np.random.rand(self.n_clusters, self.dim)
                # for dummy data
                # centroids = randomly_init_centroid(0, self.n_clusters + 1, self.dim, self.n_clusters, self.random_state)
                centroids = random_initialize_centroids(X, self.n_clusters, self.random_state)
            elif self.init_centroids == 'kmeans++':
                centroids, _ = kmeans_plusplus(X, n_clusters=self.n_clusters, random_state=self.random_state)
            elif self.init_centroids == 'true':
                centroids = self.true_centroids['train']
            else:
                raise NotImplementedError
        elif self.init_centroids.shape == (self.n_clusters, self.dim):
            centroids = self.init_centroids
        else:
            raise NotImplementedError
        return centroids

    @timer
    def fit(self, X_dict, y_dict, splits, record_at=None):
        self.is_train_finished = False
        X = X_dict['train']
        # self.n_clients = len(X)
        self.n_points, self.dim = X.shape

        # if in 5 consecutive times, the difference between new and old centroids is less than self.tol,
        # then the model converges and we stop the training.
        self.n_consecutive = 0
        means_record = []
        stds_record = []
        self.history = []
        to_reassign = np.zeros(self.n_clusters)
        self.training_iterations = self.max_iter
        for iteration in range(0, self.max_iter):
            self.training_iterations = iteration
            if self.verbose >= 2:
                print(f'iteration: {iteration}')
            if iteration == 0:
                self.initial_centroids = self.do_init_centroids(X)
                # print(f'initialization method: {self.init_centroids}, centers: {centroids}')
                self.centroids = self.initial_centroids
                print(f'initial_centroids: \n{self.centroids}')
                # testing after each iteration
                scores = evaluate2(
                    kmeans=self,
                    x=X_dict, y=y_dict,
                    splits=splits,
                    federated=False,
                    verbose=self.verbose,
                )
                centroids_diff = {}
                for split in splits:
                    if self.centroids.shape != self.true_centroids[split].shape:
                        continue
                    centroids_diff[split] = self.centroids - self.true_centroids[split]
                centroids_update = self.centroids-np.zeros((self.n_clusters, self.dim))  # centroids(t+1) - centroid(t)
                self.history.append({'iteration': iteration, 'centroids': self.centroids, 'scores': scores,
                                     'centroids_update': centroids_update, 'centroids_diff': centroids_diff})
                if self.verbose >=3:
                    for split in splits:
                        print(f'{split}:')
                        pprint(scores[split])
                continue
            # compute distances
            # computationally efficient
            # differences = np.expand_dims(x, axis=1) - np.expand_dims(centroids, axis=0)
            # sq_dist = np.sum(np.square(differences), axis=2)
            # memory efficient
            sq_dist = np.zeros((self.n_points, self.n_clusters))
            for i in range(self.n_clusters):
                sq_dist[:, i] = np.sum(np.square(X - self.centroids[i, :]), axis=1)

            labels = np.argmin(sq_dist, axis=1)
            # update centroids
            centroids_update = np.zeros((self.n_clusters, self.dim))
            counts = np.zeros((self.n_clusters,))
            for i in range(self.n_clusters):
                mask = np.equal(labels, i)
                size = np.sum(mask)
                counts[i] = size
                if size > 0:
                    # new_centroids[i, :] = np.mean(X[mask], axis=0)
                    update = np.sum(X[mask] - self.centroids[i, :], axis=0)
                    centroids_update[i, :] = update / size
                # if self.reassign_min is not None:
                #     if size < X.shape[0] * self.reassign_min:
                #         to_reassign[i] += 1
                #     else:
                #         to_reassign[i] = 0

            # np.sum(np.square(centroids - (centroids + centroid_updates)), axis=1)
            # print(iteration, centroid_updates, centroids)
            delta = np.sum(np.square(centroids_update))
            if self.verbose>=2:
                print(f'iteration: {iteration}, np.sum(np.square(centroids_update)): {delta}')
            if delta < self.tol:
                if self.n_consecutive >= self.params['n_consecutive']:
                    self.training_iterations = iteration
                    # training finishes in advance
                    break
                else:
                    self.n_consecutive += 1
            else:
                self.n_consecutive = 0

            # centroids = centroids + centroids_update
            self.centroids = self.centroids + centroids_update
            if self.verbose >= 4:
                print(f'server\'s centroids_update: {centroids_update} and n_points per cluster: {counts}')
                print(f'new centroids: {self.centroids}')

            # testing after each iteration
            scores = evaluate2(
                kmeans=self,
                x=X_dict, y=y_dict,
                splits=splits,
                federated=False,
                verbose=self.verbose,
            )
            centroids_diff = {}
            for split in splits:
                if self.centroids.shape != self.true_centroids[split].shape:
                    continue
                centroids_diff[split] = self.centroids - self.true_centroids[split]
            self.history.append({'iteration': iteration, 'centroids': self.centroids, 'scores': scores,
                                 'centroids_update': centroids_update, 'centroids_diff': centroids_diff})
            if self.verbose>=3:
                for split in splits:
                    print(f'{split}:')
                    pprint(scores[split])

            # changed = np.any(np.absolute(centroid_updates) > self.tol)
            # for i, num_no_change in enumerate(to_reassign):
            #     if num_no_change >= self.reassign_after:
            #         centroids[i] = randomly_init_centroid(0, self.n_clusters + 1, self.dim, 1, self.random_state)
            #         to_reassign[i] = 0
            #         changed = True
            #
            # if record_at is not None and iteration in record_at:
            #     means, stds = record_state(centroids, X)
            #     means_record.append(means)
            #     stds_record.append(stds)
            # # if not changed:
            # #     break

        if record_at is not None:
            #  NOTE: only for dummy data
            plot_progress(means_record, stds_record, record_at)

        # print(sq_dist.shape)
        # print(labels.shape)
        # print(centroids.shape)
        # self.labels_ = labels

        # print(f'Training result: centers: {centroids}')
        self.is_train_finished = True
        return

    def predict(self, x):
        # before predicting, check if you already preprocessed x (e.g., std).
        # memory efficient
        sq_dist = np.zeros((x.shape[0], self.n_clusters))
        for i in range(self.n_clusters):
            sq_dist[:, i] = np.sum(np.square(x - self.centroids[i, :]), axis=1)
        labels = np.argmin(sq_dist, axis=1)
        return labels


"""
    Centralized K-means

    PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 centralized_kmeans.py --dataset 'FEMNIST' \
                --data_details '1client_multiwriters_1digit' --algorithm 'Centralized_random'
"""
# # Email: Kun.bj@outlook.com
# import argparse
# import traceback
# from pprint import pprint
# import numpy as np
# from fkm.cluster.Centralized_Kmeans import KMeans
# from fkm._main import run_clustering_federated
# from fkm.experiment_cases import get_experiment_params
#
# # These options determine the way floating point numbers, arrays and
# # other NumPy objects are displayed.
# # np.set_printoptions(precision=3, suppress=True)
# np.set_printoptions(precision=3, suppress=True, formatter={'float': '{:20.3f}'.format}, edgeitems = 120, linewidth=100000)
#
# if __name__ == '__main__':
#     print(__file__)
#     parser = argparse.ArgumentParser(description='Description of your program')
#     # parser.add_argument('-C', '--config_file', help='A configuration file (yaml) that includes all parameters',
#     #                     default='config.yaml')
#     # parser.add_argument('-p', '--py_name', help='python file name', required=True)
#     parser.add_argument('-S', '--dataset', help='dataset', default='NBAIOT')
#     # parser.add_argument('-T', '--data_details', help='data details', default='n1_5000-sigma1_3+n2_5000-sigma2_3+n3_5000-sigma3_3+n4_5000-sigma4_3:ratio_0.0:diff_sigma_n')
#     # parser.add_argument('-T', '--data_details', help='data details', default='n1_5000-sigma1_0.3_0.3+n2_5000-sigma2_0.3_0.3+n3_10000-sigma3_0.3_0.3+n4_0.1-sigma4_0.3_0.3:ratio_0.0:diff_sigma_n')
#     parser.add_argument('-T', '--data_details', help='data details',
#                         default='nbaiot_user_percent_client11')
#     parser.add_argument('-M', '--algorithm', help='algorithm', default='Centralized_kmeans++')
#     parser.add_argument('-K', '--n_clusters', help='number of clusters', default= 2)    # 9 or 11
#     parser.add_argument('-C', '--n_clients', help='number of clients', default= 11)
#     # args = vars(parser.parse_args())
#     args = parser.parse_args()
#     pprint(args)
#     p3 = __file__.split('/')[-1]
#     params = get_experiment_params(p0=args.dataset, p1=args.data_details, p2=args.algorithm, p3 =p3,
#                                    n_clusters = int(args.n_clusters), n_clients=int(args.n_clients))
#     pprint(params)
#     try:
#         run_clustering_federated(
#             params,
#             KMeans,
#             verbose=5 if args.dataset == 'FEMNIST' else 10,
#         )
#     except Exception as e:
#         print(f'Error: {e}')
#         traceback.print_exc()
