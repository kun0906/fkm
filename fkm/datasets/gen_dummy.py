"""

https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_plusplus.html#sphx-glr-auto-examples-cluster-plot-kmeans-plusplus-py

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs

from fkm.datasets.femnist import femnist_1client_1writer_multidigits, femnist_1client_multiwriters_multidigits, \
    femnist_1client_multiwriters_1digit
from fkm.datasets.gaussian2 import *
from fkm.datasets.gaussian3 import *
from fkm.datasets.gaussian4 import *
from fkm.datasets.gaussian5 import gaussian5_5clients_5clusters
from fkm.datasets.moon import moons_dataset
from fkm.utils.utils_func import timer


def _gen_data(t, n_clusters=5, n_samples_per_cluster=1000, random_state=42):
    """ generated cluster data

    Parameters
    ----------
    t:
    n_clusters
    n_samples_per_cluster

    Returns
    -------

    """
    # centroid2label={(1,1): 0, (5,5): 1, (5, 10): 2, (10, 10): 3, (10, 5): 4}
    cluster_std = 1
    if t == 0:
        centers = np.asarray([(1, 1)])
        n_sampes = n_samples_per_cluster * 1
        X, y_true = make_blobs(
            n_samples=n_sampes, centers=centers, cluster_std=cluster_std, random_state=random_state
        )
        n_points = [n_samples_per_cluster]
    elif t == 1:
        centers = np.asarray([(5, 5), (5, 10)])
        n_sampes = n_samples_per_cluster * 2
        X, y_true = make_blobs(
            n_samples=n_sampes, centers=centers, cluster_std=cluster_std, random_state=random_state
        )
        # be careful of the reassigned order
        y_true[y_true == 1] = 2
        y_true[y_true == 0] = 1
        # print(set(y_true))
        n_points = [n_samples_per_cluster, n_samples_per_cluster]
    elif t == 2:
        centers = np.asarray([(5, 10), (10, 10), (10, 5)])
        n_sampes = n_samples_per_cluster * 3
        X, y_true = make_blobs(
            n_samples=n_sampes, centers=centers, cluster_std=cluster_std, random_state=random_state
        )
        y_true[y_true == 2] = 4
        y_true[y_true == 1] = 3
        y_true[y_true == 0] = 2
        n_points = [n_samples_per_cluster, n_samples_per_cluster, n_samples_per_cluster]
    elif t == 3:
        centers = np.asarray([(10, 10), (10, 5)])
        n_sampes = n_samples_per_cluster * 2
        X, y_true = make_blobs(
            n_samples=n_sampes, centers=centers, cluster_std=cluster_std, random_state=random_state
        )
        y_true[y_true == 1] = 4
        y_true[y_true == 0] = 3
        n_points = [n_samples_per_cluster, n_samples_per_cluster]
    else:
        centers = np.asarray([(10, 5)])
        n_sampes = n_samples_per_cluster * 1
        X, y_true = make_blobs(
            n_samples=n_sampes, centers=centers, cluster_std=cluster_std, random_state=random_state
        )
        y_true[y_true == 0] = 4
        n_points = [n_samples_per_cluster]
    return X, y_true, centers, n_points


def gen_data(n_samples_per_cluster=1000, n_clusters=5, n_clients=25, random_state=42, is_show=True):
    data = []
    means = []
    labels = []
    # n_samples_per_cluster* n_clusters //n_clients # how many data points per client?, each client has a single clusters
    for i in range(n_clients):
        t = i % n_clusters
        X, y_true, centroids, n_points = _gen_data(t, n_clusters, n_samples_per_cluster, random_state)
        data.append(X)
        means.append(np.mean(X, axis=0))
        labels.append(y_true)

        if is_show:
            # Plot init seeds along side sample data
            plt.figure(1)
            # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
            colors = ["r", "g", "b", "m", 'black']
            for k, col in enumerate(colors):
                cluster_data = y_true == k
                plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=".", s=10)

            # plt.scatter(centroids[:, 0], centroids[:, 1], c="b", s=50)
            plt.title(f"Client_{i}")
            plt.xlim([-2, 15])
            plt.ylim([-2, 15])
            # plt.xticks([])
            # plt.yticks([])
            plt.show()

    return data, labels


def load_federated_dummy_2D(random_state=42, verbose=False, clients_per_cluster=10, clusters=5):
    M = 25
    K = clusters
    # 1. Generate data
    is_show = False
    # generate train sets
    clients_train_x, clients_train_y = gen_data(n_clients=M, n_clusters=K, n_samples_per_cluster=5000,
                                                random_state=random_state, is_show=is_show)
    # generate test sets
    clients_test_x, clients_test_y = gen_data(n_clients=M, n_clusters=K, n_samples_per_cluster=1000,
                                              random_state=random_state, is_show=is_show)
    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}
    return x, labels


def _blobs(n_samples_per_cluster=1000, random_state=42, rho=1.0, cluster_std=0.1):
    M = 3  # number of clients
    # 1. Generate data
    # generate train sets
    data = []
    means = []
    labels = []
    for i in range(M):
        # if i == 0:
        #     centers = np.asarray([0, 1])
        #     n_sampes = n_samples_per_cluster * (i + 1)
        # elif i == 1:
        #     centers = np.asarray([0, -1])
        #     n_sampes = n_samples_per_cluster * (i + 1)
        # elif i == 2:
        #     centers = np.asarray([rho, 0])
        #     n_sampes = n_samples_per_cluster * (i + 1)
        if i == 0:
            centers = np.asarray([0, 0])
            n_sampes = n_samples_per_cluster * (i + 1)
        elif i == 1:
            centers = np.asarray([1, 0])
            n_sampes = n_samples_per_cluster * (i + 1)
        elif i == 2:
            centers = np.asarray([rho, 0])
            n_sampes = n_samples_per_cluster * (i + 1)
        else:
            raise NotImplementedError

        X, y = make_blobs(
            n_samples=n_sampes, centers=centers.reshape((1, 2)), cluster_std=cluster_std, random_state=random_state
        )
        data.append(X)
        # means.append(np.mean(X, axis=0))
        labels.append(y)

    is_show = True
    if is_show:
        # Plot init seeds along side sample data
        plt.figure(1)
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        for k, X in enumerate(data):
            # cluster_data = y_true == k
            # plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=".", s=10)
            plt.scatter(X[:, 0], X[:, 1], c=colors[k], marker=".", s=10)
        plt.axvline(x=0, color='k', linestyle='--')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.title(f"")
        # plt.xlim([-2, 15])
        # plt.ylim([-2, 15])
        plt.xlim([-2, 5])
        plt.ylim([-2, 5])
        # plt.xticks([])
        # plt.yticks([])
        plt.show()

    return data, labels


def load_federated_dummy_2D_M3(clusters=None, random_state=42, verbose=False, clients_per_cluster=10, K=2, rho=1):
    clients_train_x, clients_train_y = _blobs(n_samples_per_cluster=1000, rho=rho, cluster_std=0.1,
                                              random_state=random_state)
    # generate test sets
    clients_test_x, clients_test_y = _blobs(n_samples_per_cluster=1000, rho=rho, cluster_std=0.1,
                                            random_state=random_state * 2)
    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}
    return x, labels


def _blobs_C5(n_clients=5, n_clusters=4, n_samples_per_cluster=1000, random_state=42, rho=1.0, cluster_std=1.0):
    M = n_clients  # number of clients
    # 1. Generate data
    # generate train sets
    data = []
    means = []
    labels = []
    for i in range(M):
        # if i == 0:
        #     centers = np.asarray([0, 0])
        #     n_sampes = n_samples_per_cluster * 4
        # elif i == 1:
        #     centers = np.asarray([5, 0])
        #     n_sampes = n_samples_per_cluster
        # elif i == 2:
        #     centers = np.asarray([0, 5])
        #     n_sampes = n_samples_per_cluster
        # elif i == 3:
        #     centers = np.asarray([-5, 0])
        #     n_sampes = n_samples_per_cluster
        # elif i == 4:
        #     centers = np.asarray([0, -5])
        #     n_sampes = n_samples_per_cluster
        # else:
        #     raise NotImplementedError
        # X, y = make_blobs(
        #             n_samples=n_sampes, centers=centers.reshape((1, 2)), cluster_std=cluster_std, random_state=random_state
        #         )

        if i == 0:
            n_sampes = n_samples_per_cluster * 4
            centers = np.asarray([0, 0])
            X, y = make_blobs(
                n_samples=n_sampes, centers=centers.reshape((1, 2)), cluster_std=cluster_std, random_state=random_state
            )
        elif i % 2 == 0:
            n_sampes = n_samples_per_cluster * 2
            X, y = make_blobs(
                n_samples=n_sampes, centers=np.asarray([[5, 0], [-5, 0]]), cluster_std=cluster_std,
                random_state=i * random_state
            )
        else:
            n_sampes = n_samples_per_cluster * 2
            X, y = make_blobs(
                n_samples=n_sampes, centers=np.asarray([[0, 5], [0, -5]]), cluster_std=cluster_std,
                random_state=random_state * i
            )
        data.append(X)
        # means.append(np.mean(X, axis=0))
        labels.append(y)

    is_show = False
    if is_show:
        # Plot init seeds along side sample data
        plt.figure(1)
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        for k, X in enumerate(data):
            # cluster_data = y_true == k
            # plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=".", s=10)
            plt.scatter(X[:, 0], X[:, 1], c=colors[k], marker=".", s=10)
        plt.axvline(x=0, color='k', linestyle='--')
        plt.axhline(y=0, color='k', linestyle='--')
        plt.title(f"")
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        # plt.xticks([])
        # plt.yticks([])
        plt.show()

    return data, labels


def load_federated_dummy_2D_C5(n_clients=5, n_clusters=4, random_state=42, rho=1, verbose=None):
    clients_train_x, clients_train_y = _blobs_C5(n_clients=n_clients, n_clusters=n_clusters, n_samples_per_cluster=1000,
                                                 rho=rho, cluster_std=1,
                                                 random_state=random_state)
    # generate test sets
    clients_test_x, clients_test_y = _blobs_C5(n_clients=n_clients, n_clusters=n_clusters, n_samples_per_cluster=1000,
                                               rho=rho, cluster_std=1,
                                               random_state=random_state * 2)
    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}
    return x, labels


#
# def load_femnist(params={}, random_state=42):
#     # importing the module
#     import json
#
#     def femnist(in_dir='../leaf/data/femnist/data/train'):
#
#         X = []
#         y = []
#         for i, f in enumerate(os.listdir(in_dir)):  # number of clients
#             # Opening JSON file
#             f = os.path.join(in_dir, f)
#             with open(f) as json_file:
#                 data = json.load(json_file)
#                 x_ = []
#                 y_ = []
#                 for u_name, u_vs in data['user_data'].items():
#                     # only keep 0-9 digitals
#                     ab = [(v, l) for v, l in zip(u_vs['x'], u_vs['y']) if l in list(range(10))]
#                     if len(ab) == 0: continue
#                     a, b = zip(*ab)
#                     x_.extend(list(a))
#                     y_.extend(list(b))
#                 # print(i, f, Counter(y_))
#             X.append(np.asarray(x_))
#             y.append(np.asarray(y_))
#
#             # if i >= 5: break
#
#         return X, y
#
#     clients_train_x, clients_train_y = femnist(in_dir='../datasets/femnist_sampled/train')
#     # generate test sets
#     clients_test_x, clients_test_y = femnist(in_dir='../datasets/femnist_sampled/test')
#     x = {'train': clients_train_x,
#          'test': clients_test_x}
#     labels = {'train': clients_train_y,
#               'test': clients_test_y}
#
#     return x, labels

@timer
def load_federated(limit_csv=None, verbose=False, seed=None, clusters=None, n_clients=5, n_clusters=4, rho=2.0,
                   params={}):
    data_name = params['data_name']
    if data_name == 'dummy_2D':  #
        return load_federated_dummy_2D(random_state=seed, verbose=verbose, clusters=clusters)
    elif data_name == 'dummy_C2M3':
        return load_federated_dummy_2D_M3(random_state=seed, verbose=verbose, clusters=clusters, rho=rho)
    elif data_name == 'dummy_C5':
        return load_federated_dummy_2D_C5(n_clients=n_clients, n_clusters=n_clusters, random_state=seed,
                                          verbose=verbose)
    # elif data_name == 'femnist':
    #     return load_femnist(params, random_state=seed)
    # elif data_name == 'femnist_0.1users_0.3user_testset':
    #     return load_femnist_user_percent(params, random_state=seed)
    # elif data_name == 'femnist_0.1users_0.3data_testset':
    #     return load_femnist_data_percent_per_user(params, random_state=seed)
    # elif data_name == 'femnist_0.1users_20clients0.3data_testset':
    #     return load_femnist_user_percent_clients(params, random_state=seed)
    # elif data_name == 'femnist_10clients':
    #     return load_femnist_10clients(params, random_state=seed)
    elif params['p0'] == 'FEMNIST':
        if params['p1'] == '1client_1writer_multidigits':
            return femnist_1client_1writer_multidigits(params, random_state=seed)
        elif params['p1'] == '1client_multiwriters_multidigits':
            return femnist_1client_multiwriters_multidigits(params, random_state=seed)
        elif params['p1'] == '1client_multiwriters_1digit':
            return femnist_1client_multiwriters_1digit(params, random_state=seed)
    elif params['p0'] == '2GAUSSIANS':
        # here is only for the same sigma. for different sigmas, not implement yet.
        if params['p1'] == '1client_1cluster':
            # 2 clusters ((-1,0), (1, 0)) in R^2, each client has one cluster. 2 clusters has overlaps.
            return gaussian2_1client_1cluster(params, random_state=seed)
        elif params['p1'].split(':')[-1] == 'mix_clusters_per_client':
            # 2 clusters in R^2:
            # 1) client1 has 70% data from cluster 1 and 30% data from cluster2
            # 2) client2 has 30% data from cluster 1 and 70% data from cluster2
            return gaussian2_mix_clusters_per_client(params, random_state=seed)
        elif params['p1'] == '1client_ylt0':
            # lt0 means all 'y's are larger than 0
            # 2 clusters in R^2
            # 1) client 1 has all data (y>0) from cluster1 and cluster2
            # 1) client 2 has all data (y<=0) from cluster1 and cluster2
            return gaussian2_1client_ylt0(params, random_state=seed)
        elif params['p1'] == '1client_xlt0':
            # lt0 means all 'x's are larger than 0
            # 2 clusters in R^2
            # 1) client 1 has all data (x>0) from cluster1 and cluster2
            # 1) client 2 has all data (x<=0) from cluster1 and cluster2
            return gaussian2_1client_xlt0(params, random_state=seed)

        elif params['p1'] == '1client_1cluster_diff_sigma':
            """
            # 2 clusters ((-1,0), (1, 0)) in R^2, each client has one cluster. 2 clusters has no overlaps.
            cluster 1: sigma = 0.5
            cluster 2: sigma = 1
            params['p1'] == '1client_1cluster_diff_sigma':
            Parameters
            """
            return gaussian2_1client_1cluster_diff_sigma(params, random_state=seed)
        elif params['p1'].split(':')[-1] == 'diff_sigma_n':
            """
            # 2 clusters ((-1,0), (1, 0)) in R^2, each client has one cluster. 2 clusters has no overlaps.
            cluster 1: sigma = 0.5 and n_points = 5000
            cluster 2: sigma = 1    and n_points = 15000
            params['p1'] == 'diff_sigma_n':
            Parameters
            """
            return gaussian2_diff_sigma_n(params, random_state=seed)
        elif params['p1'] == '1client_xlt0_2':
            """
            # 2 clusters ((-1,0), (1, 0)) in R^2, each client has one cluster. 2 clusters has no overlaps.
             # lt0 means all 'x's are larger than 0
            # 2 clusters in R^2
            # 1) client 1 has all data (x>0) from cluster1 and cluster2
            # 1) client 2 has all data (x<=0) from cluster1 and cluster2
            Parameters
            """
            return gaussian2_1client_xlt0_2(params, random_state=seed)


    elif params['p0'] == '3GAUSSIANS':
        # here is only for the same sigma. for different sigmas, not implement yet.
        if params['p1'] == '1client_1cluster':
            return gaussian3_1client_1cluster(params, random_state=seed)
        elif params['p1'].split(':')[-1] == 'mix_clusters_per_client':
            return gaussian3_mix_clusters_per_client(params, random_state=seed)
        elif params['p1'] == '1client_ylt0':
            return gaussian3_1client_ylt0(params, random_state=seed)
        elif params['p1'] == '1client_xlt0':
            return gaussian3_1client_xlt0(params, random_state=seed)
        elif params['p1'] == '1client_1cluster_diff_sigma':
            return gaussian3_1client_1cluster_diff_sigma(params, random_state=seed)
        elif params['p1'].split(':')[-1] == 'diff_sigma_n':
            return gaussian3_diff_sigma_n(params, random_state=seed)
        elif params['p1'] == '1client_xlt0_2':
            return gaussian3_1client_xlt0_2(params, random_state=seed)

    elif params['p0'] == '4GAUSSIANS':
        # here is only for the same sigma. for different sigmas, not implement yet.
        # if params['p1'] == '1client_1cluster':
        #     return gaussian4_1client_1cluster(params, random_state=seed)
        # elif params['p1'].split(':')[-1] == 'mix_clusters_per_client':
        #     return gaussian4_mix_clusters_per_client(params, random_state=seed)
        # elif params['p1'] == '1client_ylt0':
        #     return gaussian4_1client_ylt0(params, random_state=seed)
        # elif params['p1'] == '1client_xlt0':
        #     return gaussian4_1client_xlt0(params, random_state=seed)
        # elif params['p1'] == '1client_1cluster_diff_sigma':
        #     return gaussian4_1client_1cluster_diff_sigma(params, random_state=seed)
        if params['p1'].split(':')[-1] == 'diff_sigma_n':
            return gaussian4_diff_sigma_n(params, random_state=seed)
        # elif params['p1'] == '1client_xlt0_2':
        #     return gaussian4_1client_xlt0_2(params, random_state=seed)

    elif params['p0'] == '5GAUSSIANS':
        # here is only for the same sigma. for different sigmas, not implement yet.
        if params['p1'] == '5clients_5clusters':
            return gaussian5_5clients_5clusters(params, random_state=seed)
        elif params['p1'] == '5clients_4clusters':
            return gaussian5_5clients_5clusters(params, random_state=seed)
        elif params['p1'] == '5clients_3clusters':
            return gaussian5_5clients_5clusters(params, random_state=seed)

    elif params['p0'] == '2MOONS':
        if params['p1'] == '2moons':
            """
            # 2 clusters ((-1,0), (1, 0)) in R^2, each client has one cluster. 2 clusters has no overlaps.
             # lt0 means all 'x's are larger than 0
            # 2 clusters in R^2
            # 1) client 1 has all data (x>0) from cluster1 and cluster2
            # 1) client 2 has all data (x<=0) from cluster1 and cluster2
            Parameters
            """
            return moons_dataset(params, random_state=seed)
        else:
            # TODO : different sigmas
            pass

