"""

https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_plusplus.html#sphx-glr-auto-examples-cluster-plot-kmeans-plusplus-py

"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


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
    # Generate sample data
    # n_samples_per_cluster = 4000
    # n_components = 4
    # X, y_true = make_blobs(
    #     n_samples=n_sampes, centers=centers, cluster_std=cluster_std, random_state=0
    # )

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


# gen_data()


def load_federated_dummy_2D(random_state=42, verbose=False, clients_per_cluster=10, clusters=5):
    # # assert dims == 1, "only one dimension implemented"
    # np.random.seed(seed)
    # x = {}
    # ids = {}
    # # data, means = create_dummy_data(clients_per_cluster=2*clients_per_cluster, clusters=clusters, verbose=verbose)
    # mid = clients_per_cluster * clusters
    # x["train"], ids["train"] = data[:mid], means[:mid]
    # x["test"], ids["test"] = data[mid:], means[mid:]
    # # print(len(x['train']), x['train'][0].shape)

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


def _blobs_C5(n_samples_per_cluster=1000, random_state=42, rho=1.0, cluster_std=1.0):
    M = 5  # number of clients
    # 1. Generate data
    # generate train sets
    data = []
    means = []
    labels = []
    for i in range(M):
        if i == 0:
            centers = np.asarray([0, 0])
            n_sampes = n_samples_per_cluster * 4
        elif i == 1:
            centers = np.asarray([5, 0])
            n_sampes = n_samples_per_cluster
        elif i == 2:
            centers = np.asarray([0, 5])
            n_sampes = n_samples_per_cluster
        elif i == 3:
            centers = np.asarray([-5, 0])
            n_sampes = n_samples_per_cluster
        elif i == 4:
            centers = np.asarray([0, -5])
            n_sampes = n_samples_per_cluster
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
        plt.xlim([-10, 10])
        plt.ylim([-10, 10])
        # plt.xticks([])
        # plt.yticks([])
        plt.show()

    return data, labels


def load_federated_dummy_2D_C5(clusters=None, random_state=42, verbose=False, clients_per_cluster=10, K=2, rho=1):
    clients_train_x, clients_train_y = _blobs_C5(n_samples_per_cluster=1000, rho=rho, cluster_std=1,
                                              random_state=random_state)
    # generate test sets
    clients_test_x, clients_test_y = _blobs_C5(n_samples_per_cluster=1000, rho=rho, cluster_std=1,
                                            random_state=random_state * 2)
    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}
    return x, labels


def load_federated(limit_csv=None, verbose=False, seed=None, dummy='dummy_2D', clusters=None, rho=2.0):
    if dummy == 'dummy_2D':  #
        return load_federated_dummy_2D(random_state=seed, verbose=verbose, clusters=clusters)
    elif dummy == 'dummy_2D_M3':
        return load_federated_dummy_2D_M3(random_state=seed, verbose=verbose, clusters=clusters, rho=rho)
    elif dummy == 'dummy_2D_C5':
        return load_federated_dummy_2D_C5(random_state=seed, verbose=verbose, clusters=clusters, rho=rho)
