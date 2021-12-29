import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class Client:

    def __init__(self):
        pass

    def gen_data(self):
        pass


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


def gen_data(n_clients=25, n_clusters=5, n_samples_per_cluster=1000, is_show=True, random_state = 42):
    clients = []  # [(X, y)]
    for i in range(n_clients):
        t = i % n_clusters
        X, y, centroids, n_points = _gen_data(t, n_clusters, n_samples_per_cluster, random_state)
        client = Client()
        client.X = X
        client.y = y
        client.n_points = n_points
        client.centroids = centroids
        clients.append(client)

        if is_show:
            # Plot init seeds along side sample data
            plt.figure(1)
            colors = ["r", "g", "b", "m", 'black']
            for k in set(y):
                cluster_data = y == k
                plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=colors[k], marker=".", s=10)
            plt.title(f"Client_{i}")
            plt.xlim([0, 15])
            plt.ylim([0, 15])
            # plt.xticks([])
            # plt.yticks([])
            plt.show()
    return clients


