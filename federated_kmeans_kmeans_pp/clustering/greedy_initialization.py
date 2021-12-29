import numpy as np


def distance(x1, x2):
    # return np.sqrt(np.sum(np.square(x1 - x2), axis=1))
    return np.sqrt(np.sum(np.square(x1-x2)))

def distance_sq(x1, x2):
    # return np.sqrt(np.sum(np.square(x1 - x2), axis=1))
    return np.sum(np.square(x1-x2))

def pick_centroid(KM_centroids, KM_ns, KM_ds, centroids, indices):
    # choose the one centroid when f_i is the smallest one
    d = np.inf
    idx = None
    for i, y_i in enumerate(KM_centroids):
        if i in indices: continue
        f_i = 0
        for j, y_j in enumerate(KM_centroids):
            if (i == j) or (j in indices): continue
            # n_j * min(dlj for l in [i] + indices)
            f_i += KM_ns[j] * min(distance_sq(KM_centroids[l], KM_centroids[j]) for l in [i] + indices)
        if f_i < d:
            d = f_i
            idx = i
    return idx


def pick_first_centorid(KM_centroids, KM_ns, KM_ds):
    d = np.inf
    idx = None
    for i, y_i in enumerate(KM_centroids):
        d_i = KM_ns[i] * KM_ds[i]  # n_i * d_ii
        for j, y_j in enumerate(KM_centroids):
            if i == j: continue
            d_i += KM_ns[j] * distance_sq(y_i, y_j)  # n_j * d_ij
        if d_i < d:
            d = d_i
            idx = i

    return idx


def pick_first_centorid2(KM_centroids, KM_ns, KM_ds):
    d = np.inf
    idx = None
    for i, y_i in enumerate(KM_centroids):
        # d_i = KM_ns[i] * KM_ds[i]  # n_i * d_ii
        d_i = 0
        for j, y_j in enumerate(KM_centroids):
            if i == j: continue
            d_i += KM_ns[j] * distance_sq(y_i, y_j)  # n_j * d_ij
        if d_i < d:
            d = d_i
            idx = i

    return idx


def pick_first_centorid3(KM_centroids, KM_ns, KM_ds):
    d = np.inf
    idx = None
    alpha = 0.3
    for i, y_i in enumerate(KM_centroids):
        d_i = KM_ns[i] * KM_ds[i]  # n_i * d_ii
        for j, y_j in enumerate(KM_centroids):
            if i == j: continue
            d_i += KM_ns[j] * distance_sq(y_i, y_j) + alpha * KM_ds[j]  # n_j * d_ij + alpha * d_jj
        if d_i < d:
            d = d_i
            idx = i
    return idx


def greedily_initialize(KM_centroids, KM_ns, KM_ds, n_clusters):
    """ Find K centroids greedily from the parameters (includes centroids, ... ) uploaded from clients.

        Potential issues:
            1. if time complexity is too high?
    Parameters
    ----------
    KM_centroids:  ndarray of shape (K*M, n_features)
        K*M centroids collected from the clients who participate in the training phase.
    KM_ns:  ndarray of shape (K*M, )
        K*M list/array,where each item is an integer that is the number of data points of one cluster.
    KM_ds:  ndarray of shape (K*M, )
        K*M list/array, where each item is an integer that is the average distance of all data points of a cluster
        to its centroid
    n_clusters: Number of clusters

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.


    """
    centroids = []
    indices = []
    # choose the first centroid when d_i is the smallest one
    idx = pick_first_centorid(KM_centroids, KM_ns, KM_ds)  # version 1
    # idx = pick_first_centorid2(KM_centroids, KM_ns, KM_ds)    # version 2
    # idx = pick_first_centorid3(KM_centroids, KM_ns, KM_ds)    # version 3
    centroids.append(KM_centroids[idx])
    indices.append(idx)

    # choose the rest of centroids:
    for _ in range(1, n_clusters):
        j = pick_centroid(KM_centroids, KM_ns, KM_ds, centroids, indices)
        centroids.append(KM_centroids[j])
        indices.append(j)

    return np.asarray(centroids), np.array(indices)
