import pickle
import time
from datetime import datetime

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt

def randomly_init_centroid(min_value, max_value, n_dims, repeats=1, random_state=42):
    r = np.random.RandomState(random_state)
    if repeats == 1:
        # return min_value + (max_value - min_value) * np.random.rand(n_dims)
        return min_value + (max_value - min_value) * r.rand(n_dims)
    else:
        # return min_value + (max_value - min_value) * np.random.rand(repeats, n_dims)
        return min_value + (max_value - min_value) * r.rand(repeats, n_dims)


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


def init_kmeans_sklearn(n_clusters, batch_size, seed, init_centroids='random'):
    # init kmeans
    if batch_size is not None:
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=seed,
            init=init_centroids,  # 'random', 'k-means++', ndarray (n_clusters, n_features)
            max_iter=100,
            tol=0,  # 0.0001 (if not zero, adds compute overhead)
            n_init=1,
            # verbose=True,
            batch_size=batch_size,
            compute_labels=True,
            max_no_improvement=100,  # None
            init_size=None,
            reassignment_ratio=0.1 / n_clusters,
        )
    else:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=seed,
            init=init_centroids,  # 'random', 'k-means++', ndarray (n_clusters, n_features)
            max_iter=100,
            tol=0.001,
            n_init=1,
            # verbose=True,
            precompute_distances=True,
            algorithm='full',  # 'full',  # 'elkan',
        )
    return kmeans


def init_kmeans(implementation, **kwargs):
    if implementation == 'sklearn':
        return init_kmeans_sklearn(**kwargs)
    elif implementation == 'python':
        return init_kmeans_python(**kwargs)
    else:
        raise NotImplementedError


def load(in_file):
    with open(in_file, 'rb') as f:
        data = pickle.load(f)
    return data


def dump(data, out_file):
    with open(out_file, 'wb') as out:
        pickle.dump(data, out)


def timer(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        print(f'{func.__name__}() starts at {datetime.now()}')
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'{func.__name__}() ends at {datetime.now()}')
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


def plot_centroids(history, title = ''):
    res = history['res']

    seeds = []
    initial_centroids = []
    final_centroids = []
    scores = []
    iterations = []
    for vs in res:
        seeds.append(vs['seed'])
        initial_centroids.append(vs['initial_centroids'])
        final_centroids.append(vs['final_centroids'])
        scores.append(vs['scores'])
        iterations.append(vs['training_iterations'])

    nrows, ncols = 6, 5
    fig, axes = plt.subplots(nrows, ncols,sharex=True, sharey=True, figsize=(15, 15))

    for i, seed in enumerate(seeds):
        r, c = divmod(i, ncols)
        # print(i, seed, r, c)
        ax = axes[r, c]
        ps = initial_centroids[i]
        for j, p in enumerate(ps):
            ax.scatter(p[0], p[1], c='gray', marker="o", s=100, label='initial' if j == 0 else '')
            ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=(p[0], p[1]+2*(j+1)), fontsize= 8, color='gray',
                        bbox=dict(facecolor='none', edgecolor='gray', pad=0.3),
                        arrowprops=dict(arrowstyle="->", color='gray', shrinkA = 1,
                                        connectionstyle="angle3, angleA=90,angleB=0"))

        ps = final_centroids[i]
        for j, p in enumerate(ps):
            ax.scatter(p[0], p[1], c='r', marker="*", s=100, label='final' if j == 0 else '')
            ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]),xytext=(p[0], p[1]-2*(j+1)), fontsize= 8, color='r',
                        bbox=dict(facecolor='none', edgecolor='red', pad=0.3),
                        arrowprops=dict(arrowstyle="->", color='r', shrinkA=1,
                                        connectionstyle="angle3, angleA=90,angleB=0"))

        train_db, test_db = scores[i]['train'], scores[i]['test']
        ax.set_title(f'Train: {train_db:.2f}, Test: {test_db:.2f}\nIterations: {iterations[i]}')

        # ax.set_xlim([-10, 10]) # [-3, 7]
        # ax.set_ylim([-15, 15])  # [-3, 7]
        ax.set_xlim([-5, 5])  # [-3, 7]
        ax.set_ylim([-5, 5])  # [-3, 7]

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
    fig.suptitle(title, fontsize = 20)
    # # Put a legend below current axis
    # plt.legend(loc='lower center', bbox_to_anchor=(-.5, -0.5),   # (x0,y0, width, height)=(0,0,1,1)).
    #           fancybox=False, shadow=False, ncol=2)
    # plt.xlim([-2, 15])
    # plt.ylim([-2, 15])
    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout()
    plt.show()


def plot_rhos(history, title = ''):
    res = history['res']

    seeds = []
    initial_centroids = []
    final_centroids = []
    scores = []
    iterations = []
    for vs in res:
        seeds.append(vs['seed'])
        initial_centroids.append(vs['initial_centroids'])
        final_centroids.append(vs['final_centroids'])
        scores.append(vs['scores'])
        iterations.append(vs['training_iterations'])

    nrows, ncols = 6, 5
    fig, axes = plt.subplots(nrows, ncols,sharex=True, sharey=True, figsize=(15, 15))

    for i, seed in enumerate(seeds):
        r, c = divmod(i, ncols)
        # print(i, seed, r, c)
        ax = axes[r, c]
        ps = initial_centroids[i]
        for j, p in enumerate(ps):
            ax.scatter(p[0], p[1], c='gray', marker="o", s=50, label='initial' if j == 0 else '')
            ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=(p[0], p[1]+2*(j+1)),
                        arrowprops=dict(arrowstyle="->", facecolor='r', shrinkA = 1,
                                        connectionstyle="angle3, angleA=90,angleB=0"))

        ps = final_centroids[i]
        for j, p in enumerate(ps):
            ax.scatter(p[0], p[1], c='r', marker="*", s=50, label='final' if j == 0 else '')
            ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]),xytext=(p[0], p[1]-2*(j+1)),
                        arrowprops=dict(arrowstyle="->", facecolor='r', shrinkA=1,
                                        connectionstyle="angle3, angleA=90,angleB=0"))

        train_db, test_db = scores[i]['train'], scores[i]['test']
        ax.set_title(f'Train: {train_db:.2f}, Test: {test_db:.2f}\nIterations: {iterations[i]}')

        ax.set_xlim([-3, 7])
        ax.set_ylim([-3, 7])

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
    fig.suptitle(title, fontsize = 20)
    # # Put a legend below current axis
    # plt.legend(loc='lower center', bbox_to_anchor=(-.5, -0.5),   # (x0,y0, width, height)=(0,0,1,1)).
    #           fancybox=False, shadow=False, ncol=2)
    # plt.xlim([-2, 15])
    # plt.ylim([-2, 15])
    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout()
    plt.show()



