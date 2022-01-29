"""

"""
import os
import pickle
import shutil
import time
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

# project_dir = os.path.dirname(os.getcwd())
project_dir = os.getcwd()
project_dir = project_dir if os.path.isdir(project_dir) else os.path.dirname(project_dir)


def randomly_init_centroid(min_value, max_value, n_dims, repeats=1, random_state=42):
    r = np.random.RandomState(random_state)
    if repeats == 1:
        # return min_value + (max_value - min_value) * np.random.rand(n_dims)
        return min_value + (max_value - min_value) * r.rand(n_dims)  # range [min_val, max_val]
    else:
        # return min_value + (max_value - min_value) * np.random.rand(repeats, n_dims)
        return min_value + (max_value - min_value) * r.rand(repeats, n_dims)


def random_initialize_centroids(X, n_clusters, random_state=42):
    n, dim = X.shape
    r = np.random.RandomState(max(1, random_state))
    # centroids = np.zeros((n_clusters, dim))
    # for i in range(dim):
    #     centroids[:, i] = r.choice(X[:, i], size= n_clusters ,replace=False)  # without replacement and random

    indices = r.choice(range(0, n), size=n_clusters, replace=False)  # without replacement and random
    centroids = X[indices]
    return centroids


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
    out_dir = os.path.dirname(out_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
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


@timer
def plot_centroids(history, out_dir='results', title='', fig_name='fig', params={}, is_show=False):
    plt.close()
    res = history['results']

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
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(15, 15))

    for i, seed in enumerate(seeds):
        if i >= nrows * ncols: break
        r, c = divmod(i, ncols)
        # print(i, seed, r, c)
        ax = axes[r, c]
        ps = initial_centroids[i]
        for j, p in enumerate(ps):
            ax.scatter(p[0], p[1], c='gray', marker="o", s=100, label='initial' if j == 0 else '')
            # offset = 0.9 * (j+1)
            offset = 0.9
            xytext = (p[0] , p[1] + offset)
            ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=8, color='black',
                        ha='center', va='center',  # textcoords='offset points',
                        bbox=dict(facecolor='none', edgecolor='gray', pad=1),
                        arrowprops=dict(arrowstyle="->", color='gray', shrinkA=1,
                                        connectionstyle="angle3, angleA=90,angleB=0"))

        ps = final_centroids[i]
        for j, p in enumerate(ps):
            ax.scatter(p[0], p[1], c='r', marker="*", s=100, label='final' if j == 0 else '')
            # offset = 0.9 * (j+1)
            offset = 0.9
            xytext = (p[0], p[1] - offset)
            ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=8, color='r',
                        ha='center', va='center',
                        bbox=dict(facecolor='none', edgecolor='red', pad=1),
                        arrowprops=dict(arrowstyle="->", color='r', shrinkA=1,
                                        connectionstyle="angle3, angleA=90,angleB=0"))

        train_db, test_db = scores[i]['train']['davies_bouldin'], scores[i]['test']['davies_bouldin']
        ax.set_title(f'Train: {train_db:.2f}, Test: {test_db:.2f}\nIterations: {iterations[i]}, Seed: {seed}')

        # # ax.set_xlim([-10, 10]) # [-3, 7]
        # # ax.set_ylim([-15, 15])  # [-3, 7]
        ax.set_xlim([-4, 4])  # [-3, 7]
        ax.set_ylim([-4, 4])  # [-3, 7]

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
    fig.suptitle(title, fontsize=20)
    # # Put a legend below current axis
    # plt.legend(loc='lower center', bbox_to_anchor=(-.5, -0.5),   # (x0,y0, width, height)=(0,0,1,1)).
    #           fancybox=False, shadow=False, ncol=2)
    # plt.xlim([-2, 15])
    # plt.ylim([-2, 15])
    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{fig_name}.png")
    tmp_dir = os.path.dirname(fig_path)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    if is_show:
        plt.show()

    return fig_path


def plot_rhos(history, title='', fig_name=''):
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
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(15, 15))

    for i, seed in enumerate(seeds):
        if i >= nrows * ncols: break
        r, c = divmod(i, ncols)
        # print(i, seed, r, c)
        ax = axes[r, c]
        ps = initial_centroids[i]
        for j, p in enumerate(ps):
            ax.scatter(p[0], p[1], c='gray', marker="o", s=50, label='initial' if j == 0 else '')
            ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=(p[0], p[1]),
                        arrowprops=dict(arrowstyle="->", facecolor='r', shrinkA=1,
                                        connectionstyle="angle3, angleA=90,angleB=0"))

        ps = final_centroids[i]
        for j, p in enumerate(ps):
            ax.scatter(p[0], p[1], c='r', marker="*", s=50, label='final' if j == 0 else '')
            ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=(p[0], p[1]),
                        arrowprops=dict(arrowstyle="->", facecolor='r', shrinkA=1,
                                        connectionstyle="angle3, angleA=90,angleB=0"))

        train_db, test_db = scores[i]['train'], scores[i]['test']
        ax.set_title(f'Train: {train_db:.2f}, Test: {test_db:.2f}\nIterations: {iterations[i]}')

        ax.set_xlim([-3, 7])
        ax.set_ylim([-3, 7])

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
    fig.suptitle(title, fontsize=20)
    # # Put a legend below current axis
    # plt.legend(loc='lower center', bbox_to_anchor=(-.5, -0.5),   # (x0,y0, width, height)=(0,0,1,1)).
    #           fancybox=False, shadow=False, ncol=2)
    # plt.xlim([-2, 15])
    # plt.ylim([-2, 15])
    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout()
    fig_path = os.path.join(project_dir, "results")
    plt.savefig(os.path.join(fig_path, f"{fig_name}.png"), dpi=600, bbox_inches='tight')
    plt.show()

    return fig_path


def figs2movie(images, out_file='video.mp4'):
    """ # If width > 8000 or height > 8000, the writer won't work

    Parameters
    ----------
    images
    out_file

    Returns
    -------

    """
    import cv2

    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    # width, height = 9000, 9000

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # fourcc = -1
    s = 0  # If width > 8000, it can't work
    if width > 8000 and height > 8000:
        s = max(width, height) - 8000
    fps = 0.1
    video = cv2.VideoWriter(out_file, fourcc, fps, (width - s, height - s))

    for image in images:
        print(image)
        # video.write()
        im = cv2.imread(image)
        imS = cv2.resize(im, (width - s, height - s))
        video.write(imS)

    cv2.destroyAllWindows()
    video.release()
    print(out_file)


# images = ['/Users/kun/PycharmProjects/fkm/fkm/results/M=2.png',
#            ] * 3
# figs2movie(images, out_file='res.mp4')
@timer
def plot_metric_over_time_femnist(history, out_dir='results', title='', fig_name='fig', params={}, is_show=True):
    res = history['results']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    seeds = []
    initial_centroids = []
    final_centroids = []
    scores = []
    iterations = []
    training_histories = []
    for vs in res:
        seeds.append(vs['seed'])
        training_histories.append(vs['history'])
        initial_centroids.append(vs['initial_centroids'])
        final_centroids.append(vs['final_centroids'])
        scores.append(vs['scores'])
        iterations.append(vs['training_iterations'])

    # only show the first 3 results
    # show 'initial centroids' and 'final centroids'
    if params['is_crop_image']:
        image_shape = params['image_shape']
    else:
        image_shape = (28, 28)

    nrows, ncols = 2 * 3, initial_centroids[0].shape[0]
    fig, axes = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(15, 15))
    r = 0
    for i, seed in enumerate(seeds[:3]):
        for j, d in enumerate(initial_centroids[i]):
            ax = axes[i * 2, j]
            d = d.reshape(image_shape)
            ax.imshow(d, cmap='gray')
            if j == 0:
                ax.set_title(f'initial, seed:{seed},\nCentroid: {j}')
            else:
                ax.set_title(f'Centroid: {j}')

        for j, d in enumerate(final_centroids[i]):
            ax = axes[i * 2 + 1, j]
            d = d.reshape(image_shape)
            ax.imshow(d, cmap='gray')
            if j == 0:
                ax.set_title(f'final, seed:{seed},\nCentroid: {j}')
            else:
                ax.set_title(f'Centroid: {j}')
    # fig.set_facecolor("black")
    fig.suptitle(title, fontsize=20)
    # # Put a legend below current axis
    # plt.legend(loc='lower center', bbox_to_anchor=(-.5, -0.5),   # (x0,y0, width, height)=(0,0,1,1)).
    #           fancybox=False, shadow=False, ncol=2)
    # plt.xlim([-2, 15])
    # plt.ylim([-2, 15])
    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{fig_name}-centroids.png")
    tmp_dir = os.path.dirname(fig_path)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    if is_show:
        plt.show()

    # show screos
    nrows, ncols = 6, 5
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(15, 15))
    axes = axes.reshape((nrows, ncols))
    for i, seed in enumerate(seeds):
        if i >= nrows * ncols: break
        r, c = divmod(i, ncols)
        # print(i, seed, r, c)
        ax = axes[r, c]
        # iter_, centroids_, scores_ = zip(*training_histories[i])
        iter_ = []
        centroids_ = []
        scores_ = []
        for t_i, training_h in enumerate(training_histories[i]):
            iter_.append(training_h['iteration'])
            centroids_.append(training_h['centroids'])
            scores_.append(training_h['scores'])

        ax.plot(iter_, [vs['train']['davies_bouldin'] for vs in scores_], label='train')
        ax.plot(iter_, [vs['test']['davies_bouldin'] for vs in scores_], label='test')

        train_db, test_db = scores[i]['train']['davies_bouldin'], scores[i]['test']['davies_bouldin']
        ax.set_title(f'Train: {train_db:.2f}, Test: {test_db:.2f}\nIterations: {iterations[i]}. Seed: {seed}')
        ax.set_ylabel('Davies Bouldin Score')
        ax.set_xlabel('Iterations')

        # # print(i, seed, r, c)
        # ax = axes[r+1, c]
        # iter_, centroids_, scores_ = zip(*training_histories[i])
        # ax.plot(iter_, [vs['train']['euclidean'] for vs in scores_], label='train')
        # ax.plot(iter_, [vs['test']['euclidean'] for vs in scores_], label='test')
        #
        # train_db, test_db = scores[i]['train']['euclidean'], scores[i]['test']['euclidean']
        # ax.set_title(f'Train: {train_db:.2f}, Test: {test_db:.2f}\nIterations: {iterations[i]}')
        # ax.set_ylabel('Average of within-clusters\' distances')
        # ax.set_xlabel('Iterations')

        # ax.set_xlim([0, 10]) # [-3, 7]
        # ax.set_ylim([0, 5])  # [-3, 7]
        # ax.set_xlim([-5, 5])  # [-3, 7]
        # ax.set_ylim([-5, 5])  # [-3, 7]

        # ax.axvline(x=0, color='k', linestyle='--')
        # ax.axhline(y=0, color='k', linestyle='--')
        if i == 0:
            ax.legend(loc='upper right')
    fig.suptitle(title, fontsize=20)
    # # Put a legend below current axis
    # plt.legend(loc='lower center', bbox_to_anchor=(-.5, -0.5),   # (x0,y0, width, height)=(0,0,1,1)).
    #           fancybox=False, shadow=False, ncol=2)
    # plt.xlim([-2, 15])
    # plt.ylim([-2, 15])
    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{fig_name}-scores.png")
    tmp_dir = os.path.dirname(fig_path)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    if is_show:
        plt.show()

    return fig_path


@timer
def plot_metric_over_time_2gaussian(history, out_dir='results', title='', fig_name='fig', params={}, is_show=True):
    res = history['results']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    seeds = []
    initial_centroids = []
    final_centroids = []
    scores = []
    iterations = []
    training_histories = []
    for vs in res:
        seeds.append(vs['seed'])
        training_histories.append(vs['history'])
        initial_centroids.append(vs['initial_centroids'])
        final_centroids.append(vs['final_centroids'])
        scores.append(vs['scores'])
        iterations.append(vs['training_iterations'])

    # only show the first 3 results
    # show 'initial centroids' and 'final centroids'
    nrows, ncols = 2 * 3, initial_centroids[0].shape[0]
    fig, axes = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(15, 15))
    r = 0
    for i, seed in enumerate(seeds[:3]):
        for j, p in enumerate(initial_centroids[i]):
            ax = axes[i * 2, j]

            ax.scatter(p[0], p[1], c='gray', marker="o", s=50, label='initial' if j == 0 else '')
            ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=(p[0], p[1]),
                        ha='center', va='center',
                        arrowprops=dict(arrowstyle="->", facecolor='r', shrinkA=1,
                                        connectionstyle="angle3, angleA=90,angleB=0"))
            if j == 0:
                ax.set_title(f'initial, seed:{seed},\nCentroid: {j}')
            else:
                ax.set_title(f'Centroid: {j}')

        for j, p in enumerate(final_centroids[i]):
            ax = axes[i * 2 + 1, j]
            ax.scatter(p[0], p[1], c='r', marker="*", s=50, label='final' if j == 0 else '')
            ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=(p[0], p[1]),
                        ha='center', va='center',
                        arrowprops=dict(arrowstyle="->", facecolor='r', shrinkA=1,
                                        connectionstyle="angle3, angleA=90,angleB=0"))
            if j == 0:
                ax.set_title(f'final, seed:{seed},\nCentroid: {j}')
            else:
                ax.set_title(f'Centroid: {j}')

    # fig.set_facecolor("black")
    fig.suptitle(title, fontsize=20)
    # # Put a legend below current axis
    # plt.legend(loc='lower center', bbox_to_anchor=(-.5, -0.5),   # (x0,y0, width, height)=(0,0,1,1)).
    #           fancybox=False, shadow=False, ncol=2)
    # plt.xlim([-2, 15])
    # plt.ylim([-2, 15])
    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{fig_name}-centroids.png")
    tmp_dir = os.path.dirname(fig_path)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    if is_show:
        plt.show()

    # show screos
    nrows, ncols = 6, 5
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(15, 15))
    axes = axes.reshape((nrows, ncols))
    for i, seed in enumerate(seeds):
        if i >= nrows * ncols: break
        r, c = divmod(i, ncols)
        # print(i, seed, r, c)
        ax = axes[r, c]
        # iter_, centroids_, scores_ = zip(*training_histories[i])
        iter_ = []
        centroids_ = []
        scores_ = []
        for t_i, training_h in enumerate(training_histories[i]):
            iter_.append(training_h['iteration'])
            centroids_.append(training_h['centroids'])
            scores_.append(training_h['scores'])

        ax.plot(iter_, [vs['train']['davies_bouldin'] for vs in scores_], 'g-', label='train')
        ax.plot(iter_, [vs['test']['davies_bouldin'] for vs in scores_], 'r-', label='test')

        train_db, test_db = scores[i]['train']['davies_bouldin'], scores[i]['test']['davies_bouldin']
        ax.set_title(f'Train: {train_db:.2f}, Test: {test_db:.2f}\nIterations: {iterations[i]}. Seed: {seed}')
        ax.set_ylabel('Davies Bouldin Score')
        ax.set_xlabel('Iterations')

        # # print(i, seed, r, c)
        # ax = axes[r+1, c]
        # iter_, centroids_, scores_ = zip(*training_histories[i])
        # ax.plot(iter_, [vs['train']['euclidean'] for vs in scores_], label='train')
        # ax.plot(iter_, [vs['test']['euclidean'] for vs in scores_], label='test')
        #
        # train_db, test_db = scores[i]['train']['euclidean'], scores[i]['test']['euclidean']
        # ax.set_title(f'Train: {train_db:.2f}, Test: {test_db:.2f}\nIterations: {iterations[i]}')
        # ax.set_ylabel('Average of within-clusters\' distances')
        # ax.set_xlabel('Iterations')

        # ax.set_xlim([0, 10]) # [-3, 7]
        # ax.set_ylim([0, 5])  # [-3, 7]
        # ax.set_xlim([-5, 5])  # [-3, 7]
        # ax.set_ylim([-5, 5])  # [-3, 7]

        # ax.axvline(x=0, color='k', linestyle='--')
        # ax.axhline(y=0, color='k', linestyle='--')
        if i == 0:
            ax.legend(loc='upper right')
    fig.suptitle(title, fontsize=20)
    # # Put a legend below current axis
    # plt.legend(loc='lower center', bbox_to_anchor=(-.5, -0.5),   # (x0,y0, width, height)=(0,0,1,1)).
    #           fancybox=False, shadow=False, ncol=2)
    # plt.xlim([-2, 15])
    # plt.ylim([-2, 15])
    # plt.xticks([])
    # plt.yticks([])
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{fig_name}-scores.png")
    tmp_dir = os.path.dirname(fig_path)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    if is_show:
        plt.show()

    return fig_path


def save_image2disk(data, out_dir, params):
    X, Y = data
    if params['is_crop_image']:
        image_shape = params['image_shape']
    else:
        image_shape = (28, 28)
    for k, clients in X.items():  # x={'train': clients, 'test': clients}
        print(k)
        k_dir = os.path.join(out_dir, k, 'image')
        if os.path.exists(k_dir):
            shutil.rmtree(k_dir, ignore_errors=True)
        for idx, (client, ys) in enumerate(zip(clients, Y[k])):
            for img_idx, (data, y) in enumerate(zip(client, ys)):
                # ax.imshow(data, cmap='gray')
                # im = Image.fromarray(data)
                # im.save(f)
                f = os.path.join(k_dir, f'client_{idx}/{y}_{img_idx}.png')
                client_dir = os.path.dirname(f)
                if not os.path.exists(client_dir):
                    os.makedirs(client_dir)
                matplotlib.image.imsave(f, data.reshape(image_shape), cmap='gray')
                if img_idx + 1 == len(ys):
                    print(f)


def predict_n_saveimg(kmeans, X, Y=None, splits=['train', 'test'], SEED=42, federated=False,
                      verbose=False, out_dir='', params={}, is_show=False):
    initial_centroids = kmeans.initial_centroids
    final_centroids = kmeans.cluster_centers_
    for k, clients in X.items():  # x={'train': clients, 'test': clients}
        print(f'predic_n_saveimg: {k}')
        k_dir = os.path.join(out_dir, k, f'SEED_{SEED}-image_pred')
        if os.path.exists(k_dir):
            shutil.rmtree(k_dir, ignore_errors=True)

        if not os.path.exists(k_dir):
            os.makedirs(k_dir)

        ######################################################################
        # save centroids to images
        if params['is_crop_image']:
            image_shape = params['image_shape']
        else:
            image_shape = (28, 28)

        nrows, ncols = 2, 10
        fig, axes = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(15, 6))  # width, height
        f = os.path.join(k_dir, f'random_state_{SEED}-initial_final_centroids.png')
        for i in range(final_centroids.shape[0]):
            # data = final_centroids[i]
            # matplotlib.image.imsave(f, data.reshape((28, 28)), cmap='gray')
            # initial centroid
            ax = axes[0, i]
            d = initial_centroids[i].reshape(image_shape)
            ax.imshow(d, cmap='gray')
            if i == 0:
                ax.set_title(f'initial centroid\ncentroid: {i}')
            else:
                ax.set_title(f'centroid: {i}')
            # final centroid
            ax = axes[1, i]
            d = final_centroids[i].reshape(image_shape)
            ax.imshow(d, cmap='gray')
            if i == 0:
                ax.set_title(f'final centroid\ncentroid: {i}')
            else:
                ax.set_title(f'centroid: {i}')

        plt.tight_layout()
        plt.savefig(f, dpi=600, bbox_inches='tight')
        if is_show:
            plt.show()
        # plt.clf()
        plt.close(fig)

        ######################################################################
        if federated == False:
            clients = [clients]
            Y[k] = [Y[k]]
        for idx, (client, ys) in enumerate(zip(clients, Y[k])):
            print(f'process client {idx}')
            y_preds = kmeans.predict(client)  # y and labels misalign, so you can't use y directly
            step = max(1, len(ys) // 10)  # only save 10 images
            for img_idx, (data, y, y_pred) in enumerate(zip(client, ys, y_preds)):
                if img_idx % step != 0: continue
                # ax.imshow(data, cmap='gray')
                # im = Image.fromarray(data)
                # im.save(f)
                f = os.path.join(k_dir, f'client_{idx}/y{y}-y_pred{y_pred}-{img_idx}.png')
                client_dir = os.path.dirname(f)
                if not os.path.exists(client_dir):
                    os.makedirs(client_dir)
                # matplotlib.image.imsave(f, data.reshape((28, 28)), cmap='gray')

                nrows, ncols = 1, 2
                fig, axes = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(6, 3))  # width, height
                # y_ture
                ax = axes[0]
                d = data.reshape(image_shape)
                ax.imshow(d, cmap='gray')
                ax.set_title(f'y_true: {y}')

                # y_pred
                ax = axes[1]
                d = final_centroids[y_pred].reshape(image_shape)
                ax.imshow(d, cmap='gray')
                ax.set_title(f'y_pred: {y_pred}')

                # fig.set_facecolor("black")
                fig.suptitle(f'img_idx: {img_idx}', fontsize=20)

                plt.tight_layout()

                plt.savefig(f, dpi=600, bbox_inches='tight')
                # plt.clf()
                if is_show:
                    plt.show()

                if img_idx + 1 == len(ys):
                    print(f)
                print(f)
            plt.close(fig)


@timer
def obtain_true_centroids(X_dict, y_dict, splits=['train', 'test'], params=None):
    all_centroids = {}
    for split in splits:
        x = np.concatenate(X_dict[split], axis=0)
        y = np.concatenate(y_dict[split], axis=0)

        labels = np.unique(y)
        n_clusters = len(labels)
        centroids = np.zeros((n_clusters, len(x[0])))
        for j, label in enumerate(sorted(labels, reverse=False)):
            mask = np.equal(y, label)
            cnt = np.sum(mask)
            if cnt > 0:
                centroids[j] = np.sum(x[mask], axis=0) / cnt
        all_centroids[split] = centroids[:params['n_clusters']]
    return all_centroids


def plot_centroids_diff_over_time(history,
                                  out_dir='',
                                  title='', fig_name='',
                                  params={}, is_show=True):
    results = history['results']
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    seeds = []
    initial_centroids = []
    final_centroids = []
    scores = []
    iterations = []
    training_histories = []
    true_centroids = []
    for vs in results:
        seeds.append(vs['seed'])
        training_histories.append(vs['history'])
        initial_centroids.append(vs['initial_centroids'])
        true_centroids.append(vs['true_centroids'])
        final_centroids.append(vs['final_centroids'])
        scores.append(vs['scores'])
        iterations.append(vs['training_iterations'])

    ######################################################################
    # save centroids to images
    nrows, ncols = 6, 5
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(15, 15))  # width, height
    # axes = axes.reshape((nrows, ncols))
    f = os.path.join(out_dir, f'centroids_diff.png')
    colors = ["r", "g", "b", "m", 'black']
    for i, training_h in enumerate(training_histories):
        if i >= nrows * ncols: break
        seed = seeds[i]
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        #
        # # training_h = [{}, {}]
        # # training_h.append({'iteration': iteration, 'centroids': centroids, 'scores': scores,
        # #                      'centroids_update': centroids_update, 'centroids_diff': centroids_diff})
        x = [vs['iteration'] for vs in training_h]
        y = [np.sum(np.square(vs['centroids_update'])) for vs in training_h]
        ax.plot(x, y, f'b-', label=f'centroids_update')

        split = 'train'
        y = [np.sum(np.square(vs['centroids_diff'][split])) for vs in training_h]
        ax.plot(x, y, f'g-', label=f'{split}, centroids_diff')

        split = 'test'
        y = [np.sum(np.square(vs['centroids_diff'][split])) for vs in training_h]
        ax.plot(x, y, f'r-', label=f'{split}, centroids_diff')

        ax.set_title(f'Iterations: {iterations[i]}, SEED: {seed}')
        ax.set_ylabel('')
        ax.set_xlabel('Iterations')
        if i == 0:
            ax.legend(loc="upper right")
    fig.suptitle(title + fig_name + ', centroids diff')
    plt.tight_layout()
    plt.savefig(f, dpi=600, bbox_inches='tight')
    if is_show:
        plt.show()
    # plt.clf()
    plt.close(fig)

    ######################################################################
    # save updated centroids over time
    nrows, ncols = 6, 5
    # fig, axes = plt.subplots(nrows, ncols,figsize=(15, 15))  # width, height
    fig = plt.figure(figsize=(15, 15))  # width, height
    f = os.path.join(out_dir, f'centroids_updates.png')
    for i, training_h in enumerate(training_histories):
        if i >= nrows * ncols: break
        seed = seeds[i]
        # r, c = divmod(i, ncols)
        # ax = axes[r, c]
        ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
        #
        # # training_h = [{}, {}]
        # # training_h.append({'iteration': iteration, 'centroids': centroids, 'scores': scores,
        # #                      'centroids_update': centroids_update, 'centroids_diff': centroids_diff})
        x = [vs['iteration'] for vs in training_h]
        y = [vs['centroids'] for vs in training_h]  # [centroids.shape = (K, dim), ..., ]
        true_c = true_centroids[i]['train']

        # plot the first 2 centroids
        # y[i]: only plot the first 2 centroids

        # plot the first centroid and for each centroid, only show the first 2 dimensional data.
        y1 = [y_[0][:2] for y_ in y]
        y11, y12 = list(zip(*y1))
        ax.scatter(x, y11, y12, c='r', marker='o', label=f'centroid_1?')
        p = true_c[0]
        # ax.scatter(x, y, z, c='gray', marker="x", s=50)
        ax.scatter(x[0], p[0], p[1], c='gray', marker="o", s=50)
        ax.text(x[0], p[0], p[1], f'({p[0]:.1f},{p[1]:.1f})', fontsize=8, color='b',
                # bbox=dict(facecolor='none', edgecolor='red', pad=1),
                ha='center', va='center')
        # final centroid
        p = y1[-1]
        # ax.scatter(x, y, z, c='gray', marker="x", s=50)
        ax.scatter(x[-1], p[0], p[1], c='r', marker="o", s=50)
        ax.text(x[-1], p[0], p[1], f'({p[0]:.1f},{p[1]:.1f})', fontsize=8, color='b',
                # bbox=dict(facecolor='none', edgecolor='red', pad=1),
                ha='center', va='center')

        # plot the second centroid
        y2 = [y_[1][:2] for y_ in y]
        # ax.scatter(x, y2, f'b-', label=f'centroids_2')
        y21, y22 = list(zip(*y2))
        ax.scatter(x, y21, y22, c='g', marker='x', label=f'centroid_2?')
        p = true_c[1]
        ax.scatter(x[0], p[0], p[1], c='gray', marker="x", s=50)
        # ax.annotate(f'({p[0]:.1f}, {x[0]:.1f}, {p[1]:.1f})', xy=(p[0], x[0], p[1]), xytext=(p[0], x[0], p[1]),
        #             ha='center', va='center',
        #             arrowprops=dict(arrowstyle="->", facecolor='r', shrinkA=1,
        #                             connectionstyle="angle3, angleA=90,angleB=0"))
        ax.text(x[0], p[0], p[1], f'({p[0]:.1f},{p[1]:.1f})', fontsize=8, color='r',
                # bbox=dict(facecolor='none', edgecolor='b', pad=1),
                ha='center', va='center')
        # final centroid
        p = y2[-1]
        # ax.scatter(x, y, z, c='gray', marker="x", s=50)
        ax.scatter(x[-1], p[0], p[1], c='g', marker="x", s=50)
        ax.text(x[-1], p[0], p[1], f'({p[0]:.1f},{p[1]:.1f})', fontsize=8, color='r',
                # bbox=dict(facecolor='none', edgecolor='red', pad=1),
                ha='center', va='center')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('x')
        ax.set_zlabel('y')
        ax.set_title(f'Iterations: {iterations[i]}, SEED: {seed}')
        if i == 0:
            ax.legend(loc="upper right")
        plt.tight_layout()
    fig.suptitle(title + fig_name + ', centroids update')
    plt.tight_layout()
    plt.savefig(f, dpi=600, bbox_inches='tight')
    if is_show:
        plt.show()
    # plt.clf()
    plt.close(fig)


@timer
def history2movie(history, out_dir='.',
                  title='', fig_name='demo',
                  params={}, is_show=True):
    """ # If width > 8000 or height > 8000, the writer won't work

    Parameters
    ----------
    images
    out_file

    Returns
    -------

    """
    import matplotlib.animation as animation

    plt.close()
    res = history['results']

    seeds_list = []
    initial_centroids_list = []
    final_centroids_list = []
    scores_list = []
    training_iterations_list = []
    max_iterations = 0
    for vs in res:
        seeds_list.append(vs['seed'])
        initial_centroids_list.append(vs['initial_centroids'])
        final_centroids_list.append(vs['final_centroids'])
        scores_list.append(vs['scores'])
        training_iterations_list.append(vs['history'])
        max_iterations = max(max_iterations, vs['training_iterations'])
    print(f'max_iterations: {max_iterations}')
    nrows, ncols = 6, 5
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(15, 15))

    fig.suptitle(title, fontsize=20)

    # # Put a legend below current axis
    # plt.legend(loc='lower center', bbox_to_anchor=(-.5, -0.5),   # (x0,y0, width, height)=(0,0,1,1)).
    #           fancybox=False, shadow=False, ncol=2)
    # plt.xlim([-2, 15])
    # plt.ylim([-2, 15])
    # plt.xticks([])
    # plt.yticks([])
    # plt.tight_layout()
    # plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    # plt.tight_layout()

    def animate(ith_iteration):  # for the ith iteration for all repeats
        for i in range(len(seeds_list)):  # for the ith repeat
            if i >= nrows * ncols: break
            if ith_iteration + 1 > len(training_iterations_list[i]): continue
            r, c = divmod(i, ncols)
            # print(i, seed, r, c)
            ax = axes[r, c]
            ax.cla()  # clear the previous plot
            ps = initial_centroids_list[i]
            for j, p in enumerate(ps):
                ax.scatter(p[0], p[1], c='gray', marker="o", s=100, label='initial' if j == 0 else '')
                offset = 1.0
                xytext = (p[0], p[1] + offset)
                ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]),
                            xytext=xytext, fontsize=10, color='black',
                            ha='center', va='center',
                            bbox=dict(facecolor='none', edgecolor='gray', pad=1),
                            arrowprops=dict(arrowstyle="->", color='gray', shrinkA=1,
                                            connectionstyle="angle3, angleA=90,angleB=0"))

            ps = training_iterations_list[i][ith_iteration]['centroids']
            for j, p in enumerate(ps):
                ax.scatter(p[0], p[1], c='r', marker="*", s=100, label='final' if j == 0 else '')
                # offset = 0.9 * (j + 1)
                offset = 1.0
                xytext = (p[0], p[1] + offset)
                ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]),
                            xytext=xytext, fontsize=10, color='r',
                            ha='center', va='center',
                            bbox=dict(facecolor='none', edgecolor='gray', pad=1),
                            arrowprops=dict(arrowstyle="->", color='r', shrinkA=1,
                                            connectionstyle="angle3, angleA=90,angleB=0"))

            train_db, test_db = scores_list[i]['train']['davies_bouldin'], scores_list[i]['test']['davies_bouldin']
            ax.set_title(f'Train: {train_db:.2f}, Test: {test_db:.2f}\nIterations: {ith_iteration + 1}')

            ax.set_xlim([-3, 3])  # [-3, 7]
            ax.set_ylim([-3, 3])  # [-3, 7]

            ax.axvline(x=0, color='k', linestyle='--')
            ax.axhline(y=0, color='k', linestyle='--')
        plt.tight_layout()
        # print(ith_iteration)
        # return ax

    out_file = os.path.join(out_dir, f"{fig_name}.mp4")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ani = animation.FuncAnimation(fig, animate, frames=max_iterations,
                                  interval=2000)  # 1000: milliseconds
    # To save the animation, use e.g.
    ani.save(out_file)
    # or
    # writer = animation.FFMpegWriter(
    #     fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save("movie.mp4", writer=writer)
    # plt.tight_layout()
    if is_show:
        plt.show()

    return out_file
