import copy
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from fkm.utils.utils_func import timer

project_dir = os.path.dirname(os.getcwd())


def davies_bouldin(x, labels, centroids, verbose=False):
    # print(f'davies_bouldin, centroids: {centroids}, when two centroids are same, there will be Nan.')
    # DIY
    if len(np.unique(labels)) != centroids.shape[0]:
        print(f'len(np.unique(labels)):{len(np.unique(labels))}!= centroids.shape[0]: {centroids.shape[0]}')
        return 10.0
    NUM_CLUSTERS = centroids.shape[0]
    distances = np.sqrt(np.sum(np.square(x - centroids[labels]), axis=1))  # each point to its centroid's distance

    # centroid distances: the distance between each two centroids
    centroid_dist_matrix = np.expand_dims(centroids, axis=0) - np.expand_dims(centroids, axis=1)
    centroid_dist_matrix = np.sqrt(np.sum(np.square(centroid_dist_matrix), axis=2))
    # print(centroid_dist_matrix - metrics.pairwise.euclidean_distances(X=centroids, Y=centroids))
    # reassign the diagonal
    centroid_dist_matrix[range(NUM_CLUSTERS), range(NUM_CLUSTERS)] = float("inf")
    for i in range(NUM_CLUSTERS):
        if len([1 for v in centroid_dist_matrix[i] if (np.isnan(v) or np.isinf(v))]) > 1:
            print('***WARNING: db score may be not correct.')
            print(centroid_dist_matrix)
            break
    # print(centroid_dist_matrix)

    # intra cluster dist
    intra_dist = np.zeros(NUM_CLUSTERS)

    for i in range(NUM_CLUSTERS):
        # if len(distances[i == labels]) > 0 :
        intra_dist[i] = np.mean(distances[i == labels])  # the average distance of all points to its centroid.
        # wiki: q = 1, p = 2 for db score
    s_ij = np.expand_dims(intra_dist, axis=0) + np.expand_dims(intra_dist, axis=1)
    d_i = np.nanmax(s_ij / centroid_dist_matrix, axis=1)
    db_score = np.nanmean(d_i)
    if verbose > 5:
        print("centroid_min_dist", np.amin(centroid_dist_matrix, axis=1))
        print("intra_dist", intra_dist)
    return db_score


def euclidean_dist(x, labels, centroids):
    # distances = np.sqrt(np.sum(np.square(x - centroids[labels]), axis=1))
    distances = np.sum(np.square(x - centroids[labels]), axis=1)
    dist = np.mean(distances)
    return dist


def evaluate(kmeans, x, splits=['train', 'test'], use_metric='euclidean', federated=False, verbose=False):
    scores = {}
    centroids = kmeans.cluster_centers_
    for split in splits:
        if federated:
            x[split] = np.concatenate(x[split], axis=0)
        labels = kmeans.predict(x[split])  # y and labels misalign, so you can't use y directly
        if verbose:
            print(split, use_metric)
        if "davies_bouldin" == use_metric:
            score = davies_bouldin(x[split], labels, centroids, verbose)
        elif "silhouette" == use_metric:
            # silhouette (takes forever) -> need metric with linear execution time wrt data size
            score = metrics.silhouette_score(x[split], labels)
        else:
            assert use_metric == 'euclidean'
            score = euclidean_dist(x[split], labels, centroids)
        scores[split] = score
        if verbose:
            print(score)
    return scores

@timer
def evaluate2(kmeans, x, y=None, splits=['train', 'test'], federated=False, verbose=False):
    scores = {}
    centroids = kmeans.cluster_centers_
    x = copy.deepcopy(x)
    y = copy.deepcopy(y)
    for split in splits:
        if federated:
            # for federated KMeans, we collect all clients' data together as test set.
            # Then evaluate the model on the whole test set.
            x[split] = np.concatenate(x[split], axis=0)
            if y is not None:
                y[split] = np.concatenate(y[split], axis=0)
        labels = kmeans.predict(x[split])  # y and labels misalign, so you can't use y directly
        try:
            db = davies_bouldin(x[split], labels, centroids, verbose)
        except Exception as e:
            db = f'Error: {e}'

        try:
            sil = metrics.silhouette_score(x[split], labels)
        except Exception as e:
            sil = f'Error: {e}'

        try:
            euclidean = euclidean_dist(x[split], labels, centroids)
        except Exception as e:
            euclidean = f'Error: {e}'

        score = {
            'davies_bouldin': db,
            'silhouette': sil,
            'euclidean': euclidean
        }
        scores[split] = score
        if verbose > 5:
            print(score)
    return scores


def plot_stats(stats, x_variable, x_variable_name, metric_name, title=''):
    for spl, spl_dict in stats.items():
        for stat, stat_values in spl_dict.items():
            stats[spl][stat] = np.array(stat_values)

    if x_variable[-1] is None:
        x_variable[-1] = 1
    x_variable = ["single" if i == 0.0 else i for i in x_variable]

    x_axis = np.array(range(len(x_variable)))
    plt.plot(stats['train']['avg'], 'ro-', label='Train')
    plt.plot(stats['test']['avg'], 'b*-', label='Test')
    plt.fill_between(
        x_axis,
        stats['train']['avg'] - stats['train']['std'],
        stats['train']['avg'] + stats['train']['std'],
        facecolor='r',
        alpha=0.3,
    )
    plt.fill_between(
        x_axis,
        stats['test']['avg'] - stats['test']['std'],
        stats['test']['avg'] + stats['test']['std'],
        facecolor='b',
        alpha=0.2,
    )
    # capsize = 4
    # plt.errorbar(x_axis, stats['train']['avg'], yerr=stats['train']['std'], fmt='g*-',
    #              capsize=capsize, lw=2, capthick=2, ecolor='r', label='Train', alpha=0.3)
    # plt.errorbar(x_axis, stats['test']['avg'], yerr=stats['test']['std'], fmt='bo-',
    #              capsize=capsize, lw=2, capthick=2, ecolor='m', label='Test', alpha=0.3)
    plt.xticks(x_axis, x_variable)
    plt.xlabel(x_variable_name)
    # plt.ylabel('Davies-Bouldin Score')
    plt.ylabel(metric_name)
    plt.legend(loc='upper right')
    plt.ylim((0, 3))
    plt.title(title)
    fig_path = os.path.join(project_dir, "results")
    # plt.savefig(os.path.join(fig_path, "stats_{}.png".format(x_variable_name)), dpi=600, bbox_inches='tight')
    plt.savefig(f'{fig_path}/{title}.pdf', dpi=600, bbox_inches='tight')
    plt.show()


def plot_stats2(stats, x_variable, x_variable_name, metric_name, title=''):
    res = {}
    # stats = {C: {'train': {'davies_bouldin': (mean, std), 'silhouette':() , 'euclidean': () }, 'test': }}
    nrows, ncols = 1, 3
    fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=False, figsize=(10, 5))  # (width, height)
    axes = axes.reshape((nrows, ncols))
    for i, metric_name in enumerate(['davies_bouldin', 'silhouette', 'euclidean']):
        metric = {}
        for split in ['train', 'test']:
            avgs = []
            stds = []
            for c, vs in stats.items():
                mean, std = vs[split][metric_name]
                avgs.append(mean)
                stds.append(std)
            metric[split] = {'avg': avgs[:], 'std': stds[:]}
        if x_variable[-1] is None:
            x_variable[-1] = 1
        x_variable = ["single" if i == 0.0 else i for i in x_variable]

        x_axis = np.array(range(len(x_variable)))
        # plt.plot(metric['train']['avg'], 'ro-', label='Train')
        # plt.plot(metric['test']['avg'], 'b*-', label='Test')
        # plt.fill_between(
        #     x_axis,
        #     metric['train']['avg'] - metric['train']['std'],
        #     metric['train']['avg'] + metric['train']['std'],
        #     facecolor='r',
        #     alpha=0.3,
        # )
        # plt.fill_between(
        #     x_axis,
        #     metric['test']['avg'] - metric['test']['std'],
        #     metric['test']['avg'] + metric['test']['std'],
        #     facecolor='b',
        #     alpha=0.2,
        # )
        capsize = 4
        ax = axes[0, i]
        ax.errorbar(x_axis, metric['train']['avg'], yerr=metric['train']['std'], fmt='g*-',
                    capsize=capsize, lw=2, capthick=2, ecolor='r', label=f'Train', alpha=0.3)
        ax.errorbar(x_axis, metric['test']['avg'], yerr=metric['test']['std'], fmt='bo-',
                    capsize=capsize, lw=2, capthick=2, ecolor='m', label=f'Test', alpha=0.3)

        # plt.ylabel('Davies-Bouldin Score')
        ax.set_ylabel(metric_name)
        # plt.xticks(x_axis, x_variable)
        # plt.xlabel(x_variable_name)
        ax.legend(loc='upper right')
        # ax.ylim((0, 3))
        # ax.set_title(title)

    fig.suptitle(title, fontsize=20)
    plt.tight_layout()
    fig_path = os.path.join(project_dir, "results")
    # plt.savefig(os.path.join(fig_path, "stats_{}.png".format(x_variable_name)), dpi=600, bbox_inches='tight')
    plt.savefig(f'{fig_path}/{title}.pdf', dpi=600, bbox_inches='tight')
    plt.show()


def plot_progress(progress_means, progress_stds, record_at):
    #  NOTE: only for dummy data
    # print(len(progress_means), progress_means[0].shape)
    # print(len(progress_stds), progress_stds[0].shape)
    num_clusters = progress_means[0].shape[0]
    num_records = len(progress_means)
    true_means = np.arange(1, num_clusters + 1)
    fig = plt.figure()
    for i in range(num_clusters):
        ax = fig.add_subplot(1, 1, 1)
        x_axis = np.array(range(num_records))
        true_means_i = np.repeat(true_means[i], num_records)
        means = np.array([x[i] for x in progress_means])
        stds = np.array([x[i] for x in progress_stds])
        ax.plot(means, 'r-', label='centroid mean')
        ax.plot(true_means_i, 'b-', label='true mean')
        ax.fill_between(
            x_axis,
            means - stds,
            means + stds,
            facecolor='r',
            alpha=0.4,
            label='centroid std',
        )
        # ax.fill_between(
        #     x_axis,
        #     true_means_i - 0.1,
        #     true_means_i + 0.1,
        #     facecolor='b',
        #     alpha=0.1,
        #     label='true std',
        # )
        plt.xticks(x_axis, record_at)
    plt.xlabel("Round")
    plt.ylabel("Cluster distribution")
    # plt.legend()
    fig_path = os.path.join(project_dir, "results")
    plt.savefig(os.path.join(fig_path, "stats_{}.png".format("progress")), dpi=600, bbox_inches='tight')
    plt.show()
