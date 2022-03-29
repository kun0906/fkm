import collections
import copy
import os
import shutil
import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn import metrics

from fkm.utils.utils_func import timer

project_dir = os.path.dirname(os.getcwd())

def davies_bouldin(x, labels, centroids, verbose=False):
    """
        https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index

        {\displaystyle S_{i}=\left({\frac {1}{T_{i}}}\sum _{j=1}^{T_{i}}{\left|\left|X_{j}-A_{i}\right|\right|_{p}^{q}}\right)^{1/q}}

        here, q = 1 and p = 2

    Parameters
    ----------
    x
    labels
    centroids
    verbose

    Returns
    -------

    """
    if len(np.unique(labels)) != centroids.shape[0]:
        msg = f'***WARNING: len(np.unique(labels)):{len(np.unique(labels))}!= centroids.shape[0]: {centroids.shape[0]}'
        # traceback.print_exc()
        raise ValueError(msg)
        # return
    # Step 1: Compute S_ij
    NUM_CLUSTERS = centroids.shape[0]
    # the sqrt distance of each point x_i to its centroid: || x - \mu||^( 1/2)
    distances = np.sqrt(np.sum(np.square(x - centroids[labels]), axis=1))
    # intra/within cluster dist
    intra_dist = np.zeros(NUM_CLUSTERS)
    for i in range(NUM_CLUSTERS):
        mask = (labels == i)
        if np.sum(mask) > 0:
            intra_dist[i] = np.mean(distances[mask])  # the average "sqrt distance" of all points to its centroid
        else:
            # intra_dist[i] = 0, if set it as 0, then the final db_score will be reduced by average
            intra_dist[i] = np.nan  # ignore this cluster when we compute the final DB score.
    # S_ij = S_i + S_j
    # S_ij = row vector + column vector => matrix (nxn)
    # https://jakevdp.github.io/PythonDataScienceHandbook/02.05-computation-on-arrays-broadcasting.html
    S_ij = np.expand_dims(intra_dist, axis=0) + np.expand_dims(intra_dist, axis=1)

    # Step 2: Compute M_ij
    # centroid distances: the distance between each two centroids (||C_i - C_j||^(1/2))
    centroid_dist_matrix = np.expand_dims(centroids, axis=0) - np.expand_dims(centroids, axis=1)
    M_ij = np.sqrt(np.sum(np.square(centroid_dist_matrix), axis=2))
    # print(centroid_dist_matrix - metrics.pairwise.euclidean_distances(X=centroids, Y=centroids))
    # reassign the diagonal
    M_ij[range(NUM_CLUSTERS), range(NUM_CLUSTERS)] = float("inf")
    for i in range(NUM_CLUSTERS):
        if len([1 for v in M_ij[i] if (np.isnan(v) or np.isinf(v))]) > 1:
            warnings.warn('***WARNING: db score may be not correct.')
            print(M_ij)
            break
    # print(M_ij)

    # Step3: max R_ij = (S_i + S_j)/M_ij
    # for each cluster i. for each row, return the maximum (ignore nan)
    D_i = np.nanmax(S_ij / M_ij, axis=1) # element-wise division.
    db_score = np.nanmean(D_i)  # compute the mean, however, ignore nan
    if verbose > 5:
        print("centroid_min_dist", np.amin(M_ij, axis=1))
        print("intra_dist", intra_dist)

    return db_score


def euclidean_dist(x, labels, centroids):
    labels = [int(v) for v in labels]
    # distances = np.sqrt(np.sum(np.square(x - centroids[labels]), axis=1))
    distances = np.sum(np.square(x - centroids[labels]), axis=1)
    dist = np.mean(distances)
    return dist

#
# def evaluate(kmeans, x, splits=['train', 'test'], use_metric='euclidean', federated=False, verbose=False):
#     scores = {}
#     centroids = kmeans.centroids
#     for split in splits:
#         if federated:
#             x[split] = np.concatenate(x[split], axis=0)
#         labels = kmeans.predict(x[split])  # y and labels misalign, so you can't use y directly
#         if verbose:
#             print(split, use_metric)
#         if "davies_bouldin" == use_metric:
#             score = davies_bouldin(x[split], labels, centroids, verbose)
#         elif "silhouette" == use_metric:
#             # silhouette (takes forever) -> need metric with linear execution time wrt data size
#             score = metrics.silhouette_score(x[split], labels)
#         else:
#             assert use_metric == 'euclidean'
#             score = euclidean_dist(x[split], labels, centroids)
#         scores[split] = score
#         if verbose:
#             print(score)
#     return scores

@timer
def evaluate2(kmeans, x, y=None, splits=['train', 'test'], federated=False, verbose=False):
    scores = {}
    centroids = kmeans.centroids
    x = copy.deepcopy(x)
    y = copy.deepcopy(y)
    for split in splits:
        if federated:
            # for federated KMeans, we collect all clients' data together as test set.
            # Then evaluate the model on the whole test set.
            x[split] = np.concatenate(x[split], axis=0)
            if y is not None:
                y[split] = np.concatenate(y[split], axis=0)
        labels_pred = kmeans.predict(x[split])  # y and labels misalign, so you can't use y directly
        labels_true = [str(v) for v in y[split]]
        labels_pred = [str(v) for v in labels_pred]
        _true = dict(collections.Counter(labels_true))
        _pred = dict(collections.Counter(labels_pred))
        if verbose >= 5: print(f'labels_pred:', _pred)

        if len(_true.items()) != len(_pred.items()):
            msg = f'*** Error: the number of predicted labels is wrong (label_true({len(_true.items())})' \
                  f'!=label_pred({len(_pred.items())}))\n'
            msg += f'label_true: {_true.items()}\n'
            msg += f'label_pred: {_pred.items()}'
            warnings.warn(msg)
            # # traceback.print_exc()
            # # raise ValueError(msg)
            # # require label_true
            # ari = f'length is not match'
            # ami = f'length is not match'
            # fm = f'length is not match'
            # vm = f'length is not match'
            #
            # # no need label_true
            # db = f'length is not match'
            # sil = f'length is not match'
            # ch = f'length is not match'
            # euclidean = f'length is not match'

        try: # need groud truth
            ## Rand Index
            # ri = sklearn.metrics.rand_score(labels_true, labels_pred)
            # Adjusted Rand Index
            ari = metrics.adjusted_rand_score(labels_true, labels_pred)
        except Exception as e:
            msg = f'Error: {e}'
            warnings.warn(msg)
            # ri = np.nan
            ari = f'Error: {e}'

        try:
            # adjust mutual information
            ami = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
        except Exception as e:
            msg = f'Error: {e}'
            warnings.warn(msg)
            # ri = np.nan
            ami = f'Error: {e}'

        try:
            # adjust mutual information
            fm = metrics.fowlkes_mallows_score(labels_true, labels_pred)
        except Exception as e:
            msg = f'Error: {e}'
            warnings.warn(msg)
            # ri = np.nan
            fm = f'Error: {e}'

        try:
            # Compute the Calinski and Harabasz score.
            vm = metrics.v_measure_score(labels_true, labels_pred)
        except Exception as e:
            msg = f'Error: {e}'
            warnings.warn(msg)
            vm = f'Error: {e}'

        try:
            # Compute the Calinski and Harabasz score.
            ch = metrics.calinski_harabasz_score(x[split], labels_pred)  # np.sqrt(recall * precision)
        except Exception as e:
            msg = f'Error: {e}'
            warnings.warn(msg)
            ch = f'Error: {e}'

        try:
            # db = davies_bouldin(x[split], labels, centroids, verbose)
            db = metrics.davies_bouldin_score(x[split], labels_pred)
            # print(f'db: {db}, db2: {db2}')
        except Exception as e:
            db = f'Error: {e}'

        try:
            sil = metrics.silhouette_score(x[split], labels_pred)
        except Exception as e:
            sil = f'Error: {e}'

        try:
            euclidean = euclidean_dist(x[split], labels_pred, centroids)
        except Exception as e:
            euclidean = f'Error: {e}'

        score = {
            'davies_bouldin': db,
            'silhouette': sil,
            'ch': ch,
            'euclidean': euclidean,
            'n_clusters': len(centroids),
            'n_clusters_pred': len(np.unique(labels_pred)),
            'ari': ari,
            'ami': ami,
            'fm':fm,
            'vm': vm,
            'labels_true': _true,
            'labels_pred': _pred
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
