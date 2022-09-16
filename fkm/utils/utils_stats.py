# Source Generated with Decompyle++
# File: utils_stats.cpython-39.pyc (Python 3.9)

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
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster._unsupervised import check_number_of_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_X_y, _safe_indexing
from fkm.utils.utils_func import timer
from fkm.utils.silhouette_plot import silhouette_plot
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


def get_min_max(X):
    mn = np.inf
    mx = -(np.inf)
    (m, n) = X.shape
    for i in range(m):
        for j in range(i + 1, n):
            mn = min(mn, X[i][j])
            mx = max(mx, X[i][j])
    return (mn, mx)


def davies_bouldin_score_normalized(X, labels):
    """Compute the Davies-Bouldin score.

    The score is defined as the average similarity measure of each cluster with
    its most similar cluster, where similarity is the ratio of within-cluster
    distances to between-cluster distances. Thus, clusters which are farther
    apart and less dispersed will result in a better score.

    The minimum score is zero, with lower values indicating better clustering.

    Read more in the :ref:`User Guide <davies-bouldin_index>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    score: float
        The resulting Davies-Bouldin score.
    """
    (X, labels) = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    (n_samples, _) = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)
    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])))
    for k in range(n_labels):
        cluster_k = _safe_indexing(X, labels == k)
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.average(pairwise_distances(cluster_k, [centroid]))
    centroid_distances = pairwise_distances(centroids)
    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0
    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    # print(combined_intra_dists)
    # print(centroid_distances)
    (mn, mx) = get_min_max(centroid_distances)
    if mn == mx:
        mn = 0
    centroid_distances = centroid_distances / (mx - mn)
    (mn, mx) = get_min_max(combined_intra_dists)
    if mn == mx:
        mn = 0
    combined_intra_dists = combined_intra_dists / (mx - mn)
    scores = np.max(combined_intra_dists / centroid_distances,axis=1)
    return np.mean(scores)


def davies_bouldin_score_weighted(X, labels):
    """Compute the Davies-Bouldin score.
        weighted DB score 
        # (n1 * s1 + n2 * s2) / ((n1+n2) M12) 
        # weighted average and consider about cluster sizes. 

    The score is defined as the average similarity measure of each cluster with
    its most similar cluster, where similarity is the ratio of within-cluster
    distances to between-cluster distances. Thus, clusters which are farther
    apart and less dispersed will result in a better score.

    The minimum score is zero, with lower values indicating better clustering.

    Read more in the :ref:`User Guide <davies-bouldin_index>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    score: float
        The resulting Davies-Bouldin score.
    """
    (X, labels) = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    (n_samples, _) = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)
    intra_dists = np.zeros(n_labels)
    centroids = np.zeros((n_labels, len(X[0])))
    centroid_sizes = np.zeros((n_labels, ))
    for k in range(n_labels):
        cluster_k = _safe_indexing(X, labels == k)
        centroid_sizes[k] = cluster_k.shape[0]
        centroid = np.mean(cluster_k, axis=0)
        centroids[k] = centroid
        intra_dists[k] = np.sum(pairwise_distances(cluster_k, [centroid]))  # i.e., n1 * s1

    centroid_distances = np.zeros((n_labels, n_labels))
    for k in range(n_labels):
        for k2 in range(k+1, n_labels):
            centroid_distances[k][k2] = np.sqrt(np.sum(np.square(centroids[k]-centroids[k2]))) * (centroid_sizes[k] + centroid_sizes[k2])
            centroid_distances[k2][k] = centroid_distances[k][k2]

    # centroid_distances = pairwise_distances(centroids)
    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0

    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists  # n1 * s1 + n2 * s2 
    # print(combined_intra_dists)
    # print(centroid_distances)
    scores = np.max(combined_intra_dists / centroid_distances,axis=1) # (n1 * s1 + n2 * s2) / ((n1+n2) M12) # weighted average and consider about cluster sizes. 

    return np.mean(scores)


def _euclidean(X, C):
    return np.sqrt(np.sum(np.square(X-C), axis=1))

def davies_bouldin_score_weighted2(X, labels, eps = 1e-5):
    """Compute the Davies-Bouldin score.
        weighted DB score 
        # 1/ ((s12 - (n1*s1 + n2*s2))/(n1+n2)) + 1 / M12 
        # weighted average and consider about cluster sizes. 
        
    The score is defined as the average similarity measure of each cluster with
    its most similar cluster, where similarity is the ratio of within-cluster
    distances to between-cluster distances. Thus, clusters which are farther
    apart and less dispersed will result in a better score.

    The minimum score is zero, with lower values indicating better clustering.

    Read more in the :ref:`User Guide <davies-bouldin_index>`.

    .. versionadded:: 0.20

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        A list of ``n_features``-dimensional data points. Each row corresponds
        to a single data point.

    labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    Returns
    -------
    score: float
        The resulting Davies-Bouldin score.
    """

    (X, labels) = check_X_y(X, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    (n_samples, _) = X.shape
    n_labels = len(le.classes_)
    check_number_of_labels(n_labels, n_samples)
    
    centroid_distances = np.zeros((n_labels, n_labels))
    for k in range(n_labels):
        cluster_k = _safe_indexing(X, labels == k)
        n_k = cluster_k.shape[0]
        centroid_k = np.mean(cluster_k, axis=0)
        s_k = np.sum(_euclidean(cluster_k, centroid_k))  # euclidean distance 

        for k2 in range(k+1, n_labels):
            cluster_k2 = _safe_indexing(X, labels == k2)
            n_k2 = cluster_k2.shape[0]
            centroid_k2 = np.mean(cluster_k2, axis=0)   
            s_k2 = np.sum(_euclidean(cluster_k2, centroid_k2)) # euclidean distance

            m_k_k2 = np.sum(_euclidean(centroid_k.reshape((-1, 1)), centroid_k2.reshape((-1, 1))))  # * (n_k + n_k2)

            X_ = np.concatenate([cluster_k, cluster_k2], axis=0)
            s_k_k2 = np.sum(_euclidean(X_, np.mean(X_, axis=0)))    # # np.median(X_)
            centroid_distances[k][k2] = 1 / ((s_k_k2 - (s_k + s_k2))/(n_k+n_k2) + eps)  + 1 / (m_k_k2) 

            centroid_distances[k2][k] = centroid_distances[k][k2]

    # centroid_distances = pairwise_distances(centroids)
    if np.allclose(centroid_distances, 0):
        return 0

    centroid_distances[centroid_distances == 0] = 0
    scores = np.max(centroid_distances,axis=1) # (n1 * s1 + n2 * s2) / ((n1+n2) M12) # weighted average and consider about cluster sizes. 

    return np.mean(scores)

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
        labels_true = np.asarray([str(v) for v in y[split]])
        labels_pred = np.asarray([str(v) for v in labels_pred])
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
            db_normalized = davies_bouldin_score_normalized(x[split], labels_pred)
            db_weighted = davies_bouldin_score_weighted(x[split], labels_pred)
            db_weighted2 = davies_bouldin_score_weighted2(x[split], labels_pred)
            # print(f'db: {db}, db2: {db2}')
        except Exception as e:
            db = f'Error: {e}'
            db_normalized = f'Error: {e}'
            db_weighted = f'Error: {e}'
            db_weighted2 = f'Error: {e}'
            traceback.print_exc()

        try:
            sil = metrics.silhouette_score(x[split], labels_pred)
            sil_weighted_ = []
            # le = LabelEncoder()
            # labels = le.fit_transform(y_pred)
            # n_samples = len(labels)
            sample_silhouette_values = metrics.silhouette_samples(x[split], labels_pred)
            for i in sorted(np.unique(labels_pred)):
                ith_cluster_silhouette_values = sample_silhouette_values[labels_pred == i]
                sil_weighted_.append((ith_cluster_silhouette_values, ith_cluster_silhouette_values.shape[0]))
            sil_weighted = np.mean([np.sum(v_)/n_ for v_, n_ in sil_weighted_])

            training_iterations = kmeans.training_iterations
            if training_iterations % 5 == 0 or kmeans.is_train_finished:
                out_file = os.path.join(kmeans.params['OUT_DIR'], f'SEED_' + str(kmeans.params['SEED']), f'{training_iterations}.png')
                silhouette_plot(x[split], labels_pred, centroids, out_file, is_show=kmeans.params['IS_SHOW'])

        except Exception as e:
            sil = f'Error: {e}'
            sil_weighted = f'Error: {e}'

        try:
            euclidean = euclidean_dist(x[split], labels_pred, centroids)
        except Exception as e:
            euclidean = f'Error: {e}'

        score = {
            'davies_bouldin': db,
            'db_normalized': db_normalized,
            'db_weighted': db_weighted, 
            'db_weighted2': db_weighted2, 
            'silhouette': sil,
            'sil_weighted': sil_weighted,
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
