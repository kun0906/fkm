import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def gaussian2_1client_1cluster(params, random_state=42):
    """
    # 2 clusters ((-1,0), (1, 0)) in R^2, each client has one cluster. 2 clusters has overlaps.
    params['p1'] == '1client_1cluster':
    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    is_show = params['is_show']

    def get_xy(n=10000, is_show=True):

        # client 1
        mus = [-1, 0]
        sigma = 1
        cov = np.asarray([[sigma, 0], [0, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n)
        y1 = np.asarray([0] * n)

        # client 2
        mus = [1, 0]
        sigma = 1
        cov = np.asarray([[sigma, 0], [0, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n)
        y2 = np.asarray([1] * n)

        return X1, y1, X2, y2

    X1, y1, X2, y2 = get_xy(n=10000)

    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='centroid_1')
        p = np.mean(X1, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 1.0
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] - offset)
        print(xytext)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
                    ha='center', va='center',  # textcoords='offset points',
                    bbox=dict(facecolor='none', edgecolor='b', pad=1),
                    arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        # angleA : starting angle of the path
        # angleB : ending angle of the path

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='centroid_2')
        p = np.mean(X2, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 1.0
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right')
        plt.title(params['p1'])
        # # plt.xlim([-2, 15])
        # # plt.ylim([-2, 15])
        # plt.xlim([-2, 5])
        # plt.ylim([-2, 5])
        # # plt.xticks([])
        # # plt.yticks([])
        if not os.path.exists(params['out_dir']):
            os.makedirs(params['out_dir'])
        f = os.path.join(params['out_dir'], params['p1'] + '.png')
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                            random_state=random_state)
        clients_train_x.append(train_x)
        clients_train_y.append(train_y)
        clients_test_x.append(test_x)
        clients_test_y.append(test_y)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    return x, labels


# gaussian2_1client_1cluster({})


def gaussian2_mix_clusters_per_client(params, random_state=42):
    """
     if params['p1'] == 'mix_clusters_per_client':
    # 2 clusters in R^2:
    # 1) client1 has 70% data from cluster 1 and 30% data from cluster2
    # 2) client2 has 30% data from cluster 1 and 70% data from cluster2
    return gaussian2_mix_clusters_per_client(params, random_state=seed)


    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    is_show = params['is_show']

    def get_xy(n=10000):

        # client 1
        mus = [-1, 0]
        sigma = 1
        cov = np.asarray([[sigma, 0], [0, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n)
        y1 = np.asarray([0] * n)

        # client 2
        mus = [1, 0]
        sigma = 1
        cov = np.asarray([[sigma, 0], [0, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n)
        y2 = np.asarray([1] * n)

        return X1, y1, X2, y2

    X1, y1, X2, y2 = get_xy(n=10000)
    train_x1, test_x1, train_y1, test_y1 = train_test_split(X1, y1, test_size=0.1, shuffle=True,
                                                            random_state=random_state)

    train_x2, test_x2, train_y2, test_y2 = train_test_split(X2, y2, test_size=0.1, shuffle=True,
                                                            random_state=random_state)

    X1 = np.concatenate([train_x1, test_x2], axis=0)
    # y1 = np.concatenate([train_y1, test_y2], axis=0) # be careful of this
    y1 = np.zeros((X1.shape[0],))
    X2 = np.concatenate([test_x1, train_x2], axis=0)
    # y2 = np.concatenate([test_y1, train_y2], axis=0)
    y2 = np.ones((X2.shape[0],))

    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='centroid_1')
        p = np.mean(X1, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 1.0
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] - offset)
        print(xytext)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
                    ha='center', va='center',  # textcoords='offset points',
                    bbox=dict(facecolor='none', edgecolor='b', pad=1),
                    arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        # angleA : starting angle of the path
        # angleB : ending angle of the path

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='centroid_2')
        p = np.mean(X2, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 1.0
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right')
        plt.title(params['p1'])
        # # plt.xlim([-2, 15])
        # # plt.ylim([-2, 15])
        # plt.xlim([-2, 5])
        # plt.ylim([-2, 5])
        # # plt.xticks([])
        # # plt.yticks([])
        if not os.path.exists(params['out_dir']):
            os.makedirs(params['out_dir'])
        f = os.path.join(params['out_dir'], params['p1'] + '.png')
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                            random_state=random_state)
        clients_train_x.append(train_x)
        clients_train_y.append(train_y)
        clients_test_x.append(test_x)
        clients_test_y.append(test_y)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    # y_tmp = []
    # for vs in clients_train_y:
    #     y_tmp.extend(vs)
    # print(f'n_train_clients: {len(clients_train_x)}, n_datapoints: {sum(len(vs) for vs in clients_train_y)}, '
    #       f'cluster_size: {sorted(Counter(y_tmp).items(), key=lambda kv: kv[0], reverse=False)}')
    # y_tmp = []
    # for vs in clients_test_y:
    #     y_tmp.extend(vs)
    # print(f'n_test_clients: {len(clients_test_x)}, n_datapoints: {sum(len(vs) for vs in clients_test_y)}, '
    #       f'cluster_size: {sorted(Counter(y_tmp).items(), key=lambda kv: kv[0], reverse=False)}')
    return x, labels


# gaussian2_mix_clusters_per_client({}, random_state=42)


def gaussian2_1client_ylt0(params, random_state=42):
    """
     if params['p1'] == '1client_ylt0':
    # lt0 means all 'y's are larger than 0
    # 2 clusters in R^2
    # 1) client 1 has all data (y>0) from cluster1 and cluster2
    # 1) client 2 has all data (y<=0) from cluster1 and cluster2
    return gaussian2_1client_ylt0(params, random_state=seed)


    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    is_show = params['is_show']

    def get_xy(n=10000):

        # client 1
        mus = [-1, 0]
        sigma = 1
        cov = np.asarray([[sigma, 0], [0, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n)
        y1 = np.asarray([0] * n)

        # client 2
        mus = [1, 0]
        sigma = 1
        cov = np.asarray([[sigma, 0], [0, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n)
        y2 = np.asarray([1] * n)

        return X1, y1, X2, y2

    X1, y1, X2, y2 = get_xy(n=10000)
    indices1 = np.where(X1[:, 1] > 0)
    indices2 = np.where(X2[:, 1] > 0)
    indices3 = np.where(X1[:, 1] <= 0)
    indices4 = np.where(X2[:, 1] <= 0)
    new_X1 = np.concatenate([X1[indices1], X2[indices2]], axis=0)
    # new_y1 = np.concatenate([y1[indices1], y2[indices2]], axis=0)
    new_y1 = np.zeros((new_X1.shape[0],))
    new_X2 = np.concatenate([X1[indices3], X2[indices4]], axis=0)
    # new_y2 = np.concatenate([y1[indices3], y2[indices4]], axis=0)
    new_y2 = np.ones((new_X2.shape[0],))
    X1, y1, X2, y2 = new_X1, new_y1, new_X2, new_y2

    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='centroid_1')
        p = np.mean(X1, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 1.0
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] + offset)
        print(xytext)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
                    ha='center', va='center',  # textcoords='offset points',
                    bbox=dict(facecolor='none', edgecolor='b', pad=1),
                    arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        # angleA : starting angle of the path
        # angleB : ending angle of the path

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='centroid_2')
        p = np.mean(X2, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 1.0
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right')
        plt.title(params['p1'])
        # # plt.xlim([-2, 15])
        # # plt.ylim([-2, 15])
        # plt.xlim([-2, 5])
        # plt.ylim([-2, 5])
        # # plt.xticks([])
        # # plt.yticks([])
        if not os.path.exists(params['out_dir']):
            os.makedirs(params['out_dir'])
        f = os.path.join(params['out_dir'], params['p1'] + '.png')
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                            random_state=random_state)
        clients_train_x.append(train_x)
        clients_train_y.append(train_y)
        clients_test_x.append(test_x)
        clients_test_y.append(test_y)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    return x, labels


# gaussian2_1client_ylt0({}, random_state=42)

def gaussian2_1client_xlt0(params, random_state=42):
    """
     if params['p1'] == '1client_xlt0':
    # lt0 means all 'x's are larger than 0
    # 2 clusters in R^2
    # 1) client 1 has all data (x>0) from cluster1 and cluster2
    # 1) client 2 has all data (x<=0) from cluster1 and cluster2
    return gaussian2_1client_xlt0(params, random_state=seed)

    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """

    def get_xy(n=10000):

        # client 1
        mus = [-1, 0]
        sigma = 1
        cov = np.asarray([[sigma, 0], [0, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n)
        y1 = np.asarray([0] * n)

        # client 2
        mus = [1, 0]
        sigma = 1
        cov = np.asarray([[sigma, 0], [0, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n)
        y2 = np.asarray([1] * n)

        return X1, y1, X2, y2

    X1, y1, X2, y2 = get_xy(n=10000)
    indices1 = np.where(X1[:, 0] < 0)
    indices2 = np.where(X2[:, 0] < 0)
    indices3 = np.where(X1[:, 0] >= 0)
    indices4 = np.where(X2[:, 0] >= 0)
    new_X1 = np.concatenate([X1[indices1], X2[indices2]], axis=0)
    # new_y1 = np.concatenate([y1[indices1], y2[indices2]], axis=0)
    new_y1 = np.zeros((new_X1.shape[0],))
    new_X2 = np.concatenate([X1[indices3], X2[indices4]], axis=0)
    # new_y2 = np.concatenate([y1[indices3], y2[indices4]], axis=0)
    new_y2 = np.ones((new_X2.shape[0],))
    X1, y1, X2, y2 = new_X1, new_y1, new_X2, new_y2

    is_show = params['is_show']
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='centroid_1')
        p = np.mean(X1, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 1.0
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] - offset)
        print(xytext)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
                    ha='center', va='center',  # textcoords='offset points',
                    bbox=dict(facecolor='none', edgecolor='b', pad=1),
                    arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        # angleA : starting angle of the path
        # angleB : ending angle of the path

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='centroid_2')
        p = np.mean(X2, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 1.0
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right')
        plt.title(params['p1'])
        # # plt.xlim([-2, 15])
        # # plt.ylim([-2, 15])
        # plt.xlim([-2, 5])
        # plt.ylim([-2, 5])
        # # plt.xticks([])
        # # plt.yticks([])
        if not os.path.exists(params['out_dir']):
            os.makedirs(params['out_dir'])
        f = os.path.join(params['out_dir'], params['p1'] + '.png')
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                            random_state=random_state)
        clients_train_x.append(train_x)
        clients_train_y.append(train_y)
        clients_test_x.append(test_x)
        clients_test_y.append(test_y)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    return x, labels


# gaussian2_1client_xlt0({}, random_state=42)


def gaussian2_1client_1cluster_diff_sigma(params, random_state=42):
    """
    # 2 clusters ((-1,0), (1, 0)) in R^2, each client has one cluster. 2 clusters has no overlaps.
    cluster 1: sigma = 0.5
    cluster 2: sigma = 1
    params['p1'] == '1client_1cluster_diff_sigma':
    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    is_show = params['is_show']

    def get_xy(n=10000, is_show=True):

        # client 1
        mus = [-1, 0]
        sigma = 0.5
        cov = np.asarray([[sigma, 0], [0, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n)
        y1 = np.asarray([0] * n)

        # client 2
        mus = [1, 0]
        sigma = 1
        cov = np.asarray([[sigma, 0], [0, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n)
        y2 = np.asarray([1] * n)

        return X1, y1, X2, y2

    X1, y1, X2, y2 = get_xy(n=10000)

    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='centroid_1')
        p = np.mean(X1, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 1.0
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] - offset)
        print(xytext)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
                    ha='center', va='center',  # textcoords='offset points',
                    bbox=dict(facecolor='none', edgecolor='b', pad=1),
                    arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        # angleA : starting angle of the path
        # angleB : ending angle of the path

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='centroid_2')
        p = np.mean(X2, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 1.0
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right')
        plt.title(params['p1'])
        # # plt.xlim([-2, 15])
        # # plt.ylim([-2, 15])
        # plt.xlim([-2, 5])
        # plt.ylim([-2, 5])
        # # plt.xticks([])
        # # plt.yticks([])
        if not os.path.exists(params['out_dir']):
            os.makedirs(params['out_dir'])
        f = os.path.join(params['out_dir'], params['p1'] + '.png')
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                            random_state=random_state)
        clients_train_x.append(train_x)
        clients_train_y.append(train_y)
        clients_test_x.append(test_x)
        clients_test_y.append(test_y)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    return x, labels


# gaussian2_1client_1cluster({})


def gaussian2_diff_sigma_n(params, random_state=42):
    """
    # 2 clusters ((-1,0), (1, 0)) in R^2, each client has one cluster. 2 clusters has no overlaps.
    cluster 1: sigma = 0.1 and n_points = 5000
    cluster 2: sigma = 1    and n_points = 15000
    params['p1'] == 'diff_sigma_n':
    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    is_show = params['is_show']

    def get_xy(n=10000, is_show=True):

        # client 1
        mus = [-1, 0]
        sigma = 0.1
        n1 = 5000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n1)
        y1 = np.asarray([0] * n1)

        # client 2
        mus = [1, 0]
        sigma = 1
        n2 = 15000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n2)
        y2 = np.asarray([1] * n2)

        return X1, y1, X2, y2

    X1, y1, X2, y2 = get_xy(n=10000)

    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='centroid_1')
        p = np.mean(X1, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 1.0
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] - offset)
        print(xytext)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
                    ha='center', va='center',  # textcoords='offset points',
                    bbox=dict(facecolor='none', edgecolor='b', pad=1),
                    arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        # angleA : starting angle of the path
        # angleB : ending angle of the path

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='centroid_2')
        p = np.mean(X2, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 1.0
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right')
        plt.title(params['p1'])
        # # plt.xlim([-2, 15])
        # # plt.ylim([-2, 15])
        # plt.xlim([-2, 2])
        # plt.ylim([-2, 2])
        # # plt.xticks([])
        # # plt.yticks([])
        if not os.path.exists(params['out_dir']):
            os.makedirs(params['out_dir'])
        f = os.path.join(params['out_dir'], params['p1'] + '.png')
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                            random_state=random_state)
        clients_train_x.append(train_x)
        clients_train_y.append(train_y)
        clients_test_x.append(test_x)
        clients_test_y.append(test_y)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    return x, labels


def gaussian2_1client_xlt0_2(params, random_state=42):
    """
     if params['p1'] == '1client_xlt0':
    # lt0 means all 'x's are larger than 0
    # 2 clusters in R^2
    # 1) client 1 has all data (x>0) from cluster1 and cluster2
    # 1) client 2 has all data (x<=0) from cluster1 and cluster2
    return gaussian2_1client_xlt0(params, random_state=seed)

    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """

    def get_xy(n=10000):

        # client 1
        mus = [-1, 0]
        sigma = 0.1
        n1 = 5000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, sigma], [sigma, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n1)
        y1 = np.asarray([0] * n1)

        # client 2
        mus = [1, 0]
        sigma = 0.3
        n2 = 10000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, -sigma], [-sigma, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n2)
        y2 = np.asarray([1] * n2)

        # mus = [0, 3]
        # n1 = 10000
        # sigma = 0.1
        # cov = np.asarray([[sigma+0.9, 0], [0, 0]])
        # X3 = r.multivariate_normal(mus, cov, size=n1)
        # y3 = np.asarray([2] * n1)

        # # mus = [0, -3]
        # cov = np.asarray([[sigma, 0], [0, sigma]])
        # X4 = r.multivariate_normal(mus, cov, size=n1)
        # y4 = np.asarray([1] * n1)

        # X1 = np.concatenate([X1, X2], axis=0)
        # y1 = np.concatenate([y1, y2], axis=0)
        # X3 = np.concatenate([X3, X4], axis=0)
        # y3 = np.concatenate([y3, y4], axis=0)

        return X1, y1, X2, y2

    X1, y1, X2, y2 = get_xy(n=10000)
    # indices1 = np.where(X1[:, 0] < 0)
    # indices2 = np.where(X2[:, 0] < 0)
    # indices3 = np.where(X1[:, 0] >= 0)
    # indices4 = np.where(X2[:, 0] >= 0)
    # new_X1 = np.concatenate([X1[indices1], X2[indices2]], axis=0)
    # # new_y1 = np.concatenate([y1[indices1], y2[indices2]], axis=0)
    # new_y1 = np.zeros((new_X1.shape[0],))
    # new_X2 = np.concatenate([X1[indices3], X2[indices4]], axis=0)
    # # new_y2 = np.concatenate([y1[indices3], y2[indices4]], axis=0)
    # new_y2 = np.ones((new_X2.shape[0],))
    # X1, y1, X2, y2 = new_X1, new_y1, new_X2, new_y2

    # train_x1, test_x1, train_y1, test_y1 = train_test_split(X1, y1, test_size=0.1, shuffle=True,
    #                                                         random_state=random_state)
    #
    # train_x2, test_x2, train_y2, test_y2 = train_test_split(X2, y2, test_size=0.1, shuffle=True,
    #                                                         random_state=random_state)
    #
    # train_x3, test_x3, train_y3, test_y3 = train_test_split(X3, y3, test_size=0.1, shuffle=True,
    #                                                         random_state=random_state)
    #
    # X1 = np.concatenate([train_x1, test_x2, test_x3], axis=0)
    # # y1 = np.concatenate([train_y1, test_y2], axis=0) # be careful of this
    # y1 = np.zeros((X1.shape[0],))
    # X2 = np.concatenate([test_x1, train_x2, test_x3], axis=0)
    # # y2 = np.concatenate([test_y1, train_y2], axis=0)
    # y2 = np.ones((X2.shape[0],))
    #
    # X3 = np.concatenate([test_x1, test_x2, train_x3], axis=0)
    # y3 = np.ones((X3.shape[0],)) * 2


    is_show = params['is_show']
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='centroid_1')
        p = np.mean(X1, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] - offset)
        print(xytext)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
                    ha='center', va='center',  # textcoords='offset points',
                    bbox=dict(facecolor='none', edgecolor='b', pad=1),
                    arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        # angleA : starting angle of the path
        # angleB : ending angle of the path

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='centroid_2')
        p = np.mean(X2, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        # ax.scatter(X3[:, 0], X3[:, 1], c=colors[2], marker="o", s=10, alpha=0.3, label='centroid_3')
        # p = np.mean(X3, axis=0)
        # ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        # offset = 0.3
        # # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        # xytext = (p[0] + offset, p[1] - offset)
        # ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
        #             ha='center', va='center',  # textcoords='offset points', va='bottom',
        #             bbox=dict(facecolor='none', edgecolor='red', pad=1),
        #             arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
        #                             connectionstyle="angle3, angleA=90,angleB=0"))

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right')
        plt.title(params['p1'])
        # # plt.xlim([-2, 15])
        # # plt.ylim([-2, 15])
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        # # plt.xticks([])
        # # plt.yticks([])
        if not os.path.exists(params['out_dir']):
            os.makedirs(params['out_dir'])
        f = os.path.join(params['out_dir'], params['p1'] + '.png')
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                            random_state=random_state)
        clients_train_x.append(train_x)
        clients_train_y.append(train_y)
        clients_test_x.append(test_x)
        clients_test_y.append(test_y)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    return x, labels



def gaussian2_diff_sigma_n(params, random_state=42):
    """
    # 2 clusters ((-1,0), (1, 0)) in R^2, each client has one cluster. 2 clusters has no overlaps.
    cluster 1: sigma = 0.1 and n_points = 5000
    cluster 2: sigma = 1    and n_points = 15000
    params['p1'] == 'diff_sigma_n':
    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    p1 = params['p1'].split(':')   # 'n1_100-sigma1_0.1+n2_5000-sigma2_0.2+n3_10000-sigma3_0.3:ratio_0.1:diff_sigma_n'
    ratio = float(p1[1].split('_')[1])

    p1_0 = p1[0].split('+')     # 'n1_100-sigma1_0.1, n2_1000-sigma2_0.2, n3_10000-sigma3_0.3
    p1_0_c1 = p1_0[0].split('-')
    n1 = int(p1_0_c1[0].split('_')[1])
    sigma1 = float(p1_0_c1[1].split('_')[1])

    p1_0_c2 = p1_0[1].split('-')
    n2 = int(p1_0_c2[0].split('_')[1])
    sigma2 = float(p1_0_c2[1].split('_')[1])

    def get_xy(n=0):

        # client 1
        mus = [-0.3, 0]
        # sigma = 0.1
        # n1 = 2000
        cov = np.asarray([[0.0, 0], [0, sigma1]])
        # cov = np.asarray([[sigma, sigma], [sigma, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n1)
        y1 = np.asarray([0] * n1)

        # client 2
        mus = [0.3, 0]
        # sigma = 0.2
        # n2 = 5000
        cov = np.asarray([[0.0, 0], [0, sigma2]])
        # cov = np.asarray([[sigma, -sigma], [-sigma, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n2)
        y2 = np.asarray([1] * n2)


        return X1, y1, X2, y2

    X1, y1, X2, y2 = get_xy()
    if ratio <= 0 or ratio >= 1:
        pass
    else:
        # client 1: 90% cluster1, 10 % cluster2
        # client 2: 10% cluster1, 90 % cluster2
        train_x1, test_x1, train_y1, test_y1 = train_test_split(X1, y1, test_size= ratio, shuffle=True,
                                                      random_state=random_state)  # train set = 1-ratio

        train_x2, test_x2, train_y2, test_y2 = train_test_split(X2, y2, test_size= ratio, shuffle=True,
                                                      random_state=random_state)

        X1 = np.concatenate([train_x1, test_x2], axis=0)
        y1 = np.zeros((X1.shape[0],))

        X2 = np.concatenate([test_x1, train_x2], axis=0)
        y2 = np.ones((X2.shape[0],))

    is_show = params['is_show']
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='centroid_1')
        p = np.mean(X1, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] - offset, p[1] - offset)
        print(xytext)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
                    ha='center', va='center',  # textcoords='offset points',
                    bbox=dict(facecolor='none', edgecolor='b', pad=1),
                    arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))
        # angleA : starting angle of the path
        # angleB : ending angle of the path

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='centroid_2')
        p = np.mean(X2, axis=0)
        ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
        offset = 0.3
        # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
        xytext = (p[0] + offset, p[1] - offset)
        ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
                    ha='center', va='center',  # textcoords='offset points', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='red', pad=1),
                    arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
                                    connectionstyle="angle3, angleA=90,angleB=0"))

        ax.axvline(x=0, color='k', linestyle='--')
        ax.axhline(y=0, color='k', linestyle='--')
        ax.legend(loc='upper right')
        plt.title(params['p1'].replace(':', '\n'))
        # # plt.xlim([-2, 15])
        # # plt.ylim([-2, 15])
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        # # plt.xticks([])
        # # plt.yticks([])
        plt.tight_layout()
        if not os.path.exists(params['out_dir']):
            os.makedirs(params['out_dir'])
        f = os.path.join(params['out_dir'], params['p1']+'.png')
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, shuffle=True,
                                                            random_state=random_state)
        clients_train_x.append(train_x)
        clients_train_y.append(train_y)
        clients_test_x.append(test_x)
        clients_test_y.append(test_y)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    return x, labels
