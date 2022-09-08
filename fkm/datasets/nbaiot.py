"""
NBAIOT
Data Set Information:

(a) Attribute being predicted:
-- Originally we aimed at distinguishing between benign and Malicious traffic data by means of anomaly detection techniques.
-- However, as the malicious data can be divided into 10 attacks carried by 2 botnets, the dataset can also be used for multi-class classification: 10 classes of attacks, plus 1 class of 'benign'.

(b) The study's results:
-- For each of the 9 IoT devices we trained and optimized a deep autoencoder on 2/3 of its benign data (i.e., the training set of each device). This was done to capture normal network traffic patterns.
-- The test data of each device comprised of the remaining 1/3 of benign data plus all the malicious data. On each test set we applied the respective trained (deep) autoencoder as an anomaly detector. The detection of anomalies (i.e., the cyberattacks launched from each of the above IoT devices) concluded with 100% TPR.


https://archive.ics.uci.edu/ml/datasets/detection_of_IoT_botnet_attacks_N_BaIoT#

"""
import os

import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def nbaiot_user_percent(args={}, random_state=42):
    in_dir = 'datasets/NBAIOT/Danmini_Doorbell'
    # n_clusters = args['n_clusters']
    csvs = ['benign_traffic.csv', os.path.join('gafgyt_attacks', "tcp.csv")]

    X = []
    Y = []
    labels = []
    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    n_users = 0
    n_data_points = 0
    dim = 0
    for i, csv in enumerate(csvs):
        if ".DS_Store" in csv: continue
        n_users +=1
        labels.append(csv)
        f = os.path.join(in_dir, csv)
        print(i, f)
        df = pd.read_csv(f)
        X_, y_ = df.values, np.asarray([i] * df.shape[0])
        X.extend(X_)
        Y.extend(y_)
        X_, y_ = sklearn.utils.resample(X_, y_, replace=False, n_samples=5000, random_state=random_state)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_, y_, test_size=2, shuffle=True,
                                                                                    random_state=random_state)
        n_data_points += len(X_train)
        dim = len(X_train[0])
        clients_train_x.append(X_train)
        clients_train_y.append(y_train)
        clients_test_x.append(X_test)
        clients_test_y.append(y_test)

    print(f'n_users: {n_users}, n_data_points: {n_data_points}, dim: {dim}')

    is_show = True
    if is_show:
        import matplotlib.pyplot as plt
        X_, y_ = sklearn.utils.resample(X, Y, replace=False, n_samples=2000, stratify=Y, random_state=random_state)
        mp = {v: i/(len(set(y_))-1) * 100 for i, v in enumerate(sorted(set(y_)))} # corlormap range from 0 to 100.
        v2labels = {v:labels[k] for k, v in mp.items()}
        std = sklearn.preprocessing.StandardScaler()
        X_ = std.fit_transform(X_)
        tsne = TSNE(n_components=2, random_state=0)
        X_2 = tsne.fit_transform(X_)
        c = [mp[v] for v in y_]
        plt.scatter(X_2[:, 0], X_2[:, 1], c = c, cmap= 'gist_rainbow', alpha=0.5)
        # set colorbar
        ticks = sorted(mp.values())
        cbar = plt.colorbar(ticks = ticks, fraction=0.3, pad=0.04)
        cbar.ax.set_yticklabels([v2labels[v] for v in ticks], fontsize=8)  # horizontal colorbar
        # for label, x_tmp, y_tmp in zip(y_, X_2[:, 0], X_2[:, 1]):
        #     plt.annotate(label, xy=(x_tmp, y_tmp), xytext=(0.1, 0.1), fontsize = 20, textcoords="offset points")
        # f = os.path.join(in_dir, 'nbaiot.png')
        f = args['data_file'] + '.png'
        print(f)
        plt.savefig(f, dpi=300)
        plt.show()

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    print(f'n_train_clients: {len(clients_train_x)}, n_datapoints: {sum(len(vs) for vs in clients_train_y)}')
    print(f'n_test_clients: {len(clients_test_x)}, n_datapoints: {sum(len(vs) for vs in clients_test_y)}')
    return x, labels


def nbaiot_user_percent_client11(args={}, random_state=42):
    in_dir = 'datasets/NBAIOT/Danmini_Doorbell'

    # dataset_detail = args['DATASET']['detail']
    # data_file = os.path.join(in_dir, f'{dataset_detail}.dat')
    # if os.path.exists(data_file):
    #     return utils_func.load(data_file)

    N_CLUSTERS = args['N_CLUSTERS']
    csvs = ['benign_traffic.csv'] + \
           [os.path.join('gafgyt_attacks', f) for f in sorted(os.listdir(os.path.join(in_dir, 'gafgyt_attacks')))] + \
           [os.path.join('mirai_attacks', f) for f in sorted(os.listdir(os.path.join(in_dir, 'mirai_attacks')))]
    X = []
    Y = []
    labels = []
    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    n_users = 0
    n_data_points = 0
    dim = 0
    for i, csv in enumerate(csvs):
        if ".DS_Store" in csv: continue
        n_users +=1
        labels.append(csv)
        f = os.path.join(in_dir, csv)
        print(i, f)
        df = pd.read_csv(f)
        if N_CLUSTERS == 2:
            if 'benign_traffic' in csv:
                label = 0
            else:
                label = 1
            X_, y_ = df.values, np.asarray([label] * df.shape[0])
            X.extend(X_)
            Y.extend(y_)
            if label == 0:
                X_, y_ = sklearn.utils.resample(X_, y_, replace=False, n_samples=5000, random_state=random_state)
            else:
                X_, y_ = sklearn.utils.resample(X_, y_, replace=False, n_samples=int(5000//11), random_state=random_state)
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_, y_, test_size=2,
                                                                                        shuffle=True,
                                                                                        random_state=random_state)
        elif N_CLUSTERS == 11:
            X_, y_ = df.values, np.asarray([i] * df.shape[0])
            X.extend(X_)
            Y.extend(y_)
            X_, y_ = sklearn.utils.resample(X_, y_, replace=False, n_samples=5000, random_state=random_state)
            X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_, y_, test_size=2, shuffle=True,
                                                                                        random_state=random_state)
        else:
            raise NotImplementedError(f'Error: {N_CLUSTERS}')
        n_data_points += len(X_train)
        dim = len(X_train[0])
        clients_train_x.append(X_train)
        clients_train_y.append(y_train)
        clients_test_x.append(X_test)
        clients_test_y.append(y_test)

    print(f'n_users: {n_users}, n_data_points: {n_data_points}, dim: {dim}')

    is_show = True
    if is_show:
        X_, y_ = sklearn.utils.resample(X, Y, replace=False, n_samples=2000, stratify=Y, random_state=random_state)
        mp = {v: i/(len(set(y_))-1) * 100 for i, v in enumerate(sorted(set(y_)))} # corlormap range from 0 to 100.
        v2labels = {v:labels[k] for k, v in mp.items()}
        std = sklearn.preprocessing.StandardScaler()
        X_ = std.fit_transform(X_)
        tsne = TSNE(n_components=2, random_state=0)
        X_2 = tsne.fit_transform(X_)
        c = [mp[v] for v in y_]
        plt.scatter(X_2[:, 0], X_2[:, 1], c = c, cmap= 'gist_rainbow', alpha=0.5)
        # set colorbar
        ticks = sorted(mp.values())
        cbar = plt.colorbar(ticks = ticks, fraction=0.3, pad=0.04)
        cbar.ax.set_yticklabels([v2labels[v] for v in ticks], fontsize=8)  # horizontal colorbar
        # for label, x_tmp, y_tmp in zip(y_, X_2[:, 0], X_2[:, 1]):
        #     plt.annotate(label, xy=(x_tmp, y_tmp), xytext=(0.1, 0.1), fontsize = 20, textcoords="offset points")
        # f = os.path.join(in_dir, 'nbaiot.png')
        f = args['data_file'] + '.png'
        print(f)
        plt.savefig(f, dpi=300)
        plt.show()

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    print(f'n_train_clients: {len(clients_train_x)}, n_datapoints: {sum(len(vs) for vs in clients_train_y)}')
    print(f'n_test_clients: {len(clients_test_x)}, n_datapoints: {sum(len(vs) for vs in clients_test_y)}')
    return x, labels

#
# def nbaiot_user_percent(args={}, random_state=42):
#     # in_dir = 'datasets/NBAIOT/Danmini_Doorbell'
#     return _nbaiot_user_percent(args={}, n_clusters=2, random_state=42)
#
# def nbaiot11_user_percent(args={}, random_state=42):
#     # in_dir = 'datasets/NBAIOT/Danmini_Doorbell'
#     return _nbaiot_user_percent(args={}, n_clusters = 11, random_state=42)



def nbaiot_diff_sigma_n(args, random_state=42):
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
    n_clusters = args['N_CLUSTERS']
    dataset_detail = args['DATASET']['detail']   # 'nbaiot_user_percent_client:ratio_0.1'
    p1 = dataset_detail.split(':')
    ratio = float(p1[1].split('_')[1])

    p1_0 = p1[0].split('+')  # 'n1_100-sigma1_0.1, n2_1000-sigma2_0.2, n3_10000-sigma3_0.3
    p1_0_c1 = p1_0[0].split('-')
    n1 = int(p1_0_c1[0].split('_')[1])
    # tmp = p1_0_c1[1].split('_')
    # sigma1_0, sigma1_1 = float(tmp[1]), float(tmp[2])

    p1_0_c2 = p1_0[1].split('-')
    n2 = int(p1_0_c2[0].split('_')[1])
    # tmp = p1_0_c2[1].split('_')
    # sigma2_0, sigma2_1 = float(tmp[1]), float(tmp[2])

    p1_0_c3 = p1_0[2].split('-')
    n3 = int(p1_0_c3[0].split('_')[1])
    # tmp = p1_0_c3[1].split('_')
    # sigma3_0, sigma3_1 = float(tmp[1]), float(tmp[2])

    in_dir = 'datasets/NBAIOT/Danmini_Doorbell'
    def get_xy(n=0):

        # client 1
        f = os.path.join(in_dir, 'benign_traffic.csv')
        X1 = pd.read_csv(f).values
        print('normal: ', X1.shape)
        X1 = sklearn.utils.resample(X1, replace=False, n_samples=n1, random_state=random_state)
        y1 = np.asarray([0] * X1.shape[0])

        # client 2
        f = os.path.join(in_dir, os.path.join('gafgyt_attacks', "tcp.csv"))
        X2 = pd.read_csv(f).values
        print('abnormal:', X2.shape)
        X2 = sklearn.utils.resample(X2, replace=False, n_samples=n2, random_state=random_state)
        y2 = np.asarray([1] * X2.shape[0])

        # client 3
        f = os.path.join(in_dir, os.path.join('gafgyt_attacks', "junk.csv"))
        X3 = pd.read_csv(f).values
        X3 = sklearn.utils.resample(X3, replace=False, n_samples=n3, random_state=random_state)
        if n_clusters == 2:
            y3 = np.asarray([1] * X2.shape[0])
        elif n_clusters == 3:
            y3 = np.asarray([2] * X2.shape[0])
        else:
            raise NotImplementedError
        # # mus = [0, -3]
        # cov = np.asarray([[sigma, 0], [0, sigma]])
        # X4 = r.multivariate_normal(mus, cov, size=n1)
        # y4 = np.asarray([1] * n1)

        # X1 = np.concatenate([X1, X2], axis=0)
        # y1 = np.concatenate([y1, y2], axis=0)
        # X3 = np.concatenate([X3, X4], axis=0)
        # y3 = np.concatenate([y3, y4], axis=0)

        return X1, y1, X2, y2, X3, y3

    X1, y1, X2, y2, X3, y3 = get_xy()
    if 'Centralized' in args['ALGORITHM']['py_name']:   # for Centralized Kmeans, ratios should have no impact.
        pass
    elif 2* ratio <= 0 or 2* ratio >= 1:
        pass
    else:
        # client 1: 90% cluster1, 10 % cluster2, 10 % cluster3
        # client 2: 10% cluster1, 90 % cluster2, 10 % cluster3
        # client 3: 10% cluster1, 10 % cluster2, 90 % cluster3
        train_x1, X1, train_y1, y1 = train_test_split(X1, y1, test_size=2 * ratio, shuffle=True,
                                                      random_state=random_state)  # train set = 1-ratio
        test_x11, test_x12, test_y11, test_y12 = train_test_split(X1, y1, test_size=0.5, shuffle=True,
                                                                  random_state=random_state)  # each test set = 50% of rest data

        train_x2, X2, train_y2, y2 = train_test_split(X2, y2, test_size=2* ratio, shuffle=True,
                                                      random_state=random_state)
        test_x21, test_x22, test_y21, test_y22 = train_test_split(X2, y2, test_size=0.5, shuffle=True,
                                                                  random_state=random_state)

        train_x3, X3, train_y3, y3 = train_test_split(X3, y3, test_size=2* ratio, shuffle=True,
                                                      random_state=random_state)
        test_x31, test_x32, test_y31, test_y32 = train_test_split(X3, y3, test_size=0.5, shuffle=True,
                                                                  random_state=random_state)

        X1 = np.concatenate([train_x1, test_x21, test_x31], axis=0)
        y1 = np.concatenate([train_y1, test_y21, test_y31], axis=0) # be careful of this
        # y1 = np.zeros((X1.shape[0],))

        X2 = np.concatenate([test_x11, train_x2, test_x32], axis=0)
        y2 = np.concatenate([test_y11, train_y2, test_y32], axis=0)
        # y2 = np.ones((X2.shape[0],))

        X3 = np.concatenate([test_x12, test_x22, train_x3], axis=0)
        y3 = np.concatenate([test_y12, test_y22, train_y3], axis=0)
        # y3 = np.ones((X3.shape[0],)) * 2


    is_show = args['IS_SHOW']
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='$G_1$')
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

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='$G_2$')
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

        ax.scatter(X3[:, 0], X3[:, 1], c=colors[2], marker="o", s=10, alpha=0.3, label='$G_3$')
        p = np.mean(X3, axis=0)
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
        ax.legend(loc='upper right', fontsize = 13)
        if args['SHOW_TITLE']:
            plt.title(dataset_detail.replace(':', '\n'))

        # if 'xlim' in kwargs:
        #     plt.xlim(kwargs['xlim'])
        # else:
        #     plt.xlim([-6, 6])
        # if 'ylim' in kwargs:
        #     plt.ylim(kwargs['ylim'])
        # else:
        #     plt.ylim([-6, 6])

        fontsize = 13
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.tight_layout()
        # if not os.path.exists(params['OUT_DIR']):
        #     os.makedirs(params['OUT_DIR'])
        # f = os.path.join(args['OUT_DIR'], dataset_detail+'.png')
        f = args['data_file'] + '.png'
        print(f)
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2), (X3, y3)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=2, shuffle=True,
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




def nbaiot_C_2_diff_sigma_n(args, random_state=42):
    """
    2 clients
    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """
    dataset_detail = args['DATASET']['detail']   # 'nbaiot_user_percent_client:ratio_0.1'
    p1 = dataset_detail.split(':')
    ratio = float(p1[1].split('_')[1])

    p1_0 = p1[0].split('+')  # 'n1_100-sigma1_0.1, n2_1000-sigma2_0.2, n3_10000-sigma3_0.3
    p1_0_c1 = p1_0[0].split('-')
    n1 = int(p1_0_c1[0].split('_')[1])
    # tmp = p1_0_c1[1].split('_')
    # sigma1_0, sigma1_1 = float(tmp[1]), float(tmp[2])

    p1_0_c2 = p1_0[1].split('-')
    n2 = int(p1_0_c2[0].split('_')[1])
    # tmp = p1_0_c2[1].split('_')
    # sigma2_0, sigma2_1 = float(tmp[1]), float(tmp[2])

    # p1_0_c3 = p1_0[2].split('-')
    # n3 = int(p1_0_c3[0].split('_')[1])
    # tmp = p1_0_c3[1].split('_')
    # sigma3_0, sigma3_1 = float(tmp[1]), float(tmp[2])

    in_dir = 'datasets/NBAIOT/Danmini_Doorbell'
    def get_xy(n=0):

        # client 1
        f = os.path.join(in_dir, 'benign_traffic.csv')
        X1 = pd.read_csv(f).values
        X1 = sklearn.utils.resample(X1, replace=False, n_samples=n1, random_state=random_state)
        y1 = np.asarray([0] * X1.shape[0])

        # client 2
        f = os.path.join(in_dir, os.path.join('gafgyt_attacks', "tcp.csv"))
        X2 = pd.read_csv(f).values
        X2 = sklearn.utils.resample(X2, replace=False, n_samples=n2, random_state=random_state)
        y2 = np.asarray([1] * X2.shape[0])

        # # client 3
        # f = os.path.join(in_dir, os.path.join('gafgyt_attacks', "junk.csv"))
        # X3 = pd.read_csv(f).values
        # X3 = sklearn.utils.resample(X3, replace=False, n_samples=n3, random_state=random_state)
        # y3 = np.asarray([1] * X2.shape[0])

        # # mus = [0, -3]
        # cov = np.asarray([[sigma, 0], [0, sigma]])
        # X4 = r.multivariate_normal(mus, cov, size=n1)
        # y4 = np.asarray([1] * n1)

        # X1 = np.concatenate([X1, X2], axis=0)
        # y1 = np.concatenate([y1, y2], axis=0)
        # X3 = np.concatenate([X3, X4], axis=0)
        # y3 = np.concatenate([y3, y4], axis=0)

        return X1, y1, X2, y2

    X1, y1, X2, y2 = get_xy()
    if 'Centralized' in args['ALGORITHM']['py_name']:   # for Centralized Kmeans, ratios should have no impact.
        pass
    elif 2* ratio <= 0 or 2* ratio >= 1:
        pass
    else:
        # client 1: 90% cluster1, 10 % cluster2, 10 % cluster3
        # client 2: 10% cluster1, 90 % cluster2, 10 % cluster3
        # client 3: 10% cluster1, 10 % cluster2, 90 % cluster3
        train_x1, test_x1, train_y1, test_y1 = train_test_split(X1, y1, test_size= ratio, shuffle=True,
                                                      random_state=random_state)  # train set = 1-ratio
        # test_x11, test_x12, test_y11, test_y12 = train_test_split(X1, y1, test_size=0.5, shuffle=True,
        #                                                           random_state=random_state)  # each test set = 50% of rest data

        train_x2, test_x2, train_y2, test_y2 = train_test_split(X2, y2, test_size= ratio, shuffle=True,
                                                      random_state=random_state)
        # test_x21, test_x22, test_y21, test_y22 = train_test_split(X2, y2, test_size=0.5, shuffle=True,
        #                                                           random_state=random_state)

        # train_x3, X3, train_y3, y3 = train_test_split(X3, y3, test_size=2* ratio, shuffle=True,
        #                                               random_state=random_state)
        # test_x31, test_x32, test_y31, test_y32 = train_test_split(X3, y3, test_size=0.5, shuffle=True,
        #                                                           random_state=random_state)

        X1 = np.concatenate([train_x1, test_x2], axis=0)
        y1 = np.concatenate([train_y1, test_y2], axis=0) # be careful of this
        # y1 = np.zeros((X1.shape[0],))

        X2 = np.concatenate([test_x1, train_x2], axis=0)
        y2 = np.concatenate([test_y1, train_y2], axis=0)
        # y2 = np.ones((X2.shape[0],))



    is_show = False
    if is_show:
        # Plot init seeds along side sample data
        fig, ax = plt.subplots()
        # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
        colors = ["r", "g", "b", "m", 'black']
        ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='$G_1$')
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

        ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='$G_2$')
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

        ax.scatter(X3[:, 0], X3[:, 1], c=colors[2], marker="o", s=10, alpha=0.3, label='$G_3$')
        p = np.mean(X3, axis=0)
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
        ax.legend(loc='upper right', fontsize = 13)
        if args['SHOW_TITLE']:
            plt.title(dataset_detail.replace(':', '\n'))

        # if 'xlim' in kwargs:
        #     plt.xlim(kwargs['xlim'])
        # else:
        #     plt.xlim([-6, 6])
        # if 'ylim' in kwargs:
        #     plt.ylim(kwargs['ylim'])
        # else:
        #     plt.ylim([-6, 6])

        fontsize = 13
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.tight_layout()
        # if not os.path.exists(params['OUT_DIR']):
        #     os.makedirs(params['OUT_DIR'])
        # f = os.path.join(args['OUT_DIR'], dataset_detail+'.png')
        f = args['data_file'] + '.png'
        print(f)
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2)]):
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=2, shuffle=True,
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

def stats( in_dir = 'datasets/NBAIOT/Danmini_Doorbell'):
    csvs = ['benign_traffic.csv'] + \
           [os.path.join('gafgyt_attacks', f) for f in sorted(os.listdir(os.path.join(in_dir, 'gafgyt_attacks')))] + \
           [os.path.join('mirai_attacks', f) for f in sorted(os.listdir(os.path.join(in_dir, 'mirai_attacks')))]
    for csv in csvs:
        f = os.path.join(in_dir, csv)
        df = pd.read_csv(f)
        print(df.shape, f)

if __name__ == '__main__':
    stats()
	# nbaiot_diff_sigma_n({'N_CLIENTS': 0, 'N_CLUSTERS': 2, 'IS_PCA':True, 'DATASET': {'detail': 'n1_100+n2_100+n3_100:ratio_0.00:diff_sigma_n'}})

