
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def gaussian5_5clients_5clusters(params, random_state=42):
    """
     if params['p1'] == '1client_5clusters':
    # 5 clusters in R^2

    Parameters
    ----------
    params
    random_state

    Returns
    -------

    """

    def get_xy(n=10000):

        r = np.random.RandomState(random_state)

        data = []
        # client 1
        mus = [0, 0]
        sigma = 0.5
        n1 = 10000
        cov = np.asarray([[sigma, 0], [0, sigma]])
        # cov = np.asarray([[sigma, -sigma], [-sigma, sigma]])
        X1 = r.multivariate_normal(mus, cov, size=n1)
        y1 = np.asarray([0] * n1)
        data.append((X1, y1))

        # client 2, 3, 4, 5
        for i, mus in enumerate([[-3, 0], [0, -3], [3, 0],  [0, 3]]):
            sigma = 0.1
            n2 = 2000
            cov = np.asarray([[sigma, 0], [0, sigma]])
            # cov = np.asarray([[sigma, sigma], [sigma, sigma]])
            X2 = r.multivariate_normal(mus, cov, size=n2)
            y2 = np.asarray([i+1] * n2)
            data.append((X2, y2))

        return data

    data = get_xy(n=10000)
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
        for i, (x, y) in enumerate(data):
            ax.scatter(x[:, 0], x[:, 1], c=colors[i], marker="x", s=10, alpha=0.3, label=f'centroid_{i+1}')
            p = np.mean(x, axis=0)
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
    for i, (x, y) in enumerate(data):
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

