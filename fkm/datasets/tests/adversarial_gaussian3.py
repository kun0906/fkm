import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

def adversarial_gaussian3_diff_sigma_n(params, random_state=42, **kwargs):
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
    p1 = params['DATASET']['params'].split('|')
    ratio = float(p1[-1].split('_')[1])

    p1_0_c1 = p1[0].split('-')
    n1 = int(p1_0_c1[0].split('_')[1])
    tmp = p1_0_c1[1].split('_')
    sigma1_0, sigma1_1 = float(tmp[1]), float(tmp[2])

    p1_0_c2 = p1[1].split('-')
    n2 = int(p1_0_c2[0].split('_')[1])
    tmp = p1_0_c2[1].split('_')
    sigma2_0, sigma2_1 = float(tmp[1]), float(tmp[2])

    p1_0_c3 = p1[2].split('-')
    n3 = int(p1_0_c3[0].split('_')[1])
    tmp = p1_0_c3[1].split('_')
    sigma3_0, sigma3_1 = float(tmp[1]), float(tmp[2])

    p1_0_c4 = p1[3].split('-')
    # n4 = int(p1_0_c4[0].split('_')[1])
    p4 = float(p1_0_c4[0].split('_')[1])
    tmp = p1_0_c4[1].split('_')
    sigma4_0, sigma4_1 = float(tmp[1]), float(tmp[2])

    def gen_noise(n4, random_state=10):
        # noise:
        mus = [0, 0]
        cov = np.asarray([[sigma4_0, 0], [0, sigma4_1]])
        r = np.random.RandomState(random_state)
        X_noise = r.multivariate_normal(mus, cov, size=n4)
        # X_noise = r.uniform(-5, 5, size=(n4, 2))    # 2 is the dimension
        y_noise = np.asarray([4] * n4)

        return X_noise, y_noise

    def gen_noise1(n4,mus = [0, 0], random_state=10):
        # noise:
        cov = np.asarray([[sigma4_0, 0], [0, sigma4_1]])
        r = np.random.RandomState(random_state)
        X_noise = r.multivariate_normal(mus, cov, size=n4)
        # X_noise = r.uniform(-5, 5, size=(n4, 2))    # 2 is the dimension
        y_noise = np.asarray([4] * n4)

        return X_noise, y_noise

    def gen_noise2(n4, mus = [-1, -1], random_state=10):
        # noise:
        cov = np.asarray([[sigma4_0, 0], [0, sigma4_1]])
        r = np.random.RandomState(random_state)
        X_noise = r.multivariate_normal(mus, cov, size=n4)
        # X_noise = r.uniform(-5, 5, size=(n4, 2))    # 2 is the dimension
        y_noise = np.asarray([4] * n4)

        return X_noise, y_noise

    def gen_noise3(n4,  mus = [1, 1], random_state=10):
        # noise:
        cov = np.asarray([[sigma4_0, 0], [0, sigma4_1]])
        r = np.random.RandomState(random_state)
        X_noise = r.multivariate_normal(mus, cov, size=n4)
        # X_noise = r.uniform(-5, 5, size=(n4, 2))    # 2 is the dimension
        y_noise = np.asarray([4] * n4)

        return X_noise, y_noise

    def get_xy(n=0):

        # client 1
        mus = [-3, 0]
        # sigma = 0.1
        # n1 = 2000
        cov = np.asarray([[sigma1_0, 0], [0, sigma1_1]])
        # cov = np.asarray([[sigma, sigma], [sigma, sigma]])
        r = np.random.RandomState(random_state)
        X1 = r.multivariate_normal(mus, cov, size=n1)
        y1 = np.asarray([0] * n1)
        # X1_, y1_ = gen_noise(n4, random_state)
        # X1_, y1_ = gen_noise(int(np.ceil(n1*p4)),random_state)
        X1_, y1_ = gen_noise1(int(np.ceil(n1 * p4)), mus = [-1, 0], random_state=random_state)
        X4 = X1_[:]
        y4 = y1_[:]

        # client 2
        mus = [3, 0]
        # sigma = 0.2
        # n2 = 5000
        cov = np.asarray([[sigma2_0, 0], [0, sigma2_1]])
        # cov = np.asarray([[sigma, -sigma], [-sigma, sigma]])
        X2 = r.multivariate_normal(mus, cov, size=n2)
        y2 = np.asarray([1] * n2)
        # X2_, y2_ = gen_noise(n4, random_state*50)
        # X2_, y2_ = gen_noise(int(np.ceil(n2*p4)), random_state *50)
        X2_, y2_ = gen_noise2(int(np.ceil(n2 * p4)), mus = [5, 0], random_state=random_state * 50)
        X4 = np.concatenate([X4, X2_], axis=0)
        y4 = np.concatenate([y4, y2_], axis=0)

        # client 3
        mus = [0, 5]
        # n3 = 10000
        # sigma = 0.3
        cov = np.asarray([[sigma3_0, 0], [0, sigma3_1]])
        X3 = r.multivariate_normal(mus, cov, size=n3)
        y3 = np.asarray([2] * n3)
        # X3_, y3_ = gen_noise(n4, random_state*200)
        # X3_, y3_ = gen_noise(int(np.ceil(n3*p4)), random_state * 200)
        X3_, y3_ = gen_noise3(int(np.ceil(n3 * p4)), mus=[0, 8], random_state=random_state * 200)
        X4 = np.concatenate([X4, X3_], axis=0)
        y4 = np.concatenate([y4, y3_], axis=0)


        SHOW = params['SHOW']
        if SHOW:
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

            # show the noise n4
            ax.scatter(X4[:, 0], X4[:, 1], c=colors[3], marker="o", s=10, alpha=0.3)
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
            ax.legend(loc='upper right', fontsize=13)
            if params['show_plt_title']:
                title = params['DATASET']['params']
                plt.title(title[:30] + '\n' + title[30:])

            if 'xlim' in kwargs:
                plt.xlim(kwargs['xlim'])
            else:
                plt.xlim([-10, 10])
            if 'ylim' in kwargs:
                plt.ylim(kwargs['ylim'])
            else:
                plt.ylim([-10, 10])

            fontsize = 13
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)

            plt.tight_layout()
            if not os.path.exists(params['OUT_DIR']):
                os.makedirs(params['OUT_DIR'])
            f = os.path.join(params['OUT_DIR'],  'noise-sep.png')
            print(f)
            plt.savefig(f, dpi=600, bbox_inches='tight')
            plt.show()


        # add noise to each Gaussian
        X1 = np.concatenate([X1, X1_], axis=0)
        y1 = np.concatenate([y1, y1_], axis=0)

        X2 = np.concatenate([X2, X2_], axis=0)
        y2 = np.concatenate([y2, y2_], axis=0)


        X3 = np.concatenate([X3, X3_], axis=0)
        y3 = np.concatenate([y3, y3_], axis=0)


        return X1, y1, X2, y2, X3, y3

    X1, y1, X2, y2, X3, y3 = get_xy()
    if 'ckm' in params['DATASET']['name']:   # for Centralized Kmeans, ratios should have no impact.
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
        # y1 = np.concatenate([train_y1, test_y2], axis=0) # be careful of this
        y1 = np.zeros((X1.shape[0],))

        X2 = np.concatenate([test_x11, train_x2, test_x32], axis=0)
        # y2 = np.concatenate([test_y1, train_y2], axis=0)
        y2 = np.ones((X2.shape[0],))

        X3 = np.concatenate([test_x12, test_x22, train_x3], axis=0)
        y3 = np.ones((X3.shape[0],)) * 2


    SHOW = params['SHOW']
    if SHOW:
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

        # # show the noise n4
        # ax.scatter(X3[:, 0], X3[:, 1], c=colors[2], marker="o", s=10, alpha=0.3, label='$G_3$')
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
        ax.legend(loc='upper right', fontsize = 13)
        if params['show_plt_title']:
            title = params['DATASET']['params']
            plt.title(title[:30]+'\n' + title[30:])

        if 'xlim' in kwargs:
            plt.xlim(kwargs['xlim'])
        else:
            plt.xlim([-10, 10])
        if 'ylim' in kwargs:
            plt.ylim(kwargs['ylim'])
        else:
            plt.ylim([-10, 10])

        fontsize = 13
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        plt.tight_layout()
        if not os.path.exists(params['OUT_DIR']):
            os.makedirs(params['OUT_DIR'])
        f = os.path.join(params['OUT_DIR'], 'noise.png')
        print(f)
        plt.savefig(f, dpi=600, bbox_inches='tight')
        plt.show()

    clients_train_x = []
    clients_train_y = []
    clients_test_x = []
    clients_test_y = []
    for i, (x, y) in enumerate([(X1, y1), (X2, y2), (X3, y3)]):
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

