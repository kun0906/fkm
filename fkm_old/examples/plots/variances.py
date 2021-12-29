import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import os

def gen(cluster_std=1, n_sampes=1000, random_state=42):
    centers = np.asarray([(10, 10)])
    # centers = np.asarray([(0, 0), (10, 10)])
    X, y = make_blobs(
        n_samples=n_sampes, centers=centers, cluster_std=cluster_std, random_state=random_state
    )
    return X, y


def distance(x1, x2):
    return np.sqrt(sum([(v1 - v2) ** 2 for v1, v2 in zip(x1, x2)]))


def main():
    data_txt = 'variance_50.txt'
    flg = True
    if not os.path.exists(data_txt):
        c1 = (0, 0)
        c2 = (10, 10)
        data = []
        stds = [v/10 for v in list(range(1, 100+1, 10))]
        # print(stds)
        for std in stds:
            # std = (i+1)
            n2 = 5 * 10**3
            X, y = gen(cluster_std=std, n_sampes=n2)
            s = sum([distance(c1, v1) for v1 in X])
            d12 = distance(c1, c2)
            d22 = sum([distance(c2, v1) for v1 in X]) / n2
            alpha = (s - n2 * d12) / (n2 * d22)
            print(std, d12 / d22, d22/d12, alpha)
            data.append((std, d12 / d22, d22/d12, alpha))
        np.savetxt(data_txt, data)
    data = np.loadtxt(data_txt)
    print(data)
    print(len(data))
    # Plot init seeds along side sample data
    plt.figure(1)
    x1, x2, x3, y = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    plt.plot(x2, y, 'b*', label='True')

    # curve_fitting with polynormial
    coefs = np.polyfit(x2, y, 6)
    print(coefs)
    y_pred = np.poly1d(coefs)(x2)
    plt.plot(x2, y_pred, 'ro', alpha=0.1, label='Poly')

    # curve_fitting with exp
    from scipy.optimize import curve_fit
    def f(x, a, b, c, d):
        # return a * np.exp(-x*b) + 2* c - b* np.exp(-x)
        return a / np.power(x*b, 2/3) #+ d*c
    def f2(x, a, b, c):
        return a * np.log(-x*b) + c

    popt, pcov = curve_fit(f, x2, y)
    # plt.plot(x2, f(np.asarray(x2), *popt), 'g+')

    # plt.plot(x3, y, marker = 'o')
    # plt.plot(x3, y)
    # plt.title(f"")
    # plt.xlim([0, 15])
    # plt.ylim([0, 15])
    plt.legend()
    plt.xlabel('d12/d22')
    # plt.xlabel('d22/d12')
    plt.ylabel('$\\alpha^{*}$')
    plt.show()

def plot_data():
    cluster_std = 1000
    random_state = 42
    centers = np.asarray([(0, 0), (10, 10)])
    n_sampes = 5 * 10**5
    X, y_true = make_blobs(
        n_samples=n_sampes, centers=centers, cluster_std=cluster_std, random_state=random_state
    )

    # Plot init seeds along side sample data
    plt.figure(1)
    # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
    colors = ["r", "g", "b", "m", 'black']
    for k, col in enumerate(colors):
        cluster_data = y_true == k
        plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=".", s=10)

    # # plt.title(f"Client_{i}")
    # plt.xlim([0, 15])
    # plt.ylim([0, 15])
    # plt.xticks([])
    # plt.yticks([])
    plt.show()

if __name__ == '__main__':
    main()
    # plot_data()
