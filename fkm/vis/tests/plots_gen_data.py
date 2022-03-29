import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


def main():
    random_state = 100
    n_samples_per_cluster = 1000
    cluster_std = 1

    centers = np.asarray([(1, 1), (5, 5), (5, 10), (10, 10), (10, 5)])
    n_sampes = n_samples_per_cluster * 2
    X, y_true = make_blobs(
        n_samples=n_sampes, centers=centers, cluster_std=cluster_std, random_state=random_state
    )

    # Plot init seeds along side sample data
    plt.figure(1)
    colors = ["r", "g", "b", "m", 'black']
    for k in set(y_true):
        cluster_data = y_true == k
        plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=colors[k], marker=".", s=10)
    # plt.title(f"Client_{i}")
    plt.xlim([-3, 15])
    plt.ylim([-3, 15])
    # plt.xticks([])
    # plt.yticks([])
    plt.savefig('gen_data.pdf', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()
