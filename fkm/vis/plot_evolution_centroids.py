"""
    run:
        module load anaconda3/2021.5
        cd /scratch/gpfs/ky8517/fkm/fkm
        PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 results/latex_plot.py

"""
import copy
import os
import pickle
import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
from fkm import config

def plot_evolution_centroids_fig(history, f_name='vis/out/', title=''):
    true_centroids = history['true_centroids']['train']
    iteration_histories = history['history']
    n_clusters, dim = history['initial_centroids'].shape
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    # plt.show()
    colors = ['green', 'blue', 'purple']
    markers = ['o', 'p', '*']
    for i in range(n_clusters):
        x = []
        y = []
        for j in range(len(iteration_histories)):
            cent = iteration_histories[j]['centroids'][i]
            x.append(cent[0])
            y.append(cent[1])
        # df = pd.DataFrame.from_dict({'x': [0, 3, 8, 7, 5, 3, 2, 1],
        #                              'y': [0, 1, 3, 5, 9, 8, 7, 5]})
        df = pd.DataFrame.from_dict({'x': x,
                                     'y': y})
        x = df['x'].values
        y = df['y'].values

        u = np.diff(x)  # direction from previous point to current one.
        v = np.diff(y)
        pos_x = x[:-1] + u / 2  # arrow position
        pos_y = y[:-1] + v / 2
        norm = np.sqrt(u ** 2 + v ** 2)

        # ax.plot(x, y, linestyle='-.', color=colors[i], marker=markers[i], alpha=0.1 *)
        alpha_values = np.linspace(0, 1, num=len(x))    # alpha \in [0, 1]
        # rgb = mcolors.to_rgb(colors[i])
        # current_colors = [tuple(list(rgb) + [alpha]) for alpha in np.linspace(0, 1, num=len(x))]  # Varying alpha for transparency
        # Vary arrow color intensity over time
        for t in range(len(x)):
            alpha_value = alpha_values[t]  # Using the random color values as alpha for varying intensities
            color = mcolors.to_rgba(colors[i], alpha=alpha_value)  # Setting the color with varying alpha
            ax.scatter(x[t], y[t], c=[color], marker=markers[i], s=100)
            if t < len(pos_x):
                if norm[t] == 0:
                    norm[t] += 1e-6
                ax.quiver(pos_x[t], pos_y[t], u[t] / norm[t], v[t] / norm[t], angles="xy", pivot='mid', color=color,
                      scale=25, headwidth=15, headlength=8, width=1e-3*2, linestyle='-.', linewidth=1)

        # ax.scatter(x, y, c=c, marker=markers[i], s=100,cmap=ListedColormap(current_colors))
        # # plt.show()
        # ax.quiver(pos_x, pos_y, u / norm, v / norm,  angles="xy",pivot='mid', color='gray',
        #           scale=40)

    # true centroids
    ax.scatter(true_centroids[:, 0], true_centroids[:, 1], color='red', marker='x', s=30)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    out_dir = 'out'
    fig_name = 'centroids'
    # f = os.path.join(out_dir, f'{fig_name}.png')
    f = f_name + f'_{fig_name}.png'
    plt.savefig(f, dpi=600, bbox_inches='tight')
    print('img:', os.path.abspath(f))
    if is_show:
        plt.show()


def plot_evolution_centroids(ax, history, f_name='vis/out/', n_iters=3, title=''):
    true_centroids = history['true_centroids']['train']
    iteration_histories = history['history']
    n_clusters, dim = history['initial_centroids'].shape
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    n_iters = min(n_iters, len(iteration_histories))
    colors = ['green', 'blue', 'purple', 'tab:blue', 'tab:orange','tab:green','tab:purple','tab:brown','tab:pink','tab:gray']
    markers = ['o', 'p', '*',',','.','^', 'v','s', 'p', 'h','3','4','8']
    aris = [it['scores']['train']['ari'] for it in iteration_histories]
    euclideans = [it['scores']['train']['euclidean'] for it in iteration_histories]
    # print(f'aris: {aris[-1]}')
    # print(f'euclideans: {euclideans[-1]}')
    for i in range(n_clusters):
        x = []
        y = []
        for j in range(n_iters):
            cent = iteration_histories[j]['centroids'][i]
            x.append(cent[0])
            y.append(cent[1])
        # df = pd.DataFrame.from_dict({'x': [0, 3, 8, 7, 5, 3, 2, 1],
        #                              'y': [0, 1, 3, 5, 9, 8, 7, 5]})
        df = pd.DataFrame.from_dict({'x': x,
                                     'y': y})
        x = df['x'].values
        y = df['y'].values

        u = np.diff(x)  # direction from previous point to current one.
        v = np.diff(y)
        pos_x = x[:-1] + u / 2  # arrow position
        pos_y = y[:-1] + v / 2
        norm = np.sqrt(u ** 2 + v ** 2)

        # ax.plot(x, y, linestyle='-.', color=colors[i], marker=markers[i], alpha=0.1 *)
        alpha_values = np.linspace(0.3, 1, num=len(x))    # alpha \in [0, 1]
        # rgb = mcolors.to_rgb(colors[i])
        # current_colors = [tuple(list(rgb) + [alpha]) for alpha in np.linspace(0, 1, num=len(x))]  # Varying alpha for transparency
        # Vary arrow color intensity over time
        for t in range(len(pos_x)):
            if t == 0:
                alpha_value = 0.3  # Using the random color values as alpha for varying intensities
                color = mcolors.to_rgba(colors[i], alpha=alpha_value)  # Setting the color with varying alpha
                ax.scatter(x[t], y[t], c=[color], marker=markers[i], s=100, facecolor='yellow', edgecolor='blue', linestyle='dotted')
                ax.scatter(x[t], y[t], c='brown', s=10, marker='o')  # initial centroids
            else:
                alpha_value = alpha_values[t]  # Using the random color values as alpha for varying intensities
                color = mcolors.to_rgba(colors[i], alpha=alpha_value)  # Setting the color with varying alpha
                ax.scatter(x[t], y[t], c=[color], marker=markers[i], s=100)
            if t < len(pos_x):
                if norm[t] == 0:
                    norm[t] += 1e-6
                color2 = mcolors.to_rgba(colors[i], alpha=0.8)
                ax.quiver(pos_x[t], pos_y[t], u[t] / norm[t], v[t] / norm[t], angles="xy", pivot='mid', color=color2,
                      scale=10, headwidth=15, headlength=8, width=2e-3*2, linestyle='-', linewidth=100)
            if t == len(pos_x)-1: # final centroid
                ax.scatter(x[t], y[t], c='yellow', s=10, marker='s')

        # ax.scatter(x, y, c=c, marker=markers[i], s=100,cmap=ListedColormap(current_colors))
        # # plt.show()
        # ax.quiver(pos_x, pos_y, u / norm, v / norm,  angles="xy",pivot='mid', color='gray',
        #           scale=40)

    # true centroids
    ax.scatter(true_centroids[:, 0], true_centroids[:, 1], color='red', marker='x', s=30)
    ax.set_title(title + f'({n_iters})')

    # plt.tight_layout()
    # plt.show()
    # out_dir = 'out'
    # fig_name = 'centroids'
    # # f = os.path.join(out_dir, f'{fig_name}.png')
    # f = f_name + f'_{fig_name}.png'
    # plt.savefig(f, dpi=600, bbox_inches='tight')
    # print('img:', os.path.abspath(f))
    # if is_show:
    #     plt.show()


def plot_real_case_P(in_dir, out_dir, alg2abbrev, csv_files):
    # dataset_name = dataset['name']
    # dataset_detail = dataset['detail']
    # n_clusters = dataset['n_clusters']
    # n_clients = dataset['n_clients']
    args = csv_files[0]
    n_repeats = args['N_REPEATS']

    rows = 2
    fig, axes = plt.subplots(rows, 7, sharex=False, sharey=False, figsize=(15, 5))  # (width, height)
    # axes = axes.reshape((1, 7))
    # fig_name = f'{dataset_name}_K_{n_clusters}_M_{n_clients}_R_{n_repeats}_diff_n_fixed_p_metrics'
    n_iterations = [3, 20]
    for row_idx in range(rows):
        n_iters = n_iterations[row_idx]
        for col_idx, args in enumerate(csv_files):
            ax = axes[row_idx][col_idx]
            try:
                f = args['csv_file']
                p = args['p']
                title = args['title']
                with open(f, 'rb') as file:
                    data = pickle.load(file)

                # centroids = data['true_centroids']
                # iteration_histories = data['history']
                plot_evolution_centroids(ax, data, f, n_iters=n_iters, title=title)

            except Exception as e:
                print(e)

    plt.tight_layout()

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig_name = 'centroids_evolution'
    f = os.path.join(out_dir, f'{fig_name}.png')
    plt.savefig(f, dpi=1000, bbox_inches='tight')
    print(os.path.abspath(f))
    if is_show:
        plt.show()



if __name__ == '__main__':
    config_file = 'config.yaml'
    args = config.load(config_file)

    # in_dir = os.path.abspath('~/Downloads/xlsx')
    IN_DIR = os.path.expanduser('out')
    # IN_DIR = os.path.expanduser('~/Downloads/xlsx')
    OUT_DIR = f'{IN_DIR}/latex_plot/'
    N_REPEATS = 50  # args['N_REPEATS']
    TOLERANCE = args['TOLERANCE']
    NORMALIZE_METHOD = args['NORMALIZE_METHOD']
    IS_REMOVE_OUTLIERS = args['IS_REMOVE_OUTLIERS']
    args['OUT_DIR'] = OUT_DIR
    SEPERTOR = args['SEPERTOR']

    ALG2ABBREV = {
        f'centralized_kmeans|R_{N_REPEATS}|random|None|{TOLERANCE}|{NORMALIZE_METHOD}': 'CKM-Random',
        f'centralized_kmeans|R_{N_REPEATS}|kmeans++|None|{TOLERANCE}|{NORMALIZE_METHOD}': 'CKM++',
        # f'federated_server_init_first|R_{N_REPEATS}|random|None|{TOLERANCE}|{NORMALIZE_METHOD}': 'Server-Initialized',
        f'federated_server_init_first|R_{N_REPEATS}|min_max|None|{TOLERANCE}|{NORMALIZE_METHOD}': 'Server-MinMax',
        f'federated_client_init_first|R_{N_REPEATS}|average|random|{TOLERANCE}|{NORMALIZE_METHOD}': 'Average-Random',
        f'federated_client_init_first|R_{N_REPEATS}|average|kmeans++|{TOLERANCE}|{NORMALIZE_METHOD}': 'Average-KM++',
        f'federated_greedy_kmeans|R_{N_REPEATS}|greedy|random|{TOLERANCE}|{NORMALIZE_METHOD}': 'Greedy-Random',
        f'federated_greedy_kmeans|R_{N_REPEATS}|greedy|kmeans++|{TOLERANCE}|{NORMALIZE_METHOD}': 'Greedy-KM++',
    }
    is_show = True
    ratios = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]

    # dataset_name = 'FEMNIST'
    # dataset_name = 'NBAIOT'
    IS_PCA = False  # args['IS_PCA']
    dataset_name = '3GAUSSIANS'

    if dataset_name == '3GAUSSIANS':
        n_clusters = 3
        n_clients = 3
        # csv_files = []
        N = 1000  # total cases: 8*7
        # for ratio in [0.0]: #[0, 0.1, 0.3, 0.5]:
        #     for n1 in [N]:
        #         # for n1 in [500, 2000, 3000, 5000, 8000]:
        #         for sigma1 in ["0.1_0.1"]:  # sigma  = [[0.1, 0], [0, 0.1]]
        #             for n2 in [N]:
        #                 for sigma2 in ["0.1_0.1"]:  # sigma  = [[0.1, 0], [0, 0.1]]
        #                     for n3 in [N]:  # [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
        #                         for sigma3 in ["1.0_0.1"]:  # sigma  = [[1, 0], [0, 0.1]]
        #                             dataset_detail = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
        #                             # f = f'{IN_DIR}/{dataset_name}/{dataset_detail}|{NORMALIZE_METHOD}|PCA_{IS_PCA}|M_{n_clients}|K_{n_clusters}|REMOVE_OUTLIERS_{IS_REMOVE_OUTLIERS}/SEED_DATA_0/centralized_kmeans/R_{N_REPEATS}|random|None|{TOLERANCE}|std/SEED_42/~history.dat'
        #                             f = f'{IN_DIR}/{dataset_name}/{dataset_detail}|{NORMALIZE_METHOD}|PCA_{IS_PCA}|M_{n_clients}|K_{n_clusters}|REMOVE_OUTLIERS_{IS_REMOVE_OUTLIERS}/SEED_DATA_0/federated_greedy_kmeans/R_{N_REPEATS}|greedy|kmeans++|{TOLERANCE}|std/SEED_42/~history.dat'
        #                             # dataset = {'name': dataset_name, 'detail': dataset_detail,  'p': n1, 'n_clients': n_clients, 'n_clusters':n_clusters, 'csv_file':f}
        #                             args1 = copy.deepcopy(args)
        #                             args1['csv_file'] = f
        #                             args1['p'] = ratio
        #                             SEED = args1['SEED']
        #                             args1['DATASET']['name'] = dataset_name
        #                             args1['DATASET']['detail'] = dataset_detail
        #                             args1['N_CLIENTS'] = n_clients
        #                             args1['N_CLUSTERS'] = n_clusters
        #                             N_CLIENTS = args1['N_CLIENTS']
        #                             N_CLUSTERS = args1['N_CLUSTERS']
        #                             args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'],
        #                                                                              f'M_{N_CLIENTS}',
        #                                                                              f'K_{N_CLUSTERS}', f'SEED_{SEED}'])
        #                         csv_files.append(copy.deepcopy(args1))
        p = 0.10
        csv_file1 = f'out/3GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_1.0_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0/centralized_kmeans/R_50|random|None|0.0001|std/SEED_42/~history.dat'
        csv_file2 = f'out/3GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_1.0_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0/centralized_kmeans/R_50|kmeans++|None|0.0001|std/SEED_42/~history.dat'
        csv_file3 = f'out/3GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_1.0_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_server_init_first/R_50|min_max|None|0.0001|std/SEED_42/~history.dat'
        csv_file4 = f'out/3GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_1.0_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_client_init_first/R_50|average|random|0.0001|std/SEED_42/~history.dat'
        csv_file5 = f'out/3GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_1.0_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_client_init_first/R_50|average|kmeans++|0.0001|std/SEED_42/~history.dat'
        csv_file6 = f'out/3GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_1.0_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_greedy_kmeans/R_50|greedy|random|0.0001|std/SEED_42/~history.dat'
        csv_file7 = f'out/3GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_1.0_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_greedy_kmeans/R_50|greedy|kmeans++|0.0001|std/SEED_42/~history.dat'
        # config_file = 'out/MNIST/n1_50-sigma1_0.1_0.1+n2_50-sigma2_0.1_0.1+n3_50-sigma3_1.0_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0/centralized_kmeans/R_50|random|None|0.0001|std/SEED_42/~history.dat'

        # # 10Gaussians
        # p = 0.10
        # csv_file1 = f'out/10GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_0.1_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_10|K_10|REMOVE_OUTLIERS_False/SEED_DATA_0/centralized_kmeans/R_50|random|None|0.0001|std/SEED_42/~history.dat'
        # csv_file2 = f'out/10GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_0.1_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_10|K_10|REMOVE_OUTLIERS_False/SEED_DATA_0/centralized_kmeans/R_50|kmeans++|None|0.0001|std/SEED_42/~history.dat'
        # csv_file3 = f'out/10GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_0.1_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_10|K_10|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_server_init_first/R_50|min_max|None|0.0001|std/SEED_42/~history.dat'
        # csv_file4 = f'out/10GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_0.1_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_10|K_10|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_client_init_first/R_50|average|random|0.0001|std/SEED_42/~history.dat'
        # csv_file5 = f'out/10GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_0.1_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_10|K_10|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_client_init_first/R_50|average|kmeans++|0.0001|std/SEED_42/~history.dat'
        # csv_file6 = f'out/10GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_0.1_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_10|K_10|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_greedy_kmeans/R_50|greedy|random|0.0001|std/SEED_42/~history.dat'
        # csv_file7 = f'out/10GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_0.1_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_10|K_10|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_greedy_kmeans/R_50|greedy|kmeans++|0.0001|std/SEED_42/~history.dat'

        csvs = [(csv_file1, 'CKM-Random'), (csv_file2, 'CKM++'),
                (csv_file3, 'Server-MinMax'),
                (csv_file4, 'Average-Random'), (csv_file5, 'Average-KM++'),
                (csv_file6, 'Greedy-Random'), (csv_file7, 'Greedy-KM++'),]
        csv_files = []
        for f,title in csvs:
            csv_files.append({'N_REPEATS': N_REPEATS, 'p':0.00, 'csv_file':f, 'title': title})
        plot_real_case_P(IN_DIR, OUT_DIR, ALG2ABBREV, csv_files)
        exit()

