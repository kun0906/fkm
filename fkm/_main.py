# Email: Kun.bj@outlook.com
import argparse
import copy
import json
import os
import shutil
import time
import traceback
from collections import Counter
from pprint import pprint

import numpy as np

from fkm.clustering.my_kmeans import KMeans
from fkm.datasets.gen_dummy import load_federated
from fkm.experiment_cases import get_experiment_params
from fkm.utils.utils_func import plot_centroids, dump, save_image2disk, predict_n_saveimg, \
    plot_metric_over_time_2gaussian, plot_metric_over_time_femnist, obtain_true_centroids, \
    plot_centroids_diff_over_time, history2movie
from fkm.utils.utils_func import timer
from fkm.utils.utils_stats import evaluate2


def save_history2txt(seed_history, out_file='.txt'):
    """
        with open(seed_file + '.txt', 'w') as file:
            file.write(json.dumps(seed_history))  # not working
    Returns
    -------

    """

    def format(data):
        res = ''
        if type(data) == dict:
            for k, v in data.items():
                res += f'{k}: ' + format(v) + '\n'
        elif type(data) == list:
            res += f'{data} \n'
        else:
            res += f'{data} \n'

        return res

    with open(out_file, 'w') as f:
        f.write('***Save data with pprint\n')
        pprint(seed_history, stream=f, sort_dicts=False)    # 'sort_dicts=False' works when python version >= 3.8

        f.write('\n\n***Save data with recursion')
        res = format(seed_history)
        f.write(res)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@timer
def run_clustering_federated(params, KMeansFederated, verbose=5):
    """

    Parameters
    ----------
    params
    KMeansFederated
    verbose:
        0 < verbose <= 5: info
        5 < verbose <= 10: debug

    Returns
    -------

    """
    # settings
    np.random.seed(1234)  # set the global seed for numpy
    is_show = True
    params['is_show'] = is_show
    OUT_DIR = params['out_dir']
    REPEATS = params['repeats']
    print(f'OUT_DIR: {OUT_DIR}')
    seeds = [10 * v ** 2 for v in range(1, REPEATS + 1, 1)]
    print(f'REPEATS: {REPEATS}, {seeds}')
    N_CLUSTERS = params['n_clusters']
    LIMIT_CSV = None
    EPOCHS = params['client_epochs']
    ROUNDS = 500
    TOLERANCE = 1e-10   # 1e-4, 1e-6
    LR = None  # 0.5
    LR_AD = None
    EPOCH_LR = None  # 0.5
    MOMENTUM = None  # 0.5
    # RECORD = range(1, ROUNDS+1)
    RECORD = None
    REASSIGN = (None, None)  # (0.01, 10)
    histories = {}  # Store all results and needed data
    stats = {}  # store all average results
    n_clients = [params['n_clients']]  # Currently, we use all clients
    fig_paths = []
    server_init_centroids = params['server_init_centroids']
    client_init_centroids = params['client_init_centroids']
    for grid_i, C in enumerate(n_clients):
        print(f'grid_i: {grid_i}, clients_per_round_fraction (C): {C}')

        results = {}
        splits = ['train', 'test']
        for split in splits:
            results[split] = []

        raw_x, raw_y = load_federated(
            limit_csv=LIMIT_CSV,
            verbose=verbose,
            seed=100,  # the seed is fixed.
            clusters=N_CLUSTERS,
            n_clients=C,
            params=params
        )
        n_clients = len(raw_x['train']) if params['is_federated'] else 0
        out_dir_i = os.path.join(OUT_DIR, f'Clients_{n_clients}')
        # if os.path.exists(out_dir_i):
        #     shutil.rmtree(out_dir_i, ignore_errors=True)
        if not os.path.exists(out_dir_i):
            os.makedirs(out_dir_i)

        if verbose > 0:
            print(f'n_clients={n_clients} when C={C}')
            # print raw_x and raw_y distribution
            for split in splits:
                print(f'{split} set:')
                clients_x, clients_y = raw_x[split], raw_y[split]
                if verbose >= 5:
                    # print each client distribution
                    for c_i, (c_x, c_y) in enumerate(zip(clients_x, clients_y)):
                        print(f'\tClient_{c_i}, n_datapoints: {len(c_y)}, '
                              f'cluster_size: {sorted(Counter(c_y).items(), key=lambda kv: kv[0], reverse=False)}')

                y_tmp = []
                for vs in clients_y:
                    y_tmp.extend(vs)
                print(f'n_{split}_clients: {len(clients_x)}, n_datapoints: {sum(len(vs) for vs in clients_y)}, '
                      f'cluster_size: {sorted(Counter(y_tmp).items(), key=lambda kv: kv[0], reverse=False)}')

        # obtain the true centroids given raw_x and raw_y
        true_centroids = obtain_true_centroids(raw_x, raw_y, splits, params)

        if verbose:
            # print true centroids
            for split in splits:
                true_c = true_centroids[split]
                print(f'{split}_true_centroids: {true_c}')
        histories[n_clients] = {'n_clients': n_clients, 'C': C, 'true_centroids': true_centroids}

        if params['p0'] == 'FEMNIST':
            save_image2disk((raw_x, raw_y), out_dir_i, params)

        history = {'x': raw_x, 'y': raw_y, 'results': []}
        for s_i, SEED in enumerate(seeds):  # repetitions:  to obtain average and std score.
            t1 = time.time()
            if verbose > 5:
                print(f'\n***{s_i}th repeat with seed: {SEED}:')
            x = copy.deepcopy(raw_x)
            y = copy.deepcopy(raw_y)
            if not params['is_federated']:  # or C == "central":
                # collects all clients' data together
                for spl in splits:
                    x[spl] = np.concatenate(x[spl], axis=0)
                    y[spl] = np.concatenate(y[spl], axis=0)

                # for Centralized Kmeans, we use sever_init_centroids as init_centroids.
                kmeans = KMeans(
                    n_clusters=N_CLUSTERS,
                    # batch_size=BATCH_SIZE,
                    init_centroids=server_init_centroids,
                    true_centroids=true_centroids,
                    random_state=SEED,
                    max_iter=ROUNDS,
                    reassign_min=REASSIGN[0],
                    reassign_after=REASSIGN[1],
                    verbose=verbose,
                    tol= TOLERANCE,
                    params=params
                )
            else:
                kmeans = KMeansFederated(
                    n_clusters=N_CLUSTERS,
                    # batch_size=BATCH_SIZE,
                    sample_fraction=C,
                    epochs_per_round=EPOCHS,
                    max_iter=ROUNDS,
                    server_init_centroids=server_init_centroids,
                    client_init_centroids=client_init_centroids,
                    true_centroids=true_centroids,
                    random_state=SEED,
                    learning_rate=LR,
                    adaptive_lr=LR_AD,
                    epoch_lr=EPOCH_LR,
                    momentum=MOMENTUM,
                    reassign_min=REASSIGN[0],
                    reassign_after=REASSIGN[1],
                    verbose=verbose,
                    tol=TOLERANCE,
                    params = params
                )
            if verbose > 5:
                # print all kmeans's variables.
                pprint(vars(kmeans))

            # During the training, we also evaluate the model on the test set at each iteration
            kmeans.fit(
                x, y, splits,
                record_at=RECORD,
            )

            # After training, we obtain the final scores on the test set.
            scores = evaluate2(
                kmeans=kmeans,
                x=x, y=y,
                splits=splits,
                federated=params['is_federated'],
                verbose=verbose,
            )
            # To save the disk storage, we only save the first repeat results.
            if params['p0'] == 'FEMNIST' and s_i == 0:
                try:
                    predict_n_saveimg(kmeans, x, y, splits, SEED,
                                      federated=params['is_federated'], verbose=verbose,
                                      out_dir=os.path.join(out_dir_i, f'SEED_{SEED}'),
                                      params=params, is_show=is_show)
                except Exception as e:
                    print(f'Error: {e}')
                    # traceback.print_exc()

            # for each seed, we will save the results.
            seed_history = {'seed': SEED, 'initial_centroids': kmeans.initial_centroids,
                            'true_centroids': kmeans.true_centroids,
                            'final_centroids': kmeans.cluster_centers_,
                            'training_iterations': kmeans.training_iterations,
                            'history': kmeans.history,
                            'scores': scores}
            # save the current 'history' to disk before plotting.
            seed_file = os.path.join(out_dir_i, f'SEED_{SEED}', f'~history.dat')
            dump(seed_history, out_file=seed_file)
            try:
                save_history2txt(seed_history, out_file=seed_file + '.txt')
            except Exception as e:
                print(f'save_history2txt() fails when SEED={SEED}, Error: {e}')

            history['results'].append(seed_history)
            if verbose > 5:
                print(f'grid_i:{grid_i}, SEED:{SEED}, training_iterations:{kmeans.training_iterations}, '
                      f'scores:{scores}')

            t2 = time.time()
            print(f'{s_i}th iteration takes {(t2 - t1):.4f}s')

        # save the current 'history' to disk before plotting.
        dump(history, out_file=os.path.join(out_dir_i, f'Clients_{n_clients}.dat'))

        # get the average and std for each clients_per_round_fraction: C
        results_avg = {}
        for split in splits:
            metric_names = history['results'][0]['scores'][split].keys()
            if split == 'train':
                training_iterations = [vs['training_iterations'] for vs in history['results']]
                results_avg[split] = {'Iterations': (np.mean(training_iterations), np.std(training_iterations))}
            else:
                results_avg[split] = {'Iterations': ('', '')}
            for k in metric_names:
                value = [vs['scores'][split][k] for vs in history['results']]
                try:
                    score_mean = np.mean(value)
                    score_std = np.std(value)
                    # score_mean = np.around(np.mean(value), decimals=3)
                    # score_std = np.around(np.std(value), decimals=3)
                except Exception as e:
                    print(f'Error: {e}, {split}, {k}')
                    score_mean = -1
                    score_std = - 1
                results_avg[split][k] = (score_mean, score_std)
        if verbose > 5:
            print(f'results_avg:')
            pprint(results_avg)
        stats[n_clients] = results_avg
        histories[n_clients]['history'] = history
        histories[n_clients]['results_avg'] = results_avg
        # save the current 'history' to disk before plotting.
        dump(histories, out_file=os.path.join(OUT_DIR, f'~histories.dat'))

        try:
            title = f'Centralized KMeans with {server_init_centroids} initialization' if not params['is_federated'] \
                else f'Federated KMeans with {server_init_centroids} (Server) and {client_init_centroids} (Clients)'
            fig_path = plot_centroids(history, out_dir=out_dir_i,
                                      title=title + f'. {n_clients} Clients',
                                      fig_name=f'M={n_clients}-Centroids', params=params, is_show=is_show)
            fig_paths.append(fig_path)
            if params['p0'] == 'FEMNIST':
                plot_metric_over_time_femnist(history, out_dir=f'{out_dir_i}/over_time',
                                              title=title + f'. {n_clients} Clients', fig_name=f'M={n_clients}',
                                              params=params, is_show=is_show)
            elif params['p0'] in ['2GAUSSIANS','3GAUSSIANS', '5GAUSSIANS' ]:
                plot_metric_over_time_2gaussian(history, out_dir=f'{out_dir_i}/over_time',
                                                title=title + f'. {n_clients} Clients', fig_name=f'M={n_clients}',
                                                params=params, is_show=is_show)
            # plot centroids_update and centroids_diff over time.
            plot_centroids_diff_over_time(history,
                                          out_dir=f'{out_dir_i}/over_time',
                                          title=title + f'. {n_clients} Clients', fig_name=f'M={n_clients}-Scores',
                                          params=params, is_show=is_show)
            # save history as animation
            history2movie(history, out_dir=f'{out_dir_i}/over_time',
                                            title=title + f'. {n_clients} Clients', fig_name=f'M={n_clients}',
                                            params=params, is_show=is_show)

        except Exception as e:
            print(f'Error: {e}')
            traceback.print_exc()

    if verbose > 0:
        print('stats:')
        pprint(stats)
    out_file = os.path.join(OUT_DIR, f'varied_clients-Server_{server_init_centroids}-Client_{client_init_centroids}')
    print(out_file)
    dump(stats, out_file=out_file + '.dat')
    with open(out_file + '.txt', 'w') as file:
        file.write(json.dumps(stats))  # use `json.loads` to do the reverse

    dump(histories, out_file=out_file + '-histories.dat')
    with open(out_file + '-histories.txt', 'w') as file:
        file.write(json.dumps(histories, cls=NumpyEncoder))  # use `json.loads` to do the reverse


    # will be needed.
    # figs2movie(fig_paths, out_file=out_file + '.mp4')
    #
    # plot_stats2(
    #     stats,
    #     x_variable=n_clients,
    #     x_variable_name="Fraction of Clients per Round",
    #     metric_name=None,
    #     title='Random Initialization'
    # )

#
# def main(KMeansFederated):
#     print(__file__)
#     parser = argparse.ArgumentParser(description='Description of your program')
#     # parser.add_argument('-p', '--py_name', help='python file name', required=True)
#     parser.add_argument('-S', '--dataset', help='dataset', default='FEMNIST')
#     parser.add_argument('-T', '--data_details', help='data details', default='1client_multiwriters_1digit')
#     parser.add_argument('-M', '--algorithm', help='algorithm', default='Centralized_random')
#     # args = vars(parser.parse_args())
#     args = parser.parse_args()
#     print(args)
#     params = get_experiment_params(p0=args.dataset, p1=args.data_details, p2=args.algorithm)
#     pprint(params)
#     try:
#         run_clustering_federated(
#             params,
#             KMeansFederated,
#             verbose=10,
#         )
#     except Exception as e:
#         print(f'Error: {e}')
#         traceback.print_exc()
