"""
    srun --nodes=1 --ntasks-per-node=1 --time=01:00:00 --pty bash -i
    module load anaconda3/2021.5

    cd fkm/fkm/
    PYTHONPATH='..' python3 plot_results.py

"""
import os

from fkm.experiment_cases import get_experiment_params
from fkm.sbatch import gen_cases
# from fkm.utils.utils_func import load, plot_metric_over_time
from fkm.utils.utils_func import load, history2movie


def save2txt(data, out=''):
    with open(out, 'w') as f:
        # f.write(data)
        for vs in data:
            print(vs, file=f)

    return out

def main():
    # data_name = 'femnist_0.1users_0.3data_testset'
    # out_dir = 'results-epoch=5'
    # in_files = [
    #     # Server_average-Clients_random
    #     f'{out_dir}/{data_name}/Server_average-Clients_random/0.1user_0.3data_testset_C10/varied clients-random-histories.dat',
    #     # Server_average-Clients_kmeans_++
    #     f'{out_dir}/{data_name}/Server_average-Clients_kmeans_pp/0.1user_0.3data_testset_C10/varied clients-kmeans++-histories.dat',
    #
    #     # Server_random
    #     f'{out_dir}/{data_name}/Server_random/0.1user_0.3data_testset_C10/varied clients-random-histories.dat',
    #
    #     # Server_greedy-Clients_random
    #     f'{out_dir}/{data_name}/Server_greedy-Clients_random/0.1user_0.3data_testset_C10/varied clients-random-histories.dat',
    #     # Server_greedy-Clients_kmeans_++
    #     f'{out_dir}/{data_name}/Server_greedy-Clients_kmeans_pp/0.1user_0.3data_testset_C10/varied clients-kmeans++-histories.dat',
    #     ]
    #
    # data_name = 'femnist_0.1users_20clients0.3data_testset'
    # in_files_ = [
    #     # Server_average-Clients_random
    #     f'{out_dir}/{data_name}/Server_average-Clients_random/0.1user_0.3data_testset_Clients20_C10/varied clients-random-histories.dat',
    #     # Server_average-Clients_kmeans_++
    #     f'{out_dir}/{data_name}/Server_average-Clients_kmeans_pp/0.1user_0.3data_testset_Clients20_C10/varied clients-kmeans++-histories.dat',
    #
    #     # Server_random
    #     f'{out_dir}/{data_name}/Server_random/0.1user_0.3data_testset_Clients20_C10/varied clients-random-histories.dat',
    #
    #     # Server_greedy-Clients_random
    #     f'{out_dir}/{data_name}/Server_greedy-Clients_random/0.1user_0.3data_testset_Clients20_C10/varied clients-random-histories.dat',
    #     # Server_greedy-Clients_kmeans_++
    #     f'{out_dir}/{data_name}/Server_greedy-Clients_kmeans_pp/0.1user_0.3data_testset_Clients20_C10/varied clients-kmeans++-histories.dat',
    # ]
    # in_files += in_files_
    #
    # data_name = 'femnist_10clients'
    # in_files_ = [
    #     # Server_average-Clients_random
    #     f'{out_dir}/{data_name}/Server_average-Clients_random/0.1user_0.3data_per_digit_testset_C10/varied clients-random-histories.dat',
    #     # Server_average-Clients_kmeans_++
    #     f'{out_dir}/{data_name}/Server_average-Clients_kmeans_pp/0.1user_0.3data_per_digit_testset_C10/varied clients-kmeans++-histories.dat',
    #
    #     # Server_random
    #     f'{out_dir}/{data_name}/Server_random/0.1user_0.3data_per_digit_testset_C10/varied clients-random-histories.dat',
    #
    #     # Server_greedy-Clients_random
    #      f'{out_dir}/{data_name}/Server_greedy-Clients_random/0.1user_0.3data_per_digit_testset_C10/varied clients-random-histories.dat',
    #     # Server_greedy-Clients_kmeans_++
    #      f'{out_dir}/{data_name}/Server_greedy-Clients_kmeans_pp/0.1user_0.3data_per_digit_testset_C10/varied clients-kmeans++-histories.dat',
    # ]
    # in_files += in_files_
    # in_files = []
    # for in_file in in_files:
    #     print(f'\n{in_file}')
    #     try:
    #         results = load(in_file)
    #         print(results.keys())
    #         out_dir = os.path.dirname(in_file)
    #
    #         for n_clients in results.keys():
    #             (results_avg, history) = results[n_clients]
    #             print(f'n_clients: {n_clients}, results_avg: {results_avg}')
    #             x, y = history['x'], history['y']
    #             arr = in_file.split('/')
    #             title = f'{arr[1]}\n{arr[2]}'
    #             fig_path = plot_metric_over_time(history, out_dir=f'{out_dir}/report',
    #                                       title=title, fig_name=f'M={n_clients}')
    #             save2txt((in_file, n_clients, results_avg, history), out=fig_path+'.txt')
    #
    #             # # save history as animation
    #             # history2movie(history, out_dir=f'{out_dir_i}/over_time',
    #             #               title=title + f'. {n_clients} Clients', fig_name=f'M={n_clients}',
    #             #               params=params, is_show=is_show)
    #
    #             print(n_clients, fig_path)
    #     except Exception as e:
    #         print(f'Error: {e}, {in_file}')
    #
    #

    OUT_DIR = 'results-repeats_50'
    tot_cnt = 0
    for dataset in ['2GAUSSIANS']:  # ['FEMNIST', '2GAUSSIANS']:
        # dataset = 'FEMNIST'
        py_names = [
            'Centralized_Kmeans',
            'Stanford_server_random_initialization',
            'Stanford_client_initialization',
            'Our_greedy_initialization'
        ]
        if dataset == 'FEMNIST':
            cnt = 0
            for data_details in ['1client_1writer_multidigits', '1client_multiwriters_multidigits',
                                 '1client_multiwriters_1digit']:
                cnt_ = 0
                for py_name in py_names:
                    for case in gen_cases(py_name, dataset, data_details):
                        try:
                            # gen_sh(py_name, case)
                            params = get_experiment_params(p0=case['dataset'], p1=case['data_details'],
                                                           p2=case['algorithm'])
                            # print(params)
                            out_dir = os.path.join(OUT_DIR, params['out_dir'])
                            histories_file = os.path.join(out_dir, '~histories.dat')
                            histories = load(histories_file)
                            # save history as animation
                            n_clients = params['n_clients'] if 'Centralized' not in case['algorithm'] else 0
                            out_dir_i = os.path.join(out_dir, f'Clients_{n_clients}')
                            print(out_dir_i)
                            history = histories[n_clients]['history']
                            # fig_path = plot_metric_over_time(history, out_dir=f'{out_dir}/report',
                            #                                  title=title, fig_name=f'M={n_clients}')
                            # save2txt((in_file, n_clients, results_avg, history), out=fig_path + '.txt')

                            history2movie(history, out_dir=f'{out_dir_i}/over_time',
                                          title=f'{n_clients} Clients', fig_name=f'~M={n_clients}',
                                          params=params, is_show=True)
                        except Exception as e:
                            print(f'Error: {e}')

                        cnt_ += 1
                print(f'{py_name}: {cnt_} cases.\n')
                cnt += cnt_
        elif dataset == '2GAUSSIANS':
            cnt = 0
            for data_details in ['1client_1cluster', 'mix_clusters_per_client',
                                 '1client_ylt0', '1client_xlt0']:
                cnt_ = 0
                for py_name in py_names:
                    for case in gen_cases(py_name, dataset, data_details):
                        try:
                            # gen_sh(py_name, case)
                            params = get_experiment_params(p0=case['dataset'], p1=case['data_details'],
                                                           p2=case['algorithm'])
                            # print(params)
                            out_dir = os.path.join(OUT_DIR, params['out_dir'])
                            histories_file = os.path.join(out_dir, '~histories.dat')
                            histories = load(histories_file)
                            # save history as animation
                            n_clients = params['n_clients'] if 'Centralized' not in case['algorithm'] else 0
                            out_dir_i = os.path.join(out_dir, f'Clients_{n_clients}')
                            print(out_dir_i)
                            history = histories[n_clients]['history']
                            # fig_path = plot_metric_over_time(history, out_dir=f'{out_dir}/report',
                            #                                  title=title, fig_name=f'M={n_clients}')
                            # save2txt((in_file, n_clients, results_avg, history), out=fig_path + '.txt')

                            history2movie(history, out_dir=f'{out_dir_i}/over_time',
                                          title=f'{n_clients} Clients', fig_name=f'~M={n_clients}',
                                          params=params, is_show=True)
                        except Exception as e:
                            print(f'Error: {e}')
                        cnt_ += 1
                print(f'{py_name}: {cnt_} cases.\n')
                cnt += cnt_
        tot_cnt += cnt
        print(f'* {dataset} cases: {cnt}\n')
    print(f'*** Total cases: {tot_cnt}')


if __name__ == '__main__':
    main()
