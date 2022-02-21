"""
    run:
        module load anaconda3/2021.5
        cd /scratch/gpfs/ky8517/fkm/fkm
        PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 sbatch.py

"""
# Email: kun.bj@outlook.com

import os

from fkm.experiment_cases import get_experiment_params


def gen_sh(cases=None):
    """

    Parameters
    ----------
    py_name
    case

    Returns
    -------

    """
    # check_arguments()
    n_tasks = len(cases)
    case = cases[0]
    dataset = case['dataset']
    data_details = case['data_details']
    py_name = case['py_name']

    job_name = f'{py_name}-{dataset}-{data_details}'
    tmp_dir = '~tmp'
    if not os.path.exists(tmp_dir):
        os.system(f'mkdir {tmp_dir}')
    if 'FEMNIST' in dataset:
        t = 24
    else:
        t = 24
    content = fr"""#!/bin/bash
#SBATCH --job-name={job_name}         # create a short name for your job
#SBATCH --nodes=1                # node count
# # SBATCH --ntasks={n_tasks}       # total number of tasks across all nodes
#SBATCH --ntasks-per-node={n_tasks}   # Number of task (cores/ppn) per node
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time={t}:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --output={tmp_dir}/%j-{job_name}-out.txt
#SBATCH --error={tmp_dir}/%j-{job_name}-err.txt
# #SBATCH --mail-user=kun.bj@cloud.com # not work
#SBATCH --mail-user=ky8517@princeton.edu

module purge
module load anaconda3/2021.5

cd /scratch/gpfs/ky8517/fkm/fkm
pwd
python3 -V
    """
    for case in cases:
        py_name = case['py_name']
        dataset = case['dataset']
        data_details = case['data_details']
        algorithm = case['algorithm']

        params = get_experiment_params(p0=dataset, p1=data_details, p2=algorithm, p3=py_name)
        # print(params)
        out_dir = params['out_dir']
        print(f'out_dir: {out_dir}')
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        content += f"\nPYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 {py_name}.py --dataset '{dataset}' " \
                          f"--data_details '{data_details}' --algorithm '{algorithm}' > {out_dir}/out.txt 2>&1 & \n"
        # not work with '&' running in background > {job_name}.txt 2>&1 &
    # must append wait to the end!
    # The wait command at the end is important so the shell waits until the background processes are complete
    # (otherwise the batch job will exit right away).
    content +='\nwait'
    content += '\necho \'done\''

    sh_file = f'{tmp_dir}/{py_name}-{dataset}-{data_details}.sh'
    with open(sh_file, 'w') as f:
        f.write(content)
    cmd = f'sbatch {sh_file}'
    print(cmd)
    os.system(cmd)


def gen_cases(py_name, dataset='FEMNIST', data_details='1client_1writer_multidigits'):
    cases = []
    if py_name == 'Stanford_server_random_initialization':
        # p0, p1, p2
        algorithms = [
            'Federated-Server_true',  # use true centroids as initial centroids.
            'Federated-Server_random',
        ]
    elif py_name == 'Stanford_client_initialization':
        algorithms = [
            'Federated-Server_average-Client_true',
            'Federated-Server_average-Client_random',
            'Federated-Server_average-Client_kmeans++',
        ]
    elif py_name == 'Our_greedy_initialization':
        algorithms = [
            'Federated-Server_greedy-Client_true',
            'Federated-Server_greedy-Client_random',
            'Federated-Server_greedy-Client_kmeans++',
        ]
    elif py_name == 'Our_greedy_center':
        algorithms = [
            'Federated-Server_greedy-Client_true',
            'Federated-Server_greedy-Client_random',
            'Federated-Server_greedy-Client_kmeans++',
        ]
    elif py_name == 'Our_greedy_2K':
        algorithms = [
            'Federated-Server_greedy-Client_true',
            'Federated-Server_greedy-Client_random',
            'Federated-Server_greedy-Client_kmeans++',
        ]
    elif py_name == 'Our_greedy_K_K':
        algorithms = [
            'Federated-Server_greedy-Client_true',
            'Federated-Server_greedy-Client_random',
            'Federated-Server_greedy-Client_kmeans++',
        ]
    elif py_name == 'Centralized_Kmeans':
        algorithms = ['Centralized_true',
                      'Centralized_random',
                      'Centralized_kmeans++',
                      ]
    else:
        msg = py_name
        raise NotImplementedError(msg)
    for algorithm in algorithms:
        # if 'true' not in algorithm: continue
        tmp = {'py_name':py_name, 'dataset': dataset, 'data_details': data_details, 'algorithm': algorithm}
        cases.append(tmp)

    return cases


def main():
    tot_cnt = 0
    for dataset in ['3GAUSSIANS']:  # [ '2GAUSSIANS', '5GAUSSIANS', 'FEMNIST']:
        # dataset = 'FEMNIST'
        py_names = [
            'Centralized_Kmeans',
            'Stanford_server_random_initialization',
            'Stanford_client_initialization',
            'Our_greedy_initialization',
            'Our_greedy_center',
            'Our_greedy_2K',
            'Our_greedy_K_K',
        ]
        cnt = 0
        if dataset == 'FEMNIST':
            data_details_lst = ['1client_1writer_multidigits', '1client_multiwriters_multidigits',
                             '1client_multiwriters_1digit']
        elif dataset == '2GAUSSIANS':
            data_details_lst = [
                # '1client_1cluster', 'mix_clusters_per_client',
                # '1client_ylt0', '1client_xlt0',
                # '1client_1cluster_diff_sigma', 'diff_sigma_n',
                '1client_xlt0_2',
            ]
        elif dataset == '3GAUSSIANS':
            data_details_lst = [
                '1client_1cluster', 'mix_clusters_per_client',
                '1client_ylt0', '1client_xlt0',
                '1client_1cluster_diff_sigma', 'diff_sigma_n',
                '1client_xlt0_2',
            ]
        elif dataset == '5GAUSSIANS':
            data_details_lst = [
                    '5clients_5clusters', '5clients_4clusters', '5clients_3clusters',
                ]
        else:
            msg = f'{dataset}'
            raise NotImplementedError(msg)
        for data_details in data_details_lst:
            cnt_ = 0
            for py_name in py_names:
                cases = gen_cases(py_name, dataset, data_details)
                # generate sbatches
                gen_sh(cases)
                cnt_ = len(cases)
                print(f'{py_name}: {cnt_} cases.\n')
            cnt += cnt_
        tot_cnt += cnt
        print(f'* {dataset} cases: {cnt}\n')
    print(f'*** Total cases: {tot_cnt}')


if __name__ == '__main__':
    main()
