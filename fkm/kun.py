"""
    run:
        module load anaconda3/2021.5
        cd /scratch/gpfs/ky8517/fkm/fkm
        PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 kun.py

"""
# Email: kun.bj@outlook.com

import os

from fkm.experiment_cases import get_experiment_params


def gen_sh(py_name='', case=None):
    """

    Parameters
    ----------
    py_name
    case

    Returns
    -------

    """
    # check_arguments()
    dataset = case['dataset']
    data_details = case['data_details']
    algorithm = case['algorithm']
    params = get_experiment_params(p0=dataset, p1=data_details, p2=algorithm, p3 = py_name)
    # print(params)
    out_dir = params['out_dir']
    print(f'out_dir: {out_dir}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    job_name = f'{py_name}-{dataset}-{data_details}-{algorithm}'
    tmp_dir = '~tmp'
    if not os.path.exists(tmp_dir):
        os.system(f'mkdir {tmp_dir}')
    if '2GAUSSIANS' in dataset:
        t = 24
    elif 'FEMNIST' in dataset and 'greedy' in py_name:
        t = 24
    else:
        t = 24
    content = fr"""#!/bin/bash
#SBATCH --job-name={job_name}-{out_dir}         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=5        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time={t}:00:00          # total run time limit (HH:MM:SS)
# # SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
## SBATCH --output={tmp_dir}/%j-{job_name}-out.txt
## SBATCH --error={tmp_dir}/%j-{job_name}-err.txt
#SBATCH --output={out_dir}/out.txt
#SBATCH --error={out_dir}/err.txt
# #SBATCH --mail-user=kun.bj@cloud.com # not work \
# #SBATCH --mail-user=<YourNetID>@princeton.edu
#SBATCH --mail-user=ky8517@princeton.edu

module purge
module load anaconda3/2021.5

cd /scratch/gpfs/ky8517/fkm/fkm
pwd
python3 -V
    """
    content += '\n' + f"PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 {py_name}.py --dataset '{dataset}' " \
                      f"--data_details '{data_details}' --algorithm '{algorithm}' \n"
    # not work with '&' running in background > {job_name}.txt 2>&1 &
    content += '\necho \'done\''
    sh_file = f'{tmp_dir}/{py_name}-{dataset}-{data_details}-{algorithm}.sh'
    with open(sh_file, 'w') as f:
        f.write(content)
    cmd = f'sbatch {sh_file}'
    print(cmd)
    os.system(cmd)


def gen_cases(py_name, dataset='FEMNIST', data_details='1client_1writer_multidigits'):
    cases = []
    if py_name == 'Stanford_random_initialization':
        # p0, p1, p2
        algorithms = [
            'Federated-Server_true',  # use true centroids as initial centroids.
            'Federated-Server_random',
        ]
    elif py_name == 'Stanford_average_initialization':
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
    elif py_name == 'Our_greedy_concat_Ks':
        algorithms = [
            'Federated-Server_greedy-Client_true',
            'Federated-Server_greedy-Client_random',
            'Federated-Server_greedy-Client_kmeans++',
        ]
    elif py_name == 'Our_weighted_kmeans_initialization':
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
        tmp = {'dataset': dataset, 'data_details': data_details, 'algorithm': algorithm}
        cases.append(tmp)
    return cases


def main():
    tot_cnt = 0
    for dataset in ['3GAUSSIANS', '2GAUSSIANS', '5GAUSSIANS' ]:  # [ '2GAUSSIANS', '5GAUSSIANS', 'FEMNIST']:
        # dataset = 'FEMNIST'
        py_names = [
            # 'Centralized_Kmeans',
            # 'Stanford_random_initialization',
            # 'Stanford_average_initialization',
            # 'Our_greedy_initialization',
            # 'Our_greedy_center',
            # 'Our_greedy_2K',
            # 'Our_greedy_K_K',
            # 'Our_greedy_concat_Ks',
            'Our_weighted_kmeans_initialization',
        ]
        cnt = 0
        if dataset == 'FEMNIST':
            data_details_lst = ['1client_1writer_multidigits', '1client_multiwriters_multidigits',
                             '1client_multiwriters_1digit']
        elif dataset == '2GAUSSIANS':
            data_details_lst = [
                '1client_1cluster', '1client_0.7cluster1_0.3cluster2',
                '1client_ylt0', '1client_xlt0',
                '1client_1cluster_diff_sigma', '1client_1cluster_diff_sigma_n',
                '1client_xlt0_2',
            ]
        elif dataset == '3GAUSSIANS':
            data_details_lst = [
                '1client_1cluster', '1client_0.7cluster1_0.3cluster2',
                '1client_ylt0', '1client_xlt0',
                '1client_1cluster_diff_sigma', '1client_1cluster_diff_sigma_n',
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
                for case in gen_cases(py_name, dataset, data_details):
                    gen_sh(py_name, case)
                    cnt_ += 1
                print(f'{py_name}: {cnt_} cases.\n')
            cnt += cnt_
        tot_cnt += cnt
        print(f'* {dataset} cases: {cnt}\n')
    print(f'*** Total cases: {tot_cnt}')


if __name__ == '__main__':
    main()
