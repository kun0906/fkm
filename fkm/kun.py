"""
    run:
        module load anaconda3/2021.5
        cd /scratch/gpfs/ky8517/fkm/fkm
        PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 kun.py

"""
import os

from fkm.experiment_cases import get_experiment_params


def gen_sh(py_name='', case={}):
    dataset = case['dataset']
    data_details = case['data_details']
    algorithm = case['algorithm']
    params = get_experiment_params(p0=dataset, p1=data_details, p2=algorithm)
    # pprint(params)
    out_dir = params['out_dir']
    print(f'out_dir: {out_dir}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    job_name = f'{py_name}-{dataset}-{data_details}-{algorithm}'
    tmp_dir = '~tmp'
    if not os.path.exists(tmp_dir):
        os.system(f'mkdir {tmp_dir}')
    if '2GAUSSIANS' in dataset:
        t = 3
    elif 'FEMNIST' in dataset and 'greedy' in py_name:
        t = 24
    else:
        t = 12
    content = fr"""#!/bin/bash
#SBATCH --job-name={job_name}-{out_dir}         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time={t}:00:00          # total run time limit (HH:MM:SS)
# # SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --output={tmp_dir}/%j-{job_name}-out.txt
#SBATCH --error={tmp_dir}/%j-{job_name}-err.txt
#SBATCH --mail-user=ky8517@princeton.edu

module purge
module load anaconda3/2021.5

cd /scratch/gpfs/ky8517/fkm/fkm
pwd
python3 -V
    """
    content += '\n' + f"PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 {py_name}.py --dataset '{dataset}' " \
                      f"--data_details '{data_details}' --algorithm '{algorithm}' > {out_dir}/out.txt 2>&1 \n"
    # not work with redirect > {job_name}.txt 2>&1 &
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
            'Federated-Server_random',
            'Federated-Server_true',    # use true centroids as initial centroids.
        ]
    elif py_name == 'Stanford_average_initialization':
        algorithms = [
            'Federated-Server_average-Client_random',
            'Federated-Server_average-Client_kmeans++',
            'Federated-Server_average-Client_true',
        ]
    elif py_name == 'Our_greedy_initialization':
        algorithms = [
            'Federated-Server_greedy-Client_random',
            'Federated-Server_greedy-Client_kmeans++',
            'Federated-Server_greedy-Client_true'
        ]
    else:  # Centralized Kmeans
        algorithms = ['Centralized_random',
                      'Centralized_kmeans++',
                      'Centralized_true',
                      ]

    for algorithm in algorithms:
        if 'true' not in algorithm: continue
        tmp = {'dataset': dataset, 'data_details': data_details, 'algorithm': algorithm}
        cases.append(tmp)
    return cases


if __name__ == '__main__':

    # experiments = {
    #     'Centralized_Kmeans': gen_cases(dataset='FEMNIST', data_details='1client_1writer_multidigits'),
    #     'Centralized_Kmeans': gen_cases(dataset='FEMNIST', data_details='1client_multiwriters_multidigits'),
    #     'Centralized_Kmeans': gen_cases(dataset='FEMNIST', data_details='1client_multiwriters_1digit'),
    #
    #     'Stanford_random_initialization':gen_cases(dataset='FEMNIST', data_details='1client_1writer_multidigits'),
    #     'Stanford_random_initialization':gen_cases(dataset='FEMNIST', data_details='1client_multiwriters_multidigits'),
    #     'Stanford_random_initialization':gen_cases(dataset='FEMNIST', data_details='1client_multiwriters_1digit'),
    #
    #     'Stanford_average_initialization': gen_cases(dataset='FEMNIST', data_details='1client_1writer_multidigits'),
    #     'Stanford_average_initialization': gen_cases(dataset='FEMNIST', data_details='1client_multiwriters_multidigits'),
    #     'Stanford_average_initialization': gen_cases(dataset='FEMNIST', data_details='1client_multiwriters_1digit'),
    #
    #     'Our_greedy_initialization': gen_cases(dataset='FEMNIST', data_details='1client_1writer_multidigits'),
    #     'Our_greedy_initialization': gen_cases(dataset='FEMNIST',data_details='1client_multiwriters_multidigits'),
    #     'Our_greedy_initialization': gen_cases(dataset='FEMNIST', data_details='1client_multiwriters_1digit'),
    #
    #     # 'Our_greedy_initialization': [ '20', '21'],
    #     # 'Stanford_random_initialization': ['00', '10', '20'],
    # }
    tot_cnt = 0
    for dataset in ['2GAUSSIANS']:  # [ '2GAUSSIANS', 'FEMNIST']:
        # dataset = 'FEMNIST'
        py_names = [
            'Centralized_Kmeans',
            'Stanford_random_initialization',
            'Stanford_average_initialization',
            'Our_greedy_initialization'
        ]
        if dataset == 'FEMNIST':
            cnt = 0
            for data_details in ['1client_1writer_multidigits', '1client_multiwriters_multidigits',
                                 '1client_multiwriters_1digit']:
                cnt_ = 0
                for py_name in py_names:
                    for case in gen_cases(py_name, dataset, data_details):
                        gen_sh(py_name, case)
                        cnt_ += 1
                print(f'{py_name}: {cnt_} cases.\n')
                cnt += cnt_
        elif dataset == '2GAUSSIANS':
            cnt = 0
            for data_details in ['1client_1cluster', '1client_0.7cluster1_0.3cluster2',
                                 '1client_ylt0', '1client_xlt0',
                                 '1client_1cluster_diff_sigma', '1client_1cluster_diff_sigma_n'
                                 ]:
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
