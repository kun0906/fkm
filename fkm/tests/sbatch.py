"""
    run:
        module load anaconda3/2021.5
        cd /scratch/gpfs/ky8517/fkm/fkm
        PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 sbatch.py

"""
# Email: kun.bj@outlook.com

import os

from fkm.experiment_cases import get_experiment_params


def gen_sh(algorithm=None, dataset=None):
	"""

	Parameters
	----------
	py_name
	case

	Returns
	-------

	"""
	# check_arguments()
	dataset_name = dataset['name']
	dataset_detail = dataset['detail']
	n_clients = dataset['n_clients']
	algorithm_py_name = algorithm['py_name']
	algorithm_name = algorithm['name']
	n_clusters = algorithm['n_clusters']
	params = get_experiment_params(p0=dataset_name, p1=dataset_detail, p2=algorithm_name, p3=algorithm_py_name, n_clusters=n_clusters, n_clients = n_clients)
	# print(params)
	out_dir = params['out_dir']
	print(f'out_dir: {out_dir}')
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	job_name = f'{algorithm_py_name}-{dataset_name}-{dataset_detail}-{algorithm_name}'
	tmp_dir = '~tmp'
	if not os.path.exists(tmp_dir):
		os.system(f'mkdir {tmp_dir}')
	if '2GAUSSIANS' in dataset:
		t = 24
	elif 'FEMNIST' in dataset and 'greedy' in algorithm_py_name:
		t = 48
	else:
		t = 48
	content = fr"""#!/bin/bash
#SBATCH --job-name={job_name}\n{out_dir}         # create a short name for your job
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
pip3 install nltk 

cd /scratch/gpfs/ky8517/fkm/fkm 
pwd
python3 -V
    """
	content += '\n' + f"PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 {algorithm_py_name} --dataset '{dataset_name}' " \
	                  f"--data_details '{dataset_detail}' --algorithm '{algorithm_name}' --n_clusters '{n_clusters}' --n_clients '{n_clients}' \n"
	# not work with '&' running in background > {job_name}.txt 2>&1 &
	content += '\necho \'done\''
	sh_file = f'{tmp_dir}/{algorithm_py_name}-{dataset_name}-{dataset_detail}-{algorithm_name}-K_{n_clusters}-M_{n_clients}.sh'
	with open(sh_file, 'w') as f:
		f.write(content)
	cmd = f'sbatch {sh_file}'
	print(cmd)
	os.system(cmd)


#
# def gen_cases(algorithm, dataset):
#     cases = []
#
#     for algorithm in algorithms:
#         # if 'true' not in algorithm: continue
#         tmp = {'dataset': dataset, 'data_details': data_details, 'algorithm': algorithm}
#         cases.append(tmp)
#     return cases

def get_datasets_config_lst():
	datasets = []
	for dataset_name in ['3GAUSSIANS', '10GAUSSIANS', 'NBAIOT', 'SENT140','FEMNIST']: #['NBAIOT', 'SENT140','FEMNIST']:
	                     # [ '2GAUSSIANS','3GAUSSIANS' '5GAUSSIANS',  'NBAIOT', 'SENT140', 'FEMNIST']:
		cnt = 0
		if dataset_name == 'FEMNIST':
			# data_details_lst = [
			# 	'femnist_user_percent']  # ['1client_1writer_multidigits', '1client_multiwriters_multidigits',
			# # '1client_multiwriters_1digit']
			for data_detail, n_clusters, n_clients in [('femnist_user_percent', 62, 178)]:
				datasets.append({'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients':n_clients})
		elif dataset_name == 'NBAIOT':
			for data_detail, n_clusters, n_clients in [ ('nbaiot_user_percent', 2, 2), ('nbaiot_user_percent_client11', 2, 11), ('nbaiot_user_percent_client11', 11, 11)]: #[('nbaiot_user_percent', 2, 2), ('nbaiot11_user_percent', 11, 2)]:
				datasets.append({'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients':n_clients})
		elif dataset_name == 'SENT140':
			for data_detail, n_clusters, n_clients in [('sent140_user_percent', 2, 471)]:
				datasets.append({'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients':n_clients})
		elif dataset_name == '2GAUSSIANS':
			data_details_lst = [
				'1client_1cluster', 'mix_clusters_per_client',
				'1client_ylt0', '1client_xlt0',
				'1client_1cluster_diff_sigma', 'diff_sigma_n',
				'1client_xlt0_2',
			]
		elif dataset_name == '3GAUSSIANS':
			n_clusters = 3
			n_clients = 3
			# ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]] # each test set has 10%, 20%, 30% and 40% all data.
			ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
			# tot_cnt: 7 (ratios) * 5 (n1) * 2(sigma1) * 2 (n2) * 2 (sigma2) * 4 (alg) * 2 (per alg) = 2240 / 15 = 150 hrs = 6 days
			# 8 (ratios) * 5 (n1) * 1 (sigma1) * 2 (n2) * 1 (sigma2) * 4 (alg) * 2 (per alg) = 640 / 15 = 42 hrs
			data_details_lst = []
			# data_details_lst = [
			#     # '1client_1cluster',  this case is included in '0.0:mix_clusters_per_client'
			#     # '1client_ylt0', '1client_xlt0',
			#     # '1client_1cluster_diff_sigma',    this case is included in 'diff_sigma_n'
			#     # 'diff_sigma_n',
			#     # '1client_xlt0_2',
			# ] + [f'ratio_{ratio:.2f}:mix_clusters_per_client' for ratio in ratios] # ratio in [0, 1)

			"""
			Case 1 (Same N , but various P(ratios)
				G1 with mu1 = (-1, 0) and sigma1 = (0.1, 0.1)
				G2 with mu2 = (1, 0) and sigma2 = (0.1, 0.1)
				G3 with mu3 = (0, 3) and sigma3 = (1, 0.1)
				N1=N2=N3 = 10,000
				P = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]
			"""
			# n1_100-sigma1_0.1_0.1+n2_5000-sigma2_0.1_0.1+n3_10000-sigma3_1.0_0.1:ratio_0.1:diff_sigma_n
			# same sigma
			tmp_list = []
			N = 10000
			for ratio in ratios:
				for n1 in [10000]:
					# for n1 in [500, 2000, 3000, 5000, 8000]:
					for sigma1 in ["0.1_0.1"]:  # sigma  = [[0.1, 0], [0, 0.1]]
						for n2 in [N]:
							for sigma2 in ["0.1_0.1"]:  # sigma  = [[0.1, 0], [0, 0.1]]
								for n3 in [N]:
									for sigma3 in ["1.0_0.1"]:  # sigma  = [[1, 0], [0, 0.1]]
										p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
										tmp_list.append((p1, n_clusters, n_clients))

			data_details_lst += tmp_list

			"""
			Case 2 (Various N1 and P , but same σ)
				G1 with mu1 = (-1, 0) and sigma1 = (0.3, 0.3)
				G2 with mu2 = (1, 0) and sigma2 = (0.3, 0.3)
				G3 with mu3 = (0, 3) and sigma3 = (0.3, 0.3)
				sigma1 = sigma2 = sigma3 = 0.3
				N2 = 5000, N3 = 10,000, and N1 = [100, 500, 1000, 2000, 3000, 5000, 8000, 10000]
				P = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]
			"""
			# n1_100-sigma1_0.1+n2_5000-sigma2_0.2+n3_10000-sigma3_0.3:ratio_0.1:diff_sigma_n
			# same sigma
			tmp_list = []
			N = 10000
			for ratio in ratios:
				for n1 in [100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
					# for n1 in [500, 2000, 3000, 5000, 8000]:
					for sigma1 in ["0.3_0.3"]:
						for n2 in [5000]:
							for sigma2 in ["0.3_0.3"]:
								for n3 in [N]:
									for sigma3 in ["0.3_0.3"]:
										p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
										tmp_list.append((p1, n_clusters, n_clients))
			data_details_lst += tmp_list

			"""
			Case 3 (Various N1 and P , but different σ)
				G1 with mu1 = (-1, 0) and sigma1 = (0.1, 0.1)
				G2 with mu2 = (1, 0) and sigma2 = (0.2, 0.2)
				G3 with mu3 = (0, 3) and sigma3 = (0.3, 0.3)
				N2 = 5000, N3 = 10,000, and N1 = [100, 500, 1000, 2000, 3000, 5000, 8000, 10000]
				P = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]
			"""
			# n1_100-sigma1_0.1+n2_5000-sigma2_0.2+n3_10000-sigma3_0.3:ratio_0.1:diff_sigma_n
			tmp_list = []
			# N = 10000
			for ratio in ratios:
				for n1 in [100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
					# for n1 in [500, 2000, 3000, 5000, 8000]:
					for sigma1 in ["0.1_0.1"]:
						for n2 in [5000]:
							for sigma2 in ["0.2_0.2"]:
								for n3 in [N]:
									for sigma3 in ["0.3_0.3"]:
										p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
										tmp_list.append((p1, n_clusters, n_clients))

			data_details_lst += tmp_list
			# # different sigmas
			# tmp_list = []
			# N = 10000
			# for ratio in ratios:
			#     for n1 in [100, 1000, 5000, N]:
			#         # for n1 in [500, 2000, 3000, 5000, 8000]:
			#         for sigma1 in [0.1]:
			#             for n2 in [5000]:
			#                 for sigma2 in [0.2]:
			#                     for n3 in [N]:
			#                         for sigma3 in [0.3]:
			#                             p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
			#                             tmp_list.append(p1)

			# data_details_lst += tmp_list
			for data_detail, n_clusters, n_clients in data_details_lst:
				datasets.append({'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})

		elif dataset_name == '10GAUSSIANS':
			n_clusters = 10
			n_clients = 10
			# ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]] # each test set has 10%, 20%, 30% and 40% all data.
			ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
			# tot_cnt: 7 (ratios) * 5 (n1) * 2(sigma1) * 2 (n2) * 2 (sigma2) * 4 (alg) * 2 (per alg) = 2240 / 15 = 150 hrs = 6 days
			# 8 (ratios) * 5 (n1) * 1 (sigma1) * 2 (n2) * 1 (sigma2) * 4 (alg) * 2 (per alg) = 640 / 15 = 42 hrs
			data_details_lst = []
			# data_details_lst = [
			#     # '1client_1cluster',  this case is included in '0.0:mix_clusters_per_client'
			#     # '1client_ylt0', '1client_xlt0',
			#     # '1client_1cluster_diff_sigma',    this case is included in 'diff_sigma_n'
			#     # 'diff_sigma_n',
			#     # '1client_xlt0_2',
			# ] + [f'ratio_{ratio:.2f}:mix_clusters_per_client' for ratio in ratios] # ratio in [0, 1)

			"""
			Case 1 (Same N , but various P(ratios)
				G1 with mu1 = (-1, 0) and sigma1 = (0.1, 0.1)
				G2 with mu2 = (1, 0) and sigma2 = (0.1, 0.1)
				G3 with mu3 = (0, 3) and sigma3 = (1, 0.1)
				N1=N2=N3 = 10,000
				P = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]
			"""
			# n1_100-sigma1_0.1_0.1+n2_5000-sigma2_0.1_0.1+n3_10000-sigma3_1.0_0.1:ratio_0.1:diff_sigma_n
			# same sigma
			tmp_list = []
			N = 5000
			for ratio in ratios:
				for n1 in [5000]:
					# for n1 in [500, 2000, 3000, 5000, 8000]:
					for sigma1 in ["0.3_0.3"]:  # sigma  = [[0.1, 0], [0, 0.1]]
						for n2 in [1000]:
							for sigma2 in ["0.3_0.3"]:  # sigma  = [[0.1, 0], [0, 0.1]]
								for n3 in [500]:
									for sigma3 in ["0.3_0.3"]:  # sigma  = [[1, 0], [0, 0.1]]
										p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
										tmp_list.append((p1, n_clusters, n_clients))

			data_details_lst += tmp_list
			for data_detail, n_clusters, n_clients in data_details_lst:
				datasets.append({'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients':n_clients})
		elif dataset_name == '3GAUSSIANS-ADVERSARIAL':
			n_clients = 3
			n_clusters = 3
			# ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]] # each test set has 10%, 20%, 30% and 40% all data.
			# ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
			# tot_cnt: 7 (ratios) * 5 (n1) * 2(sigma1) * 2 (n2) * 2 (sigma2) * 4 (alg) * 2 (per alg) = 2240 / 15 = 150 hrs = 6 days
			# 8 (ratios) * 5 (n1) * 1 (sigma1) * 2 (n2) * 1 (sigma2) * 4 (alg) * 2 (per alg) = 640 / 15 = 42 hrs
			data_details_lst = []
			ratios = [0, 0.01]

			"""
			Case 1 (Same N , but various P(ratios)
				G1 with mu1 = (-1, 0) and sigma1 = (0.1, 0.1)
				G2 with mu2 = (1, 0) and sigma2 = (0.1, 0.1)
				G3 with mu3 = (0, 3) and sigma3 = (1, 0.1)
				N1=N2=N3 = 10,000
				P = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]
			"""
			# n1_100-sigma1_0.1_0.1+n2_5000-sigma2_0.1_0.1+n3_10000-sigma3_1.0_0.1:ratio_0.1:diff_sigma_n
			# same sigma
			tmp_list = []
			N = 5000
			# n4s = [10, 100, 200, 300, 500, 1000, 2000, 5000]
			n4s = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 10]  # list(range(0, 10, 1)) * 0.1
			for n4 in n4s:
				for sigma4 in ["0.3_0.3", "0.5_0.5"]:
					for ratio in ratios:
						for n1 in [1000, 5000]:
							# for n1 in [500, 2000, 3000, 5000, 8000]:
							for sigma1 in ["0.3_0.3"]:  # sigma  = [[0.1, 0], [0, 0.1]]
								for n2 in [5000]:
									for sigma2 in ["0.3_0.3"]:  # sigma  = [[0.1, 0], [0, 0.1]]
										for n3 in [10000]:
											for sigma3 in ["0.3_0.3"]:  # sigma  = [[1, 0], [0, 0.1]]
												p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}+n4_{n4}-sigma4_{sigma4}:ratio_{ratio:.2f}:diff_sigma_n'
												tmp_list.append((p1, n_clusters, n_clients))

			data_details_lst += tmp_list
			for data_detail, n_clusters in data_details_lst:
				datasets.append({'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients':n_clients})
		elif dataset_name == '5GAUSSIANS':
			data_details_lst = [
				'5clients_5clusters', '5clients_4clusters', '5clients_3clusters',
			]
		else:
			msg = f'{dataset_name}'
			raise NotImplementedError(msg)

	return datasets


def get_algorithms_config_lst(n_clusters=2):
	algorithms = []
	for py_name in [
		'centralized_kmeans.py',
		'federated_server_init_first.py',  # server first: min-max per each dimension
		'federated_client_init_first.py',  # client initialization first : server average
		'federated_greedy_kmeans.py',  # client initialization first: greedy: server average
		# 'Our_greedy_center',
		# 'Our_greedy_2K',
		# 'Our_greedy_K_K',
		# 'Our_greedy_concat_Ks',
		# 'Our_weighted_kmeans_initialization',
	]:
		cnt = 0
		if py_name == 'centralized_kmeans.py':
			for name in [  # 'Centralized_true',
				# 'Centralized_random',
				'Centralized_kmeans++',
			]:
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters})

		elif py_name == 'federated_server_init_first.py':
			# p0, p1, p2
			for name in [
				# 'Federated-Server_true',  # use true centroids as initial centroids.
				# 'Federated-Server_random',
				'Federated-Server_random_min_max',
				# 'Federated-Server_gaussian',
			]:
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters})
		elif py_name == 'federated_client_init_first.py':
			for name in [
				# 'Federated-Server_average-Client_true',
				'Federated-Server_average-Client_random',
				'Federated-Server_average-Client_kmeans++',
			]:
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters})
		elif py_name == 'federated_greedy_kmeans.py':
			for name in [
				# 'Federated-Server_greedy-Client_true',
				'Federated-Server_greedy-Client_random',
				'Federated-Server_greedy-Client_kmeans++',
			]:
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters})
		# elif py_name == 'Our_greedy_center':
		#     algorithms = [
		#         # 'Federated-Server_greedy-Client_true',
		#         'Federated-Server_greedy-Client_random',
		#         'Federated-Server_greedy-Client_kmeans++',
		#     ]
		# elif py_name == 'Our_greedy_2K':
		#     algorithms = [
		#         # 'Federated-Server_greedy-Client_true',
		#         'Federated-Server_greedy-Client_random',
		#         'Federated-Server_greedy-Client_kmeans++',
		#     ]
		# elif py_name == 'Our_greedy_K_K':
		#     algorithms = [
		#         # 'Federated-Server_greedy-Client_true',
		#         'Federated-Server_greedy-Client_random',
		#         'Federated-Server_greedy-Client_kmeans++',
		#     ]
		# elif py_name == 'Our_greedy_concat_Ks':
		#     algorithms = [
		#         # 'Federated-Server_greedy-Client_true',
		#         'Federated-Server_greedy-Client_random',
		#         'Federated-Server_greedy-Client_kmeans++',
		#     ]
		# elif py_name == 'Our_weighted_kmeans_initialization':
		#     algorithms = [
		#         # 'Federated-Server_greedy-Client_true',
		#         'Federated-Server_greedy-Client_random',
		#         'Federated-Server_greedy-Client_kmeans++',
		#     ]
		else:
			msg = py_name
			raise NotImplementedError(msg)

	return algorithms


def main():
	tot_cnt = 0
	datasets = get_datasets_config_lst()
	for dataset in datasets:
		cnt_ = 0
		algorithms = get_algorithms_config_lst(dataset['n_clusters'])
		for algorithm in algorithms:
			gen_sh(algorithm, dataset)
			cnt_ += 1
			print(f'{algorithm}: {cnt_} cases.\n')
		tot_cnt += cnt_
	print(f'*** Total cases: {tot_cnt}')


if __name__ == '__main__':
	main()
