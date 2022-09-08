""" Run this main file for all experiments

#!/usr/bin/env bash

: ' multiline comment
  running the shell with source will change the current bash path
  e.g., source stat.sh

  check cuda and cudnn version for tensorflow_gpu==1.13.1
  https://www.tensorflow.org/install/source#linux
'
#ssh ky8517@tigergpu.princeton.edu
srun --nodes=1  --mem=128G --ntasks-per-node=1 --time=20:00:00 --pty bash -i
#srun --nodes=1 --gres=gpu:1 --mem=128G --ntasks-per-node=1 --time=20:00:00 --pty bash -i
cd /scratch/gpfs/ky8517/fkm/fkm
module load anaconda3/2021.11

"""
# Email: kun.bj@outllok.com
import copy
import os
import shutil
from pprint import pprint

from fkm import config, main_single
from fkm.datasets.dataset import generate_dataset


def gen_sh(args):
	"""

	Parameters
	----------
	py_name
	case

	Returns
	-------

	"""

	# check_arguments()
	dataset_name = args['DATASET']['name']
	dataset_detail = args['DATASET']['detail']
	# n_clients = args['N_CLIENTS']
	algorithm_py_name = args['ALGORITHM']['py_name']
	# algorithm_name = args['ALGORITHM']['name']
	algorithm_detail = args['ALGORITHM']['detail']
	# n_clusters = args['ALGORITHM']['n_clusters']
	config_file = args['config_file']
	OUT_DIR = args['OUT_DIR']

	job_name = f'{dataset_name}-{dataset_detail}-{algorithm_py_name}-{algorithm_detail}'
	# tmp_dir = '~tmp'
	# if not os.path.exists(tmp_dir):
	# 	os.system(f'mkdir {tmp_dir}')
	if '2GAUSSIANS' in dataset_name:
		t = 24
	elif 'FEMNIST' in dataset_name and 'greedy' in algorithm_py_name:
		t = 48
	else:
		t = 48
	content = fr"""#!/bin/bash
#SBATCH --job-name={OUT_DIR}         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=5        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time={t}:00:00          # total run time limit (HH:MM:SS)
## SBATCH --output={OUT_DIR}/%j-{job_name}-out.txt
## SBATCH --error={OUT_DIR}/%j-{job_name}-err.txt
#SBATCH --output={OUT_DIR}/out.txt
#SBATCH --error={OUT_DIR}/err.txt

### SBATCH --mail-type=begin        # send email when job begins
### SBATCH --mail-type=end          # send email when job ends\
### SBATCH --mail-user=kun.bj@cloud.com # not work \
### SBATCH --mail-user=<YourNetID>@princeton.edu
### SBATCH --mail-user=ky8517@princeton.edu     # which will cause too much email notification. 

module purge
module load anaconda3/2021.11   # Python 3.9.7
pip3 install nltk 

cd /scratch/gpfs/ky8517/fkm/fkm 
pwd
python3 -V
    """

	# content += '\n' + f"PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 {algorithm_py_name} --dataset '{dataset_name}' " \
	#                   f"--data_details '{dataset_detail}' --algorithm '{algorithm_name}' --n_clusters '{n_clusters}' --n_clients '{n_clients}' \n"
	content += '\n' + f"PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_single.py --config_file '{config_file}'\n"

	# not work with '&' running in background > {job_name}.txt 2>&1 &
	content += '\necho \'done\''
	# sh_file = f'{OUT_DIR}/{dataset_name}-{dataset_detail}-{algorithm_name}-{algorithm_detail}.sh'
	sh_file = f'{OUT_DIR}/sbatch.sh'
	with open(sh_file, 'w') as f:
		f.write(content)
	cmd = f"sbatch '{sh_file}'"
	print(cmd)
	os.system(cmd)


def get_datasets_config_lst(dataset_names=['3GAUSSIANS', '10GAUSSIANS', 'NBAIOT', 'SENT140', 'FEMNIST']):
	datasets = []
	for dataset_name in dataset_names:  # ['NBAIOT', 'SENT140','FEMNIST']:
		# [ '2GAUSSIANS','3GAUSSIANS' '5GAUSSIANS',  'NBAIOT', 'SENT140', 'FEMNIST']:
		cnt = 0
		if dataset_name == 'FEMNIST':
			# data_details_lst = [
			# 	'femnist_user_percent']  # ['1client_1writer_multidigits', '1client_multiwriters_multidigits',
			# # '1client_multiwriters_1digit']
			# for data_detail, n_clusters, n_clients in [('femnist_user_percent', 62, 178)]:
			# 	datasets.append(
			# 		{'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})

			data_details_lst = []
			n_clients = 10
			n_clusters = 2
			tmp_list = []
			for n in [1, 3, 5, 10, 15, 17]:
				# there are 20 clients and each client has n users' data.
				p1 = f'n_{n}:ratio_0.00:diff_sigma_n'
				tmp_list.append((p1, n_clusters, n_clients))
			data_details_lst += tmp_list

			for data_detail, n_clusters, n_clients in data_details_lst:
				datasets.append(
					{'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})

		elif dataset_name == 'MNIST':
			data_details_lst = []
			n_clients = 10
			n_clusters = 10
			tmp_list = []
			for n in [50, 100, 500, 1000, 3000, 5000]:
				p1 = f'n_{n}:ratio_0.00:diff_sigma_n'
				tmp_list.append((p1, n_clusters, n_clients))
			data_details_lst += tmp_list

			for data_detail, n_clusters, n_clients in data_details_lst:
				datasets.append(
					{'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})

		elif dataset_name == 'GASSENSOR':
			data_details_lst = []
			n_clients = 6
			n_clusters = 6
			tmp_list = []
			for n in [50, 100, 500, 800, 1000, 1500]:
				# there are 20 clients and each client has n users' data.
				p1 = f'n_{n}:ratio_0.00:diff_sigma_n'
				tmp_list.append((p1, n_clusters, n_clients))
			data_details_lst += tmp_list

			for data_detail, n_clusters, n_clients in data_details_lst:
				datasets.append(
					{'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})

		elif dataset_name == 'CHARFONT':
			data_details_lst = []
			n_clients = 17
			n_clusters = 17
			tmp_list = []
			for n in [50, 100, 500, 800, 1000, 1500, 3000]:
				# there are 20 clients and each client has n users' data.
				p1 = f'n_{n}:ratio_0.00:diff_sigma_n'
				tmp_list.append((p1, n_clusters, n_clients))
			data_details_lst += tmp_list

			for data_detail, n_clusters, n_clients in data_details_lst:
				datasets.append(
					{'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})

		elif dataset_name == 'DRYBEAN':
			data_details_lst = []
			n_clients = 6
			n_clusters = 6
			tmp_list = []
			for n in [50, 100, 500, 800, 1000]:
				# there are 20 clients and each client has n users' data.
				p1 = f'n_{n}:ratio_0.00:diff_sigma_n'
				tmp_list.append((p1, n_clusters, n_clients))
			data_details_lst += tmp_list

			for data_detail, n_clusters, n_clients in data_details_lst:
				datasets.append(
					{'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})

		elif dataset_name == 'BITCOIN':
			data_details_lst = []
			n_clients = 4
			n_clusters = 4
			tmp_list = []
			for n in [50, 500, 1000, 2000, 3000, 5000]:
				# there are 20 clients and each client has n users' data.
				p1 = f'n_{n}:ratio_0.00:diff_sigma_n'
				tmp_list.append((p1, n_clusters, n_clients))
			data_details_lst += tmp_list

			for data_detail, n_clusters, n_clients in data_details_lst:
				datasets.append(
					{'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})

		elif dataset_name == 'SELFBACK':
			data_details_lst = []
			n_clients = 9
			n_clusters = 9
			tmp_list = []
			for n in [50, 100, 500, 800, 1000, 1500, 2000]:
				# there are 20 clients and each client has n users' data.
				p1 = f'n_{n}:ratio_0.00:diff_sigma_n'
				tmp_list.append((p1, n_clusters, n_clients))
			data_details_lst += tmp_list

			for data_detail, n_clusters, n_clients in data_details_lst:
				datasets.append(
					{'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})


		elif dataset_name == 'NBAIOT':
			# for data_detail, n_clusters, n_clients in [ # ('nbaiot_user_percent', 2, 2),
			#                                            # ('nbaiot_user_percent_client11', 2, 11),
			# 											('nbaiot_user_percent_client11', 11,
			# 		                                           11)]:  # [('nbaiot_user_percent', 2, 2), ('nbaiot11_user_percent', 11, 2)]:
			# 	datasets.append(
			# 		{'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})
			data_details_lst = []
			# tmp_list = []
			# ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
			# N = 5000
			# n_clusters = 2
			# n_clients = 3
			# for ratio in ratios:
			# 	for n1 in [N]:
			# 		for n2 in [2500]:
			# 			for n3 in [2500]:
			# 				p1 = f'n1_{n1}+n2_{n2}+n3_{n3}:ratio_{ratio:.2f}:diff_sigma_n'
			# 				tmp_list.append((p1, n_clusters, n_clients))
			# data_details_lst += tmp_list

			# tmp_list = []
			# ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
			# N = 5000
			# n_clusters = 3
			# n_clients = 3
			# for ratio in ratios:
			# 	for n1 in [N]:
			# 		for n2 in [2500]:
			# 			for n3 in [2500]:
			# 				p1 = f'n1_{n1}+n2_{n2}+n3_{n3}:ratio_{ratio:.2f}:diff_sigma_n'
			# 				tmp_list.append((p1, n_clusters, n_clients))
			# data_details_lst += tmp_list

			# tmp_list = []
			# ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
			# N = 5000
			# n_clusters = 2
			# n_clients = 2
			# for ratio in ratios:
			# 	for n1 in [N]:
			# 		for n2 in [N]:
			# 			p1 = f'n1_{n1}+n2_{n2}:ratio_{ratio:.2f}:C_2_diff_sigma_n'
			# 			tmp_list.append((p1, n_clusters, n_clients))
			# data_details_lst += tmp_list

			tmp_list = []
			N1S = [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]
			N = 5000
			n_clusters = 2
			n_clients = 2
			for n1 in N1S:
				for n2 in [N]:
					p1 = f'n1_{n1}+n2_{n2}:ratio_0.00:C_2_diff_sigma_n'
					tmp_list.append((p1, n_clusters, n_clients))
			data_details_lst += tmp_list

			for data_detail, n_clusters, n_clients in data_details_lst:
				datasets.append(
					{'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})

		elif dataset_name == 'SENT140':
			# for data_detail, n_clusters, n_clients in [('sent140_user_percent', 2, 471)]:
			# 	datasets.append(
			# 		{'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})

			data_details_lst = []
			n_clients = 20
			n_clusters = 2
			tmp_list = []
			for n in [1, 3, 5, 10, 15, 20]:
				# there are 20 clients and each client has n users' data.
				p1 = f'n_{n}:ratio_0.00:diff_sigma_n'
				tmp_list.append((p1, n_clusters, n_clients))
			data_details_lst += tmp_list

			for data_detail, n_clusters, n_clients in data_details_lst:
				datasets.append(
					{'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})
		# elif dataset_name == '2GAUSSIANS':
		# 	data_details_lst = [
		# 		'1client_1cluster', 'mix_clusters_per_client',
		# 		'1client_ylt0', '1client_xlt0',
		# 		'1client_1cluster_diff_sigma', 'diff_sigma_n',
		# 		'1client_xlt0_2',
		# 	]
		elif dataset_name == '3GAUSSIANS':
			n_clusters = 3
			n_clients = 3
			# ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]] # each test set has 10%, 20%, 30% and 40% all data.
			ratios = [ratio for ratio in [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
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
			# # same sigma
			# tmp_list = []
			# N = 5000    # total cases: 8*7
			# for ratio in ratios:
			# 	for n1 in [N]:
			# 		# for n1 in [500, 2000, 3000, 5000, 8000]:
			# 		for sigma1 in ["0.3_0.3"]:  # sigma  = [[0.1, 0], [0, 0.1]]
			# 			for n2 in [N]:
			# 				for sigma2 in ["0.3_0.3"]:  # sigma  = [[0.1, 0], [0, 0.1]]
			# 					for n3 in [N]: #[50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
			# 						for sigma3 in ["0.3_0.3"]:  # sigma  = [[1, 0], [0, 0.1]]
			# 							p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
			# 							tmp_list.append((p1, n_clusters, n_clients))
			# #
			# data_details_lst += tmp_list
			#
			# tmp_list = []
			# N = 5000    # total cases: 9*7
			# for ratio in [0.0]:     # ratios:
			# 	for n1 in [N]:
			# 		for sigma1 in ["0.1_0.1"]:  # sigma  = [[0.1, 0], [0, 0.1]]
			# 			for n2 in [N]:
			# 				for sigma2 in ["0.2_0.2"]:  # sigma  = [[0.1, 0], [0, 0.1]]
			# 					for n3 in [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
			# 						for sigma3 in ["0.3_0.3"]:  # sigma  = [[1, 0], [0, 0.1]]
			# 							p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
			# 							tmp_list.append((p1, n_clusters, n_clients))
			#
			# data_details_lst += tmp_list

			tmp_list = []
			N = 5000  # total cases: 9*7
			for ratio in ratios:  # ratios:
				for n1 in [N]:
					for sigma1 in ["0.1_0.1"]:  # sigma  = [[0.1, 0], [0, 0.1]]
						for n2 in [N]:
							for sigma2 in ["0.1_0.1"]:  # sigma  = [[0.1, 0], [0, 0.1]]
								for n3 in [N]:
									for sigma3 in ["1.0_0.1"]:  # sigma  = [[1, 0], [0, 0.1]]
										p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
										tmp_list.append((p1, n_clusters, n_clients))

			data_details_lst += tmp_list

			tmp_list = []
			N = 5000  # total cases: 9*7
			for ratio in [0.0]:  # ratios:
				for n1 in [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
					for sigma1 in ["0.1_0.1"]:  # sigma  = [[0.1, 0], [0, 0.1]]
						for n2 in [N]:
							for sigma2 in ["0.1_0.1"]:  # sigma  = [[0.1, 0], [0, 0.1]]
								for n3 in [N]:
									for sigma3 in ["1.0_0.1"]:  # sigma  = [[1, 0], [0, 0.1]]
										p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
										tmp_list.append((p1, n_clusters, n_clients))

			data_details_lst += tmp_list

			# tmp_list = []
			# N = 5000  # total cases: 9*7
			# for ratio in [0.0]:  # ratios:
			# 	for n1 in [N]:
			# 		for sigma1 in ["0.1_0.1"]:  # sigma  = [[0.1, 0], [0, 0.1]]
			# 			for n2 in [N]:
			# 				for sigma2 in ["0.1_0.1"]:  # sigma  = [[0.1, 0], [0, 0.1]]
			# 					for n3 in [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
			# 						for sigma3 in ["1.0_0.1"]:  # sigma  = [[1, 0], [0, 0.1]]
			# 							p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
			# 							tmp_list.append((p1, n_clusters, n_clients))
			#
			# data_details_lst += tmp_list
			#
			# tmp_list = []
			# N = 5000  # total cases: 9*7
			# for ratio in [0.0]:  # ratios:
			# 	for n1 in [N]:
			# 		for sigma1 in ["0.3_0.3"]:  # sigma  = [[0.1, 0], [0, 0.1]]
			# 			for n2 in [N]:
			# 				for sigma2 in ["0.3_0.3"]:  # sigma  = [[0.1, 0], [0, 0.1]]
			# 					for n3 in [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
			# 						for sigma3 in ["1.0_0.1"]:  # sigma  = [[1, 0], [0, 0.1]]
			# 							p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
			# 							tmp_list.append((p1, n_clusters, n_clients))
			#
			# data_details_lst += tmp_list


			# # n1_100-sigma1_0.1+n2_5000-sigma2_0.2+n3_10000-sigma3_0.3:ratio_0.1:diff_sigma_n
			# # same sigma
			# tmp_list = []
			# N = 5000
			# for ratio in [0.0]:
			# 	for n1 in [N]:
			# 		# for n1 in [500, 2000, 3000, 5000, 8000]:
			# 		for sigma1 in ["0.3_0.3"]:
			# 			for n2 in [N]:
			# 				for sigma2 in ["0.3_0.3"]:
			# 					for n3 in [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
			# 						for sigma3 in ["0.3_0.3"]:
			# 							p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
			# 							tmp_list.append((p1, n_clusters, n_clients))
			# data_details_lst += tmp_list

			# # n1_100-sigma1_0.1+n2_5000-sigma2_0.2+n3_10000-sigma3_0.3:ratio_0.1:diff_sigma_n
			# tmp_list = []
			# N = 5000
			# for ratio in [0.0]:
			# 	for n1 in [N]:
			# 		# for n1 in [500, 2000, 3000, 5000, 8000]:
			# 		for sigma1 in ["0.1_0.1"]:
			# 			for n2 in [N]:
			# 				for sigma2 in ["0.2_0.2"]:
			# 					for n3 in [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
			# 						for sigma3 in ["0.3_0.3"]:
			# 							p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
			# 							tmp_list.append((p1, n_clusters, n_clients))
			#
			# data_details_lst += tmp_list
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
				datasets.append(
					{'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})

		elif dataset_name == '10GAUSSIANS':
			n_clients = 10
			data_details_lst = []
			# ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]] # each test set has 10%, 20%, 30% and 40% all data.
			ratios = [ratio for ratio in [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
			for n_clusters in [10]:
				"""
				Case 1 (Same N , but various P(ratios)
					G1 with mu1 = (-1, 0) and sigma1 = (0.1, 0.1)
					G2 with mu2 = (1, 0) and sigma2 = (0.1, 0.1)
					G3 with mu3 = (0, 3) and sigma3 = (1, 0.1)
					N1=N2=N3 = 10,000
					P = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]
				"""
				# # n1_100-sigma1_0.1_0.1+n2_5000-sigma2_0.1_0.1+n3_10000-sigma3_1.0_0.1:ratio_0.1:diff_sigma_n
				# # same sigma

				tmp_list = []
				N = 5000
				for ratio in [0.0]:
					for n1 in [N]:
						# for n1 in [500, 2000, 3000, 5000, 8000]:
						for sigma1 in ["0.1_0.1"]:
							for n2 in [N]:
								for sigma2 in ["0.1_0.1"]:
									for n3 in [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
										for sigma3 in ["1.0_0.1"]:
											p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
											tmp_list.append((p1, n_clusters, n_clients))
				data_details_lst += tmp_list

				tmp_list = []
				N = 5000
				for ratio in ratios:
					for n1 in [N]:
						# for n1 in [500, 2000, 3000, 5000, 8000]:
						for sigma1 in ["0.3_0.3"]:  # sigma  = [[0.1, 0], [0, 0.1]]
							for n2 in [N]:
								for sigma2 in ["0.3_0.3"]:  # sigma  = [[0.1, 0], [0, 0.1]]
									for n3 in [N]: #[50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
										for sigma3 in ["0.3_0.3"]:  # sigma  = [[1, 0], [0, 0.1]]
											p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
											tmp_list.append((p1, n_clusters, n_clients))
				data_details_lst += tmp_list

				tmp_list = []
				N = 5000
				for ratio in [0.0]:
					for n1 in [N]:
						# for n1 in [500, 2000, 3000, 5000, 8000]:
						for sigma1 in ["0.1_0.1"]:
							for n2 in [N]:
								for sigma2 in ["0.2_0.2"]:
									for n3 in [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
										for sigma3 in ["0.3_0.3"]:
											p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
											tmp_list.append((p1, n_clusters, n_clients))
				data_details_lst += tmp_list

			for data_detail, n_clusters, n_clients in data_details_lst:
				datasets.append(
					{'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})

		# elif dataset_name == '3GAUSSIANS-ADVERSARIAL':
		# 	n_clients = 3
		# 	n_clusters = 3
		# 	# ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]] # each test set has 10%, 20%, 30% and 40% all data.
		# 	# ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
		# 	# tot_cnt: 7 (ratios) * 5 (n1) * 2(sigma1) * 2 (n2) * 2 (sigma2) * 4 (alg) * 2 (per alg) = 2240 / 15 = 150 hrs = 6 days
		# 	# 8 (ratios) * 5 (n1) * 1 (sigma1) * 2 (n2) * 1 (sigma2) * 4 (alg) * 2 (per alg) = 640 / 15 = 42 hrs
		# 	data_details_lst = []
		# 	ratios = [0, 0.01]
		#
		# 	"""
		# 	Case 1 (Same N , but various P(ratios)
		# 		G1 with mu1 = (-1, 0) and sigma1 = (0.1, 0.1)
		# 		G2 with mu2 = (1, 0) and sigma2 = (0.1, 0.1)
		# 		G3 with mu3 = (0, 3) and sigma3 = (1, 0.1)
		# 		N1=N2=N3 = 10,000
		# 		P = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]
		# 	"""
		# 	# n1_100-sigma1_0.1_0.1+n2_5000-sigma2_0.1_0.1+n3_10000-sigma3_1.0_0.1:ratio_0.1:diff_sigma_n
		# 	# same sigma
		# 	tmp_list = []
		# 	N = 5000
		# 	# n4s = [10, 100, 200, 300, 500, 1000, 2000, 5000]
		# 	n4s = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 10]  # list(range(0, 10, 1)) * 0.1
		# 	for n4 in n4s:
		# 		for sigma4 in ["0.3_0.3", "0.5_0.5"]:
		# 			for ratio in ratios:
		# 				for n1 in [1000, 5000]:
		# 					# for n1 in [500, 2000, 3000, 5000, 8000]:
		# 					for sigma1 in ["0.3_0.3"]:  # sigma  = [[0.1, 0], [0, 0.1]]
		# 						for n2 in [5000]:
		# 							for sigma2 in ["0.3_0.3"]:  # sigma  = [[0.1, 0], [0, 0.1]]
		# 								for n3 in [10000]:
		# 									for sigma3 in ["0.3_0.3"]:  # sigma  = [[1, 0], [0, 0.1]]
		# 										p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}+n4_{n4}-sigma4_{sigma4}:ratio_{ratio:.2f}:diff_sigma_n'
		# 										tmp_list.append((p1, n_clusters, n_clients))
		#
		# 	data_details_lst += tmp_list
		# 	for data_detail, n_clusters in data_details_lst:
		# 		datasets.append(
		# 			{'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})
		# elif dataset_name == '5GAUSSIANS':
		# 	data_details_lst = [
		# 		'5clients_5clusters', '5clients_4clusters', '5clients_3clusters',
		# 	]
		else:
			msg = f'{dataset_name}'
			raise NotImplementedError(msg)

	return datasets


def get_algorithms_config_lst(py_names, n_clusters=2):
	algorithms = []
	for py_name in py_names:
		cnt = 0
		name = None
		if py_name == 'centralized_kmeans':
			for server_init_method, client_init_method in [('kmeans++', None)]:  # ('random', None)
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters,
				                   'server_init_method': server_init_method, 'client_init_method': client_init_method,
				                   'IS_FEDERATED': False})

		elif py_name == 'federated_server_init_first':
			for server_init_method, client_init_method in [('min_max', None), ('random', None)]: #[('min_max', None)]:
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters,
				                   'server_init_method': server_init_method, 'client_init_method': client_init_method,
				                   'IS_FEDERATED': True})
		elif py_name == 'federated_client_init_first':
			for server_init_method, client_init_method in [('average', 'random'), ('average', 'kmeans++')]:
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters,
				                   'server_init_method': server_init_method, 'client_init_method': client_init_method,
				                   'IS_FEDERATED': True})
		elif py_name == 'federated_greedy_kmeans':
			for server_init_method, client_init_method in [('greedy', 'random'), ('greedy', 'kmeans++')]:
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters,
				                   'server_init_method': server_init_method, 'client_init_method': client_init_method,
				                   'IS_FEDERATED': True})
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


def main(N_REPEATS=1, OVERWRITE=True, IS_DEBUG=False, VERBOSE = 5):
	# get default config.yaml
	config_file = 'config.yaml'
	args = config.load(config_file)
	OUT_DIR = args['OUT_DIR']
	SEPERTOR = args['SEPERTOR']
	args['N_REPEATS'] = N_REPEATS
	args['OVERWRITE'] = OVERWRITE
	args['VERBOSE'] = VERBOSE

	tot_cnt = 0
	# ['NBAIOT',  'FEMNIST', 'SENT140', '3GAUSSIANS', '10GAUSSIANS']
	# dataset_names = ['NBAIOT',  'FEMNIST', 'SENT140', '3GAUSSIANS', '10GAUSSIANS'] # ['NBAIOT'] # '3GAUSSIANS', '10GAUSSIANS', 'NBAIOT',  'FEMNIST', 'SENT140'
	dataset_names = ['NBAIOT',  '3GAUSSIANS', '10GAUSSIANS', 'SENT140', 'FEMNIST', 'BITCOIN', 'CHARFONT', 'DRYBEAN','GASSENSOR','SELFBACK', 'MNIST']  #
	dataset_names = ['MNIST', 'BITCOIN', 'CHARFONT','DRYBEAN', 'GASSENSOR','SELFBACK']  #
	dataset_names = ['FEMNIST']
	py_names = [
		'centralized_kmeans',
		'federated_server_init_first',  # server first: min-max per each dimension
		'federated_client_init_first',  # client initialization first : server average
		'federated_greedy_kmeans',  # client initialization first: greedy: server average
		# 'Our_greedy_center',
		# 'Our_greedy_2K',
		# 'Our_greedy_K_K',
		# 'Our_greedy_concat_Ks',
		# 'Our_weighted_kmeans_initialization',
	]
	datasets = get_datasets_config_lst(dataset_names)
	for dataset in datasets:
		args1 = copy.deepcopy(args)
		SEED = args1['SEED']
		args1['DATASET']['name'] = dataset['name']
		args1['DATASET']['detail'] = dataset['detail']
		args1['N_CLIENTS'] = dataset['n_clients']
		args1['N_CLUSTERS'] = dataset['n_clusters']
		N_CLIENTS = args1['N_CLIENTS']
		N_REPEATS = args1['N_REPEATS']
		N_CLUSTERS = args1['N_CLUSTERS']
		NORMALIZE_METHOD = args1['NORMALIZE_METHOD']
		IS_PCA = args1['IS_PCA']
		if args1['DATASET']['name']  == 'MNIST' and IS_PCA:
			args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'], NORMALIZE_METHOD, f'PCA_{IS_PCA}',
			                                                 f'M_{N_CLIENTS}', f'K_{N_CLUSTERS}', f'SEED_{SEED}'])
		else:
			args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'], f'M_{N_CLIENTS}', f'K_{N_CLUSTERS}', f'SEED_{SEED}'])
		dataset_detail = args1['DATASET']['detail']

		### generate dataset first to save time.
		args1['data_file'] = generate_dataset(args1)

		algorithms = get_algorithms_config_lst(py_names, dataset['n_clusters'])
		for algorithm in algorithms:
			args2 = copy.deepcopy(args1)
			print(f'\n*** {tot_cnt}th experiment ***')
			args2['IS_FEDERATED'] = algorithm['IS_FEDERATED']
			args2['ALGORITHM']['py_name'] = algorithm['py_name']
			# initial_method = args2['ALGORITHM']['initial_method']
			args2['ALGORITHM']['server_init_method'] = algorithm['server_init_method']
			server_init_method = args2['ALGORITHM']['server_init_method']
			args2['ALGORITHM']['client_init_method'] = algorithm['client_init_method']
			client_init_method = args2['ALGORITHM']['client_init_method']
			# args2['ALGORITHM']['name'] = algorithm['py_name'] + '_' + f'{server_init_method}|{client_init_method}'
			TOLERANCE = args2['TOLERANCE']
			NORMALIZE_METHOD = args2['NORMALIZE_METHOD']
			args2['ALGORITHM']['detail'] = f'{SEPERTOR}'.join([f'R_{N_REPEATS}',
			                                                   f'{server_init_method}|{client_init_method}',
			                                                   f'{TOLERANCE}', f'{NORMALIZE_METHOD}'])
			args2['OUT_DIR'] = os.path.join(OUT_DIR, args2['DATASET']['name'], f'{dataset_detail}',
			                                args2['ALGORITHM']['py_name'], args2['ALGORITHM']['detail'])
			if os.path.exists(args2['OUT_DIR']):
				shutil.rmtree(args2['OUT_DIR'])
				# shutil.rmtree(os.path.join(OUT_DIR, args2['DATASET']['name'], f'{dataset_detail}'))
			else:
				os.makedirs(args2['OUT_DIR'])
			new_config_file = os.path.join(args2['OUT_DIR'], 'config_file.yaml')
			if VERBOSE >= 2:
				pprint(new_config_file)
			config.dump(new_config_file, args2)
			args2['config_file'] = new_config_file
			if VERBOSE >= 5:
				pprint(args2, sort_dicts=False)

			if IS_DEBUG:
				### run a single experiment for debugging the code
				main_single.main(new_config_file)
				return

			# get sbatch.sh
			gen_sh(args2)
			tot_cnt += 1
	print(f'*** Total cases: {tot_cnt}')


if __name__ == '__main__':
	main(N_REPEATS=1, OVERWRITE=True, IS_DEBUG=True, VERBOSE=5)
	main(N_REPEATS=5, OVERWRITE=True, IS_DEBUG=False, VERBOSE=1)
