""" Run this main file for all experiments

#!/usr/bin/env bash

: ' multiline comment
  running the shell with source will change the current bash path
  e.g., source stat.sh

  check cuda and cudnn version for tensorflow_gpu==1.13.1
  https://www.tensorflow.org/install/source#linux
'
#ssh ky8517@tigergpu.princeton.edu
srun --time=2:00:00 --pty bash -i
srun --nodes=1  --mem=128G --ntasks-per-node=1 --time=20:00:00 --pty bash -i
#srun --nodes=1 --gres=gpu:1 --mem=128G --ntasks-per-node=1 --time=20:00:00 --pty bash -i
cd /scratch/gpfs/ky8517/fkm/fkm
module load anaconda3/2021.11

export PYTHONPATH="${PYTHONPATH}:..\"

# single quote (just treat the content inside it as a string) vs double quote in bash
# $ echo "$(echo "upg")"
# upg
# $ echo '$(echo "upg")'
# $(echo "upg")
#
As as result, if the file path contains \, &, $, please use single quote

linux: Difference between terms: "option", "argument", and "parameter"?
https://stackoverflow.com/questions/36495669/difference-between-terms-option-argument-and-parameter


# check why the process is killed
dmesg -T| grep -E -i -B100 'killed process'


PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_all.py


# For MNIST with GPU (more can be seen in datasets/mnist.py)
	ssh ky8517@tigergpu.princeton.edu
	# tmux new -s fkm
	tmux attach -t fkm
	srun --nodes=1 --gres=gpu:1 --mem=128G --ntasks-per-node=1 --time=20:00:00 --pty bash -i
	cd /scratch/gpfs/ky8517/fkm/fkm
	module load anaconda3/2021.11
	source activate tf2_10_0-gpu-py397
	module load cudatoolkit/11.2
	module load cudnn/cuda-11.x/8.2.0
	./sbatch

	PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 datasets/mnist.py

sacct -X --format="JobID, JobName%100,State"

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
		t = 24
	content = fr"""#!/bin/bash
#SBATCH --job-name={OUT_DIR}         # create a short name for your job
##SBATCH --nodes=1                # node count
##SBATCH --ntasks=1               # total number of tasks across all nodes
##SBATCH --cpus-per-task=5        # cpu-cores per task (>1 if multi-threaded tasks)
##SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time={t}:00:00          # total run time limit (HH:MM:SS)
## SBATCH --output={OUT_DIR}/%j-{job_name}-out.txt
## SBATCH --error={OUT_DIR}/%j-{job_name}-err.txt
#SBATCH --output={OUT_DIR}/out.txt
#SBATCH --error={OUT_DIR}/err.txt

### SBATCH --mail-type=begin        # send email when job begins
### SBATCH --mail-type=end          # send email when job ends\
### SBATCH --mail-user=kun.bj@cloud.com # not work \
### SBATCH --mail-user=<YourNetID>@princeton.edu
###SBATCH --mail-user=ky8517@princeton.edu     # which will cause too much email notification.

module purge
module load anaconda3/2021.11   # Python 3.9.7
pip3 install nltk

cd /scratch/gpfs/ky8517/fkm/fkm
pwd
python3 -V
    """

	# content += '\n' + f"PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 {algorithm_py_name} --dataset '{dataset_name}' " \
	#                   f"--data_details '{dataset_detail}' --algorithm '{algorithm_name}' --n_clusters '{n_clusters}' --n_clients '{n_clients}' \n"
	# content += '\n' + f"PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_single.py --config_file '{config_file}' > '{config_file}-log.txt' 2>&1 &\n"
	content += '\n' + f"PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_single.py --config_file '{config_file}' \n"

	# not work with '&' running in background > {job_name}.txt 2>&1 &
	# content += '\nwait\n'
	# content += '\necho \'done\''
	# sh_file = f'{OUT_DIR}/{dataset_name}-{dataset_detail}-{algorithm_name}-{algorithm_detail}.sh'
	sh_file = f'{OUT_DIR}/sbatch.sh'
	with open(sh_file, 'w') as f:
		f.write(content)
	cmd = f"sbatch '{sh_file}'"
	print(cmd)
	os.system(cmd)



def gen_all_sh(Args_lst):
	"""

	Parameters
	----------
	py_name
	case

	Returns
	-------

	"""
	args = Args_lst[0]
	# check_arguments()
	dataset_name = args['DATASET']['name']
	dataset_detail = os.path.dirname(args['DATASET']['detail'])
	# n_clients = args['N_CLIENTS']
	# algorithm_py_name = args['ALGORITHM']['py_name']
	# # algorithm_name = args['ALGORITHM']['name']
	# algorithm_detail = args['ALGORITHM']['detail']
	# # n_clusters = args['ALGORITHM']['n_clusters']
	# config_file = args['config_file']
	# OUT_DIR = os.path.join(*args['OUT_DIR'].split('/')[:4])
	OUT_DIR = args['OUT_DIR']

	job_name = f'{dataset_name}-{dataset_detail}'
	# tmp_dir = '~tmp'
	# if not os.path.exists(tmp_dir):
	# 	os.system(f'mkdir {tmp_dir}')
	if '2GAUSSIANS' in dataset_name:
		t = 48
	# elif 'FEMNIST' in dataset_name and 'greedy' in algorithm_py_name:
	# 	t = 48
	else:
		t = 48
	content = fr"""#!/bin/bash
#SBATCH --job-name={OUT_DIR}         # create a short name for your job
#SBATCH --time={t}:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output={OUT_DIR}/out.txt
#SBATCH --error={OUT_DIR}/err.txt
### SBATCH --mail-type=end          # send email when job ends
###SBATCH --mail-user=ky8517@princeton.edu     # which will cause too much email notification. \

## SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
## SBATCH --output={OUT_DIR}/%j-{job_name}-out.txt
## SBATCH --error={OUT_DIR}/%j-{job_name}-err.txt
## SBATCH --nodes=1                # node count
## SBATCH --ntasks=1               # total number of tasks across all nodes
## SBATCH --cpus-per-task=5        # cpu-cores per task (>1 if multi-threaded tasks)
## SBATCH --mem=40G\
### SBATCH --mail-type=begin        # send email when job begins
### SBATCH --mail-user=kun.bj@cloud.com # not work 
### SBATCH --mail-user=<YourNetID>@princeton.edu\

module purge
module load anaconda3/2021.11   # Python 3.9.7
pip3 install nltk 

cd /scratch/gpfs/ky8517/fkm/fkm 
pwd
python3 -V

"""
	content += "\nexport PYTHONPATH=\"${PYTHONPATH}:..\" \n"
	for i, args in enumerate(Args_lst):
		config_file_ = args['config_file']
		out_dir_ = args['OUT_DIR']
		# content += '\n' + f"PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 {algorithm_py_name} --dataset '{dataset_name}' " \
		#                   f"--data_details '{dataset_detail}' --algorithm '{algorithm_name}' --n_clusters '{n_clusters}' --n_clients '{n_clients}' \n"
		content +=  f"\nPYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 main_single.py --config_file '{config_file_}' > '{out_dir_}/a.txt' 2>&1  &\n"
		# if i == 0: 	# let the first one run first to generate the data and avoid unknown conflict for the rest.
		# 	content += "echo 'Finish the first one first...'\n"
		content += "\nwait\n"       # must has this line
		content += "echo $!\n"  # stores the background process PID
		content += "echo $?\n"  # $? stores the exit status.
		content += f"echo \'Finish {i}-th args.\'"  # $? stores the exit status.
	# single quote (just treat the content inside it as a string) vs double quote in bash
	# $ echo "$(echo "upg")"
	# upg
	# $ echo '$(echo "upg")'
	# $(echo "upg")
	#
	content += "\nwait\n"       # must has this line
	# The bash wait command is a Shell command that waits for background running processes to complete and returns the exit status.
	# Without any parameters, the wait command waits for all background processes to finish before continuing the script.
	content += "echo $!\n"  # stores the background process PID
	content += "echo $?\n"  # $? stores the exit status.
	content += "\necho \'done\n"
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
			# for n in [5000]: #[50, 100, 500, 1000, 3000, 5000]:
			ratios = [0, 0.1, 0.3, 0.5]  # [0, 0.1, 0.3, 0.5]
			for ratio in ratios:
				p1 = f'n_1000:ratio_{ratio:.2f}:diff_sigma_n'
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
			for n in [0]: #[50, 100, 500, 800, 1000, 1500]:
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
			for n in [3000]: #[50, 100, 500, 800, 1000, 1500, 3000]:
				# there are 20 clients and each client has n users' data.
				p1 = f'n_{n}:ratio_0.00:diff_sigma_n'
				tmp_list.append((p1, n_clusters, n_clients))
			data_details_lst += tmp_list

			for data_detail, n_clusters, n_clients in data_details_lst:
				datasets.append(
					{'name': dataset_name, 'detail': data_detail, 'n_clusters': n_clusters, 'n_clients': n_clients})

		elif dataset_name == 'DRYBEAN':
			data_details_lst = []
			n_clients = 7
			n_clusters = 7
			tmp_list = []
			for n in [0]: #[50, 100, 500, 800, 1000]:
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
			for n in [5000]: #[50, 500, 1000, 2000, 3000, 5000]:
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
			for n in [5000]: #[50, 100, 500, 800, 1000, 1500, 2000]:
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
			# N1S = [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]
			# N1S = [5000]    # (49548, 115) datasets/NBAIOT/Danmini_Doorbell/benign_traffic.csv
			# N = 9000        # (92141, 115) datasets/NBAIOT/Danmini_Doorbell/gafgyt_attacks/tcp.csv
			n_clusters = 2
			n_clients = 2

			ratios = [0, 0.1, 0.3, 0.5] #[0, 0.1, 0.3, 0.5]
			for ratio in ratios:
				p1 = f'n1_5000+n2_9000:ratio_{ratio:.2f}:C_2_diff_sigma_n'
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
			# ratios = [ratio for ratio in [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
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
			N = 1000  # total cases: 9*7
			for ratio in [0.0, 0.1, 0.3, 0.5]:  # ratios:
				for n1 in [N]:
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
			# 	for n1 in [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
			# 		for sigma1 in ["0.1_0.1"]:  # sigma  = [[0.1, 0], [0, 0.1]]
			# 			for n2 in [N]:
			# 				for sigma2 in ["0.1_0.1"]:  # sigma  = [[0.1, 0], [0, 0.1]]
			# 					for n3 in [N]:
			# 						for sigma3 in ["1.0_0.1"]:  # sigma  = [[1, 0], [0, 0.1]]
			# 							p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
			# 							tmp_list.append((p1, n_clusters, n_clients))
			#
			# data_details_lst += tmp_list

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
			# ratios = [ratio for ratio in [0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
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
				# the results will not be good due to the small size of other clusters.
				# for ratio in [0.0, 0.1, 0.3, 0.4999]:  # [0.0, 0.1, 0.3, 0.4999]:
				# 	for n1 in [5000]:
				# 		# for n1 in [500, 2000, 3000, 5000, 8000]:
				# 		for sigma1 in ["0.3_0.3"]:
				# 			for n2 in [1000]:
				# 				for sigma2 in ["0.2_0.2"]:
				# 					for n3 in [500]:  # [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
				# 						for sigma3 in ["0.1_0.1"]:
				# 							p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
				# 							tmp_list.append((p1, n_clusters, n_clients))
				# data_details_lst += tmp_list


				N = 1000
				for ratio in [0, 0.1, 0.3, 0.5]:
					for n1 in [N]:
						# for n1 in [500, 2000, 3000, 5000, 8000]:
						for sigma1 in ["0.1_0.1"]:
							for n2 in [N]:
								for sigma2 in ["0.1_0.1"]:
									for n3 in [N]:  #[50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
										for sigma3 in ["0.1_0.1"]:
											p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
											tmp_list.append((p1, n_clusters, n_clients))
				data_details_lst += tmp_list

				# tmp_list = []
				# N = 5000
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
				# data_details_lst += tmp_list
				#
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
				# data_details_lst += tmp_list

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
			for server_init_method, client_init_method in [('random', None), ('kmeans++', None)]:  # ('random', None)
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters,
				                   'server_init_method': server_init_method, 'client_init_method': client_init_method,
				                   'IS_FEDERATED': False})
		elif py_name == 'centralized_minibatch_kmeans':
			for server_init_method, client_init_method in [('random', None), ('kmeans++', None)]:  # ('random', None)
				algorithms.append({'py_name': py_name, 'name': name, 'n_clusters': n_clusters,
				                   'server_init_method': server_init_method, 'client_init_method': client_init_method,
				                   'IS_FEDERATED': False})

		elif py_name == 'federated_server_init_first':
			for server_init_method, client_init_method in [('min_max', None)]: #[('random', None)]:
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


def main(N_REPEATS=1, OVERWRITE=True, IS_DEBUG=False, IS_GEN_DATA = True, VERBOSE = 5, IS_PCA = True, IS_REMOVE_OUTLIERS = True):
	# get default config.yaml
	config_file = 'config.yaml'
	args = config.load(config_file)
	OUT_DIR = args['OUT_DIR']
	SEPERTOR = args['SEPERTOR']
	args['N_REPEATS'] = N_REPEATS
	args['OVERWRITE'] = OVERWRITE
	args['VERBOSE'] = VERBOSE
	args['IS_PCA'] = IS_PCA
	args['IS_REMOVE_OUTLIERS'] = IS_REMOVE_OUTLIERS

	tot_cnt = 0
	# ['NBAIOT',  'FEMNIST', 'SENT140', '3GAUSSIANS', '10GAUSSIANS']
	# dataset_names = ['NBAIOT',  'FEMNIST', 'SENT140', '3GAUSSIANS', '10GAUSSIANS'] # ['NBAIOT'] # '3GAUSSIANS', '10GAUSSIANS', 'NBAIOT',  'FEMNIST', 'SENT140'
	# dataset_names = ['NBAIOT',  '3GAUSSIANS', '10GAUSSIANS', 'SENT140', 'FEMNIST', 'BITCOIN', 'CHARFONT', 'SELFBACK','GASSENSOR','SELFBACK', 'MNIST']  #
	# dataset_names = ['MNIST', 'BITCOIN', 'CHARFONT','DRYBEAN', 'GASSENSOR','SELFBACK']  #
	# dataset_names = ['SELFBACK', 'GASSENSOR', 'MNIST', 'DRYBEAN'] # 'NBAIOT', '3GAUSSIANS'
	dataset_names = ['MNIST']
	py_names = [
		'centralized_kmeans',
		'federated_server_init_first',  # server first: min-max per each dimension
		'federated_client_init_first',  # client initialization first : server average
		'federated_greedy_kmeans',  # client initialization first: greedy: server average
		# 'centralized_minibatch_kmeans', # Not finish yet.
		# # 'Our_greedy_center',
		# 'Our_greedy_2K',
		# 'Our_greedy_K_K',
		# 'Our_greedy_concat_Ks',
		# 'Our_weighted_kmeans_initialization',
	]
	datasets = get_datasets_config_lst(dataset_names)
	for dataset in datasets:
		args1_ = copy.deepcopy(args)
		if dataset['name'] == '3GAUSSIANS' and IS_PCA == True: continue
		if dataset['name'] == '10GAUSSIANS' and IS_PCA == True: continue
		if dataset['name'] == 'MNIST':
			if args1_['IS_PCA'] == True:
				args1_['IS_PCA'] = 'CNN'
			else:
				continue
		# 	continue # we already have the results
		# # if dataset['name'] == 'MNIST' and args['IS_PCA'] == False:
		# 	continue
		if dataset['name'] == 'NBAIOT' and args1_['IS_PCA'] == True:
			continue
		algorithms = get_algorithms_config_lst(py_names, dataset['n_clusters'])
		for i_alg, algorithm in enumerate(algorithms):
			print(f'\n*** {tot_cnt}th experiment ***:', dataset['name'], algorithm['py_name'])
			Args_lst = []
			for i_repeat in range(N_REPEATS):
				seed_data = i_repeat * 10   # data seed
				# print('***', dataset['name'], i_repeat, seed_data)
				args1 = copy.deepcopy(args1_)
				SEED = args1['SEED'] # model seed
				args1['SEED_DATA'] = seed_data
				args1['DATASET']['name'] = dataset['name']
				args1['DATASET']['detail'] = dataset['detail']
				args1['N_CLIENTS'] = dataset['n_clients']
				args1['N_CLUSTERS'] = dataset['n_clusters']
				N_CLIENTS = args1['N_CLIENTS']
				N_REPEATS = args1['N_REPEATS']
				N_CLUSTERS = args1['N_CLUSTERS']
				NORMALIZE_METHOD = args1['NORMALIZE_METHOD']
				IS_PCA = args1['IS_PCA']
				IS_REMOVE_OUTLIERS = args1['IS_REMOVE_OUTLIERS']
				# if args1['DATASET']['name']  == 'MNIST' and IS_PCA:
				# 	args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'], NORMALIZE_METHOD, f'PCA_{IS_PCA}',
				# 	                                                 f'M_{N_CLIENTS}', f'K_{N_CLUSTERS}', f'SEED_{SEED}'])
				# else:
				args1['DATASET']['detail'] = os.path.join(f'{SEPERTOR}'.join([args1['DATASET']['detail'], NORMALIZE_METHOD, f'PCA_{IS_PCA}', f'M_{N_CLIENTS}', f'K_{N_CLUSTERS}', f'REMOVE_OUTLIERS_{IS_REMOVE_OUTLIERS}']), f'SEED_DATA_{seed_data}')
				dataset_detail = args1['DATASET']['detail']
				dataset_name = args1['DATASET']['name']
				args1['data_file'] = os.path.join(args1['IN_DIR'], dataset_name, f'{dataset_detail}.dat')

				args2 = copy.deepcopy(args1)
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
					# pass
					# shutil.rmtree(os.path.join(OUT_DIR, args2['DATASET']['name'], f'{dataset_detail}'))
				else:
					os.makedirs(args2['OUT_DIR'])

				### generate dataset first to save time. Be careful when muliti-processes are submitted to the server.
				# TODO: update data generation in a lazy way.
				# if IS_GEN_DATA and i_alg == 0:   if use this condition, it only generetate centralized data with ratio =0.0.  For other ratios, just be ignored.
				# for different algorithms, please generate its own dataset for safe.
				args2['data_file'] = generate_dataset(args2)
				# print(f'arg2.data_file:', args2['data_file'])
				# if IS_GEN_DATA: continue

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
				# if not IS_GEN_DATA:
				# 	gen_sh(args2)
				Args_lst.append(copy.deepcopy(args2))

			tot_cnt += 1
			if not IS_GEN_DATA:
				gen_all_sh(Args_lst)  # Be careful that multi-process will operate (generate and remove) the same dataset file
	print(f'*** Total cases: {tot_cnt}')


if __name__ == '__main__':
	# main(N_REPEATS=1, OVERWRITE=True, IS_DEBUG=True, VERBOSE=5, IS_PCA = False, IS_REMOVE_OUTLIERS = False)
	# exit()

	for IS_REMOVE_OUTLIERS in [False]:
		for IS_PCA in [False, True]:
			# you should run twice: the first time is to generate data and the second one is to run the models to avoid multi-processes operate on the same file.
			# the first time is to generate data
			# train and evalate the models. Note that IS_GEN_DATA = False  and OVERWRITE = False
			main(N_REPEATS=50, OVERWRITE=True, IS_DEBUG=False, IS_GEN_DATA = False, VERBOSE=3, IS_PCA = IS_PCA, IS_REMOVE_OUTLIERS = IS_REMOVE_OUTLIERS)
