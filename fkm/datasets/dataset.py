"""

https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_plusplus.html#sphx-glr-auto-examples-cluster-plot-kmeans-plusplus-py

"""
import os
import pickle

from fkm.datasets.bitcoin import bitcoin_diff_sigma_n
from fkm.datasets.charfont import charfont_diff_sigma_n
from fkm.datasets.drybean import drybean_diff_sigma_n
from fkm.datasets.femnist import femnist_multiusers_per_client, femnist_diff_sigma_n
from fkm.datasets.gassensor import gassensor_diff_sigma_n
from fkm.datasets.gaussian10 import gaussian10_diff_sigma_n
from fkm.datasets.mnist import mnist_diff_sigma_n
from fkm.datasets.nbaiot import nbaiot_user_percent, nbaiot_user_percent_client11, nbaiot_diff_sigma_n, \
	nbaiot_C_2_diff_sigma_n
from fkm.datasets.selfback import selfback_diff_sigma_n
from fkm.datasets.sent140 import sent140_user_percent, sent140_diff_sigma_n
from fkm.datasets.gaussian3 import gaussian3_diff_sigma_n
from fkm.utils.utils_func import timer, check_path


@timer
def generate_dataset(args):
	"""

	Parameters
	----------
	args

	Returns
	-------

	"""
	# SEED = args['SEED']
	SEED_DATA = args['SEED_DATA']
	dataset_name = args['DATASET']['name']
	dataset_detail = args['DATASET']['detail']
	N_CLIENTS = args['N_CLIENTS']
	N_REPEATS = args['N_REPEATS']
	N_CLUSTERS = args['N_CLUSTERS']
	data_file = os.path.join(args['IN_DIR'], dataset_name,  f'{dataset_detail}.dat')
	# print(data_file)
	check_path(data_file)
	args['data_file'] = data_file
	if args['OVERWRITE'] and os.path.exists(data_file):
		# here could be some issue for multi-tasks, please double-check before calling this function.
		os.remove(data_file)
	elif os.path.exists(data_file):
		return data_file
	else:
		print('Generate dataset')

	if dataset_name == 'FEMNIST':
		# if dataset_detail == '1client_1writer_multidigits':
		#     return femnist_1client_1writer_multidigits(params, random_state=SEED_DATA)
		# elif dataset_detail == '1client_multiwriters_multidigits':
		#     return femnist_1client_multiwriters_multidigits(params, random_state=SEED_DATA)
		# elif dataset_detail == '1client_multiwriters_1digit':
		#     return femnist_1client_multiwriters_1digit(params, random_state=SEED_DATA)
		if 'femnist_user_percent' in dataset_detail:
			data = femnist_multiusers_per_client(args, random_state=SEED_DATA)
		elif 'diff_sigma_n' in dataset_detail:
			data = femnist_diff_sigma_n(args, random_state=SEED_DATA)
		else:
			msg = f'{dataset_name}, {dataset_detail}'
			raise NotImplementedError(msg)
	elif dataset_name == 'MNIST':
		if 'diff_sigma_n' in dataset_detail:
			data = mnist_diff_sigma_n(args, random_state=SEED_DATA)
		else:
			msg = f'{dataset_name}, {dataset_detail}'
			raise NotImplementedError(msg)
	elif dataset_name == 'GASSENSOR':
		if 'diff_sigma_n' in dataset_detail:
			data = gassensor_diff_sigma_n(args, random_state=SEED_DATA)
		else:
			msg = f'{dataset_name}, {dataset_detail}'
			raise NotImplementedError(msg)
	elif dataset_name == 'CHARFONT':
		if 'diff_sigma_n' in dataset_detail:
			data = charfont_diff_sigma_n(args, random_state=SEED_DATA)
		else:
			msg = f'{dataset_name}, {dataset_detail}'
			raise NotImplementedError(msg)
	elif dataset_name == 'DRYBEAN':
		if 'diff_sigma_n' in dataset_detail:
			data = drybean_diff_sigma_n(args, random_state=SEED_DATA)
		else:
			msg = f'{dataset_name}, {dataset_detail}'
			raise NotImplementedError(msg)
	elif dataset_name == 'BITCOIN':
		if 'diff_sigma_n' in dataset_detail:
			data = bitcoin_diff_sigma_n(args, random_state=SEED_DATA)
		else:
			msg = f'{dataset_name}, {dataset_detail}'
			raise NotImplementedError(msg)
	elif dataset_name == 'SELFBACK':
		if 'diff_sigma_n' in dataset_detail:
			data = selfback_diff_sigma_n(args, random_state=SEED_DATA)
		else:
			msg = f'{dataset_name}, {dataset_detail}'
			raise NotImplementedError(msg)
	elif dataset_name == 'NBAIOT':
		if 'nbaiot_user_percent' in dataset_detail and N_CLUSTERS == 2 and N_CLIENTS == 2:
			data = nbaiot_user_percent(args, random_state=SEED_DATA)
		elif 'nbaiot_user_percent' in dataset_detail and N_CLIENTS == 11:
			data = nbaiot_user_percent_client11(args, random_state=SEED_DATA)
		elif 'C_2_diff_sigma_n' in dataset_detail:
			data = nbaiot_C_2_diff_sigma_n(args, random_state=SEED_DATA)
		elif 'diff_sigma_n' in dataset_detail:
			data = nbaiot_diff_sigma_n(args, random_state=SEED_DATA)
		else:
			msg = f'{dataset_name}, {dataset_detail}'
			raise NotImplementedError(msg)
	elif dataset_name == 'SENT140':
		if 'sent140_user_percent' in dataset_detail and N_CLUSTERS == 2:
			data = sent140_user_percent(args, random_state=SEED_DATA)
		elif 'diff_sigma_n' in dataset_detail:
			data = sent140_diff_sigma_n(args, random_state=SEED_DATA)
		else:
			msg = f'{dataset_name}, {dataset_detail}'
			raise NotImplementedError(msg)
	elif dataset_name == '3GAUSSIANS':
		if 'diff_sigma_n' in dataset_detail:
			data = gaussian3_diff_sigma_n(args, random_state=SEED_DATA)
		# elif dataset_detail == '1client_1cluster':
		# 	data = gaussian3_1client_1cluster(params, random_state=SEED_DATA)
		# elif dataset_detail.split(':')[-1] == 'mix_clusters_per_client':
		# 	data =  gaussian3_mix_clusters_per_client(params, random_state=SEED_DATA)
		# elif dataset_detail == '1client_ylt0':
		# 	data =  gaussian3_1client_ylt0(params, random_state=SEED_DATA)
		# elif dataset_detail == '1client_xlt0':
		# 	data = gaussian3_1client_xlt0(params, random_state=SEED_DATA)
		# elif dataset_detail == '1client_1cluster_diff_sigma':
		# 	data = gaussian3_1client_1cluster_diff_sigma(params, random_state=SEED_DATA)
		# elif dataset_detail == '1client_xlt0_2':
		# 	data = gaussian3_1client_xlt0_2(params, random_state=SEED_DATA)
		else:
			msg = f'{dataset_name}, {dataset_detail}'
			raise NotImplementedError(msg)
	elif dataset_name == '10GAUSSIANS':
		if 'diff_sigma_n' in dataset_detail:
			data = gaussian10_diff_sigma_n(args, random_state=SEED_DATA)
		else:
			msg = f'{dataset_name}, {dataset_detail}'
			raise NotImplementedError(msg)
	# elif dataset_name == '3GAUSSIANS-ADVERSARIAL':
	# 	if dataset_detail.split(':')[-1] == 'diff_sigma_n':
	# 		data = adversarial_gaussian3_diff_sigma_n(params, random_state=SEED_DATA)
	# 	else:
	# 		msg = f'{dataset_name}, {dataset_detail}'
	# 		raise NotImplementedError(msg)
	else:
		msg = f'{dataset_name}, {dataset_detail}'
		raise NotImplementedError(msg)

	check_path(data_file)

	with open(data_file, 'wb') as f:
		pickle.dump(data, f)

	return data_file