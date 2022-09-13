"""

"""
# Email: Kun.bj@outlook.com
import copy
import json
import os
import pickle
import time
import traceback
from collections import Counter
from pprint import pprint

import numpy as np
from sklearn.preprocessing import StandardScaler

from fkm import vis
from fkm.utils.utils_func import dump, obtain_true_centroids
from fkm.utils.utils_stats import evaluate2
# These options determine the way floating point numbers, arrays and
# other NumPy objects are displayed.
# np.set_printoptions(precision=3, suppress=True)
from fkm.vis.visualize import plot_2gaussian, plot_3gaussian

np.set_printoptions(precision=3, suppress=True, formatter={'float': '{:.3f}'.format}, edgeitems=120, linewidth=100000)

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
		pprint(seed_history, stream=f, sort_dicts=False)  # 'sort_dicts=False' works when python version >= 3.8

		f.write('\n\n***Save data with recursion')
		res = format(seed_history)
		f.write(res)


# def normalize2(raw_x, raw_y, raw_true_centroids, splits, params, is_federated=False):
# 	"""
# 	Only for diagonal covariance matrix:
# 		even for federated kmeans, we still can get global mean and std from each client data.
# 		Based on that, for centralized and federated kmeans, we can use the same standscaler.
#
# 	Parameters
# 	----------
# 	raw_x
# 	raw_y
# 	raw_true_centroids
# 	splits
# 	params
# 	is_federated
#
# 	Returns
# 	-------
#
# 	"""
# 	is_show = params['is_show']
# 	normalize_method = params['normalize_method']
# 	# do normalization
# 	if normalize_method == 'std':
# 		# collects all clients' data together
# 		x = copy.deepcopy(raw_x)
# 		y = copy.deepcopy(raw_y)
# 		new_true_centroids = copy.deepcopy(raw_true_centroids)
# 		for spl in splits:  # train and test
# 			x[spl] = np.concatenate(x[spl], axis=0)
# 		global_stdscaler = StandardScaler()
# 		global_stdscaler.fit(x['train'])
# 		for spl in splits:  # train and test
# 			new_true_centroids[spl] = global_stdscaler.transform(raw_true_centroids[spl])
#
# 		if is_federated == False:  # for centralized normalization
# 			new_x = copy.deepcopy(raw_x)
# 			new_y = copy.deepcopy(raw_y)
# 			for i_, _ in enumerate(new_x['train']):
# 				new_x['train'][i_] = global_stdscaler.transform(new_x['train'][i_])
# 				new_x['test'][i_] = global_stdscaler.transform(new_x['test'][i_])
# 			params['stdscaler'] = global_stdscaler
#
# 			if is_show:
# 				if '3GAUSSIANS' in params['p0']:  # for plotting
# 					plot_3gaussian(new_x['train'][0], new_y['train'], new_x['train'][1], new_y['train'][1],
# 					               new_x['train'][2], new_y['train'][2], params, title='std')
# 		else:  # federated kmeans
# 			# for each client, we can get mean and std and then use them to get global_std
# 			new_x = copy.deepcopy(raw_x)
# 			new_y = copy.deepcopy(raw_y)
# 			stds = [[]] * len(new_x['train'])  # get number of clients
# 			N = 0
# 			for i_, _ in enumerate(new_x['train']):  # for each client
# 				data = new_x['train'][i_]
# 				n = len(data)
# 				stds[i_] = (n, np.mean(data, axis=0), np.std(data, axis=0))
# 				N += n
#
# 			dim = new_x['train'][0].shape[1]
# 			global_mean = [[]] * dim
# 			global_std = [[]] * dim
# 			# the following method only works for diagonal covariance matrix
# 			for i_ in range(dim):
# 				# compute global mean and std given each client's mean and std
# 				# global_mean = 1/N * (\sum client1  + \sum client2 + \sum client3)
# 				# 			  = 1/N * (n1 * mu1 + n2 * mu2 + n3 * mu3)
# 				global_mean[i_] = sum([n[i_] * mu[i_] for n, mu, s in stds]) / N
#
# 				# global_var = E(x-mu)**2 = (1/N * (\sum x**2)) - mu**2
# 				# 					   = (1/N * (\sum client1**2 + \sum client2**2+ \sum client3**2)) - global_mean **2
# 				#                      = (1/N * (n1 * var1 + n2 * var2 + n3 * var3)) - global_mean**2
# 				#                      = (1/N * (n1 * std1**2 + n2 * std2** + n3 * std3**2)) - global_mean**2
# 				global_std[i_] = (sum([n[i_] * s[i_] ** 2 for n, mu, s in stds]) / N - global_mean[i_] ** 2) ** (1 / 2)
#
# 			# for each client, then normalize its data use the global mean and std
# 			global_stdscaler2 = StandardScaler()
# 			global_stdscaler2.mean_ = global_mean
# 			global_stdscaler2.scale_ = global_std
#
# 			for i_, _ in enumerate(new_x['train']):
# 				new_x['train'][i_] = global_stdscaler2.transform(new_x['train'][i_])
# 				new_x['test'][i_] = global_stdscaler2.transform(new_x['test'][i_])
# 			params['stdscaler'] = global_stdscaler2
#
# 			if is_show:
# 				if '3GAUSSIANS' in params['p0']:  # for plotting
# 					plot_3gaussian(new_x['train'][0], new_y['train'], new_x['train'][1], new_y['train'][1],
# 					               new_x['train'][2], new_y['train'][2], params, title='std')
#
# 	else:
# 		new_x, new_y, new_true_centroids = raw_x, raw_y, raw_true_centroids
#
# 	return new_x, new_y, new_true_centroids


def normalize(raw_x, raw_y, raw_true_centroids, splits, params):
	"""
	For federated kmeans, we still can get global mean and std from each client data.
		Based on that, for centralized and federated kmeans, we can use the same standscaler.

	Parameters
	----------
	raw_x
	raw_y
	raw_true_centroids
	splits
	params

	Returns
	-------

	"""
	try:
		is_show = params['is_show']
	except Exception as e:
		is_show = params['IS_SHOW']
	try:
		normalize_method = params['normalize_method']
	except Exception as e:
		normalize_method = params['NORMALIZE_METHOD']

	try:
		algorithm_name = params['p0']
	except Exception as e:
		algorithm_name = params['ALGORITHM']['name']
	# do normalization
	if normalize_method == 'std':
		# collects all clients' data together and get global stdscaler
		x = copy.deepcopy(raw_x)
		y = copy.deepcopy(raw_y)
		new_true_centroids = copy.deepcopy(raw_true_centroids)
		for spl in splits:  # train and test
			x[spl] = np.concatenate(x[spl], axis=0)

		global_stdscaler = StandardScaler()  # we can get the same global_stdscaler using each client mean and std.
		global_stdscaler.fit(x['train'])

		for spl in splits:  # train and test
			new_true_centroids[spl] = global_stdscaler.transform(new_true_centroids[spl])
			# print(new_true_centroids[spl])

		# normalize data
		new_x = copy.deepcopy(raw_x)
		new_y = copy.deepcopy(raw_y)
		for spl in splits:
			for i_ in range(len(new_x[spl])):
				new_x[spl][i_] = global_stdscaler.transform(new_x[spl][i_])  # normalize data first
		# for j_ in set(new_y[spl][i_]):
		# 	new_true_centroids[spl][j_] = np.mean(new_x[spl][i_], axis=0)  # get new centroids

		# if is_show:
		# 	if '2GAUSSIANS' in algorithm_name:  # for plotting
		# 		plot_2gaussian(new_x['train'][0], new_y['train'], new_x['train'][1], new_y['train'][1],
		# 		               params, title='std')
		# 	elif '3GAUSSIANS' in algorithm_name:  # for plotting
		# 		plot_3gaussian(new_x['train'][0], new_y['train'], new_x['train'][1], new_y['train'][1],
		# 		               new_x['train'][2], new_y['train'][2], params, title='std')
		# 	elif '4GAUSSIANS' in algorithm_name:  # for plotting
		# 		plot_2gaussian(new_x['train'][0], new_y['train'], new_x['train'][1], new_y['train'][1],
		# 		               params, title='std')
	elif normalize_method == 'min_max':
		raise NotImplementedError
	else:
		# new_x, new_y, new_true_centroids = raw_x, raw_y, raw_true_centroids
		# # collects all clients' data together and get global stdscaler
		# x = copy.deepcopy(raw_x)
		# y = copy.deepcopy(raw_y)
		# for spl in splits:  # train and test
		# 	x[spl] = np.concatenate(x[spl], axis=0)
		#
		# global_stdscaler = StandardScaler()  # we can get the same global_stdscaler using each client mean and std.
		# global_stdscaler.fit(x['train'])
		return raw_x, raw_y, raw_true_centroids, None

	return new_x, new_y, new_true_centroids, global_stdscaler


class NumpyEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		return json.JSONEncoder.default(self, obj)


def run_model(args):
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
	np.random.seed(args['SEED'])  # set the global seed for numpy
	VERBOSE = args['VERBOSE']
	SEPERTOR = args['SEPERTOR']
	SPLITS = args['SPLITS']
	dataset_name = args['DATASET']['name']
	N_CLUSTERS = args['N_CLUSTERS']
	N_CLIENTS = args['N_CLIENTS']
	# args['DATASET']['detail'] = f'{SEPERTOR}'.join(args['DATASET']['detail'], f'M_{N_CLIENTS}|K_{N_CLUSTERS}')
	# dataset_detail = args['DATASET']['detail']
	# algorithm_py_name = args['ALGORITHM']['py_name']
	# initial_method = args['ALGORITHM']['initial_method']
	server_init_method = args['ALGORITHM']['server_init_method']
	client_init_method = args['ALGORITHM']['client_init_method']

	N_REPEATS = args['N_REPEATS']
	TOLERANCE = args['TOLERANCE']
	NORMALIZE_METHOD = args['NORMALIZE_METHOD']
	# algorithm_detail = args['ALGORITHM']['detail']
	# OUT_DIR = args['OUT_DIR']
	# # if os.path.exists(OUT_DIR):
	# # 	shutil.rmtree(OUT_DIR)
	history_file = os.path.join(args['OUT_DIR'], 'history.dat')
	args['history_file'] = history_file
	# if os.path.exists(history_file):
	# # here could be some issue for multi-tasks, please double-check before calling this function.
	# 	return history_file

	# settings
	history = {'args': args}  # Store all results and needed data

	# data is fixed, however, the algorithm will have different initialization centroids with different seeds.
	with open(args['data_file'], 'rb') as f:
		raw_x, raw_y = pickle.load(f)

	if VERBOSE >= 1:
		# print raw_x and raw_y distribution
		for split in SPLITS:
			print(f'{split} set:')
			clients_x, clients_y = raw_x[split], raw_y[split]
			if VERBOSE >= 5:
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
	raw_true_centroids = obtain_true_centroids(raw_x, raw_y, SPLITS, args)
	if VERBOSE >= 3:
		# print true centroids
		for split in SPLITS:
			true_c = raw_true_centroids[split]
			print(f'{split}_true_centroids:')
			print(true_c)

	# if algorithm_name == 'FEMNIST':
	# 	save_image2disk((raw_x, raw_y), out_dir_i, params)

	# do normalization in global for kmeans and federated kmeans
	raw_x, raw_y, raw_true_centroids, global_stdscaler = normalize(raw_x, raw_y, raw_true_centroids, SPLITS, args)
	args['global_stdscaler'] = global_stdscaler
	history['raw_true_centroids'] = raw_true_centroids
	print(f'after normalization, true_centroids:\n{raw_true_centroids} \nwhen normalize_method = {NORMALIZE_METHOD}')
	# history = {'x': raw_x, 'y': raw_y, 'results': []}
	# SEEDS = [10 * v ** 2 for v in range(1, N_REPEATS + 1, 1)]
	SEEDS = [42]    # we fix the model seed; however, the data seed is different.
	history['SEEDS'] = SEEDS

	from fkm.cluster import centralized_kmeans, federated_server_init_first, federated_client_init_first, \
		federated_greedy_kmeans
	KMEANS2PY = {
		'centralized_kmeans': centralized_kmeans.KMeans,
		'federated_server_init_first': federated_server_init_first.KMeansFederated,
		'federated_client_init_first': federated_client_init_first.KMeansFederated,
		'federated_greedy_kmeans': federated_greedy_kmeans.KMeansFederated
	}
	for idx_seed, seed in enumerate(SEEDS):  # repetitions:  to obtain average and std score.

		# if VERBOSE >= 2:
		print(f'\n***{idx_seed}th repeat with seed: {seed}:')
		X = copy.deepcopy(raw_x)
		Y = copy.deepcopy(raw_y)
		true_centroids = copy.deepcopy(raw_true_centroids)

		if not args['IS_FEDERATED']:
			# collects all clients' data together
			for spl in SPLITS:
				X[spl] = np.concatenate(X[spl], axis=0)
				Y[spl] = np.concatenate(Y[spl], axis=0)
				print(spl, X[spl].shape, Y[spl].shape)

		t1 = time.time()
		# for Centralized Kmeans, we use server_init_centroids as init_centroids.
		kmeans = KMEANS2PY[args['ALGORITHM']['py_name']](
			n_clusters=N_CLUSTERS,
			# batch_size=BATCH_SIZE,
			sample_fraction=0.5,
			epochs_per_round=args['CLIENT_EPOCHS'],
			max_iter=args['ROUNDS'],
			server_init_method=server_init_method,
			client_init_method=client_init_method,
			true_centroids=true_centroids,
			random_state=seed,
			learning_rate=0,
			adaptive_lr=0,
			epoch_lr=0,
			momentum=0,
			reassign_min=0,
			reassign_after=0,
			verbose=VERBOSE,
			tol=TOLERANCE,
			params=args
		)

		if VERBOSE > 5:
			# print all kmeans's variables.
			pprint(vars(kmeans))

		# During the training, we also evaluate the model on the test set at each iteration
		kmeans.fit(X, Y, SPLITS, record_at=None)

		# After training, we obtain the final scores on the train/test set.
		scores = evaluate2(
			kmeans=kmeans,
			x=X, y=Y,
			splits=SPLITS,
			federated=args['IS_FEDERATED'],
			verbose=VERBOSE,
		)
		# # To save the disk storage, we only save the first repeat results.
		# if params['p0'] == 'FEMNIST' and s_i == 0:
		# 	try:
		# 		predict_n_saveimg(kmeans, x, y, SPLITS, SEED,
		# 		                  federated=params['is_federated'], verbose=VERBOSE,
		# 		                  out_dir=os.path.join(out_dir_i, f'SEED_{SEED}'),
		# 		                  params=params, is_show=is_show)
		# 	except Exception as e:
		# 		print(f'Error: {e}')
		# # traceback.print_exc()

		t2 = time.time()
		print(f'{idx_seed}th repeat with seed {seed} takes {(t2 - t1):.4f}s')

		# for each seed, we will save the results.
		history[seed] = {'initial_centroids': kmeans.initial_centroids,
		                 'true_centroids': kmeans.true_centroids,
		                 'final_centroids': kmeans.centroids,
		                 'training_iterations': kmeans.training_iterations,
		                 'history': kmeans.history,
		                 'scores': scores, 'duration': t2 - t1}
		if dataset_name != 'FEMNIST':
			try:
				# # save the current 'history' to disk before plotting.
				seed_file = os.path.join(args['OUT_DIR'], f'SEED_{seed}', f'~history.dat')
				dump(history[seed], out_file=seed_file)
				save_history2txt(history[seed], out_file=seed_file + '.txt')
			except Exception as e:
				print(f'save_history2txt() fails when SEED={seed}, Error: {e}')
		if VERBOSE >= 2:
			pprint(f'seed:{seed}, '
			       f'scores:{scores}')
	try:
		results_avg = vis.visualize.stats_single(history)
	except Exception as e:
		traceback.print_exc()
		results_avg = {}
	# save the current 'history' to disk before plotting.
	history['results_avg'] = results_avg
	dump(history, out_file=history_file)

	try:
		vis.visualize.plot_single(history)
	except Exception as e:
		traceback.print_exc()
	# out_file = os.path.join( args['OUT_DIR'], f'varied_clients-Server_{server_init_method}-Client_{client_init_method}')
	# print(out_file)
	# dump(stats, out_file=out_file + '.dat')

	# # dump(histories, out_file=out_file + '-histories.dat')
	# with open(history_file + '-histories.txt', 'w') as file:
	# 	file.write(json.dumps(history, cls=NumpyEncoder))  # use `json.loads` to do the reverse

	return history_file



