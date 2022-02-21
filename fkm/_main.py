# Email: Kun.bj@outlook.com
import copy
import json
import os
import shutil
import time
import traceback
from collections import Counter
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from fkm.clustering.my_kmeans import KMeans
from fkm.datasets.gen_dummy import load_federated
from fkm.utils.utils_func import plot_centroids, dump, save_image2disk, predict_n_saveimg, \
	plot_metric_over_time_2gaussian, plot_metric_over_time_femnist, obtain_true_centroids, \
	plot_centroids_diff_over_time, history2movie
from fkm.utils.utils_func import timer
from fkm.utils.utils_stats import evaluate2



def plot_2gaussian(X1, y1, X2, y2, params, title=''):
	# Plot init seeds along side sample data
	fig, ax = plt.subplots()
	# colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
	colors = ["r", "g", "b", "m", 'black']
	ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='centroid_1')
	p = np.mean(X1, axis=0)
	ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
	offset = 0.3
	# xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
	xytext = (p[0] - offset, p[1] - offset)
	# print(xytext)
	ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
	            ha='center', va='center',  # textcoords='offset points',
	            bbox=dict(facecolor='none', edgecolor='b', pad=1),
	            arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
	                            connectionstyle="angle3, angleA=90,angleB=0"))
	# angleA : starting angle of the path
	# angleB : ending angle of the path

	ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='centroid_2')
	p = np.mean(X2, axis=0)
	ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
	offset = 0.3
	# xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
	xytext = (p[0] + offset, p[1] - offset)
	ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
	            ha='center', va='center',  # textcoords='offset points', va='bottom',
	            bbox=dict(facecolor='none', edgecolor='red', pad=1),
	            arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
	                            connectionstyle="angle3, angleA=90,angleB=0"))

	ax.axvline(x=0, color='k', linestyle='--')
	ax.axhline(y=0, color='k', linestyle='--')
	ax.legend(loc='upper right')
	plt.title(params['p1'].replace(':', '\n') + f':{title}')
	# # plt.xlim([-2, 15])
	# # plt.ylim([-2, 15])
	plt.xlim([-6, 6])
	plt.ylim([-6, 6])
	# # plt.xticks([])
	# # plt.yticks([])
	plt.tight_layout()
	if not os.path.exists(params['out_dir']):
		os.makedirs(params['out_dir'])
	f = os.path.join(params['out_dir'], params['p1'] + '-' + params['normalize_method']+'.png')
	plt.savefig(f, dpi=600, bbox_inches='tight')
	plt.show()

def plot_3gaussian(X1, y1, X2, y2, X3, y3, params, title=''):
	# Plot init seeds along side sample data
	fig, ax = plt.subplots()
	# colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
	colors = ["r", "g", "b", "m", 'black']
	ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='centroid_1')
	p = np.mean(X1, axis=0)
	ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
	offset = 0.3
	# xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
	xytext = (p[0] - offset, p[1] - offset)
	# print(xytext)
	ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
	            ha='center', va='center',  # textcoords='offset points',
	            bbox=dict(facecolor='none', edgecolor='b', pad=1),
	            arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
	                            connectionstyle="angle3, angleA=90,angleB=0"))
	# angleA : starting angle of the path
	# angleB : ending angle of the path

	ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='centroid_2')
	p = np.mean(X2, axis=0)
	ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
	offset = 0.3
	# xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
	xytext = (p[0] + offset, p[1] - offset)
	ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
	            ha='center', va='center',  # textcoords='offset points', va='bottom',
	            bbox=dict(facecolor='none', edgecolor='red', pad=1),
	            arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
	                            connectionstyle="angle3, angleA=90,angleB=0"))

	ax.scatter(X3[:, 0], X3[:, 1], c=colors[2], marker="o", s=10, alpha=0.3, label='centroid_3')
	p = np.mean(X3, axis=0)
	ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
	offset = 0.3
	# xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
	xytext = (p[0] + offset, p[1] - offset)
	ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
	            ha='center', va='center',  # textcoords='offset points', va='bottom',
	            bbox=dict(facecolor='none', edgecolor='red', pad=1),
	            arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
	                            connectionstyle="angle3, angleA=90,angleB=0"))

	ax.axvline(x=0, color='k', linestyle='--')
	ax.axhline(y=0, color='k', linestyle='--')
	ax.legend(loc='upper right')
	plt.title(params['p1'].replace(':', '\n') + f':{title}')
	# # plt.xlim([-2, 15])
	# # plt.ylim([-2, 15])
	plt.xlim([-6, 6])
	plt.ylim([-6, 6])
	# # plt.xticks([])
	# # plt.yticks([])
	plt.tight_layout()
	if not os.path.exists(params['out_dir']):
		os.makedirs(params['out_dir'])
	f = os.path.join(params['out_dir'], params['p1'] + '-' + params['normalize_method']+'.png')
	plt.savefig(f, dpi=600, bbox_inches='tight')
	plt.show()


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


def normalize(raw_x, raw_y, raw_true_centroids, splits, params, is_federated=False):
	"""
	Only for diagonal covariance matrix:
		even for federated kmeans, we still can get global mean and std from each client data.
		Based on that, for centralized and federated kmeans, we can use the same standscaler.

	Parameters
	----------
	raw_x
	raw_y
	raw_true_centroids
	splits
	params
	is_federated

	Returns
	-------

	"""
	is_show = params['is_show']
	normalize_method = params['normalize_method']
	# do normalization
	if normalize_method == 'std':
		# collects all clients' data together
		x = copy.deepcopy(raw_x)
		y = copy.deepcopy(raw_y)
		new_true_centroids = copy.deepcopy(raw_true_centroids)
		for spl in splits:  # train and test
			x[spl] = np.concatenate(x[spl], axis=0)
		global_stdscaler = StandardScaler()
		global_stdscaler.fit(x['train'])
		for spl in splits: # train and test
			new_true_centroids[spl] = global_stdscaler.transform(raw_true_centroids[spl])

		if is_federated == False:   # for centralized normalization
			new_x = copy.deepcopy(raw_x)
			new_y = copy.deepcopy(raw_y)
			for i_, _ in enumerate(new_x['train']):
				new_x['train'][i_] = global_stdscaler.transform(new_x['train'][i_])
				new_x['test'][i_] = global_stdscaler.transform(new_x['test'][i_])
			params['stdscaler'] = global_stdscaler

			if is_show:
				if '3GAUSSIANS' in params['p0']:  # for plotting
					plot_3gaussian(new_x['train'][0], new_y['train'], new_x['train'][1], new_y['train'][1],
					          new_x['train'][2], new_y['train'][2], params, title='std')
		else:   # federated kmeans
			# for each client, we can get mean and std and then use them to get global_std
			new_x = copy.deepcopy(raw_x)
			new_y = copy.deepcopy(raw_y)
			stds = [[]] * len(new_x['train']) # get number of clients
			N = 0
			for i_, _ in enumerate(new_x['train']): # for each client
				data = new_x['train'][i_]
				n = len(data)
				stds[i_] = (n, np.mean(data, axis=0), np.std(data, axis=0))
				N += n

			dim = new_x['train'][0].shape[1]
			global_mean = [[]] * dim
			global_std = [[]] * dim
			# the following method only works for diagonal covariance matrix
			for i_ in range(dim):
				# compute global mean and std given each client's mean and std
				# global_mean = 1/N * (\sum client1  + \sum client2 + \sum client3)
				# 			  = 1/N * (n1 * mu1 + n2 * mu2 + n3 * mu3)
				global_mean[i_] = sum([n[i_]* mu[i_] for n, mu, s in stds])/ N

				# global_var = E(x-mu)**2 = (1/N * (\sum x**2)) - mu**2
				# 					   = (1/N * (\sum client1**2 + \sum client2**2+ \sum client3**2)) - global_mean **2
				#                      = (1/N * (n1 * var1 + n2 * var2 + n3 * var3)) - global_mean**2
				#                      = (1/N * (n1 * std1**2 + n2 * std2** + n3 * std3**2)) - global_mean**2
				global_std[i_] =  (sum([n[i_] * s[i_]**2 for n, mu, s in stds]) / N  - global_mean[i_]**2) ** (1/2)

			# for each client, then normalize its data use the global mean and std
			global_stdscaler2 = StandardScaler()
			global_stdscaler2.mean_ = global_mean
			global_stdscaler2.scale_ = global_std

			for i_, _ in enumerate(new_x['train']):
				new_x['train'][i_] = global_stdscaler2.transform(new_x['train'][i_])
				new_x['test'][i_] = global_stdscaler2.transform(new_x['test'][i_])
			params['stdscaler'] = global_stdscaler2

			if is_show:
				if '3GAUSSIANS' in params['p0']:  # for plotting
					plot_3gaussian(new_x['train'][0], new_y['train'], new_x['train'][1], new_y['train'][1],
					          new_x['train'][2], new_y['train'][2], params, title='std')

	else:
		new_x, new_y, new_true_centroids = raw_x, raw_y, raw_true_centroids

	return new_x, new_y, new_true_centroids



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
	is_show = params['is_show']
	normalize_method = params['normalize_method']
	# do normalization
	if normalize_method == 'std':
		# collects all clients' data together and get global stdscaler
		x = copy.deepcopy(raw_x)
		y = copy.deepcopy(raw_y)
		new_true_centroids = copy.deepcopy(raw_true_centroids)
		for spl in splits:  # train and test
			x[spl] = np.concatenate(x[spl], axis=0)

		global_stdscaler = StandardScaler() # we can get the same global_stdscaler using each client mean and std.
		global_stdscaler.fit(x['train'])
		for spl in splits: # train and test
			new_true_centroids[spl] = global_stdscaler.transform(new_true_centroids[spl])

		# normalize data
		new_x = copy.deepcopy(raw_x)
		new_y = copy.deepcopy(raw_y)
		for i_, _ in enumerate(new_x['train']):
			new_x['train'][i_] = global_stdscaler.transform(new_x['train'][i_])
			new_x['test'][i_] = global_stdscaler.transform(new_x['test'][i_])

		if is_show:
			if '2GAUSSIANS' in params['p0']:  # for plotting
				plot_2gaussian(new_x['train'][0], new_y['train'], new_x['train'][1], new_y['train'][1],
				           params, title='std')
			elif '3GAUSSIANS' in params['p0']:  # for plotting
				plot_3gaussian(new_x['train'][0], new_y['train'], new_x['train'][1], new_y['train'][1],
				          new_x['train'][2], new_y['train'][2], params, title='std')
			elif '4GAUSSIANS' in params['p0']:  # for plotting
				plot_2gaussian(new_x['train'][0], new_y['train'], new_x['train'][1], new_y['train'][1],
				           params, title='std')
	else:
		new_x, new_y, new_true_centroids = raw_x, raw_y, raw_true_centroids
		# collects all clients' data together and get global stdscaler
		x = copy.deepcopy(raw_x)
		y = copy.deepcopy(raw_y)
		for spl in splits:  # train and test
			x[spl] = np.concatenate(x[spl], axis=0)

		global_stdscaler = StandardScaler()  # we can get the same global_stdscaler using each client mean and std.
		global_stdscaler.fit(x['train'])

	return new_x, new_y, new_true_centroids, global_stdscaler


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
	TOLERANCE = params['tolerance']  # 1e-4, 1e-6
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
	normalize_method = params['normalize_method']
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
		if os.path.exists(out_dir_i):
			shutil.rmtree(out_dir_i, ignore_errors=True)
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
		raw_true_centroids = obtain_true_centroids(raw_x, raw_y, splits, params)

		if verbose:
			# print true centroids
			for split in splits:
				true_c = raw_true_centroids[split]
				print(f'{split}_true_centroids: {true_c}')
		histories[n_clients] = {'n_clients': n_clients, 'C': C, 'raw_true_centroids': raw_true_centroids}

		if params['p0'] == 'FEMNIST':
			save_image2disk((raw_x, raw_y), out_dir_i, params)

		# do normalization
		raw_x, raw_y, raw_true_centroids, global_stdscaler = normalize(raw_x, raw_y, raw_true_centroids, splits, params)
		params['global_stdscaler'] = global_stdscaler
		print(f'after normalization, true_centroids: {raw_true_centroids} when normalize_method = {normalize_method}')

		history = {'x': raw_x, 'y': raw_y, 'results': []}
		for s_i, SEED in enumerate(seeds):  # repetitions:  to obtain average and std score.
			t1 = time.time()
			if verbose > 5:
				print(f'\n***{s_i}th repeat with seed: {SEED}:')
			x = copy.deepcopy(raw_x)
			y = copy.deepcopy(raw_y)
			true_centroids = copy.deepcopy(raw_true_centroids)

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
					tol=TOLERANCE,
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
					params=params
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
			elif params['p0'] in ['2GAUSSIANS', '3GAUSSIANS', '5GAUSSIANS']:
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
