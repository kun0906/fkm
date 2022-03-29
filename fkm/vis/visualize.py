import json
import os
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np


def stats_single(history):
	# get the average and std
	args = history['args']
	history_file = args['history_file']
	results_avg = {}
	SEEDS = history['SEEDS']
	for split in args['SPLITS']:
		metric_names = history[SEEDS[0]]['scores'][split].keys()
		if split == 'train':
			training_iterations = [history[seed]['training_iterations'] for seed in SEEDS]
			results_avg[split] = {'Iterations': (np.mean(training_iterations), np.std(training_iterations))}
		else:
			results_avg[split] = {'Iterations': ('', '')}
		for k in metric_names:
			value = [history[seed]['scores'][split][k] for seed in SEEDS]
			if k in ['labels_pred', 'labels_true']:
				results_avg[split][k] = value
				continue
			try:
				score_mean = np.mean(value)
				score_std = np.std(value)
			# score_mean = np.around(np.mean(value), decimals=3)
			# score_std = np.around(np.std(value), decimals=3)
			except Exception as e:
				print(f'Error: {e}, {split}, {k}, {value}')
				score_mean = np.nan
				score_std = np.nan
			results_avg[split][k] = (score_mean, score_std)
	if args['VERBOSE'] >= 2:
		print(f'results_avg:')
		pprint(results_avg, sort_dicts=False)
	with open(history_file + '-results_avg.json', 'w') as file:
		file.write(json.dumps(results_avg, indent=4, sort_keys=False))  # use `json.loads` to do the reverse
	with open(history_file + '-results_avg.txt', 'w') as file:
		# print out dict to file using single quote, which cannot be load by json.load (require double quote)
		pprint(results_avg, file, sort_dicts=False)

	return results_avg


def plot_single(history):
	args = history['args']
	SEED = args['SEED']
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

	title = f'Centralized KMeans with {server_init_method} initialization' if not args['IS_FEDERATED'] \
		else f'Federated KMeans with {server_init_method} (Server) and {client_init_method} (Clients)'
	out_dir_ = args['OUT_DIR']
	# fig_path = plot_centroids(history, out_dir=out_dir_i,
	#                           title=title + f'. {n_clients} Clients',
	#                           fig_name=f'M={n_clients}-Centroids', params=params, is_show=is_show)
	# fig_paths.append(fig_path)
	# if params['p0'] == 'FEMNIST':
	# 	plot_metric_over_time_femnist(history, out_dir=f'{out_dir_i}/over_time',
	# 	                              title=title + f'. {n_clients} Clients', fig_name=f'M={n_clients}',
	# 	                              params=params, is_show=is_show)
	# elif params['p0'] in ['2GAUSSIANS', '3GAUSSIANS', '5GAUSSIANS']:
	# 	plot_metric_over_time_2gaussian(history, out_dir=f'{out_dir_i}/over_time',
	# 	                                title=title + f'. {n_clients} Clients', fig_name=f'M={n_clients}',
	# 	                                params=params, is_show=is_show)
	# plot centroids_update and centroids_diff over time.
	plot_centroids_diff_over_time(history,
	                              out_dir=f'{out_dir_}/over_time',
	                              title=title + f'. {N_CLIENTS} Clients', fig_name=f'M={N_CLIENTS}-Scores',
	                              params=args, is_show=args['IS_SHOW'])


# # save history as animation
# history2movie(history, out_dir=f'{out_dir_}',
#               title=title + f'. {N_CLIENTS} Clients', fig_name=f'M={N_CLIENTS}',
#               params=args, is_show=args['IS_SHOW'])


def plot_centroids_diff_over_time(history,
                                  out_dir='',
                                  title='', fig_name='',
                                  params={}, is_show=True):
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)
	seeds = []
	initial_centroids = []
	final_centroids = []
	scores = []
	iterations = []
	training_histories = []
	true_centroids = []
	durations = []
	seeds = history['SEEDS']
	for seed in seeds:
		vs = history[seed]
		training_histories.append(vs['history'])
		initial_centroids.append(vs['initial_centroids'])
		true_centroids.append(vs['true_centroids'])
		final_centroids.append(vs['final_centroids'])
		scores.append(vs['scores'])
		iterations.append(vs['training_iterations'])
		durations.append(vs['duration'])

	######################################################################
	# save centroids to images
	nrows, ncols = 6, 5
	fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=False, figsize=(15, 15))  # width, height
	# axes = axes.reshape((nrows, ncols))
	f = os.path.join(out_dir, f'centroids_diff.png')
	colors = ["r", "g", "b", "m", 'black']
	for i, training_h in enumerate(training_histories):
		if i >= nrows * ncols: break
		seed = seeds[i]
		r, c = divmod(i, ncols)
		ax = axes[r, c]
		#
		# # training_h = [{}, {}]
		# # training_h.append({'iteration': iteration, 'centroids': centroids, 'scores': scores,
		# #                      'centroids_update': centroids_update, 'centroids_diff': centroids_diff})
		x = [vs['iteration'] for vs in training_h]
		split = 'train'
		y = [np.sum(np.square(vs['centroids_update'])) for vs in training_h]
		ax.plot(x, y, f'b-*', label=f'{split}:||centroids(t)-centroids(t-1)||')

		y = [np.sum(np.square(vs['centroids_diff'][split])) for vs in training_h]
		ax.plot(x, y, f'g-o', label=f'{split}:||centroids(t)-true||')

		# split = 'test'
		# y = [np.sum(np.square(vs['centroids_diff'][split])) for vs in training_h]
		# ax.plot(x, y, f'r-', label=f'{split}:||centroids(t)-true||')

		ax.set_title(f'Iterations: {iterations[i]}, SEED: {seed}')
		# ax.set_ylabel('')
		ax.set_xlabel('Iterations')
		if i == 0:
			ax.legend(loc="upper right")
	fig.suptitle(title + fig_name + ', centroids update / diff over time')
	plt.tight_layout()
	plt.savefig(f, dpi=600, bbox_inches='tight')
	if is_show:
		plt.show()
	# plt.clf()
	plt.close(fig)

	######################################################################
	# save updated centroids over time
	nrows, ncols = 6, 5
	# fig, axes = plt.subplots(nrows, ncols,figsize=(15, 15))  # width, height
	fig = plt.figure(figsize=(15, 15))  # width, height
	f = os.path.join(out_dir, f'centroids_updates.png')
	for i, training_h in enumerate(training_histories):
		if i >= nrows * ncols: break
		seed = seeds[i]
		# r, c = divmod(i, ncols)
		# ax = axes[r, c]
		ax = fig.add_subplot(nrows, ncols, i + 1, projection='3d')
		#
		# # training_h = [{}, {}]
		# # training_h.append({'iteration': iteration, 'centroids': centroids, 'scores': scores,
		# #                      'centroids_update': centroids_update, 'centroids_diff': centroids_diff})
		x = [vs['iteration'] for vs in training_h]
		y = [vs['centroids'] for vs in training_h]  # [centroids.shape = (K, dim), ..., ]
		true_c = true_centroids[i]['train']

		# plot the first 2 centroids
		# y[i]: only plot the first 2 centroids

		# plot the first centroid and for each centroid, only show the first 2 dimensional data.
		y1 = [y_[0][:2] for y_ in y]
		y11, y12 = list(zip(*y1))
		ax.scatter(x, y11, y12, c='r', marker='o', label=f'centroid_1?')
		p = true_c[0]
		# ax.scatter(x, y, z, c='gray', marker="x", s=50)
		ax.scatter(x[0], p[0], p[1], c='gray', marker="o", s=50)
		ax.text(x[0], p[0], p[1], f'({p[0]:.1f},{p[1]:.1f})', fontsize=8, color='b',
		        # bbox=dict(facecolor='none', edgecolor='red', pad=1),
		        ha='center', va='center')
		# final centroid
		p = y1[-1]
		# ax.scatter(x, y, z, c='gray', marker="x", s=50)
		ax.scatter(x[-1], p[0], p[1], c='r', marker="o", s=50)
		ax.text(x[-1], p[0], p[1], f'({p[0]:.1f},{p[1]:.1f})', fontsize=8, color='b',
		        # bbox=dict(facecolor='none', edgecolor='red', pad=1),
		        ha='center', va='center')

		# plot the second centroid
		y2 = [y_[1][:2] for y_ in y]
		# ax.scatter(x, y2, f'b-', label=f'centroids_2')
		y21, y22 = list(zip(*y2))
		ax.scatter(x, y21, y22, c='g', marker='x', label=f'centroid_2?')
		p = true_c[1]
		ax.scatter(x[0], p[0], p[1], c='gray', marker="x", s=50)
		# ax.annotate(f'({p[0]:.1f}, {x[0]:.1f}, {p[1]:.1f})', xy=(p[0], x[0], p[1]), xytext=(p[0], x[0], p[1]),
		#             ha='center', va='center',
		#             arrowprops=dict(arrowstyle="->", facecolor='r', shrinkA=1,
		#                             connectionstyle="angle3, angleA=90,angleB=0"))
		ax.text(x[0], p[0], p[1], f'({p[0]:.1f},{p[1]:.1f})', fontsize=8, color='r',
		        # bbox=dict(facecolor='none', edgecolor='b', pad=1),
		        ha='center', va='center')
		# final centroid
		p = y2[-1]
		# ax.scatter(x, y, z, c='gray', marker="x", s=50)
		ax.scatter(x[-1], p[0], p[1], c='g', marker="x", s=50)
		ax.text(x[-1], p[0], p[1], f'({p[0]:.1f},{p[1]:.1f})', fontsize=8, color='r',
		        # bbox=dict(facecolor='none', edgecolor='red', pad=1),
		        ha='center', va='center')
		ax.set_xlabel('Iteration')
		ax.set_ylabel('x')
		ax.set_zlabel('y')
		ax.set_title(f'Iterations: {iterations[i]}, SEED: {seed}')
		if i == 0:
			ax.legend(loc="upper right")
		plt.tight_layout()
	fig.suptitle(title + fig_name + ', centroids update')
	plt.tight_layout()
	plt.savefig(f, dpi=600, bbox_inches='tight')
	if is_show:
		plt.show()
	# plt.clf()
	plt.close(fig)


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
	f = os.path.join(params['out_dir'], params['p1'] + '-' + params['normalize_method'] + '.png')
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
	f = os.path.join(params['out_dir'], params['p1'] + '-' + params['normalize_method'] + '.png')
	plt.savefig(f, dpi=600, bbox_inches='tight')
	plt.show()


def visualize_data(args):
	pass

	return ''
