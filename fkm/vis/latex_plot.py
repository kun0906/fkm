"""
    run:
        module load anaconda3/2021.5
        cd /scratch/gpfs/ky8517/fkm/fkm
        PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 results/latex_plot.py

"""
import copy
import os
import traceback
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fkm import config


def format_column(column):
	res = []
	for i, v in enumerate(column):
		if i == 0:
			res += [v]
		elif i == column.shape[0] - 1:
			res += [float(v)]
		else:
			res += [[float(v_) for v_ in str(v).split('+/-')]]
	return pd.Series(res, index=column.index)


def plot_P_DB(ax, result_files, algorithms=['Random-WA-FKM', 'Gaussian-WA-FKM'], fig_name='diff_n',
              out_dir='results/xlsx', params={}, n1=100, is_show=False, is_legend=False):
	df = pd.DataFrame()
	percents = []
	for i, (f, p) in enumerate(result_files):
		try:
			df_ = pd.read_csv(f, header=None).iloc[:, 1:]
		except Exception as e:
			print(f, os.path.exists(os.path.abspath(f)), flush=True)
			traceback.print_exc()
			return
		percents += [p] * df_.shape[1]
		df = pd.concat([df, df_], axis=1)
	df = df.T
	df['Percent'] = percents
	df.columns = ['Algorithm', 'Training Iterations', 'Training DB Score', 'Training Silhouette',
	              'Training Euclidean Distance',
	              'Testing Iterations', 'Testing DB Score', 'Testing Silhouette', 'Testing Euclidean Distance', '',
	              'Percent']
	print(df)
	df.reset_index(drop=True, inplace=True)
	df = df.apply(format_column, axis=1)
	# https://stackoverflow.com/questions/57485426/markers-are-not-visible-in-seaborn-plot
	# markers= is only useful when you also specify a style= parameter.
	# axes = sns.lineplot(data=df, x='Percent', y='Training DB Score', hue=df['Algorithm'], err_style="bars",
	#              style="Algorithm", markers = True)
	# markers = ['.', '+', 'o', '^'])
	# ax.fill_between(df['Percent'], 0, 1, alpha=0.2)
	x = df['Percent'].unique()
	colors = ['g', 'b', 'm', 'y']
	for i, alg in enumerate(algorithms):
		y, yerr = list(zip(*df[df['Algorithm'] == alg]['Training DB Score']))
		print(alg, yerr, x, y)
		ax.errorbar(x, y, yerr=yerr, label=alg, lw=2, color=colors[i], ecolor='r', elinewidth=1, capsize=2)
	if is_legend:
		ax.legend(fontsize=7)
	# plt.xlim([0, 0.52])
	# plt.ylim([0, 1])
	ax.set_ylim([0., 1.0])
	ax.set_xlabel(f'P (N1={n1})')
	ax.set_ylabel('DB Score')


# plt.tight_layout()
#
# if not os.path.exists(out_dir):
# 	os.makedirs(out_dir)
# f = os.path.join(out_dir, f'{fig_name}.png')
# plt.savefig(f, dpi=600, bbox_inches='tight')
# if is_show:
# 	plt.show()

#
# def plot_N_P(ax, result_files, algorithms=['Random-WA-FKM', 'Gaussian-WA-FKM'], fig_name='diff_n', CASE='',
#              out_dir='results/xlsx', params={}, p=0.00, is_show=False, is_legend=False, y_name='Training Iterations'):
# 	df = pd.DataFrame()
# 	percents = []
# 	for i, (f, n1) in enumerate(result_files):
# 		try:
# 			df_ = pd.read_csv(f, header=None).iloc[:, 1:]
# 		except Exception as e:
# 			print(f, os.path.exists(os.path.abspath(f)), flush=True)
# 			traceback.print_exc()
# 			return
# 		percents += [n1] * df_.shape[1]
# 		df = pd.concat([df, df_], axis=1)
# 	df = df.T
# 	df['n1'] = percents
# 	df.columns = ['Algorithm', 'Training Iterations', 'Training DB Score', 'Training Silhouette',
# 	              'Training Euclidean Distance',
# 	              'Testing Iterations', 'Testing DB Score', 'Testing Silhouette', 'Testing Euclidean Distance', '',
# 	              'n1']
# 	print(df)
# 	df.reset_index(drop=True, inplace=True)
# 	df = df.apply(format_column, axis=1)
# 	# https://stackoverflow.com/questions/57485426/markers-are-not-visible-in-seaborn-plot
# 	# markers= is only useful when you also specify a style= parameter.
# 	# axes = sns.lineplot(data=df, x='Percent', y='Training DB Score', hue=df['Algorithm'], err_style="bars",
# 	#              style="Algorithm", markers = True)
# 	# markers = ['.', '+', 'o', '^'])
# 	# ax.fill_between(df['Percent'], 0, 1, alpha=0.2)
# 	x = df['n1'].unique()
# 	colors = ['g', 'b', 'c', 'm', 'y', 'k']
# 	# b: blue.
# 	# g: green.
# 	# r: red.
# 	# c: cyan.
# 	# m: magenta.
# 	# y: yellow.
# 	# k: black.
# 	# w: white.
#
# 	# y_name = 'Training Iterations'
# 	for i, alg in enumerate(algorithms):
# 		if y_name == 'Training DB Score':
# 			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
# 		elif y_name == 'Training Iterations':
# 			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
# 		elif y_name == 'Training Silhouette':
# 			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
# 		elif y_name == 'Training Euclidean Distance':
# 			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
#
# 		elif y_name == 'Testing DB Score':
# 			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
# 		elif y_name == 'Testing Silhouette':
# 			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
# 		elif y_name == 'Testing Euclidean Distance':
# 			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
#
# 		print(alg, yerr, x, y)
# 		ax.errorbar(x, y, yerr=yerr, linestyle='-', marker='o', label=alg2label[alg], lw=2, color=colors[i],
# 		            ecolor=colors[i], elinewidth=1, capsize=2)
#
# 	if is_legend:
# 		ax.legend(fontsize=7)
#
# 	fontsize = 13
# 	if y_name == 'Training DB Score':
# 		if CASE == 'Case 2': ax.set_ylim([0.5, 0.9])
# 		ax.set_ylabel('DB Score', fontsize=fontsize)
# 	elif y_name == 'Training Iterations':
# 		if CASE == 'Case 2': ax.set_ylim([5, 75])  # for Case 2
# 		if CASE == 'Case 3': ax.set_ylim([5, 40])  # for Case 3
# 		# ax.set_ylabel('Training $T$', fontsize=fontsize)
# 		ax.set_ylabel('Convergence $T$', fontsize=fontsize)
# 	elif y_name == 'Training Silhouette':
# 		if CASE == 'Case 2':  ax.set_ylim([0.4, 0.62])
# 		ax.set_ylabel('Silhouette', fontsize=fontsize)
# 	elif y_name == 'Training Euclidean Distance':
# 		# ax.set_ylim([0.43, 0.62])
# 		ax.set_ylabel('Euclidean Distance', fontsize=fontsize)
#
# 	elif y_name == 'Testing DB Score':
# 		if CASE == 'Case 2': ax.set_ylim([0.5, 0.9])
# 		ax.set_ylabel('Testing DB', fontsize=fontsize)
# 	elif y_name == 'Testing Silhouette':
# 		if CASE == 'Case 2': ax.set_ylim([0.4, 0.62])
# 		ax.set_ylabel('Testing SC', fontsize=fontsize)
# 	elif y_name == 'Testing Euclidean Distance':
# 		if CASE == 'Case 2': ax.set_ylim([0.43, 0.62])
# 		ax.set_ylabel('Testing $\overline{WCSS}$', fontsize=fontsize)
#
# 	ax.set_xlabel(f'$N_1$', fontsize=fontsize)
#

def plot_P(ax, result_files, alg2abbrev, is_legend=False, y_name='Training Iterations', CASE='Case 1'):
	df = pd.DataFrame()
	percents = []
	for i, args in enumerate(result_files):
		try:
			f = args['csv_file']
			p = args['p']
			df_ = pd.read_csv(f, header=None, skiprows=[0])  # .iloc[:, 1:]
			# df_ = pd.read_csv(f)  # .iloc[:, 1:]
			flg = np.all(df_.iloc[:, 15] == df_.iloc[:, 16])
			if not flg:
				msg = f'n_clusters == n_clusters_pred ? {flg}'
				print(msg)
				return msg
			df_ = df_.iloc[:, 0:15]
			df_['p'] = [p] * df_.shape[0]
		except Exception as e:
			traceback.print_exc()
			return
		# percents += [p] * df_.shape[0]
		df = pd.concat([df, df_], axis=0)
	df.columns = ['Algorithm', 'Iterations', 'Durations', 'DB', 'db_normalized',	'db_weighted',	'db_weighted2',
	              'Silhouette', 'Sil_weighted',	'CH',	'Euclidean', 'ARI', 'AMI', 'FM', 'VM',
	              # 'n_clusters', 'n_clusters_pred',
	              'p']

	# print(df)
	df.reset_index(drop=True, inplace=True)
	# df = df.apply(format_column, axis=1)
	df.iloc[:, :] = df.iloc[:, :].apply(format_column, axis=1)
	# https://stackoverflow.com/questions/57485426/markers-are-not-visible-in-seaborn-plot
	# markers= is only useful when you also specify a style= parameter.
	# axes = sns.lineplot(data=df, x='Percent', y='Training DB Score', hue=df['Algorithm'], err_style="bars",
	#              style="Algorithm", markers = True)
	# markers = ['.', '+', 'o', '^'])
	# ax.fill_between(df['Percent'], 0, 1, alpha=0.2)
	x = df['p'].unique()
	colors = ['g', 'b', 'tab:brown', 'c', 'm', 'y', 'k']
	# b: blue.
	# g: green.
	# r: red.
	# c: cyan.
	# m: magenta.
	# y: yellow.
	# k: black.
	# w: white.

	# y_name = 'Training Iterations'
	# algorithms = ['KM++-CKM', 'Random-WA-FKM', 'C-Random-WA-FKM', 'C-KM++-WA-FKM', 'C-Random-GD-FKM',
	#               'C-KM++-GD-FKM']
	# ABBRV2ALG  = {k:(v, i) for i, (k, v) in enumerate(ABBRV2ALG.items())}
	for i, (alg_true, alg_label) in enumerate(alg2abbrev.items()):
		print(alg_true, alg_label)
		y, yerr = list(zip(*df[df['Algorithm'] == alg_true][y_name]))
		ax.errorbar(x, y, yerr=yerr, linestyle='-', marker='o', label=alg_label, lw=2, color=colors[i],
		            ecolor=colors[i], elinewidth=1, capsize=2, alpha=1)

	if is_legend:
		ax.legend(fontsize=7, loc='upper right')

	fontsize = 13
	if y_name == 'Iterations':
		# if CASE == 'Case 1': ax.set_ylim([8, 30])  # for Case 1
		# if CASE == 'Case 4': ax.set_ylim([3, 80])  # for Case 4
		# if CASE == 'Case 5': ax.set_ylim([3, 40])  # for Case 5
		# ax.set_ylabel('Training $T$', fontsize=fontsize)
		ax.set_ylabel('Convergence $T$', fontsize=fontsize)
	# elif y_name == 'Training DB Score':
	# 	ax.set_ylim([0.5, 0.9])
	# 	ax.set_ylabel('$DB$', fontsize=fontsize)
	# elif y_name == 'Training Silhouette':
	# 	ax.set_ylim([0.4, 0.62])
	# 	ax.set_ylabel('Silhouette', fontsize=fontsize)
	# elif y_name == 'Training Euclidean Distance':
	# 	ax.set_ylim([0.43, 0.62])
	# 	ax.set_ylabel('Euclidean Distance', fontsize=fontsize)

	elif y_name == 'ARI':
		# if CASE == 'Case 1': ax.set_ylim([0.4, 0.8])
		# if CASE == 'Case 4': ax.set_ylim([0.3, 0.9])
		# if CASE == 'Case 5': ax.set_ylim([0.6, 1.2])
		ax.set_ylabel('ARI', fontsize=fontsize)
	elif y_name == 'AMI':
		# if CASE == 'Case 1': ax.set_ylim([0.4, 0.8])
		# if CASE == 'Case 4': ax.set_ylim([0.4, 0.85])
		# if CASE == 'Case 5': ax.set_ylim([0.4, 0.6])
		ax.set_ylabel('AMI', fontsize=fontsize)
	elif y_name == 'VM':
		# if CASE == 'Case 1': ax.set_ylim([0.3, 0.8])
		# if CASE == 'Case 4': ax.set_ylim([0., 0.22])
		# if CASE == 'Case 5': ax.set_ylim([0.3, 1.6])
		ax.set_ylabel('VM', fontsize=fontsize)
	elif y_name == 'DB':
		# if CASE == 'Case 1': ax.set_ylim([0.4, 0.8])
		# if CASE == 'Case 4': ax.set_ylim([0.3, 0.9])
		# if CASE == 'Case 5': ax.set_ylim([0.6, 1.2])
		ax.set_ylabel('DB', fontsize=fontsize)
	elif y_name == 'Silhouette':
		# if CASE == 'Case 1': ax.set_ylim([0.4, 0.8])
		# if CASE == 'Case 4': ax.set_ylim([0.4, 0.85])
		# if CASE == 'Case 5': ax.set_ylim([0.4, 0.6])
		ax.set_ylabel('SC', fontsize=fontsize)
	elif y_name == 'Euclidean':
		# if CASE == 'Case 1': ax.set_ylim([0.3, 0.8])
		# if CASE == 'Case 4': ax.set_ylim([0., 0.22])
		# if CASE == 'Case 5': ax.set_ylim([0.3, 1.6])
		ax.set_ylabel('$\overline{WCSS}$', fontsize=fontsize)

	if CASE == 'Case 1':
		ax.set_xlabel(f'$P$', fontsize=fontsize)
	elif CASE == 'Case 2':
		ax.set_xlabel(f'$N_3$', fontsize=fontsize)


def get_df(result_files):
	df = pd.DataFrame()
	percents = []
	for i, (f, n1) in enumerate(result_files):
		try:
			df_ = pd.read_csv(f, header=None)
		except Exception as e:
			print(f, os.path.exists(os.path.abspath(f)), flush=True)
			traceback.print_exc()
			return
		percents += [n1] * df_.shape[1]
		df = pd.concat([df, df_], axis=1)
	return df


# df = df.T
# df['n1'] = percents
# df.columns = ['Algorithm', 'Training Iterations', 'Training DB Score', 'Training Silhouette',
#               'Training Euclidean Distance',
#               'Testing Iterations', 'Testing DB Score', 'Testing Silhouette', 'Testing Euclidean Distance', '',
#               'n1']
# print(df)
# df.reset_index(drop=True, inplace=True)
# df = df.apply(format_column, axis=1)
# # https://stackoverflow.com/questions/57485426/markers-are-not-visible-in-seaborn-plot
# # markers= is only useful when you also specify a style= parameter.
# # axes = sns.lineplot(data=df, x='Percent', y='Training DB Score', hue=df['Algorithm'], err_style="bars",
# #              style="Algorithm", markers = True)
# # markers = ['.', '+', 'o', '^'])
# # ax.fill_between(df['Percent'], 0, 1, alpha=0.2)
# x = df['n1'].unique()
# return x, df


def barplot_N_P(bs_df, ax, result_files, algorithms=['Random-WA-FKM', 'Gaussian-WA-FKM'], fig_name='diff_n',
                out_dir='results/xlsx', params={}, p=0.00, is_show=False, is_legend=False,
                y_name='Training Iterations'):
	x, df = get_df(result_files)
	colors = ['g', 'b', 'c', 'm', 'y', 'k']
	# y_name = 'Training Iterations'
	res = []
	for i, alg in enumerate(algorithms):
		# if y_name ==  'Training DB Score':
		# 	y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		# 	bs_y, bs_yerr = list(zip(*bs_df[bs_df['Algorithm'] == alg][y_name]))
		# elif y_name == 'Training Iterations':
		# 	y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		# 	bs_y, bs_yerr = list(zip(*bs_df[bs_df['Algorithm'] == alg][y_name]))
		# elif y_name ==  'Training Silhouette':
		# 	y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		# 	bs_y, bs_yerr = list(zip(*bs_df[bs_df['Algorithm'] == alg][y_name]))
		# elif y_name == 'Training Euclidean Distance':
		# 	y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		# 	bs_y, bs_yerr = list(zip(*bs_df[bs_df['Algorithm'] == alg][y_name]))
		#
		# elif y_name == 'Testing DB Score':
		# 	y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		# 	bs_y, bs_yerr = list(zip(*bs_df[bs_df['Algorithm'] == alg][y_name]))
		# elif y_name ==  'Testing Silhouette':
		# 	y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		# 	bs_y, bs_yerr = list(zip(*bs_df[bs_df['Algorithm'] == alg][y_name]))
		# elif y_name == 'Testing Euclidean Distance':
		# 	y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		# 	bs_y, bs_yerr = list(zip(*bs_df[bs_df['Algorithm'] == alg][y_name]))

		y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		bs_y, bs_yerr = list(zip(*bs_df[bs_df['Algorithm'] == alg][y_name]))
		l = min(len(y), len(bs_y))
		y_diff = [v1 - v2 for v1, v2 in zip(y[:l], bs_y[:l])]
		res += y_diff
		print(alg, yerr, x, y, y_diff)
	# ax.errorbar(x, y, yerr=yerr, linestyle = '-', marker = 'o', label=alg, lw=2, color=colors[i], ecolor='r', elinewidth=1, capsize=2)
	x = range(len(algorithms))
	ax.bar(x, res, color=colors)
	# ax.axvline(x=0, color='k', linestyle='--')
	ax.axhline(y=0, color='k', linestyle='--')

	fontsize = 13
	ax.set_xticks(x)
	ax.set_xticklabels(algorithms, rotation=90, ha='right', fontsize=fontsize)

	if is_legend:
		ax.legend(fontsize=7)

	if y_name == 'Training DB Score':
		ax.set_ylim([-25, 0.0])
		ax.set_ylabel('DB Score', fontsize=fontsize)
	elif y_name == 'Training Iterations':
		# ax.set_ylim([5, 40])
		# ax.set_ylabel('Training Iterations', fontsize=fontsize)
		ax.set_ylabel('Training ${T}$' + f': $\Delta$=({p}-0.0)', fontsize=fontsize)
	elif y_name == 'Training Silhouette':
		# ax.set_ylim([0.4, 0.62])
		ax.set_ylabel('Silhouette', fontsize=fontsize)
	elif y_name == 'Training Euclidean Distance':
		# ax.set_ylim([0.43, 0.62])
		ax.set_ylabel('Euclidean Distance', fontsize=fontsize)

	elif y_name == 'Testing DB Score':
		ax.set_ylim([-0.02, 0.03])
		ax.set_ylabel('${DB}$' + f': $\Delta$=({p}-0.0)', fontsize=fontsize)
	elif y_name == 'Testing Silhouette':
		ax.set_ylim([-0.025, 0.02])
		ax.set_ylabel('${SC}$:' + f': $\Delta$=({p}-0.0)', fontsize=fontsize)
	elif y_name == 'Testing Euclidean Distance':
		ax.set_ylim([-0.05, 0.06])
		ax.set_ylabel('${\overline{WCSS}}$:' + f': $\Delta$=({p}-0.0)', fontsize=fontsize)

	ax.set_xlabel(f'$P={p}$', fontsize=fontsize)


def plot_guassian_P(in_dir, out_dir, alg2abbrev, dataset, n_repeats=10, tolerance=1e-4):
	dataset_name = dataset['name']
	dataset_detail = dataset['detail']
	n_clusters = dataset['n_clusters']
	n_clients = dataset['n_clients']
	# plot case 1: fixed  n1:
	for metrics in [['Iterations'],
	                ['ARI', 'AMI', 'VM'],
	                ['DB', 'Silhouette', 'Euclidean']]:
		if 'Iterations' in metrics[0]:
			fig, axes = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(5, 3))  # (width, height)
			axes = np.asarray(axes).reshape((1, 1))
			fig_name = f'{dataset_name}_K_{n_clusters}_M_{n_clients}_R_{n_repeats}_fixed_n_diff_p_training_iterations'
		elif 'ARI' in metrics[0]:
			fig, axes = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(15, 3))  # (width, height)
			axes = np.asarray(axes).reshape((1, 3))
			fig_name = f'{dataset_name}_K_{n_clusters}_M_{n_clients}_R_{n_repeats}_fixed_n_diff_p_ground_truth_metrics'
		else:
			fig, axes = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(15, 3))  # (width, height)
			axes = axes.reshape((1, 3))
			fig_name = f'{dataset_name}_K_{n_clusters}_M_{n_clients}_R_{n_repeats}_fixed_n_diff_p_metrics'

		for i, y_name in enumerate(metrics):
			# p = float(f'{p:.2f}')
			i_, j_ = divmod(i, 4)
			if i_ == 0 and j_ == 0:
				is_legend = True
			else:
				is_legend = False
			ax = axes[i_][j_]
			try:
				# for each p, plot n1 vs DB
				result_files = []
				n1 = n2 = n3 = 5000
				ratios = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]  # [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]
				for p in ratios:
					p = float(f'{p:.2f}')
					result_files.append((
						f'{in_dir}/{dataset_name}/n1_5000-sigma1_0.3_0.3+n2_5000-sigma2_0.3_0.3+n3_5000-sigma3_0.3_0.3:ratio_{p:.2f}:diff_sigma_n|M_{n_clients}|K_{n_clusters}|SEED_42/R_{n_repeats}|kmeans++|None|{tolerance}|std.csv',
						p))
				print(y_name, p, result_files)
				plot_P(ax, result_files, alg2abbrev, is_legend, y_name, CASE='Case 1')
			except Exception as e:
				traceback.print_exc()
				print(e, flush=True)

		plt.tight_layout()

		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		f = os.path.join(out_dir, f'{fig_name}.png')
		plt.savefig(f, dpi=600, bbox_inches='tight')
		print(os.path.abspath(f))
		if is_show:
			plt.show()


def plot_guassian_N(in_dir, out_dir, alg2abbrev, dataset, n_repeats=10, tolerance=1e-4):
	dataset_name = dataset['name']
	dataset_detail = dataset['detail']
	n_clusters = dataset['n_clusters']
	n_clients = dataset['n_clients']
	# plot case 2: varied n1 and fixed_p
	for metrics in [['Iterations'],
	                ['ARI', 'AMI', 'VM'],
	                ['DB', 'Silhouette', 'Euclidean']]:
		if 'Iterations' in metrics[0]:
			fig, axes = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(5, 3))  # (width, height)
			axes = np.asarray(axes).reshape((1, 1))
			fig_name = f'{dataset_name}_K_{n_clusters}_M_{n_clients}_R_{n_repeats}_diff_n_fixed_p_training_iterations'
		elif 'ARI' in metrics[0]:
			fig, axes = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(15, 3))  # (width, height)
			axes = np.asarray(axes).reshape((1, 3))
			fig_name = f'{dataset_name}_K_{n_clusters}_M_{n_clients}_R_{n_repeats}_diff_n_fixed_p_ground_truth_metrics'
		else:
			fig, axes = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(15, 3))  # (width, height)
			axes = axes.reshape((1, 3))
			fig_name = f'{dataset_name}_K_{n_clusters}_M_{n_clients}_R_{n_repeats}_diff_n_fixed_p_metrics'

		for i, y_name in enumerate(metrics):
			# p = float(f'{p:.2f}')
			i_, j_ = divmod(i, 4)
			if i_ == 0 and j_ == 0:
				is_legend = True
			else:
				is_legend = False
			ax = axes[i_][j_]
			try:
				# for each p, plot n1 vs DB
				result_files = []
				# n1 = n2 = n3 = 5000
				# ratios = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]
				n1s = [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]
				for n1 in n1s:
					p = 0.0
					p = float(f'{p:.2f}')
					result_files.append((
						f'{in_dir}/{dataset_name}/n1_{n1}-sigma1_0.1_0.1+n2_5000-sigma2_0.2_0.2+n3_5000-sigma3_0.3_0.3:ratio_{p:.2f}:diff_sigma_n|M_{n_clients}|K_{n_clusters}|SEED_42/R_{n_repeats}|kmeans++|None|{tolerance}|std.csv',
						n1))
				print(y_name, n1, result_files)
				plot_P(ax, result_files, alg2abbrev, is_legend, y_name, CASE='Case 2')
			except Exception as e:
				traceback.print_exc()
				print(e, flush=True)

		plt.tight_layout()

		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		f = os.path.join(out_dir, f'{fig_name}.png')
		plt.savefig(f, dpi=600, bbox_inches='tight')
		print(os.path.abspath(f))
		if is_show:
			plt.show()


def plot_real_case_P(in_dir, out_dir, alg2abbrev, csv_files):
	# dataset_name = dataset['name']
	# dataset_detail = dataset['detail']
	# n_clusters = dataset['n_clusters']
	# n_clients = dataset['n_clients']
	args = csv_files[0]
	n_repeats = args['N_REPEATS']
	# plot for real data: varied n1 and fixed_p
	for metrics in [['Iterations', 'ARI', 'Euclidean'],
	                # ['ARI', 'VM', 'Euclidean'],
	                ['DB', 'Silhouette', 'AMI',]]:
		# if 'Iterations' in metrics[0]:
		# 	fig, axes = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(5, 3))  # (width, height)
		# 	axes = np.asarray(axes).reshape((1, 1))
		# 	fig_name = f'{dataset_name}_K_{n_clusters}_M_{n_clients}_R_{n_repeats}_fixed_n_diff_p_training_iterations'
		if 'ARI' in metrics[1]:
			fig, axes = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(15, 3))  # (width, height)
			axes = np.asarray(axes).reshape((1, 3))
			fig_name = f'{dataset_name}_K_{n_clusters}_M_{n_clients}_R_{n_repeats}_fixed_n_diff_p_ground_truth_metrics'
		else:
			fig, axes = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(15, 3))  # (width, height)
			axes = axes.reshape((1, 3))
			fig_name = f'{dataset_name}_K_{n_clusters}_M_{n_clients}_R_{n_repeats}_fixed_n_diff_p_metrics'

		for i, y_name in enumerate(metrics):
			# p = float(f'{p:.2f}')
			i_, j_ = divmod(i, 4)
			if i_ == 0 and j_ == 0:
				is_legend = True
			else:
				is_legend = False
			ax = axes[i_][j_]
			try:
				plot_P(ax, csv_files, alg2abbrev, is_legend, y_name, CASE='Case 1')
			except Exception as e:
				traceback.print_exc()
				# print(e, flush=True)

		plt.tight_layout()

		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		f = os.path.join(out_dir, f'{fig_name}.png')
		plt.savefig(f, dpi=600, bbox_inches='tight')
		print(os.path.abspath(f))
		if is_show:
			plt.show()

def plot_real_case_N(in_dir, out_dir, alg2abbrev, csv_files):
	# dataset_name = dataset['name']
	# dataset_detail = dataset['detail']
	# n_clusters = dataset['n_clusters']
	# n_clients = dataset['n_clients']
	args = csv_files[0]
	n_repeats = args['N_REPEATS']

	# plot for real data: varied n1 and fixed_p
	for metrics in [['Iterations'],
	                ['ARI', 'VM', 'Euclidean'],
	                ['DB', 'Silhouette', 'AMI',]]:
		if 'Iterations' in metrics[0]:
			fig, axes = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(5, 3))  # (width, height)
			axes = np.asarray(axes).reshape((1, 1))
			fig_name = f'{dataset_name}_K_{n_clusters}_M_{n_clients}_R_{n_repeats}_diff_n_fixed_p_training_iterations'
		elif 'ARI' in metrics[0]:
			fig, axes = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(15, 3))  # (width, height)
			axes = np.asarray(axes).reshape((1, 3))
			fig_name = f'{dataset_name}_K_{n_clusters}_M_{n_clients}_R_{n_repeats}_diff_n_fixed_p_ground_truth_metrics'
		else:
			fig, axes = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(15, 3))  # (width, height)
			axes = axes.reshape((1, 3))
			fig_name = f'{dataset_name}_K_{n_clusters}_M_{n_clients}_R_{n_repeats}_diff_n_fixed_p_metrics'

		for i, y_name in enumerate(metrics):
			# p = float(f'{p:.2f}')
			i_, j_ = divmod(i, 4)
			if i_ == 0 and j_ == 0:
				is_legend = True
			else:
				is_legend = False
			ax = axes[i_][j_]
			try:
				plot_P(ax, csv_files, alg2abbrev, is_legend, y_name, CASE='Case 2')
			except Exception as e:
				traceback.print_exc()
			# print(e, flush=True)

		plt.tight_layout()

		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		f = os.path.join(out_dir, f'{fig_name}.png')
		plt.savefig(f, dpi=600, bbox_inches='tight')
		print(os.path.abspath(f))
		if is_show:
			plt.show()


if __name__ == '__main__':
	config_file = 'config.yaml'
	args = config.load(config_file)

	# in_dir = os.path.abspath('~/Downloads/xlsx')
	IN_DIR = os.path.expanduser('~/Downloads/xlsx')
	OUT_DIR = f'{IN_DIR}/latex_plot/'
	N_REPEATS =  50 #   args['N_REPEATS']
	TOLERANCE = args['TOLERANCE']
	NORMALIZE_METHOD = args['NORMALIZE_METHOD']
	IS_REMOVE_OUTLIERS = args['IS_REMOVE_OUTLIERS']
	args['OUT_DIR'] = OUT_DIR
	SEPERTOR = args['SEPERTOR']

	ALG2ABBREV = {
		f'centralized_kmeans|R_{N_REPEATS}|random|None|{TOLERANCE}|{NORMALIZE_METHOD}': 'CKM-Random',
		f'centralized_kmeans|R_{N_REPEATS}|kmeans++|None|{TOLERANCE}|{NORMALIZE_METHOD}': 'CKM++',
		# f'federated_server_init_first|R_{N_REPEATS}|random|None|{TOLERANCE}|{NORMALIZE_METHOD}': 'Server-Initialized',
		f'federated_server_init_first|R_{N_REPEATS}|min_max|None|{TOLERANCE}|{NORMALIZE_METHOD}': 'Server-MinMax',
		f'federated_client_init_first|R_{N_REPEATS}|average|random|{TOLERANCE}|{NORMALIZE_METHOD}': 'Average-Random',
		f'federated_client_init_first|R_{N_REPEATS}|average|kmeans++|{TOLERANCE}|{NORMALIZE_METHOD}': 'Average-KM++',
		f'federated_greedy_kmeans|R_{N_REPEATS}|greedy|random|{TOLERANCE}|{NORMALIZE_METHOD}': 'Greedy-Random',
		f'federated_greedy_kmeans|R_{N_REPEATS}|greedy|kmeans++|{TOLERANCE}|{NORMALIZE_METHOD}': 'Greedy-KM++',
	}
	is_show = True
	ratios = [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]

	# dataset_name = 'FEMNIST'
	dataset_name = 'NBAIOT'
	IS_PCA = True    # args['IS_PCA']
	# dataset_name = '3GAUSSIANS'

	if dataset_name == '3GAUSSIANS':
		n_clusters = 3
		n_clients = 3
		csv_files = []
		N = 5000  # total cases: 8*7
		for ratio in [0, 0.1, 0.3, 0.5] :
			for n1 in [N]:
				# for n1 in [500, 2000, 3000, 5000, 8000]:
				for sigma1 in ["0.1_0.1"]:  # sigma  = [[0.1, 0], [0, 0.1]]
					for n2 in [N]:
						for sigma2 in ["0.1_0.1"]:  # sigma  = [[0.1, 0], [0, 0.1]]
							for n3 in [N]:  # [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
								for sigma3 in ["1.0_0.1"]:  # sigma  = [[1, 0], [0, 0.1]]
									dataset_detail = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
									f = f'{IN_DIR}/{dataset_name}/{dataset_detail}|{NORMALIZE_METHOD}|PCA_{IS_PCA}|M_{n_clients}|K_{n_clusters}|REMOVE_OUTLIERS_{IS_REMOVE_OUTLIERS}/R_{N_REPEATS}|random|None|{TOLERANCE}|std.csv'
									# dataset = {'name': dataset_name, 'detail': dataset_detail,  'p': n1, 'n_clients': n_clients, 'n_clusters':n_clusters, 'csv_file':f}
									args1 = copy.deepcopy(args)
									args1['csv_file'] = f
									args1['p'] = ratio
									SEED = args1['SEED']
									args1['DATASET']['name'] = dataset_name
									args1['DATASET']['detail'] = dataset_detail
									args1['N_CLIENTS'] = n_clients
									args1['N_CLUSTERS'] = n_clusters
									N_CLIENTS = args1['N_CLIENTS']
									N_CLUSTERS = args1['N_CLUSTERS']
									args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'],
									                                                 f'M_{N_CLIENTS}',
									                                                 f'K_{N_CLUSTERS}', f'SEED_{SEED}'])
								csv_files.append(copy.deepcopy(args1))
		plot_real_case_P(IN_DIR, OUT_DIR, ALG2ABBREV, csv_files)
		exit()

		# csv_files = []
		# N = 5000  # total cases: 9*7
		# for ratio in [0.0]:  # ratios:
		# 	for n1 in [N]:
		# 		for sigma1 in ["0.1_0.1"]:  # sigma  = [[0.1, 0], [0, 0.1]]
		# 			for n2 in [N]:
		# 				for sigma2 in ["0.2_0.2"]:  # sigma  = [[0.1, 0], [0, 0.1]]
		# 					for n3 in [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
		# 						for sigma3 in ["0.3_0.3"]:  # sigma  = [[1, 0], [0, 0.1]]
		# 							dataset_detail = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
		# 							f = f'{IN_DIR}/{dataset_name}/{dataset_detail}|M_{n_clients}|K_{n_clusters}|SEED_42/R_{N_REPEATS}|kmeans++|None|{TOLERANCE}|std.csv'
		# 							args1 = copy.deepcopy(args)
		# 							args1['csv_file'] = f
		# 							args1['p'] = n3
		# 							SEED = args1['SEED']
		# 							args1['DATASET']['name'] = dataset_name
		# 							args1['DATASET']['detail'] = dataset_detail
		# 							args1['N_CLIENTS'] = n_clients
		# 							args1['N_CLUSTERS'] = n_clusters
		# 							N_CLIENTS = args1['N_CLIENTS']
		# 							N_CLUSTERS = args1['N_CLUSTERS']
		# 							args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'],
		# 							                                                 f'M_{N_CLIENTS}',
		# 							                                                 f'K_{N_CLUSTERS}', f'SEED_{SEED}'])
		# 						csv_files.append(copy.deepcopy(args1))
		# plot_real_case_N(IN_DIR, OUT_DIR, ALG2ABBREV, csv_files)
		# exit()
		#
		# csv_files = []
		# N = 5000  # total cases: 9*7
		# for ratio in [0.0]:  # ratios:
		# 	for n1 in [N]:
		# 		for sigma1 in ["0.3_0.3"]:  # sigma  = [[0.1, 0], [0, 0.1]]
		# 			for n2 in [N]:
		# 				for sigma2 in ["0.3_0.3"]:  # sigma  = [[0.1, 0], [0, 0.1]]
		# 					for n3 in [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
		# 						for sigma3 in ["1.0_0.1"]:  # sigma  = [[1, 0], [0, 0.1]]
		# 							dataset_detail = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
		# 							f = f'{IN_DIR}/{dataset_name}/{dataset_detail}|M_{n_clients}|K_{n_clusters}|SEED_42/R_{N_REPEATS}|kmeans++|None|{TOLERANCE}|std.csv'
		# 							args1 = copy.deepcopy(args)
		# 							args1['csv_file'] = f
		# 							args1['p'] = n3
		# 							SEED = args1['SEED']
		# 							args1['DATASET']['name'] = dataset_name
		# 							args1['DATASET']['detail'] = dataset_detail
		# 							args1['N_CLIENTS'] = n_clients
		# 							args1['N_CLUSTERS'] = n_clusters
		# 							N_CLIENTS = args1['N_CLIENTS']
		# 							N_CLUSTERS = args1['N_CLUSTERS']
		# 							args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'],
		# 							                                                 f'M_{N_CLIENTS}',
		# 							                                                 f'K_{N_CLUSTERS}', f'SEED_{SEED}'])
		# 						csv_files.append(copy.deepcopy(args1))
		# plot_real_case_N(IN_DIR, OUT_DIR, ALG2ABBREV, csv_files)

	elif dataset_name == '10GAUSSIANS':
		n_clients = 10
		for n_clusters in [3, 10]:
			csv_files = []
			N = 5000
			for ratio in ratios:
				for n1 in [N]:
					# for n1 in [500, 2000, 3000, 5000, 8000]:
					for sigma1 in ["0.3_0.3"]:  # sigma  = [[0.1, 0], [0, 0.1]]
						for n2 in [N]:
							for sigma2 in ["0.3_0.3"]:  # sigma  = [[0.1, 0], [0, 0.1]]
								for n3 in [N]:  # [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
									for sigma3 in ["0.3_0.3"]:  # sigma  = [[1, 0], [0, 0.1]]
										dataset_detail = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
										f = f'{IN_DIR}/{dataset_name}/{dataset_detail}|M_{n_clients}|K_{n_clusters}|SEED_42/R_{N_REPEATS}|kmeans++|None|{TOLERANCE}|std.csv'
										args1 = copy.deepcopy(args)
										args1['csv_file'] = f
										args1['p'] = ratio
										SEED = args1['SEED']
										args1['DATASET']['name'] = dataset_name
										args1['DATASET']['detail'] = dataset_detail
										args1['N_CLIENTS'] = n_clients
										args1['N_CLUSTERS'] = n_clusters
										N_CLIENTS = args1['N_CLIENTS']
										N_CLUSTERS = args1['N_CLUSTERS']
										args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'],
										                                                 f'M_{N_CLIENTS}',
										                                                 f'K_{N_CLUSTERS}',
										                                                 f'SEED_{SEED}'])
									csv_files.append(copy.deepcopy(args1))
			plot_real_case_N(IN_DIR, OUT_DIR, ALG2ABBREV, csv_files)
			# exit()

			# csv_files = []
			# N = 5000
			# for ratio in [0.0]:
			# 	for n1 in [N]:
			# 		# for n1 in [500, 2000, 3000, 5000, 8000]:
			# 		for sigma1 in ["0.1_0.1"]:
			# 			for n2 in [N]:
			# 				for sigma2 in ["0.2_0.2"]:
			# 					for n3 in [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
			# 						for sigma3 in ["0.3_0.3"]:
			# 							dataset_detail = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
			# 							f = f'{IN_DIR}/{dataset_name}/{dataset_detail}|M_{n_clients}|K_{n_clusters}|SEED_42/R_{N_REPEATS}|kmeans++|None|{TOLERANCE}|std.csv'
			# 							args1 = copy.deepcopy(args)
			# 							args1['csv_file'] = f
			# 							args1['p'] = n3
			# 							SEED = args1['SEED']
			# 							args1['DATASET']['name'] = dataset_name
			# 							args1['DATASET']['detail'] = dataset_detail
			# 							args1['N_CLIENTS'] = n_clients
			# 							args1['N_CLUSTERS'] = n_clusters
			# 							N_CLIENTS = args1['N_CLIENTS']
			# 							N_CLUSTERS = args1['N_CLUSTERS']
			# 							args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'],
			# 							                                                 f'M_{N_CLIENTS}',
			# 							                                                 f'K_{N_CLUSTERS}',
			# 							                                                 f'SEED_{SEED}'])
			# 						csv_files.append(copy.deepcopy(args1))
			# plot_real_case_N(IN_DIR, OUT_DIR, ALG2ABBREV, csv_files)
			# exit()

			# csv_files = []
			# N = 5000
			# for ratio in [0.0]:
			# 	for n1 in [N]:
			# 		# for n1 in [500, 2000, 3000, 5000, 8000]:
			# 		for sigma1 in ["0.1_0.1"]:
			# 			for n2 in [N]:
			# 				for sigma2 in ["0.1_0.1"]:
			# 					for n3 in [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
			# 						for sigma3 in ["1.0_0.1"]:
			# 							dataset_detail = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
			# 							f = f'{IN_DIR}/{dataset_name}/{dataset_detail}|M_{n_clients}|K_{n_clusters}|SEED_42/R_{N_REPEATS}|kmeans++|None|{TOLERANCE}|std.csv'
			# 							args1 = copy.deepcopy(args)
			# 							args1['csv_file'] = f
			# 							args1['p'] = n3
			# 							SEED = args1['SEED']
			# 							args1['DATASET']['name'] = dataset_name
			# 							args1['DATASET']['detail'] = dataset_detail
			# 							args1['N_CLIENTS'] = n_clients
			# 							args1['N_CLUSTERS'] = n_clusters
			# 							N_CLIENTS = args1['N_CLIENTS']
			# 							N_CLUSTERS = args1['N_CLUSTERS']
			# 							args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'],
			# 							                                                 f'M_{N_CLIENTS}',
			# 							                                                 f'K_{N_CLUSTERS}',
			# 							                                                 f'SEED_{SEED}'])
			# 						csv_files.append(copy.deepcopy(args1))
			# plot_real_case_N(IN_DIR, OUT_DIR, ALG2ABBREV, csv_files)

	elif dataset_name == 'FEMNIST':
		csv_files = []
		n_clients = 10
		n_clusters = 2
		tmp_list = []
		for n in [1, 3, 5, 10, 15, 17]:
			# there are 20 clients and each client has n users' data.
			dataset_detail = f'n_{n}:ratio_0.00:diff_sigma_n'
			f = f'{IN_DIR}/{dataset_name}/{dataset_detail}|M_{n_clients}|K_{n_clusters}|SEED_42/R_{N_REPEATS}|kmeans++|None|{TOLERANCE}|std.csv'
			args1 = copy.deepcopy(args)
			args1['csv_file'] = f
			args1['p'] = n
			SEED = args1['SEED']
			args1['DATASET']['name'] = dataset_name
			args1['DATASET']['detail'] = dataset_detail
			args1['N_CLIENTS'] = n_clients
			args1['N_CLUSTERS'] = n_clusters
			N_CLIENTS = args1['N_CLIENTS']
			N_CLUSTERS = args1['N_CLUSTERS']
			args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'],
			                                                 f'M_{N_CLIENTS}',
			                                                 f'K_{N_CLUSTERS}', f'SEED_{SEED}'])
			csv_files.append(copy.deepcopy(args1))
		plot_real_case_N(IN_DIR, OUT_DIR, ALG2ABBREV, csv_files)

	elif dataset_name == 'NBAIOT':
		# csv_files = []    # not work
		# N = 5000
		# n_clusters = 2
		# n_clients = 3
		# for ratio in ratios:
		# 	for n1 in [N]:
		# 		for n2 in [2500]:
		# 			for n3 in [2500]:
		# 				dataset_detail = f'n1_{n1}+n2_{n2}+n3_{n3}:ratio_{ratio:.2f}:diff_sigma_n'
		# 				f = f'{IN_DIR}/{dataset_name}/{dataset_detail}|{NORMALIZE_METHOD}|PCA_{IS_PCA}|M_{n_clients}|K_{n_clusters}|REMOVE_OUTLIERS_{IS_REMOVE_OUTLIERS}|R_{N_REPEATS}|kmeans++|None|{TOLERANCE}|std.csv'
		# 				args1 = copy.deepcopy(args)
		# 				args1['csv_file'] = f
		# 				args1['p'] = ratio
		# 				SEED = args1['SEED']
		# 				args1['DATASET']['name'] = dataset_name
		# 				args1['DATASET']['detail'] = dataset_detail
		# 				args1['N_CLIENTS'] = n_clients
		# 				args1['N_CLUSTERS'] = n_clusters
		# 				N_CLIENTS = args1['N_CLIENTS']
		# 				N_CLUSTERS = args1['N_CLUSTERS']
		# 				args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'],
		# 				                                                 f'M_{N_CLIENTS}',
		# 				                                                 f'K_{N_CLUSTERS}', f'SEED_{SEED}'])
		# 			csv_files.append(copy.deepcopy(args1))
		# plot_real_case_P(IN_DIR, OUT_DIR, ALG2ABBREV, csv_files)
		# exit()

		# csv_files = []    # work
		# N = 5000
		# n_clusters = 3
		# n_clients = 3
		# for ratio in ratios:
		# 	for n1 in [N]:
		# 		for n2 in [2500]:
		# 			for n3 in [2500]:
		# 				dataset_detail = f'n1_{n1}+n2_{n2}+n3_{n3}:ratio_{ratio:.2f}:diff_sigma_n'
		# 				f = f'{IN_DIR}/{dataset_name}/{dataset_detail}|M_{n_clients}|K_{n_clusters}|SEED_42/R_{N_REPEATS}|kmeans++|None|{TOLERANCE}|std.csv'
		# 				args1 = copy.deepcopy(args)
		# 				args1['csv_file'] = f
		# 				args1['p'] = ratio
		# 				SEED = args1['SEED']
		# 				args1['DATASET']['name'] = dataset_name
		# 				args1['DATASET']['detail'] = dataset_detail
		# 				args1['N_CLIENTS'] = n_clients
		# 				args1['N_CLUSTERS'] = n_clusters
		# 				N_CLIENTS = args1['N_CLIENTS']
		# 				N_CLUSTERS = args1['N_CLUSTERS']
		# 				args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'],
		# 				                                                 f'M_{N_CLIENTS}',
		# 				                                                 f'K_{N_CLUSTERS}', f'SEED_{SEED}'])
		# 			csv_files.append(copy.deepcopy(args1))
		# plot_real_case_P(IN_DIR, OUT_DIR, ALG2ABBREV, csv_files)
		# exit()

		csv_files = []    # not work
		# N = 5000
		n_clusters = 2
		n_clients = 2
		for ratio in [0, 0.1, 0.3, 0.5]:
			for n1 in [5000]:
				for n2 in [9000]:
					dataset_detail = f'n1_{n1}+n2_{n2}:ratio_{ratio:.2f}:C_2_diff_sigma_n'
					f = f'{IN_DIR}/{dataset_name}/{dataset_detail}|{NORMALIZE_METHOD}|PCA_{IS_PCA}|M_{n_clients}|K_{n_clusters}|REMOVE_OUTLIERS_{IS_REMOVE_OUTLIERS}/R_{N_REPEATS}|random|None|{TOLERANCE}|std.csv'
					args1 = copy.deepcopy(args)
					args1['csv_file'] = f
					args1['p'] = ratio
					SEED = args1['SEED']
					args1['DATASET']['name'] = dataset_name
					args1['DATASET']['detail'] = dataset_detail
					args1['N_CLIENTS'] = n_clients
					args1['N_CLUSTERS'] = n_clusters
					N_CLIENTS = args1['N_CLIENTS']
					N_CLUSTERS = args1['N_CLUSTERS']
					args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'],
					                                                 f'M_{N_CLIENTS}',
					                                                 f'K_{N_CLUSTERS}', f'SEED_{SEED}'])
				csv_files.append(copy.deepcopy(args1))
		plot_real_case_P(IN_DIR, OUT_DIR, ALG2ABBREV, csv_files)
		exit()

		# csv_files = []
		# N1S = [50, 100, 500, 1000, 2000, 3000, 5000, 8000, 10000]
		# N = 5000
		# n_clusters = 2
		# n_clients = 2
		# for n1 in N1S:
		# 	for n2 in [N]:
		# 		dataset_detail = f'n1_{n1}+n2_{n2}:ratio_0.00:C_2_diff_sigma_n'
		# 		f = f'{IN_DIR}/{dataset_name}/{dataset_detail}|M_{n_clients}|K_{n_clusters}|SEED_42/R_{N_REPEATS}|kmeans++|None|{TOLERANCE}|std.csv'
		# 		args1 = copy.deepcopy(args)
		# 		args1['csv_file'] = f
		# 		args1['p'] = n1
		# 		SEED = args1['SEED']
		# 		args1['DATASET']['name'] = dataset_name
		# 		args1['DATASET']['detail'] = dataset_detail
		# 		args1['N_CLIENTS'] = n_clients
		# 		args1['N_CLUSTERS'] = n_clusters
		# 		N_CLIENTS = args1['N_CLIENTS']
		# 		N_CLUSTERS = args1['N_CLUSTERS']
		# 		args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'],
		# 		                                                 f'M_{N_CLIENTS}',
		# 		                                                 f'K_{N_CLUSTERS}', f'SEED_{SEED}'])
		# 	csv_files.append(copy.deepcopy(args1))
		# plot_real_case_N(IN_DIR, OUT_DIR, ALG2ABBREV, csv_files)
		# exit()
	elif dataset_name == 'SENT140':
		csv_files = []
		n_clients = 20
		n_clusters = 2
		for n in [1, 3, 5, 10, 15, 20]:
			# there are 20 clients and each client has n users' data.
			dataset_detail = f'n_{n}:ratio_0.00:diff_sigma_n'
			f = f'{IN_DIR}/{dataset_name}/{dataset_detail}|M_{n_clients}|K_{n_clusters}|SEED_42/R_{N_REPEATS}|kmeans++|None|{TOLERANCE}|std.csv'
			args1 = copy.deepcopy(args)
			args1['csv_file'] = f
			args1['p'] = n
			SEED = args1['SEED']
			args1['DATASET']['name'] = dataset_name
			args1['DATASET']['detail'] = dataset_detail
			args1['N_CLIENTS'] = n_clients
			args1['N_CLUSTERS'] = n_clusters
			N_CLIENTS = args1['N_CLIENTS']
			N_CLUSTERS = args1['N_CLUSTERS']
			args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'],
			                                                 f'M_{N_CLIENTS}',
			                                                 f'K_{N_CLUSTERS}', f'SEED_{SEED}'])
			csv_files.append(copy.deepcopy(args1))
		plot_real_case_N(IN_DIR, OUT_DIR, ALG2ABBREV, csv_files)


# # plot dataset
# params = {}
# # gaussian3_1client_1cluster_diff_sigma(params, random_state=42)
# gaussian3_mix_clusters_per_client(params, random_state=42, xlim=[-4, 4], ylim=[-2, 5])
# gaussian3_diff_sigma_n(params, random_state=42, xlim=[-4, 4], ylim=[-2, 6])
## # 1. GAUSSIANS
	# dataset = {'name': '3GAUSSIANS', 'detail': 'diff_sigma_n', 'n_clusters': 3, 'n_clients': 3}
	# plot_guassian_P(IN_DIR, OUT_DIR, ALG2ABBREV, dataset, N_REPEATS, TOLERANCE)
	# plot_guassian_N(IN_DIR, OUT_DIR, ALG2ABBREV, dataset, N_REPEATS, TOLERANCE)
	#
	# dataset = {'name': '10GAUSSIANS', 'detail': 'diff_sigma_n', 'n_clusters': 10, 'n_clients': 10}
	# plot_guassian_P(IN_DIR, OUT_DIR, ALG2ABBREV, dataset, N_REPEATS, TOLERANCE)
	# plot_guassian_N(IN_DIR, OUT_DIR, ALG2ABBREV, dataset, N_REPEATS, TOLERANCE)

	# 2. real data
	# dataset = {'name': 'NBAIOT', 'detail': None, 'n_clusters': 2, 'n_clients': 3}
	# dataset = {'name': 'NBAIOT', 'detail': None, 'n_clusters': 2, 'n_clients': 2}
	# dataset = {'name': 'SENT140', 'detail': None, 'n_clusters': 2, 'n_clients': 2}
