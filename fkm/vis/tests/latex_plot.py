"""
    run:
        module load anaconda3/2021.5
        cd /scratch/gpfs/ky8517/fkm/fkm
        PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 results/latex_plot.py

"""
import os
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fkm.datasets.gaussian10 import gaussian10_diff_sigma_n
from fkm.datasets.gaussian3 import gaussian3_mix_clusters_per_client, gaussian3_diff_sigma_n
from fkm.utils.utils_func import load


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


def plot_N_P(ax, result_files, algorithms=['Random-WA-FKM', 'Gaussian-WA-FKM'], fig_name='diff_n',
             out_dir='results/xlsx', params={}, p=0.00, is_show=False, is_legend=False, y_name='Training Iterations'):
	df = pd.DataFrame()
	percents = []
	for i, (f, n1) in enumerate(result_files):
		try:
			df_ = pd.read_csv(f, header=None).iloc[:, 1:]
		except Exception as e:
			print(f, os.path.exists(os.path.abspath(f)), flush=True)
			traceback.print_exc()
			return
		percents += [n1] * df_.shape[1]
		df = pd.concat([df, df_], axis=1)
	df = df.T
	df['n1'] = percents
	df.columns = ['Algorithm', 'Training Iterations', 'Training DB Score', 'Training Silhouette',
	              'Training Euclidean Distance',
	              'Testing Iterations', 'Testing DB Score', 'Testing Silhouette', 'Testing Euclidean Distance', '',
	              'n1']
	print(df)
	df.reset_index(drop=True, inplace=True)
	df = df.apply(format_column, axis=1)
	# https://stackoverflow.com/questions/57485426/markers-are-not-visible-in-seaborn-plot
	# markers= is only useful when you also specify a style= parameter.
	# axes = sns.lineplot(data=df, x='Percent', y='Training DB Score', hue=df['Algorithm'], err_style="bars",
	#              style="Algorithm", markers = True)
	# markers = ['.', '+', 'o', '^'])
	# ax.fill_between(df['Percent'], 0, 1, alpha=0.2)
	x = df['n1'].unique()
	colors = ['g', 'b', 'c', 'm', 'y', 'k']
	# b: blue.
	# g: green.
	# r: red.
	# c: cyan.
	# m: magenta.
	# y: yellow.
	# k: black.
	# w: white.

	# y_name = 'Training Iterations'
	for i, alg in enumerate(algorithms):
		if y_name == 'Training DB Score':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		elif y_name == 'Training Iterations':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		elif y_name == 'Training Silhouette':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		elif y_name == 'Training Euclidean Distance':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))

		elif y_name == 'Testing DB Score':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		elif y_name == 'Testing Silhouette':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		elif y_name == 'Testing Euclidean Distance':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))

		print(alg, yerr, x, y)
		ax.errorbar(x, y, yerr=yerr, linestyle='-', marker='o', label=alg2label[alg], lw=2, color=colors[i],
		            ecolor=colors[i], elinewidth=1, capsize=2)

	if is_legend:
		ax.legend(fontsize=7)

	fontsize = 13
	if y_name == 'Training DB Score':
		if CASE == 'Case 2': ax.set_ylim([0.5, 0.9])
		ax.set_ylabel('DB Score', fontsize=fontsize)
	elif y_name == 'Training Iterations':
		if CASE == 'Case 2': ax.set_ylim([5, 75])  # for Case 2
		if CASE == 'Case 3': ax.set_ylim([5, 40])  # for Case 3
		# ax.set_ylabel('Training $T$', fontsize=fontsize)
		ax.set_ylabel('Convergence $T$', fontsize=fontsize)
	elif y_name == 'Training Silhouette':
		if CASE == 'Case 2':  ax.set_ylim([0.4, 0.62])
		ax.set_ylabel('Silhouette', fontsize=fontsize)
	elif y_name == 'Training Euclidean Distance':
		# ax.set_ylim([0.43, 0.62])
		ax.set_ylabel('Euclidean Distance', fontsize=fontsize)

	elif y_name == 'Testing DB Score':
		if CASE == 'Case 2': ax.set_ylim([0.5, 0.9])
		ax.set_ylabel('Testing DB', fontsize=fontsize)
	elif y_name == 'Testing Silhouette':
		if CASE == 'Case 2': ax.set_ylim([0.4, 0.62])
		ax.set_ylabel('Testing SC', fontsize=fontsize)
	elif y_name == 'Testing Euclidean Distance':
		if CASE == 'Case 2': ax.set_ylim([0.43, 0.62])
		ax.set_ylabel('Testing $\overline{WCSS}$', fontsize=fontsize)

	ax.set_xlabel(f'$N_1$', fontsize=fontsize)


def plot_P(ax, result_files, algorithms=['Random-WA-FKM', 'Gaussian-WA-FKM'], fig_name='diff_n',
           out_dir='results/xlsx', params={}, p=0.00, is_show=False, is_legend=False, y_name='Training Iterations'):
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
	df['p'] = percents
	df.columns = ['Algorithm', 'Training Iterations', 'Training DB Score', 'Training Silhouette',
	              'Training Euclidean Distance',
	              'Testing Iterations', 'Testing DB Score', 'Testing Silhouette', 'Testing Euclidean Distance', '',
	              'p']
	print(df)
	df.reset_index(drop=True, inplace=True)
	df = df.apply(format_column, axis=1)
	# https://stackoverflow.com/questions/57485426/markers-are-not-visible-in-seaborn-plot
	# markers= is only useful when you also specify a style= parameter.
	# axes = sns.lineplot(data=df, x='Percent', y='Training DB Score', hue=df['Algorithm'], err_style="bars",
	#              style="Algorithm", markers = True)
	# markers = ['.', '+', 'o', '^'])
	# ax.fill_between(df['Percent'], 0, 1, alpha=0.2)
	x = df['p'].unique()
	colors = ['g', 'b', 'c', 'm', 'y', 'k']
	# b: blue.
	# g: green.
	# r: red.
	# c: cyan.
	# m: magenta.
	# y: yellow.
	# k: black.
	# w: white.

	# y_name = 'Training Iterations'
	for i, alg in enumerate(algorithms):
		if y_name == 'Training DB Score':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		elif y_name == 'Training Iterations':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		elif y_name == 'Training Silhouette':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		elif y_name == 'Training Euclidean Distance':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))

		elif y_name == 'Testing DB Score':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		elif y_name == 'Testing Silhouette':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		elif y_name == 'Testing Euclidean Distance':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))

		print(alg, yerr, x, y)
		ax.errorbar(x, y, yerr=yerr, linestyle='-', marker='o', label=alg2label[alg], lw=2, color=colors[i],
		            ecolor=colors[i], elinewidth=1, capsize=2)

	if is_legend:
		ax.legend(fontsize=7, loc='upper right')

	fontsize = 13
	if y_name == 'Training Iterations':
		if CASE == 'Case 1': ax.set_ylim([8, 30])  # for Case 1
		if CASE == 'Case 4': ax.set_ylim([3, 80])  # for Case 4
		if CASE == 'Case 5': ax.set_ylim([3, 40])  # for Case 5
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

	elif y_name == 'Testing DB Score':
		if CASE == 'Case 1': ax.set_ylim([0.4, 0.8])
		if CASE == 'Case 4': ax.set_ylim([0.3, 0.9])
		if CASE == 'Case 5': ax.set_ylim([0.6, 1.2])
		ax.set_ylabel('Testing DB', fontsize=fontsize)
	elif y_name == 'Testing Silhouette':
		if CASE == 'Case 1': ax.set_ylim([0.4, 0.8])
		if CASE == 'Case 4': ax.set_ylim([0.4, 0.85])
		if CASE == 'Case 5': ax.set_ylim([0.4, 0.6])
		ax.set_ylabel('Testing SC', fontsize=fontsize)
	elif y_name == 'Testing Euclidean Distance':
		if CASE == 'Case 1': ax.set_ylim([0.3, 0.8])
		if CASE == 'Case 4': ax.set_ylim([0., 0.22])
		if CASE == 'Case 5': ax.set_ylim([0.3, 1.6])
		ax.set_ylabel('Testing $\overline{WCSS}$', fontsize=fontsize)

	ax.set_xlabel(f'$P$', fontsize=fontsize)


def get_df(result_files):
	df = pd.DataFrame()
	percents = []
	for i, (f, n1) in enumerate(result_files):
		try:
			df_ = pd.read_csv(f, header=None).iloc[:, 1:]
		except Exception as e:
			print(f, os.path.exists(os.path.abspath(f)), flush=True)
			traceback.print_exc()
			return
		percents += [n1] * df_.shape[1]
		df = pd.concat([df, df_], axis=1)
	df = df.T
	df['n1'] = percents
	df.columns = ['Algorithm', 'Training Iterations', 'Training DB Score', 'Training Silhouette',
	              'Training Euclidean Distance',
	              'Testing Iterations', 'Testing DB Score', 'Testing Silhouette', 'Testing Euclidean Distance', '',
	              'n1']
	print(df)
	df.reset_index(drop=True, inplace=True)
	df = df.apply(format_column, axis=1)
	# https://stackoverflow.com/questions/57485426/markers-are-not-visible-in-seaborn-plot
	# markers= is only useful when you also specify a style= parameter.
	# axes = sns.lineplot(data=df, x='Percent', y='Training DB Score', hue=df['Algorithm'], err_style="bars",
	#              style="Algorithm", markers = True)
	# markers = ['.', '+', 'o', '^'])
	# ax.fill_between(df['Percent'], 0, 1, alpha=0.2)
	x = df['n1'].unique()
	return x, df


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


if __name__ == '__main__':
	# in_dir = os.path.abspath('~/Downloads/xlsx')
	in_dir = os.path.expanduser('~/Downloads/xlsx')
	tot_cnt = 0

	algorithms = ['KM++-CKM', 'Random-WA-FKM', 'C-Random-WA-FKM', 'C-KM++-WA-FKM', 'C-Random-GD-FKM',
	              'C-KM++-GD-FKM']
	alg2label = {'KM++-CKM': 'CKM++', 'Random-WA-FKM': 'Server-Initialized', 'C-Random-WA-FKM': 'Average-Random',
	             'C-KM++-WA-FKM': 'Average-KM++', 'C-Random-GD-FKM': 'Greedy-Random',
	             'C-KM++-GD-FKM': 'Greedy-KM++'}

	for dataset in ['3GAUSSIANS']:  # ['3GAUSSIANS', '2GAUSSIANS', 'FEMNIST']:
		if dataset == '3GAUSSIANS':
			out_dir = 'results/xlsx'
			show_data = False
			client_epochs = 1
			n_clusters = 3
			n_clients = 3
			if show_data:
				# show dataset
				params = {'p0': None, 'p1': 'ratio_0:mix_clusters_per_client', 'p2': 'Centralized', 'repeats': None,
				          'client_epochs': None,
				          'tolerance': None, 'normalize_method': None,
				          'is_crop_image': True, 'image_shape': (14, 14),  # For FEMNIST, crop 28x28 to 14x14
				          'data_name': None,
				          'writer_ratio': None, 'data_ratio_per_writer': None,
				          'data_ratio_per_digit': None,
				          'n_clusters': 3,
				          'is_federated': None,
				          'n_clients': None,
				          'server_init_centroids': None, 'client_init_centroids': None,
				          'out_dir': out_dir,
				          'is_show': True, 'show_title': False
				          }
				# gaussian3_1client_1cluster_diff_sigma(params, random_state=42)
				gaussian3_mix_clusters_per_client(params, random_state=42, xlim=[-4, 4], ylim=[-2, 5])

				p1 = 'n1_100-sigma1_0.1_0.1+n2_5000-sigma2_0.2_0.2+n3_10000-sigma3_0.3_0.3:ratio_0.0:diff_sigma_n'
				params = {'p0': None, 'p1': p1, 'p2': 'Centralized', 'repeats': None,
				          'client_epochs': None,
				          'tolerance': None, 'normalize_method': None,
				          'is_crop_image': True, 'image_shape': (14, 14),  # For FEMNIST, crop 28x28 to 14x14
				          'data_name': None,
				          'writer_ratio': None, 'data_ratio_per_writer': None,
				          'data_ratio_per_digit': None,
				          'n_clusters': 3,
				          'is_federated': None,
				          'n_clients': None,
				          'server_init_centroids': None, 'client_init_centroids': None,
				          'out_dir': out_dir,
				          'is_show': True, 'show_title': False
				          }
				gaussian3_diff_sigma_n(params, random_state=42, xlim=[-4, 4], ylim=[-2, 6])

			###########################################################################################
			### bar plot: Dataset 1
			fix_N1 = False
			if fix_N1:
				n1 = 10000  # fixed

				is_show = True
				out_dir = 'results/xlsx'
				ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
				n2 = 10000  # 5000   #
				sigma1 = "0.1_0.1"
				sigma2 = "0.1_0.1"
				sigma3 = "1.0_0.1"
				n3 = 10000
				n_cols = 7
				for metrics in [['Training Iterations'],
				                ['Testing DB Score', 'Testing Silhouette', 'Testing Euclidean Distance']
				                ]:
					if 'Train' in metrics[0]:
						fig, axes = plt.subplots(1, n_cols, sharex=True, sharey=True,
						                         figsize=(25, 5))  # (width, height)
						axes = np.asarray(axes).reshape((1, -1))
						fig_name = f'bar_diff_n_sigma_training_metrics'
					else:
						fig, axes = plt.subplots(3, n_cols, sharex=True, sharey=False,
						                         figsize=(25, 12))  # (width, height)
						axes = axes.reshape((3, -1))
						fig_name = f'bar_diff_n_sigma_testing_metrics'

					for idx_yaxis, y_name in enumerate(metrics):
						for i, p in enumerate(ratios):
							p = float(f'{p:.2f}')
							try:
								# for each p, plot n1 vs DB
								result_files = []
								# '3GAUSSIANS-Client_epochs_1-n1_500-sigma1_0.3+n2_5000-sigma2_0.3+n3_10000-sigma3_0.3/ratio_0.00.csv
								result_files.append((os.path.join(in_dir,
								                                  f'{dataset}-Client_epochs_{client_epochs}-Clusters_{n_clusters}-Clients_{n_clients}/n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{p:.2f}:diff_sigma_n.csv'),
								                     n1))
								print(y_name, p, result_files)
								if i == 0:
									# get baseline
									_, bs_df = get_df(result_files)
								else:
									i_, j_ = divmod(i - 1, n_cols)
									if i_ == 0 and j_ == 0:
										is_legend = True
									else:
										is_legend = False
									ax = axes[idx_yaxis + i_][j_]
									barplot_N_P(bs_df, ax, result_files, algorithms[1:], out_dir='results/xlsx',
									            fig_name=f'diff_n_sigma_{p:.2f}', p=p, is_show=True,
									            is_legend=is_legend,
									            y_name=y_name)
							except Exception as e:
								# traceback.print_exc()
								print(e, flush=True)

							tot_cnt += len(ratios)
					# axes[-1][-1].axis('off')
					plt.tight_layout()

					if not os.path.exists(out_dir):
						os.makedirs(out_dir)
					f = os.path.join(out_dir, f'{fig_name}.png')
					plt.savefig(f, dpi=600, bbox_inches='tight')
					print(f)
					if is_show:
						plt.show()

			###########################################################################################
			# Case 1: line plot, for Dataset 1
			fix_N1 = False
			CASE = 'Case 1'
			if fix_N1:
				out_dir = 'results/xlsx'
				n1 = 10000  # fixed
				is_show = True
				out_dir = 'results/xlsx'
				ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
				n2 = 10000  # 5000   #
				sigma1 = "0.1_0.1"
				sigma2 = "0.1_0.1"
				sigma3 = "1.0_0.1"
				n3 = 10000
				n_cols = 7
				for metrics in [['Training Iterations'],
				                ['Testing DB Score', 'Testing Silhouette', 'Testing Euclidean Distance']
				                ]:
					if 'Train' in metrics[0]:
						fig, axes = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(5, 3))  # (width, height)
						axes = np.asarray(axes).reshape((1, 1))
						fig_name = f'diff_p_training_metrics'
					else:
						fig, axes = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(15, 3))  # (width, height)
						axes = axes.reshape((1, 3))
						fig_name = f'diff_p_testing_metrics'

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
							for p in ratios:
								# '3GAUSSIANS-Client_epochs_1-n1_500-sigma1_0.3+n2_5000-sigma2_0.3+n3_10000-sigma3_0.3/ratio_0.00.csv
								result_files.append((os.path.join(in_dir,
								                                  f'{dataset}-Client_epochs_{client_epochs}-Clusters_{n_clusters}-Clients_{n_clients}/n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{p:.2f}:diff_sigma_n.csv'),
								                     p))
							plot_P(ax, result_files, algorithms, out_dir='results/xlsx',
							       fig_name=None, p=p, is_show=True, is_legend=is_legend,
							       y_name=y_name)
						except Exception as e:
							# traceback.print_exc()
							print(e, flush=True)
						tot_cnt += 1
					plt.tight_layout()

					if not os.path.exists(out_dir):
						os.makedirs(out_dir)
					f = os.path.join(out_dir, f'{fig_name}.png')
					plt.savefig(f, dpi=600, bbox_inches='tight')
					print(os.path.abspath(f))
					if is_show:
						plt.show()

			###########################################################################################
			# bar plot
			fix_P = False  # for Dataset 2
			if fix_P:
				p = 0.01  # fixed
				is_show = True
				out_dir = 'results/xlsx'
				ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
				# n1s = [100, 500, 1000, 2000, 3000, 5000, 8000, 10000]
				n1s = [100, 500, 1000, 2000, 3000, 5000, 8000, 10000]
				n2 = 5000
				fig_dataset2 = False
				if fig_dataset2:
					sigma1 = "0.3_0.3"
					sigma2 = "0.3_0.3"
					sigma3 = "0.3_0.3"
				else:  # for dataset 3
					sigma1 = "0.1_0.1"
					sigma2 = "0.2_0.2"
					sigma3 = "0.3_0.3"
				n3 = 10000
				for metrics in [['Training Iterations'],
				                ['Testing DB Score', 'Testing Silhouette', 'Testing Euclidean Distance']
				                ]:
					if 'Train' in metrics[0]:
						fig, axes = plt.subplots(1, 4, sharex=True, sharey=False, figsize=(15, 5))  # (width, height)
						axes = np.asarray(axes).reshape((1, -1))
						fig_name = f'bar_diff_n_sigma_training_metrics'
					else:
						fig, axes = plt.subplots(3, 4, sharex=True, sharey=False, figsize=(15, 12))  # (width, height)
						axes = axes.reshape((3, -1))
						fig_name = f'bar_diff_n_sigma_testing_metrics'

					for idx_yaxis, y_name in enumerate(metrics):
						for i, n1 in enumerate(n1s):
							p = float(f'{p:.2f}')
							try:
								# for each p, plot n1 vs DB
								result_files = []
								# '3GAUSSIANS-Client_epochs_1-n1_500-sigma1_0.3+n2_5000-sigma2_0.3+n3_10000-sigma3_0.3/ratio_0.00.csv
								result_files.append((os.path.join(in_dir,
								                                  f'{dataset}-Client_epochs_{client_epochs}-Clusters_{n_clusters}-Clients_{n_clients}/n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{p:.2f}:diff_sigma_n.csv'),
								                     n1))
								if i == 0:
									# get baseline
									_, bs_df = get_df(result_files)
								else:
									i_, j_ = divmod(i - 1, 4)
									if i_ == 0 and j_ == 0:
										is_legend = True
									else:
										is_legend = False
									ax = axes[idx_yaxis + i_][j_]
									barplot_N_P(bs_df, ax, result_files, algorithms, out_dir='results/xlsx',
									            fig_name=f'diff_n_sigma_{p:.2f}', p=p, is_show=True,
									            is_legend=is_legend,
									            y_name=y_name)
							except Exception as e:
								# traceback.print_exc()
								print(e, flush=True)

							tot_cnt += len(n1s)
					plt.tight_layout()

					if not os.path.exists(out_dir):
						os.makedirs(out_dir)
					f = os.path.join(out_dir, f'{fig_name}.png')
					plt.savefig(f, dpi=600, bbox_inches='tight')
					print(f)
					if is_show:
						plt.show()

			###########################################################################################
			# Case 2/3: line plot, for Dataset 1
			# for other results:  See latex_plot_appendix, too.
			flg = True
			CASE = 'Case 2'
			fig_dataset2 = True
			# CASE = 'Case 3'
			# fig_dataset2 = False
			if flg:
				is_show = True
				out_dir = 'results/xlsx'
				n1s = [100, 500, 1000, 2000, 3000, 5000, 8000, 10000]
				# n1s = [100, 500, 1000, 2000, 3000, 5000]
				n2 = 5000
				if fig_dataset2:
					sigma1 = "0.3_0.3"
					sigma2 = "0.3_0.3"
					sigma3 = "0.3_0.3"
					dt_name = 'd2'
				else:  # for dataset 3
					sigma1 = "0.1_0.1"
					sigma2 = "0.2_0.2"
					sigma3 = "0.3_0.3"
					dt_name = 'd3'
				n3 = 10000
				ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
				for p in ratios:
					# ['Algorithm', 'Training Iterations', 'Training DB Score', 'Training Silhouette',
					#               'Training Euclidean Distance',
					#               'Testing Iterations', 'Testing DB Score', 'Testing Silhouette',
					#               'Testing Euclidean Distance', '',
					#               'n1']
					for metrics in [['Training Iterations'],
					                ['Testing DB Score', 'Testing Silhouette', 'Testing Euclidean Distance']
					                ]:
						if 'Train' in metrics[0]:
							fig, axes = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(5, 3))  # (width, height)
							axes = np.asarray(axes).reshape((1, 1))
							fig_name = f'diff_n1_p_{p:.2f}_training_metrics_{dt_name}'
						else:
							fig, axes = plt.subplots(1, 3, sharex=True, sharey=False,
							                         figsize=(15, 3))  # (width, height)
							axes = axes.reshape((1, 3))
							fig_name = f'diff_n1_p_{p:.2f}_testing_metrics_{dt_name}'

						for i, y_name in enumerate(metrics):
							p = float(f'{p:.2f}')
							i_, j_ = divmod(i, 4)
							if i_ == 0 and j_ == 0:
								is_legend = True
							else:
								is_legend = False
							ax = axes[i_][j_]
							try:
								# for each p, plot n1 vs DB
								result_files = []
								for n1 in n1s:
									# '3GAUSSIANS-Client_epochs_1-n1_500-sigma1_0.3+n2_5000-sigma2_0.3+n3_10000-sigma3_0.3/ratio_0.00.csv
									result_files.append((os.path.join(in_dir,
									                                  f'{dataset}-Client_epochs_{client_epochs}-Clusters_{n_clusters}-Clients_{n_clients}/n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{p:.2f}:diff_sigma_n.csv'),
									                     n1))
								plot_N_P(ax, result_files, algorithms, out_dir='results/xlsx',
								         fig_name=f'diff_n_sigma_{p:.2f}', p=p, is_show=True, is_legend=is_legend,
								         y_name=y_name)
							except Exception as e:
								# traceback.print_exc()
								print(e, flush=True)
							tot_cnt += len(n1s)
						plt.tight_layout()

						if not os.path.exists(out_dir):
							os.makedirs(out_dir)
						f = os.path.join(out_dir, f'{fig_name}.png')
						plt.savefig(f, dpi=600, bbox_inches='tight')
						print(f)
						if is_show:
							plt.show()

			###########################################################################################
			# final centroids: for Dataset 2： 2.2
			flg_centroids = False
			if flg_centroids:
				plt.close()
				is_show = True
				out_dir = 'results/xlsx'
				n1s = [100, 500, 1000, 2000, 3000, 5000, 8000, 10000]
				# n1s = [100, 500, 1000, 2000, 3000, 5000]
				n2 = 5000
				# sigma1 = "0.1_0.1"
				# sigma2 = "0.1_0.1"
				# sigma3 = "1.0_0.1"
				sigma1 = "0.3_0.3"
				sigma2 = "0.3_0.3"
				sigma3 = "0.3_0.3"
				n3 = 10000
				ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
				# algorithms = ['KM++-CKM', 'Random-WA-FKM', 'C-Random-WA-FKM', 'C-KM++-WA-FKM', 'C-Random-GD-FKM',
				#               'C-KM++-GD-FKM']
				algorithms = [('KM++-CKM', 'Centralized_Kmeans', 'Centralized-kmeans++'),
				              # ('Random-WA-FKM', 'Stanford_server_random_initialization',
				              #  'Federated-Server_random_min_max-Client_None'),
				              # ('C-Random-WA-FKM', 'Stanford_client_initialization',
				              #  'Federated-Server_average-Client_random'),
				              # ('C-KM++-WA-FKM', 'Stanford_client_initialization',
				              #  'Federated-Server_average-Client_kmeans++'),
				              # ('C-Random-GD-FKM', 'Our_greedy_initialization', 'Federated-Server_greedy-Client_random'),
				              ('C-KM++-GD-FKM', 'Our_greedy_initialization', 'Federated-Server_greedy-Client_kmeans++'),
				              ]
				p = 0.0  # only for one p
				nrows, ncols = len(algorithms), len(n1s)

				if nrows == 1:
					fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(15, 2))
					axes = np.asarray(axes).reshape((1, -1))
					fig_name = 'diff_n1_p_0.00_ckm_final_centroids'
				elif nrows == 2:
					fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(15, 4.5))
					fig_name = 'diff_n1_p_0.00_ckm_gd_final_centroids'
				else:
					fig, axes = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(15, 4.5))
					fig_name = 'diff_n1_p_0.00_final_centroids'
				seed = 10
				for idx, (alg_, alg_0, alg_1) in enumerate(algorithms):
					print(f'\nidx: {idx}, {alg_1}')
					# r, c = divmod(idx, ncols)
					# print(i, seed, r, c)
					for i, n1 in enumerate(n1s):
						# run on remote server: download to local
						# rsync -azP ky8517@tiger.princeton.edu:/scratch/gpfs/ky8517/fkm/fkm/results/ ~/Downloads/
						# in_file = 'results/repeats_2-client_epochs_1-tol_1e-06-normalize_std/Centralized_Kmeans/3GAUSSIANS/n1_100-sigma1_0.1_0.1+n2_5000-sigma2_0.1_0.1+n3_{n3}-sigma3_1.0_0.1:ratio_0.00:diff_sigma_n-Testset_0.3-Clusters_3-Clients_3/Centralized-kmeans++/varied_clients-Server_kmeans++-Client_None-histories.dat'
						in_file = os.path.join(in_dir, 'results/repeats_50-client_epochs_1-tol_1e-06-normalize_std',
						                       alg_0,
						                       f'{dataset}/n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:'
						                       f'ratio_{p:.2f}:diff_sigma_n-Testset_0.3-Clusters_3-Clients_3', alg_1,
						                       '~histories.dat')
						if not os.path.exists(in_file):
							# download the remote file to local
							t_dir = os.path.dirname(in_file)
							if not os.path.exists(t_dir):
								os.makedirs(t_dir)
							cmd = f'rsync -azP ky8517@tiger.princeton.edu:/scratch/gpfs/ky8517/fkm/fkm/{in_file} {t_dir}'
							os.system(cmd)
						# in_file = 'results/varied_clients-Server_greedy-Client_random-histories.dat'
						if not os.path.exists(in_file):
							print('False', in_file)
							continue
						print('True', in_file, flush=True)
						history = load(in_file)
						if 'Centralized' in alg_1:
							n_clusters = 0
						else:
							n_clusters = 3
						res = history[n_clusters]['history']['results']
						seeds = []
						initial_centroids = []
						final_centroids = []
						scores = []
						iterations = []
						for vs in res:
							seeds.append(vs['seed'])
							initial_centroids.append(vs['initial_centroids'])
							final_centroids.append(vs['final_centroids'])
							scores.append(vs['scores'])
							iterations.append(vs['training_iterations'])

						ax = axes[idx, i]
						# ps = initial_centroids[0]
						# for j, point in enumerate(ps):
						# 	ax.scatter(point[0], point[1], c='gray', marker="o", s=100, label='initial' if j == 0 else '')
						# 	# offset = 0.9 * (j+1)
						# 	offset = 0.9
						# 	xytext = (point[0], point[1] + offset)
						# 	ax.annotate(f'({point[0]:.1f}, {point[1]:.1f})', xy=(point[0], point[1]), xytext=xytext, fontsize=8,
						# 	            color='black',
						# 	            ha='center', va='center',  # textcoords='offset points',
						# 	            bbox=dict(facecolor='none', edgecolor='gray', pad=1),
						# 	            arrowprops=dict(arrowstyle="->", color='gray', shrinkA=1,
						# 	                            connectionstyle="angle3, angleA=90,angleB=0"))

						ps = final_centroids[0]
						for j, point in enumerate(ps):
							ax.scatter(point[0], point[1], c='r', marker="*", s=100, label='final' if j == 0 else '')
							# offset = 0.9 * (j+1)
							offset = 0.99
							if j == 2:
								xytext = (point[0], point[1] - offset - 0.4)
							else:
								xytext = (point[0], point[1] - offset)
							ax.annotate(f'({point[0]:.1f}, {point[1]:.1f})', xy=(point[0], point[1]), xytext=xytext,
							            fontsize=8, color='b',
							            ha='center',
							            va='center',
							            # bbox=dict(facecolor='none', edgecolor='red', pad=1),
							            arrowprops=dict(arrowstyle="->", color='b', shrinkA=1,
							                            connectionstyle="angle3, angleA=90,angleB=0"))

						train_db, test_db = scores[0]['train']['davies_bouldin'], scores[0]['test']['davies_bouldin']
						# ax.set_title(
						# 	f'Train: {train_db:.2f}, Test: {test_db:.2f}\nIterations: {iterations[i]}, Seed: {seed}')

						# # ax.set_xlim([-10, 10]) # [-3, 7]
						# # ax.set_ylim([-15, 15])  # [-3, 7]
						ax.set_xlim([-3, 3])  # [-3, 7]
						ax.set_ylim([-3, 3])  # [-3, 7]

						ax.axvline(x=0, color='k', linestyle='--')
						ax.axhline(y=0, color='k', linestyle='--')
						if nrows == 1:
							ax.set_xlabel(f'$N_1$:{n1}')
						else:
							ax.set_xlabel(f'$N_1$:{n1}\n{alg2label[alg_]}')
				# plt.show()
				# fig.suptitle(title, fontsize=20)
				# # Put a legend below current axis
				# plt.legend(loc='lower center', bbox_to_anchor=(-.5, -0.5),   # (x0,y0, width, height)=(0,0,1,1)).
				#           fancybox=False, shadow=False, ncol=2)
				# plt.xlim([-2, 15])
				# plt.ylim([-2, 15])
				# plt.xticks([])
				# plt.yticks([])
				plt.tight_layout()
				f = os.path.join(out_dir, f'{fig_name}.png')
				plt.savefig(f, dpi=600, bbox_inches='tight')
				if is_show:
					plt.show()
				print(os.path.abspath(f))

		elif dataset == '10GAUSSIANS':
			"""
				1. run collect_results.py to get xlsx
				2. run collect_table_results.py to get csv
				3. run latex_plot.py to get plot
			"""
			out_dir = 'results/xlsx'
			show_data = False
			client_epochs = 1
			n_clusters = 3  # for case 4 when n_clusters = 10; for case 5 when n_clusters = 3
			n_clients = 10
			if show_data:
				# show dataset
				p1 = 'n1_5000-sigma1_0.3_0.3+n2_1000-sigma2_0.3_0.3+n3_500-sigma3_0.3_0.3:ratio_0.0:diff_sigma_n'
				params = {'p0': None, 'p1': p1, 'p2': 'Centralized', 'repeats': None,
				          'client_epochs': None,
				          'tolerance': None, 'normalize_method': None,
				          'is_crop_image': True, 'image_shape': (14, 14),  # For FEMNIST, crop 28x28 to 14x14
				          'data_name': None,
				          'writer_ratio': None, 'data_ratio_per_writer': None,
				          'data_ratio_per_digit': None,
				          'n_clusters': 3,
				          'is_federated': None,
				          'n_clients': None,
				          'server_init_centroids': None, 'client_init_centroids': None,
				          'out_dir': out_dir,
				          'is_show': True, 'show_title': False
				          }
				gaussian10_diff_sigma_n(params, random_state=42)  # xlim=[-4, 4], ylim=[-2, 5])

			###########################################################################################
			### bar plot: Dataset 1
			fix_N1 = False
			if fix_N1:
				n1 = 5000  # fixed
				'n1_5000-sigma1_0.3_0.3+n2_1000-sigma2_0.3_0.3+n3_500-sigma3_0.3_0.3/ratio_0.00/diff_sigma_n.xlsx'
				is_show = True
				out_dir = 'results/xlsx'
				ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
				n2 = 1000  # 5000   #
				sigma1 = "0.3_0.3"
				sigma2 = "0.3_0.3"
				sigma3 = "0.3_0.3"
				n3 = 500
				n_cols = 7
				for metrics in [['Training Iterations'],
				                ['Testing DB Score', 'Testing Silhouette', 'Testing Euclidean Distance']
				                ]:
					if 'Train' in metrics[0]:
						fig, axes = plt.subplots(1, n_cols, sharex=True, sharey=True,
						                         figsize=(25, 5))  # (width, height)
						axes = np.asarray(axes).reshape((1, -1))
						fig_name = f'bar_diff_n_sigma_training_metrics_d10'
					else:
						fig, axes = plt.subplots(3, n_cols, sharex=True, sharey=False,
						                         figsize=(25, 12))  # (width, height)
						axes = axes.reshape((3, -1))
						fig_name = f'bar_diff_n_sigma_testing_metrics_d10'

					for idx_yaxis, y_name in enumerate(metrics):
						for i, p in enumerate(ratios):
							p = float(f'{p:.2f}')
							try:
								# for each p, plot n1 vs DB
								result_files = []
								# '3GAUSSIANS-Client_epochs_1-n1_500-sigma1_0.3+n2_5000-sigma2_0.3+n3_10000-sigma3_0.3/ratio_0.00.csv
								result_files.append((os.path.join(in_dir,
								                                  f'{dataset}-Client_epochs_{client_epochs}-Clusters_{n_clusters}-Clients_{n_clients}/n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{p:.2f}:diff_sigma_n.csv'),
								                     n1))
								print(y_name, p, result_files)
								if i == 0:
									# get baseline
									_, bs_df = get_df(result_files)
								else:
									i_, j_ = divmod(i - 1, n_cols)
									if i_ == 0 and j_ == 0:
										is_legend = True
									else:
										is_legend = False
									ax = axes[idx_yaxis + i_][j_]
									barplot_N_P(bs_df, ax, result_files, algorithms[1:], out_dir='results/xlsx',
									            fig_name=f'diff_n_sigma_{p:.2f}', p=p, is_show=True,
									            is_legend=is_legend,
									            y_name=y_name)
							except Exception as e:
								# traceback.print_exc()
								print(e, flush=True)

							tot_cnt += len(ratios)
					# axes[-1][-1].axis('off')
					plt.tight_layout()

					if not os.path.exists(out_dir):
						os.makedirs(out_dir)
					f = os.path.join(out_dir, f'{fig_name}.png')
					plt.savefig(f, dpi=600, bbox_inches='tight')
					print(f)
					if is_show:
						plt.show()

			###########################################################################################
			# for case 4/5:  line plot
			fix_N1 = True  # for Dataset 4/5
			# CASE='Case 4'
			CASE = 'Case 5'
			n_clusters = 10 if CASE == 'Case 4' else 3

			if fix_N1:
				out_dir = 'results/xlsx'
				n1 = 5000  # fixed
				'n1_5000-sigma1_0.3_0.3+n2_1000-sigma2_0.3_0.3+n3_500-sigma3_0.3_0.3/ratio_0.00/diff_sigma_n.xlsx'
				is_show = True
				out_dir = 'results/xlsx'
				ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
				n2 = 1000  # 5000   #
				sigma1 = "0.3_0.3"
				sigma2 = "0.3_0.3"
				sigma3 = "0.3_0.3"
				n3 = 500
				n_cols = 7
				dt_name = f'clients_{n_clients}_clusters_{n_clusters}'
				for metrics in [['Training Iterations'],
				                ['Testing DB Score', 'Testing Silhouette', 'Testing Euclidean Distance']
				                ]:
					if 'Train' in metrics[0]:
						fig, axes = plt.subplots(1, 1, sharex=True, sharey=False, figsize=(5, 3))  # (width, height)
						axes = np.asarray(axes).reshape((1, 1))
						fig_name = f'diff_p_training_metrics_{dt_name}'
					else:
						fig, axes = plt.subplots(1, 3, sharex=True, sharey=False, figsize=(15, 3))  # (width, height)
						axes = axes.reshape((1, 3))
						fig_name = f'diff_p_testing_metrics_{dt_name}'

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
							for p in ratios:
								# '3GAUSSIANS-Client_epochs_1-n1_500-sigma1_0.3+n2_5000-sigma2_0.3+n3_10000-sigma3_0.3/ratio_0.00.csv
								result_files.append((os.path.join(in_dir,
								                                  f'{dataset}-Client_epochs_{client_epochs}-Clusters_{n_clusters}-Clients_{n_clients}/n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{p:.2f}:diff_sigma_n.csv'),
								                     p))
							plot_P(ax, result_files, algorithms, out_dir='results/xlsx',
							       fig_name=None, p=p, is_show=True, is_legend=is_legend,
							       y_name=y_name)
						except Exception as e:
							# traceback.print_exc()
							print(e, flush=True)
						tot_cnt += 1
					plt.tight_layout()

					if not os.path.exists(out_dir):
						os.makedirs(out_dir)
					f = os.path.join(out_dir, f'{fig_name}.png')
					plt.savefig(f, dpi=600, bbox_inches='tight')
					print(os.path.abspath(f))
					if is_show:
						plt.show()

	# else:
	# 	cnt = main(dataset, None, None)
	# 	tot_cnt += cnt

	print()
	print(f'*** Total cases: {tot_cnt}')
