import os
import traceback

import matplotlib.pyplot as plt
import pandas as pd


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


def plot_P_DB(ax, result_files, algorithms = ['Random-WA-FKM', 'Gaussian-WA-FKM'], fig_name='diff_n',
              out_dir='results/xlsx', params={}, n1 = 100, is_show = False, is_legend=False):
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



def plot_N_DB(ax, result_files, algorithms = ['Random-WA-FKM', 'Gaussian-WA-FKM'], fig_name='diff_n',
              out_dir='results/xlsx', params={}, p = 0.00, is_show = False, is_legend=False, y_name = 'Training Iterations'):
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
	# y_name = 'Training Iterations'
	for i, alg in enumerate(algorithms):
		if y_name ==  'Testing DB Score':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		elif y_name == 'Training Iterations':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		elif y_name == 'Testing Silhouette':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))
		elif y_name == 'Testing Euclidean Distance':
			y, yerr = list(zip(*df[df['Algorithm'] == alg][y_name]))

		print(alg, yerr, x, y)
		ax.errorbar(x, y, yerr=yerr, linestyle = '-', marker = 'o', label=alg, lw=2, color=colors[i], ecolor=colors[i], elinewidth=1, capsize=2)

	if is_legend:
		ax.legend(fontsize=7)

	fontsize = 13
	if y_name == 'Training Iterations':
		# ax.set_ylim([5, 150])
		ax.set_ylabel('Training $T$', fontsize = fontsize)
	elif y_name == 'Testing DB Score':
		# ax.set_ylim([ -0.02, 0.03])
		ax.set_ylabel('Testing ${DB}$', fontsize=fontsize)
	elif y_name == 'Testing Silhouette':
		# ax.set_ylim([-0.025, 0.02])
		ax.set_ylabel('Testing ${SC}$', fontsize=fontsize)
	elif y_name == 'Testing Euclidean Distance':
		# ax.set_ylim([-0.05, 0.06])
		ax.set_ylabel('Testing ${\overline{WCSS}}$' , fontsize=fontsize)
	ax.set_xlabel(f'$N_1$ (P={p})', fontsize=fontsize)

	# plt.tight_layout()
	#
	# if not os.path.exists(out_dir):
	# 	os.makedirs(out_dir)
	# f = os.path.join(out_dir, f'{fig_name}.png')
	# plt.savefig(f, dpi=600, bbox_inches='tight')
	# if is_show:
	# 	plt.show()

# df.plot()
# # Plot init seeds along side sample data
# fig, ax = plt.subplots()
# # colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
# colors = ["r", "g", "b", "m", 'black']
# # ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='centroid_1')
# # p = np.mean(X1, axis=0)
# # ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
#
#
# offset = 0.3
# # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
# xytext = (p[0] - offset, p[1] - offset)
# print(xytext)
# ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
#             ha='center', va='center',  # textcoords='offset points',
#             bbox=dict(facecolor='none', edgecolor='b', pad=1),
#             arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
#                             connectionstyle="angle3, angleA=90,angleB=0"))
#
#
# ax.axvline(x=0, color='k', linestyle='--')
# ax.axhline(y=0, color='k', linestyle='--')
# ax.legend(loc='upper right')
# plt.title(params['p1'])
# # # plt.xlim([-2, 15])
# # # plt.ylim([-2, 15])
# plt.xlim([-6, 6])
# plt.ylim([-6, 6])
# # # plt.xticks([])
# # # plt.yticks([])
# plt.tight_layout()
# if not os.path.exists(params['out_dir']):
# 	os.makedirs(params['out_dir'])
# f = os.path.join(params['out_dir'], params['p1'] + '.png')
# plt.savefig(f, dpi=600, bbox_inches='tight')
# plt.show()
#
#
# return out_file


if __name__ == '__main__':
	in_dir = '~/Downloads/xlsx'
	tot_cnt = 0
	for dataset in ['3GAUSSIANS']:  # [ '2GAUSSIANS', 'FEMNIST']:
		if dataset == '3GAUSSIANS':
			client_epochs = 1
			n_clusters = 3
			n_clients = 3
			# # ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
			# ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
			# result_files = []
			# # algorithms = ['Random-CKM', 'KM++-CKM', 'Random-WA-FKM', 'C-Random-WA-FKM', 'C-KM++-WA-FKM',
			# #               'C-Random-GD-FKM', 'C-KM++-GD-FKM']
			algorithms = ['KM++-CKM', 'Random-WA-FKM', 'C-Random-WA-FKM', 'C-KM++-WA-FKM', 'C-Random-GD-FKM',
			              'C-KM++-GD-FKM']
			# for p in ratios:
			# 	result_files.append((os.path.join(in_dir, f'{dataset}-Client_epochs_1-n1_None-n2_None-ratio_{p:.2f}.csv'), p))
			# # plot_P_DB(result_files, algorithms , out_dir='results/xlsx', fig_name='diff_n', is_show=True)
			# tot_cnt += len(ratios)
			# flg = True
			# if flg:
			# 	fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(12, 8))  # width, height
			# 	is_show = True
			# 	out_dir = 'results/xlsx'
			# 	fig_name = f'diff_n_sigma_p'
			# 	# ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
			# 	ratios = [ratio for ratio in [0,  0.01, 0.05,  0.3,  0.4999]]
			# 	n1s = [100, 1000, 5000, 10000]
			# 	for i, n1 in enumerate(n1s):
			# 		# for each n1, plot p vs DB
			# 		i_, j_ = divmod(i, 4)
			# 		if i_ == 0 and j_ == 0:
			# 			is_legend = True
			# 		else:
			# 			is_legend = False
			# 		ax = axes[i_][j_]
			#
			# 		result_files = []
			# 		for p in ratios:
			# 			p = float(f'{p:.2f}')
			# 			result_files.append((os.path.join(in_dir, f'{dataset}-Client_epochs_1-n1_{n1}-ratio_{p:.2f}.csv'), p))
			# 			# result_files.append((os.path.join(in_dir, f'{dataset}-Client_epochs_1-n1_{n1}-n2_10000-ratio_{p:.2f}.csv'), p))
			# 		plot_P_DB(ax, result_files, algorithms, out_dir='results/xlsx',fig_name=f'diff_n_sigma_n1_{n1}', n1 = n1, is_show=True, is_legend=is_legend)
			# 		tot_cnt += len(n1s)
			#
			# 	plt.tight_layout()
			#
			# 	if not os.path.exists(out_dir):
			# 		os.makedirs(out_dir)
			# 	f = os.path.join(out_dir, f'{fig_name}.png')
			# 	plt.savefig(f, dpi=600, bbox_inches='tight')
			#
			# 	if is_show:
			# 		plt.show()

			#########################################################################################################
			# for Dataset 2
			flg = True
			if flg:
				is_show = True
				out_dir = 'results/xlsx'
				ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
				n1s = [100, 500, 1000, 2000, 3000, 5000, 8000, 10000]
				# n1s = [500, 2000, 3000, 5000, 8000]
				n2 = 5000
				fig_dataset2 = True
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
				for y_name in ['Training Iterations',  'Testing DB Score', 'Testing Silhouette', 'Testing Euclidean Distance']:
					if y_name == 'Training Iterations':
						fig_name = f'diff_n1_p_training_iters_{dt_name}'
					elif y_name == 'Testing DB Score':
						fig_name = f'diff_n1_p_testing_db_{dt_name}'
					elif y_name == 'Testing Silhouette':
						fig_name = f'diff_n1_p_testing_sc_{dt_name}'
					elif y_name == 'Testing Euclidean Distance':
						fig_name = f'diff_n1_p_testing_wcss_{dt_name}'

					fig, axes = plt.subplots(2, 4, sharex=True, sharey=True, figsize=(12, 8))  # width, height
					for i, p in enumerate(ratios):
						p = float(f'{p:.2f}')
						i_,j_ = divmod(i, 4)
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
								result_files.append((os.path.join(in_dir,  f'{dataset}-Client_epochs_{client_epochs}-Clusters_{n_clusters}-Clients_{n_clients}/n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_10000-sigma3_{sigma3}:ratio_{p:.2f}:diff_sigma_n.csv'), n1))
							plot_N_DB(ax, result_files, algorithms, out_dir='results/xlsx', fig_name=f'diff_n_sigma_{p:.2f}', p=p, is_show=True, is_legend=is_legend, y_name=y_name)
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
		# else:
		# 	cnt = main(dataset, None, None)
		# 	tot_cnt += cnt

	print()
	print(f'*** Total cases: {tot_cnt}')




