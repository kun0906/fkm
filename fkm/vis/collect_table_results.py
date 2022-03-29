"""
    run:
        module load anaconda3/2021.5
        cd /scratch/gpfs/ky8517/fkm/fkm
        PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 collect_table_results.py
"""
# Email: kun.bj@outlook.com
import json
import os
import traceback

import numpy as np
import pandas as pd
import xlsxwriter

from fkm.experiment_cases import get_experiment_params
from fkm.sbatch import gen_cases, get_data_details_lst
from fkm.utils.utils_func import load

algorithm2abbrv = {'Centralized_true': 'True-CKM',
                   'Centralized_random': 'Random-CKM',
                   'Centralized_kmeans++': 'KM++-CKM',
                   'Federated-Server_random_min_max': 'Random-WA-FKM',
                   'Federated-Server_gaussian': 'Gaussian-WA-FKM',
                   'Federated-Server_average-Client_random': 'C-Random-WA-FKM',
                   'Federated-Server_average-Client_kmeans++': 'C-KM++-WA-FKM',
                   'Federated-Server_greedy-Client_random': 'C-Random-GD-FKM',
                   'Federated-Server_greedy-Client_kmeans++': 'C-KM++-GD-FKM',
                   }

metric2abbrv = {'iterations': 'Training iterations',
                'davies_bouldin': 'DB score',
                'silhouette': 'Silhouette',
                'euclidean': 'Euclidean distance'
                }


def save2csv(df, table_l2, py_name, case, column_idx, client_epochs):
	print(py_name, case, column_idx, client_epochs)
	dataset = case['dataset']
	data_details = case['data_details']
	algorithm = case['algorithm']
	params = get_experiment_params(p0=dataset, p1=data_details, p2=algorithm, client_epochs=client_epochs,
	                               p3=py_name)
	# pprint(params)
	out_dir = params['out_dir']
	print(f'out_dir: {out_dir}')
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	try:
		# read scores from out.txt
		server_init_centroids = params['server_init_centroids']
		client_init_centroids = params['client_init_centroids']
		if False:
			print('deprecated')
			out_txt = os.path.join(out_dir, f'varied_clients-Server_{server_init_centroids}-'
			                                f'Client_{client_init_centroids}.txt')
			with open(out_txt, 'r') as f:
				data = json.load(f)
			s = ' '
			for k, vs in data.items():
				for split in vs.keys():
					s += f'{split}:\n'
					for metric, score in vs[split].items():
						s += f'\t{metric}: ' + '+/-'.join(f'{v:.2f}' for v in score) + '\n'
					s += '\n'
		else:
			out_dat = os.path.join(out_dir, f'varied_clients-Server_{server_init_centroids}-'
			                                f'Client_{client_init_centroids}-histories.dat')
			histories = load(out_dat)
			for n_clients, history_res in histories.items():
				results_avg = history_res['results_avg']
				n_clients = history_res['n_clients']
				results = history_res['history']['results']
				s = ''
				c1 = ''
				training_iterations_lst = []
				scores_lst = []
				final_centroids_lst = []
				for vs in results:
					seed = vs['seed']
					# print(f'seed: {seed}')
					training_iterations_lst.append(vs['training_iterations'])
					scores_lst.append(vs['scores'])
				# final_centroids_lst += ['(' + ', '.join(f'{v:.5f}' for v in cen) + ')' for cen
				#                         in vs['final_centroids'].tolist()]

				for split in scores_lst[0].keys():  # ['train', 'test']
					# s += f'{split}:\n'
					# c1 += f'{split}:\n'
					if split == 'train':
						if column_idx == 0:
							c1 += 'iterations\n'
						s += f'{np.mean(training_iterations_lst):.2f} +/- ' \
						     f'{np.std(training_iterations_lst):.2f}\n'
					else:
						c1 += 'iterations\n'
						s += '\n'
					for metric in scores_lst[0][split].keys():
						metric_scores = [scores[split][metric] for scores in scores_lst]
						if column_idx == 0:
							c1 += f'{metric2abbrv[metric]}\n'
						s += f'{np.mean(metric_scores):.2f} +/- {np.std(metric_scores):.2f}\n'
				# s += '\n'

				# # final centroids distribution
				# s += 'final centroids distribution: \n'
				# ss_ = sorted(collections.Counter(final_centroids_lst).items(), key=lambda kv: kv[1], reverse=True)
				# tot_centroids = len(final_centroids_lst)
				# s += '\t\n'.join(f'{cen_}: {cnt_ / tot_centroids * 100:.2f}% - ({cnt_}/{tot_centroids})' for
				#                  cen_, cnt_ in ss_)
				if column_idx == 0:
					df['metric'] = c1.split('\n')
				df[algorithm2abbrv[algorithm]] = s.split('\n')
				break
	except Exception as e:
		print(f'Error: {e}')
		data = '-'

#
# def main(dataset, n1=None, sigma1 = 0.2,  n2 = None, sigma2 = 0.3, n3 = 10000, sigma3 = 0.3, ratio=0.1):
# 	tot_cnt = 0
# 	client_epochs = 1
# 	out_dir = 'results/xlsx'
# 	# if os.path.exists(out_dir):
# 	# 	shutil.rmtree(out_dir, ignore_errors=True)
# 	if not os.path.exists(out_dir):
# 		os.makedirs(out_dir)
#
# 	# dataset = 'FEMNIST'
# 	# Create an new Excel file and add a worksheet.
# 	# if dataset == '3GAUSSIANS' and n1 != None:
# 	table_file = f'{out_dir}/{dataset}-Client_epochs_{client_epochs}-n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}.csv'
# 	# workbook = xlsxwriter.Workbook(f'{out_dir}/{dataset}-Client_epochs_{client_epochs}-n1_{n1}-ratio_{ratio:.2f}.xlsx')
# 	df = pd.DataFrame()
# 	# f = open(table_file, 'w')
#
# 	py_names = [
# 		'Centralized_Kmeans',
# 		'Stanford_server_random_initialization',
# 		'Stanford_client_initialization',
# 		'Our_greedy_initialization',
# 		'Our_greedy_center',
# 		'Our_greedy_2K',
# 		'Our_greedy_K_K',
# 		# 'Our_greedy_concat_Ks',
# 		# 'Our_weighted_kmeans_initialization',
# 	]
# 	cnt = 0
# 	if dataset == 'FEMNIST':
# 		data_details_lst = ['1client_1writer_multidigits', '1client_multiwriters_multidigits',
# 		                    '1client_multiwriters_1digit']
# 	elif dataset == '2GAUSSIANS':
# 		data_details_lst = [
# 			'1client_1cluster', 'mix_clusters_per_client',
# 			'1client_ylt0', '1client_xlt0',
# 			'1client_1cluster_diff_sigma', 'diff_sigma_n',
# 			'1client_xlt0_2',
# 		]
# 	elif dataset == '3GAUSSIANS':
# 		if n1 == None:
# 			# data_details_lst = [
# 			# 	                   # '1client_1cluster',  this case is included in '0.0:mix_clusters_per_client'
# 			# 	                   # '1client_ylt0', '1client_xlt0',
# 			# 	                   # '1client_1cluster_diff_sigma',    this case is included in 'diff_sigma_n'
# 			# 	                   # 'diff_sigma_n',
# 			# 	                   # '1client_xlt0_2',
# 			#                    ]
# 			tmp_lst = []
# 			for ratio in [ratio]:
# 				tmp_lst.append(f'ratio_{ratio:.2f}:mix_clusters_per_client')
# 			data_details_lst = tmp_lst
# 		else:
# 			# n1_100-sigma1_0.1+n2_5000-sigma2_0.2+n3_10000-sigma3_0.3:ratio_0.1:diff_sigma_n
# 			tmp_list = []
# 			N = 10000
# 			for ratio in [ratio]:  # range(0, 10, 1):
# 				# for n1 in [0, 100, 500, 1000, 2000, 5000, N]:
# 				for n1 in [n1]:
# 					for sigma1 in [sigma1]:
# 						for n2 in [n2]:
# 							for sigma2 in [sigma2]:
# 								for n3 in [N]:
# 									for sigma3 in [sigma3]:
# 										p1 = f'n1_{n1}-sigma1_{sigma1}+n2_{n2}-sigma2_{sigma2}+n3_{n3}-sigma3_{sigma3}:ratio_{ratio:.2f}:diff_sigma_n'
# 										tmp_list.append(p1)
#
# 			data_details_lst = tmp_list
# 	elif dataset == '5GAUSSIANS':
# 		data_details_lst = [
# 			'5clients_5clusters', '5clients_4clusters', '5clients_3clusters',
# 		]
# 	else:
# 		msg = f'{dataset}'
# 		raise NotImplementedError(msg)
# 	sheet_names = set()
# 	for data_details in data_details_lst:
# 		cnt_ = 0
# 		sheet_name = data_details[:25].replace(':', ' ')
# 		if sheet_name in sheet_names:
# 			sheet_name = sheet_name + f'{len(sheet_names)}'
# 		sheet_names.add(sheet_name)
# 		print(f'xlsx_sheet_name: {sheet_name}')
# 		# worksheet = workbook.add_worksheet(name=sheet_name)
# 		cnt_ = 0
# 		for py_name in py_names:
# 			for case in gen_cases(py_name, dataset, data_details):
# 				try:
# 					save2csv(df, '', py_name, case, cnt_, client_epochs)
# 				except Exception as e:
# 					traceback.print_exc()
# 				cnt_ += 1
# 		print(f'{py_name}: {cnt_} cases.\n')
# 		cnt += cnt_
# 	tot_cnt += cnt
# 	print(f'* {dataset} cases: {cnt}\n')
#
# 	df.to_csv(table_file, index=False)
# 	print(f'** Total cases: {tot_cnt}, out_dir:{out_dir}')
# 	return tot_cnt
#
#
# if __name__ == '__main__':
# 	tot_cnt = 0
# 	for dataset in ['3GAUSSIANS']:  # [ '2GAUSSIANS', 'FEMNIST']:
# 		if dataset == '3GAUSSIANS':
# 			# ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
# 			ratios = [ratio for ratio in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.4999]]
# 			for ratio in ratios:
# 				cnt = main(dataset, None, ratio)  # if n1 == None
# 				tot_cnt += cnt
#
# 			for n1 in [100, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
# 				for n2 in [5000, 10000]:
# 					for ratio in ratios:
# 						cnt = main(dataset, n1, sigma1=0.2,  n2=n2,sigma2=0.3, n3=10000, sigma3=0.3,  ratio=ratio)
# 						tot_cnt += cnt
# 						cnt = main(dataset, n1, sigma1=0.3, n2=n2, sigma2=0.2, n3=10000, sigma3=0.3, ratio=ratio)
# 						tot_cnt += cnt
# 						cnt = main(dataset, n1, sigma1=0.3, n2=n2, sigma2=0.3, n3=10000, sigma3=0.3, ratio=ratio)
# 						tot_cnt += cnt
# 		else:
# 			cnt = main(dataset, None, None)
# 			tot_cnt += cnt
#
# 	print()
# 	print(f'*** Total cases: {tot_cnt}')


def main2():
	tot_cnt = 0
	client_epochs = 1
	n_clusters = 3
	n_clients = 10
	out_dir = 'results/xlsx'
	# if os.path.exists(out_dir):
	# 	shutil.rmtree(out_dir, ignore_errors=True)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	dataset, py_names, data_details_lst = get_data_details_lst()
	sheet_names = set()
	for data_details in data_details_lst:
		table_file = f'{out_dir}/{dataset}-Client_epochs_{client_epochs}-Clusters_{n_clusters}-Clients_{n_clients}/{data_details}.csv'
		df = pd.DataFrame()
		sheet_name = data_details[:25].replace(':', ' ')
		if sheet_name in sheet_names:
			sheet_name = sheet_name + f'{len(sheet_names)}'
		sheet_names.add(sheet_name)
		print(f'xlsx_sheet_name: {sheet_name}')
		cnt_ = 0
		for py_name in py_names:
			for case in gen_cases(py_name, dataset, data_details):
				try:
					# save2xls(workbook, worksheet, py_name, case, cnt_, client_epochs)
					save2csv(df, '', py_name, case, cnt_, client_epochs)
				except Exception as e:
					traceback.print_exc()
				cnt_ += 1
		print(f'{py_name}: {cnt_} cases.\n')
		tot_cnt += cnt_
		df.to_csv(table_file, index=False)

	print(f'** Total cases: {tot_cnt}, out_dir:{out_dir}')
	return tot_cnt

if __name__ == '__main__':
	tot_cnt = main2()
	print()
	print(f'*** Total cases: {tot_cnt}')
