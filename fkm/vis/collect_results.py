"""
    run:
        module load anaconda3/2021.5
        cd /scratch/gpfs/ky8517/fkm/fkm
        PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 vis/collect_results.py
"""
# Email: kun.bj@outlook.com
import collections
import copy
import json
import os
import traceback
import warnings
from pprint import pprint

import numpy as np
import xlsxwriter

from fkm import config
from fkm.main_all import get_datasets_config_lst, get_algorithms_config_lst
from fkm.utils.utils_func import load


def parser_history(args):

	OUT_DIR =  args['OUT_DIR']
	# read scores from out.txt
	# server_init_method = args['ALGORITHM']['server_init_method']
	# client_init_method = args['ALGORITHM']['client_init_method']
	try:
		out_dat = os.path.join(OUT_DIR, 'history.dat')
		if not os.path.exists(out_dat):
			out_dat = os.path.join(OUT_DIR, f'history.data.json')
			with open(out_dat) as f:
				history = json.load(f)
		else:
			history = load(out_dat)
	except Exception as e:
		warnings.warn(f'Load Error: {e}')
		raise FileNotFoundError

	# get the average and std
	results_avg = {}
	# N_CLUSTERS = args['N_CLUSTERS']
	try:
		SEEDS = history['SEEDS']
		for split in args['SPLITS']:
			# s = f'*{split}:\n'
			metric_names = history[SEEDS[0]]['scores'][split].keys()
			if split == 'train':
				training_iterations = []
				initial_centroids = []
				final_centroids = []
				final_centroids_lst = []
				durations = []
				for seed in SEEDS:
					# if history[seed]['scores'][split]['n_clusters'] != history[seed]['scores'][split]['n_clusters_pred']:
					# 	continue
					initial_centroids+=[[ f'{v:.5f}' for v in vs ] for vs in history[seed]['initial_centroids']]
					final_centroids += [[ f'{v:.5f}' for v in vs ] for vs in history[seed]['final_centroids']]
					final_centroids_lst += ['(' + ', '.join(f'{v:.5f}' for v in vs) + ')' for vs in history[seed]['final_centroids']]
					training_iterations.append(history[seed]['training_iterations'])
					durations.append(history[seed]['duration'])
				results_avg[split] = {'metric_names': metric_names,
									'Iterations': (np.mean(training_iterations), np.std(training_iterations)),
				                      'durations': (np.mean(durations), np.std(durations)),
				                        'initial_centroids': initial_centroids,
				                        'final_centroids': final_centroids,
				                        'final_centroids_lst': final_centroids_lst}
				# s += f'\titerations: {np.mean(training_iterations):.2f} +/- '\
				#      f'{np.std(training_iterations):.2f}\n'
				# s += f'\tdurations: {np.mean(durations):.2f} +/- ' \
				#      f'{np.std(durations):.2f}\n'
			else:
				results_avg[split] = {'Iterations': ('', '')}
			for metric_name in metric_names:
				value = []
				for seed in SEEDS:
					# if history[seed]['scores'][split]['n_clusters'] != history[seed]['scores'][split]['n_clusters_pred']:
					# 	warnings.warn(f'n_clusters!=n_clusters_pred')
					# 	continue
					value.append(history[seed]['scores'][split][metric_name])
				if metric_name in ['labels_pred', 'labels_true', 'n_clusters', 'n_clusters_pred']:
					# results_avg[split][metric_name] = [history[seed]['scores'][split][metric_name] for seed in SEEDS if history[seed]['scores'][split]['n_clusters'] == history[seed]['scores'][split]['n_clusters_pred']]
					results_avg[split][metric_name] = [history[seed]['scores'][split][metric_name] for seed in SEEDS]
					# s += f'\t{metric_name}: {value}\n'
					continue
				try:
					score_mean = np.mean(value)
					score_std = np.std(value)
				# score_mean = np.around(np.mean(value), decimals=3)
				# score_std = np.around(np.std(value), decimals=3)
				except Exception as e:
					print(f'Error: {e}, {split}, {metric_name}, {value}')
					score_mean = np.nan
					score_std = np.nan
				results_avg[split][metric_name] = (score_mean, score_std)
				# s += f'\t{metric_name}: {score_mean:.2f} +/- {score_std:.2f}\n'

			# s += f'initial_centroids:\n{initial_centroids}\n'
			# s += f'final_centroids:\n{final_centroids}\n'
			# # final centroids distribution
			# s += 'final centroids distribution: \n'
			# ss_ = sorted(collections.Counter(final_centroids_lst).items(), key=lambda kv: kv[1], reverse=True)
			# tot_centroids = len(final_centroids_lst)
			# s += '\t\n'.join(f'{cen_}: {cnt_ / tot_centroids * 100:.2f}% - ({cnt_}/{tot_centroids})' for
			#                  cen_, cnt_ in ss_)
	except Exception as e:
		warnings.warn(f'Parser Error: {e}')
		raise FileNotFoundError

	return results_avg


def parser_history_topk(args):

	OUT_DIR =  args['OUT_DIR']
	# read scores from out.txt
	# server_init_method = args['ALGORITHM']['server_init_method']
	# client_init_method = args['ALGORITHM']['client_init_method']
	try:
		out_dat = os.path.join(OUT_DIR, 'history.dat')
		if not os.path.exists(out_dat):
			out_dat = os.path.join(OUT_DIR, f'history.data.json')
			with open(out_dat) as f:
				history = json.load(f)
		else:
			history = load(out_dat)
	except Exception as e:
		warnings.warn(f'Load Error: {e}')
		raise FileNotFoundError

	# get the average and std
	results_avg = {}
	# N_CLUSTERS = args['N_CLUSTERS']
	try:
		SEEDS = history['SEEDS']
		for split in args['SPLITS']:
			# s = f'*{split}:\n'
			# only get the top 2 values for each metric
			tmp = [(i, history[seed]['scores'][split]['ari']) for i, seed in enumerate(SEEDS)]
			tmp = sorted(tmp, key = lambda x: x[1], reverse=True)
			keep_indices = set([i for i, v in tmp[:2]])
			metric_names = history[SEEDS[0]]['scores'][split].keys()
			if split == 'train':
				training_iterations = []
				initial_centroids = []
				final_centroids = []
				final_centroids_lst = []
				durations = []
				for i, seed in enumerate(SEEDS):
					if i not in keep_indices: continue
					if history[seed]['scores'][split]['n_clusters'] != history[seed]['scores'][split]['n_clusters_pred']:
						continue
					initial_centroids += [[f'{v:.5f}' for v in vs] for vs in history[seed]['initial_centroids']]
					final_centroids += [[f'{v:.5f}' for v in vs] for vs in history[seed]['final_centroids']]
					final_centroids_lst += ['(' + ', '.join(f'{v:.5f}' for v in vs) + ')' for vs in
					                        history[seed]['final_centroids']]
					training_iterations.append(history[seed]['training_iterations'])
					durations.append(history[seed]['duration'])
				results_avg[split] = {'metric_names': metric_names,
									'Iterations': (np.mean(training_iterations), np.std(training_iterations)),
				                      'durations': (np.mean(durations), np.std(durations)),
				                        'initial_centroids': initial_centroids,
				                        'final_centroids': final_centroids,
				                        'final_centroids_lst': final_centroids_lst}
				# s += f'\titerations: {np.mean(training_iterations):.2f} +/- '\
				#      f'{np.std(training_iterations):.2f}\n'
				# s += f'\tdurations: {np.mean(durations):.2f} +/- ' \
				#      f'{np.std(durations):.2f}\n'
			else:
				results_avg[split] = {'Iterations': ('', '')}
			for metric_name in metric_names:
				value = []
				for i, seed in enumerate(SEEDS):
					if i not in keep_indices: continue
					if history[seed]['scores'][split]['n_clusters'] != history[seed]['scores'][split]['n_clusters_pred']:
						warnings.warn(f'n_clusters!=n_clusters_pred')
						continue
					value.append(history[seed]['scores'][split][metric_name])
				if metric_name in ['labels_pred', 'labels_true', 'n_clusters', 'n_clusters_pred']:
					results_avg[split][metric_name] = [history[seed]['scores'][split][metric_name] for i, seed in
					                                   enumerate(SEEDS) if (history[seed]['scores'][split]['n_clusters']
					                                   == history[seed]['scores'][split]['n_clusters_pred']) and
					                                   (i in keep_indices)]
					# s += f'\t{metric_name}: {value}\n'
					continue
				try:
					score_mean = np.mean(value)
					score_std = np.std(value)
				# score_mean = np.around(np.mean(value), decimals=3)
				# score_std = np.around(np.std(value), decimals=3)
				except Exception as e:
					print(f'Error: {e}, {split}, {metric_name}, {value}')
					score_mean = np.nan
					score_std = np.nan
				results_avg[split][metric_name] = (score_mean, score_std)
				# s += f'\t{metric_name}: {score_mean:.2f} +/- {score_std:.2f}\n'

			# s += f'initial_centroids:\n{initial_centroids}\n'
			# s += f'final_centroids:\n{final_centroids}\n'
			# # final centroids distribution
			# s += 'final centroids distribution: \n'
			# ss_ = sorted(collections.Counter(final_centroids_lst).items(), key=lambda kv: kv[1], reverse=True)
			# tot_centroids = len(final_centroids_lst)
			# s += '\t\n'.join(f'{cen_}: {cnt_ / tot_centroids * 100:.2f}% - ({cnt_}/{tot_centroids})' for
			#                  cen_, cnt_ in ss_)
	except Exception as e:
		warnings.warn(f'Parser Error: {e}')
		raise FileNotFoundError

	return results_avg


def parser_history2(args):

	OUT_DIR =  args['OUT_DIR']
	# read scores from out.txt
	# server_init_method = args['ALGORITHM']['server_init_method']
	# client_init_method = args['ALGORITHM']['client_init_method']
	try:
		out_dat = os.path.join(OUT_DIR, 'history.dat')
		if not os.path.exists(out_dat):
			out_dat = os.path.join(OUT_DIR, f'history.data.json')
			with open(out_dat) as f:
				history = json.load(f)
		else:
			history = load(out_dat)
	except Exception as e:
		warnings.warn(f'Load Error: {e}')
		raise FileNotFoundError

	# get the average and std
	results_detail = {}
	# N_CLUSTERS = args['N_CLUSTERS']
	try:
		SEEDS = history['SEEDS']
		for split in args['SPLITS']:
			# s = f'*{split}:\n'
			metric_names = history[SEEDS[0]]['scores'][split].keys()
			if split == 'train':
				training_iterations = []
				initial_centroids = []
				final_centroids = []
				final_centroids_lst = []
				durations = []
				for seed in SEEDS:
					initial_centroids+=[[ f'{v:.5f}' for v in vs ] for vs in history[seed]['initial_centroids']]
					final_centroids += [[ f'{v:.5f}' for v in vs ] for vs in history[seed]['final_centroids']]
					final_centroids_lst += ['(' + ', '.join(f'{v:.5f}' for v in vs) + ')' for vs in history[seed]['final_centroids']]
					training_iterations.append(history[seed]['training_iterations'])
					durations.append(history[seed]['duration'])
				results_detail[split] = {'metric_names': metric_names,
									'Iterations': training_iterations,
				                      'durations': durations,
				                        'initial_centroids': initial_centroids,
				                        'final_centroids': final_centroids,
				                        'final_centroids_lst': final_centroids_lst}
				# s += f'\titerations: {np.mean(training_iterations):.2f} +/- '\
				#      f'{np.std(training_iterations):.2f}\n'
				# s += f'\tdurations: {np.mean(durations):.2f} +/- ' \
				#      f'{np.std(durations):.2f}\n'
			else:
				results_detail[split] = {'Iterations': ('', '')}
			for metric_name in metric_names:
				results_detail[split][metric_name] = [history[seed]['scores'][split][metric_name] for seed in SEEDS]

	except Exception as e:
		warnings.warn(f'Parser Error: {e}')
		raise FileNotFoundError

	return results_detail


def save2xls(workbook, worksheet, column_idx, args, results_avg):
	dataset_name = args['DATASET']['name']
	dataset_detail = args['DATASET']['detail']
	# algorithm_name = args['ALGORITHM']['name']
	algorithm_py_name = args['ALGORITHM']['py_name']
	algorithm_detail = args['ALGORITHM']['detail']
	# print(params)
	OUT_DIR = args['OUT_DIR']
	# print(f'OUT_DIR: {OUT_DIR}')
	# set background color
	cell_format = workbook.add_format()
	cell_format.set_pattern(1)  # This is optional when using a solid fill.
	cell_format.set_text_wrap()
	# new cell_format to add more formats to one cell
	cell_format2 = copy.deepcopy(cell_format)
	cell_format2.set_bg_color('FF0000')
	# cell_format.set_bg_color('#FFFFFE')

	row = 0
	# add dataset details, e.g., plot
	worksheet.set_row(row, 100)  # set row height to 100
	if column_idx == 1:
		scale = 0.248
		dataset_img = os.path.join(OUT_DIR, dataset_detail + '.png')
		if os.path.exists(dataset_img):
			worksheet.insert_image(row, column_idx, dataset_img,
			                       {'x_scale': scale, 'y_scale': scale, 'object_position': 1})
	row += 1

	# Widen the first column to make the text clearer.
	worksheet.set_row(row, 100)  # set row height to 100
	worksheet.set_column(column_idx, column_idx, width=50, cell_format=cell_format)
	# Insert an image offset in the cell.
	if column_idx == 0:
		s = f'{OUT_DIR}'
	else:
		s = ''
	worksheet.write(row, column_idx, s)
	if column_idx == 1:
		dataset_img = os.path.join(OUT_DIR, dataset_detail + '-' + args['NORMALIZE_METHOD'] + '.png')
		if os.path.exists(dataset_img):
			worksheet.insert_image(row, column_idx, dataset_img,
			                       {'x_scale': scale, 'y_scale': scale, 'object_position': 1})
	row += 1

	# write the second row
	s = f'{algorithm_py_name}'
	steps = 2
	start = column_idx // steps
	colors = ['#F0FEFE', '#FEF0FE', '#FEFEF0', '#F0FBFB', '#FBF0FB', '#FBFBF0']
	if column_idx == 0:
		cell_format.set_bg_color('#F0FEFE')
	elif column_idx == 1:
		cell_format.set_bg_color('#FEF0FE')
	elif start * steps <= column_idx < start * steps + steps:
		cell_format.set_bg_color(colors[start % len(colors) + 2])  # #RRGGBB
	else:
		cell_format.set_bg_color('#FFFFFF')  # white
	worksheet.write(row, column_idx, s)
	row += 1
	# Insert an image offset in the cell.
	s = f'{dataset_name}\n{dataset_detail}\n{algorithm_py_name}\n{algorithm_detail}\n'
	worksheet.set_row(row, 60)  # set row height to 100
	worksheet.write(row, column_idx, s)
	row += 1
	# try:
	# 	# read scores from out.txt
	# 	# server_init_method = args['ALGORITHM']['server_init_method']
	# 	# client_init_method = args['ALGORITHM']['client_init_method']
	# 	try:
	# 		out_dat = os.path.join(OUT_DIR, 'history.dat')
	# 		if not os.path.exists(out_dat):
	# 			out_dat = os.path.join(OUT_DIR, f'history.data.json')
	# 			with open(out_dat) as f:
	# 				history = json.load(f)
	# 		else:
	# 			history = load(out_dat)
	# 	except Exception as e:
	# 		warnings.warn(f'Load Error: {e}')
	# 		raise FileNotFoundError
	#
	# 	# get the average and std
	# 	results_avg = {}
	# 	SEEDS = history['SEEDS']
	# 	for split in args['SPLITS']:
	# 		s = f'*{split}:\n'
	# 		metric_names = history[SEEDS[0]]['scores'][split].keys()
	# 		if split == 'train':
	# 			training_iterations = [history[seed]['training_iterations'] for seed in SEEDS]
	# 			initial_centroids = []
	# 			final_centroids = []
	# 			final_centroids_lst = []
	# 			for seed in SEEDS:
	# 				initial_centroids+=[[ f'{v:.5f}' for v in vs ] for vs in history[seed]['initial_centroids']]
	# 				final_centroids += [[ f'{v:.5f}' for v in vs ] for vs in history[seed]['final_centroids']]
	# 				final_centroids_lst += ['(' + ', '.join(f'{v:.5f}' for v in vs) + ')' for vs in history[seed]['final_centroids']]
	# 			durations = [history[seed]['duration'] for seed in SEEDS]
	# 			results_avg[split] = {'Iterations': (np.mean(training_iterations), np.std(training_iterations))}
	# 			s += f'\titerations: {np.mean(training_iterations):.2f} +/- '\
	# 			     f'{np.std(training_iterations):.2f}\n'
	# 			s += f'\tdurations: {np.mean(durations):.2f} +/- ' \
	# 			     f'{np.std(durations):.2f}\n'
	# 		else:
	# 			results_avg[split] = {'Iterations': ('', '')}
	# 		for metric_name in metric_names:
	# 			value = [history[seed]['scores'][split][metric_name] for seed in SEEDS]
	# 			if metric_name in ['labels_pred', 'labels_true']:
	# 				results_avg[split][metric_name] = value
	# 				s += f'\t{metric_name}: {value}\n'
	# 				continue
	# 			try:
	# 				score_mean = np.mean(value)
	# 				score_std = np.std(value)
	# 			# score_mean = np.around(np.mean(value), decimals=3)
	# 			# score_std = np.around(np.std(value), decimals=3)
	# 			except Exception as e:
	# 				print(f'Error: {e}, {split}, {metric_name}, {value}')
	# 				score_mean = np.nan
	# 				score_std = np.nan
	# 			results_avg[split][metric_name] = (score_mean, score_std)
	# 			s += f'\t{metric_name}: {score_mean:.2f} +/- {score_std:.2f}\n'
	#
	# 		s += f'initial_centroids:\n{initial_centroids}\n'
	# 		s += f'final_centroids:\n{final_centroids}\n'
	# 		# final centroids distribution
	# 		s += 'final centroids distribution: \n'
	# 		ss_ = sorted(collections.Counter(final_centroids_lst).items(), key=lambda kv: kv[1], reverse=True)
	# 		tot_centroids = len(final_centroids_lst)
	# 		s += '\t\n'.join(f'{cen_}: {cnt_ / tot_centroids * 100:.2f}% - ({cnt_}/{tot_centroids})' for
	# 		                 cen_, cnt_ in ss_)
	#
	# 		data = s
	# 		# Insert an image offset in the cell.
	# 		worksheet.set_row(row, 400)  # set row height to 100
	# 		# cell_format = workbook.add_format({'bold': True, 'italic': True})
	# 		# cell_format2 = workbook.add_format()
	# 		cell_format.set_align('top')
	# 		worksheet.write(row, column_idx, data, cell_format)
	# 		row += 1
	# 	# break
	# except Exception as e:
	# 	print(f'Error: {e}')
	# 	traceback.print_exc()
	# 	data = '-'
	try:
		for split in args['SPLITS']:
			metric_names = results_avg[split]['metric_names']
			s = f'*{split}:\n'
			if split == 'train':
				Iterations = results_avg[split]['Iterations']
				durations = results_avg[split]['durations']
				s += f'\titerations: {Iterations[0]:.2f} +/- '\
				     f'{Iterations[1]:.2f}\n'
				s += f'\tdurations: {durations[0]:.2f} +/- ' \
				     f'{durations[1]:.2f}\n'
			else:
				s += '\n'
			for metric_name in metric_names:
				if metric_name in ['labels_pred', 'labels_true', 'n_clusters', 'n_clusters_pred']:
					value = results_avg[split][metric_name]
					s += f'\t{metric_name}: {value}\n'
					continue
				score_mean, score_std = results_avg[split][metric_name]
				s += f'\t{metric_name}: {score_mean:.2f} +/- {score_std:.2f}\n'

			initial_centroids = results_avg[split]['initial_centroids']
			final_centroids = results_avg[split]['final_centroids']
			final_centroids_lst = results_avg[split]['final_centroids_lst']
			s += f'initial_centroids:\n{initial_centroids}\n'
			s += f'final_centroids:\n{final_centroids}\n'
			# final centroids distribution
			s += 'final centroids distribution: \n'
			ss_ = sorted(collections.Counter(final_centroids_lst).items(), key=lambda kv: kv[1], reverse=True)
			tot_centroids = len(final_centroids_lst)
			s += '\t\n'.join(f'{cen_}: {cnt_ / tot_centroids * 100:.2f}% - ({cnt_}/{tot_centroids})' for
			                 cen_, cnt_ in ss_)

			data = s
			# Insert an image offset in the cell.
			worksheet.set_row(row, 400)  # set row height to 100
			# cell_format = workbook.add_format({'bold': True, 'italic': True})
			# cell_format2 = workbook.add_format()
			cell_format.set_align('top')
			worksheet.write(row, column_idx, data, cell_format)
			row += 1
		# break
	except Exception as e:
		print(f'Error: {e}')
		traceback.print_exc()
		data = '-'

	# # worksheet.write('A12', 'Insert an image with an offset:')
	# n_clients = 0 if 'Centralized' in algorithm_py_name else args['N_CLIENTS']
	# sub_dir = f'Clients_{n_clients}'
	# centroids_img = os.path.join(out_dir, sub_dir, f'M={n_clients}-Centroids.png')
	# print(f'{centroids_img} exist: {os.path.exists(centroids_img)}')
	# worksheet.set_row(row, 300)  # set row height to 30
	scale = 0.248
	# if os.path.exists(centroids_img):
	# 	worksheet.insert_image(row, column_idx, centroids_img, {'x_scale': scale, 'y_scale': scale, 'object_position': 1})
	# row += 1

	score_img = os.path.join(OUT_DIR, 'over_time', f'centroids_diff.png')
	print(f'{score_img} exist: {os.path.exists(score_img)}')
	worksheet.set_row(row, 300)  # set row height to 30
	if os.path.exists(score_img):
		worksheet.insert_image(row, column_idx, score_img, {'x_scale': scale, 'y_scale': scale, 'object_position': 1})
	row += 1


	# score_img = os.path.join(out_dir, sub_dir, 'over_time', f'M={n_clients}-scores.png')
	# print(f'{score_img} exist: {os.path.exists(score_img)}')
	# worksheet.set_row(row, 300)  # set row height to 30
	# worksheet.insert_image(row, column_idx, score_img, {'x_scale': scale, 'y_scale': scale, 'object_position': 1})
	# row += 1


# ALG2ABBRV = {'Centralized_true': 'True-CKM',
#                    'Centralized_random': 'Random-CKM',
#                    'Centralized_kmeans++': 'KM++-CKM',
#                    'Federated-Server_random_min_max': 'Random-WA-FKM',
#                    'Federated-Server_gaussian': 'Gaussian-WA-FKM',
#                    'Federated-Server_average-Client_random': 'C-Random-WA-FKM',
#                    'Federated-Server_average-Client_kmeans++': 'C-KM++-WA-FKM',
#                    'Federated-Server_greedy-Client_random': 'C-Random-GD-FKM',
#                    'Federated-Server_greedy-Client_kmeans++': 'C-KM++-GD-FKM',
#                    }

METRIC2ABBRV = {'Iterations': 'Iterations',
                'durations': 'Durations',
                'davies_bouldin': 'DB',
                'silhouette': 'Silhouette',
                'ch': 'CH',
                'euclidean': 'Euclidean',
                'n_clusters': 'N_clusters',
                'n_clusters_pred': 'N_clusters_pred',
				'ari': 'ARI',
				'ami': 'AMI',
				'fm': 'FM',
				'vm': 'VM',
                'n_repeats': 'N_REPEATS(useful)'
                }


def save2csv(csv_f, idx_alg, args, results_avg, metric_names):
	try:
		for split in args['SPLITS']:
			# metric_names = results_avg[split]['metric_names']
			if idx_alg == 0:
				s = ','.join([split] + [METRIC2ABBRV[v] if v in METRIC2ABBRV.keys() else v for v in metric_names])
				csv_f.write(s + '\n')

			alg_name = args['ALGORITHM']['py_name'] + '|'+ args["ALGORITHM"]['detail']
			s = [f'{alg_name}']
			for metric_name in metric_names:
				if metric_name in ['labels_pred', 'labels_true', 'n_clusters', 'n_clusters_pred']:
					value = results_avg[split][metric_name]
					value = str(value).replace(',', '|')
					s.append(f'{value}')
					continue
				try:
					score_mean, score_std = results_avg[split][metric_name]
					s.append(f'{score_mean:.2f} +/- {score_std:.2f}')
				except Exception as e:
					s.append(f'nan')
			s = ','.join(s)
			csv_f.write(s + '\n')
	# break
	except Exception as e:
		print(f'Error: {e}')
		traceback.print_exc()
		s = '-'
		csv_f.write(s)


def save2csv2(csv_f, idx_alg, args, results_avg, metric_names):
	try:
		for split in args['SPLITS']:
			# metric_names = results_avg[split]['metric_names']
			if idx_alg == 0:
				s = ','.join([split] + [v for v in metric_names])
				csv_f.write(s + '\n')

			alg_name = args['ALGORITHM']['py_name'] + '|'+ args["ALGORITHM"]['detail']
			s = [f'{alg_name}']
			for metric_name in metric_names:
				try:
					value = results_avg[split][metric_name]
					value = str(value).replace(',', '|')
					s.append(f'{value}')
				except Exception as e:
					s.append(f'nan')
			s = ','.join(s)
			csv_f.write(s + '\n')
	# break
	except Exception as e:
		print(f'Error: {e}')
		traceback.print_exc()
		s = '-'
		csv_f.write(s)



def main2():
	# get default config.yaml (template)
	config_file = 'config.yaml'
	args = config.load(config_file)
	OUT_DIR = args['OUT_DIR']
	# args['N_REPEATS'] = 1

	VERBOSE = 0
	SEPERTOR = args['SEPERTOR']

	tot_cnt = 0
	sheet_names = set()
	dataset_names = ['NBAIOT',  'FEMNIST', 'SENT140', '3GAUSSIANS', '10GAUSSIANS']
	dataset_names = ['10GAUSSIANS']
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
		N_CLIENTS = dataset['n_clients']
		N_REPEATS = args1['N_REPEATS']
		N_CLUSTERS = dataset['n_clusters']
		args1['N_CLIENTS'] = dataset['n_clients']
		args1['N_CLUSTERS'] = dataset['n_clusters']
		args1['DATASET']['detail'] = f'{SEPERTOR}'.join([args1['DATASET']['detail'],
		                                                 f'M_{N_CLIENTS}', f'K_{N_CLUSTERS}', f'SEED_{SEED}'])
		dataset_detail = args1['DATASET']['detail']
		args1['ALGORITHM']['n_clusters'] = dataset['n_clusters']
		algorithms = get_algorithms_config_lst(py_names, dataset['n_clusters'])
		for idx_alg, algorithm in enumerate(algorithms):
			args2 = copy.deepcopy(args1)
			if VERBOSE >= 1: print(f'\n*** {tot_cnt}th experiment ***')
			args2['ALGORITHM']['py_name'] = algorithm['py_name']
			# initial_method = args2['ALGORITHM']['initial_method']
			args2['ALGORITHM']['server_init_method'] = algorithm['server_init_method']
			server_init_method = args2['ALGORITHM']['server_init_method']
			args2['ALGORITHM']['client_init_method'] = algorithm['client_init_method']
			client_init_method = args2['ALGORITHM']['client_init_method']
			# args2['ALGORITHM']['name'] = algorithm['py_name'] + '_' + f'{server_init_method}|{client_init_method}'
			N_REPEATS = args2['N_REPEATS']
			TOLERANCE = args2['TOLERANCE']
			NORMALIZE_METHOD = args2['NORMALIZE_METHOD']
			args2['ALGORITHM']['detail'] = f'{SEPERTOR}'.join([f'R_{N_REPEATS}',
			                                                   f'{server_init_method}|{client_init_method}',
			                                                   f'{TOLERANCE}', f'{NORMALIZE_METHOD}'])
			args2['OUT_DIR'] = os.path.join(OUT_DIR, args2['DATASET']['name'], f'{dataset_detail}',
			                                args2['ALGORITHM']['py_name'], args2['ALGORITHM']['detail'])
			new_config_file = os.path.join(args2['OUT_DIR'], 'config_file.yaml')
			if VERBOSE >= 2:
				pprint(new_config_file, sort_dicts=False)
			args2['config_file'] = new_config_file
			if VERBOSE >= 5:
				pprint(args2, sort_dicts=False)

			if idx_alg == 0:
				xlsx_file = os.path.join(OUT_DIR, 'xlsx', args2['DATASET']['name'], f'{dataset_detail}',
				                         args2['ALGORITHM']['detail'] + '.xlsx')

				tmp_dir = os.path.dirname(xlsx_file)
				if not os.path.exists(tmp_dir):
					os.makedirs(tmp_dir)
				workbook = xlsxwriter.Workbook(xlsx_file)
				if VERBOSE >= 1: print(xlsx_file)
				sheet_name = dataset_detail[:25].replace(':', ' ')
				if sheet_name in sheet_names:
					sheet_name = sheet_name + f'{len(sheet_names)}'
				sheet_names.add(sheet_name)
				if VERBOSE >= 1: print(f'xlsx_sheet_name: {sheet_name}')
				worksheet = workbook.add_worksheet(name=sheet_name)

				# get csv file
				csv_file = os.path.join(os.path.dirname(xlsx_file), args2['ALGORITHM']['detail'] + '.csv')
				csv_file2 = os.path.join(os.path.dirname(xlsx_file), args2['ALGORITHM']['detail'] + '-detail.csv')
				try:
					csv_f = open(csv_file, 'w')
					csv_f2 = open(csv_file2, 'w')
				except Exception as e:
					traceback.print_exc()
					break
			try:
				results_avg = parser_history(args2)
				save2xls(workbook, worksheet, idx_alg, args2, results_avg)
				# # only save the top 2 results
				# results_avg = parser_history_topk(args2)
				save2csv(csv_f, idx_alg, args2, results_avg, metric_names=['ari', 'ami', 'fm', 'vm',
				                                                           'Iterations', 'durations', 'davies_bouldin',
				                                                           'silhouette', 'ch', 'euclidean',
				                                                           'n_clusters', 'n_clusters_pred',
				                                                           'labels_pred', 'labels_true'
				                                                           ])
				# save results detail
				results_detail = parser_history2(args2)
				save2csv2(csv_f2, idx_alg, args2, results_detail, metric_names=['ari', 'ami', 'fm', 'vm',
				                                                           'Iterations', 'durations', 'davies_bouldin',
				                                                           'silhouette', 'ch', 'euclidean',
				                                                           'n_clusters', 'n_clusters_pred',
				                                                            'labels_pred', 'labels_true'
				                                                           ])
			except Exception as e:
				traceback.print_exc()
			tot_cnt += 1
		csv_f.close()
		csv_f2.close()
		workbook.close()
	# break
	print(f'*** Total cases: {tot_cnt}')


if __name__ == '__main__':
	main2()
