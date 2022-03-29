import os

import pandas as pd
from tabulate import tabulate

from fkm.sbatch import get_data_details_lst


def table1(data=None):
	"""
	Table 1
	Centralized Kmeans, ,       ,
	random,          kmeans++,


	Table 2
	federated Kmeans (Centroids initialization at server),



	Table 3
		federated Kmeans (Centroids initialization at each client), ,
client,  random, kmeans++,   random, kmeans++,  random (K+center), kmeans++ (K+center),
server,   weighted average,   greedy,           greedy

Training iterations,
DB index,
silhouette,
Euclidean distance,

Testing
DB index
Euclidean distance

	Parameters
	----------
	data

	Returns
	-------

	metric	KM++-CKM	Random-WA-FKM	C-Random-WA-FKM	C-KM++-WA-FKM	C-Random-GD-FKM	C-KM++-GD-FKM
	iterations	11.22 +/- 1.62	15.38 +/- 2.31	13.40 +/- 1.90	13.14 +/- 1.37	10.54 +/- 1.19	10.12 +/- 0.86
	DB score	0.50 +/- 0.03	0.54 +/- 0.08	0.50 +/- 0.03	0.50 +/- 0.00	0.51 +/- 0.04	0.50 +/- 0.00
	Silhouette	0.65 +/- 0.03	0.62 +/- 0.08	0.65 +/- 0.03	0.66 +/- 0.00	0.65 +/- 0.04	0.66 +/- 0.00
	Euclidean distance	0.43 +/- 0.06	0.50 +/- 0.17	0.43 +/- 0.06	0.42 +/- 0.00	0.43 +/- 0.08	0.42 +/- 0.00
	iterations
	DB score	0.51 +/- 0.03	0.54 +/- 0.08	0.51 +/- 0.03	0.51 +/- 0.00	0.51 +/- 0.04	0.51 +/- 0.00
	Silhouette	0.65 +/- 0.03	0.62 +/- 0.08	0.65 +/- 0.03	0.65 +/- 0.00	0.65 +/- 0.04	0.65 +/- 0.00
	Euclidean distance	0.44 +/- 0.06	0.52 +/- 0.17	0.44 +/- 0.06	0.43 +/- 0.00	0.45 +/- 0.08	0.43 +/- 0.00


	"""

	# algorithm -> abbreviation
	algorithm2abbrv = {
		# 'Centralized_true': 'True-CKM',
		'Centralized_random': 'Random-CKM',
		# 'Centralized_kmeans++':   'KM++-CKM',
		'Federated-Server_random_min_max': 'Random-WA-FKM',
		# 'Federated-Server_gaussian': 'Gaussian-WA-FKM',
		'Federated-Server_average-Client_random': 'C-Random-WA-FKM',
		'Federated-Server_average-Client_kmeans++': 'C-KM++-WA-FKM',
		'Federated-Server_greedy-Client_random': 'C-Random-GD-FKM',
		'Federated-Server_greedy-Client_kmeans++': 'C-KM++-GD-FKM',
	}

	table = [["spam", 42], ["eggs", 451], ["bacon", 0]]
	headers = ["item", "qty"]
	out = tabulate(table, headers, tablefmt='latex_raw')
	print(out)


metric2abbrv = {'iterations': 'Training iterations',
                'davies_bouldin': 'DB score',
                'silhouette': 'Silhouette',
                'euclidean': 'Euclidean distance'
                }


def main2():
	tot_cnt = 0
	client_epochs = 1
	n_clusters = 10
	n_clients = 10
	out_dir = '~/Downloads/xlsx'
	# out_dir = 'results/xlsx'
	# if os.path.exists(out_dir):
	# 	shutil.rmtree(out_dir, ignore_errors=True)
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	dataset, py_names, data_details_lst = get_data_details_lst()

	for data_details in data_details_lst:
		table_file = f'{out_dir}/{dataset}-Client_epochs_{client_epochs}-Clusters_{n_clusters}-Clients_{n_clients}/{data_details}.csv'
		df = pd.read_csv(table_file)
		out_df = df.iloc[[0, 5, 6, 7], :].copy(deep=True)
		out_df.iloc[0, 0] = 'Training $T$'
		out_df.iloc[1, 0] = 'Testing $DB$'
		out_df.iloc[2, 0] = 'Testing $SC$'
		out_df.iloc[3, 0] = 'Testing $\overline{WCSS}$'
		out_df.replace(to_replace='\ \+/\-\ ', value='$\\\pm$', regex=True, inplace=True)
		out_file = os.path.splitext(table_file)[0] + '-latex.csv'
		# out_df.to_csv(out_file, index=False, sep=',')
		with open(os.path.expanduser(out_file), 'w') as out:
			line = r"\begin{tabular}[c]{@{}l@{}}\\\end{tabular}         & KM++-CKM       & \NLcell{Random-WA\\-FKM} & \NLcell{C-Random-\\WA-FKM} & \NLcell{C-KM++-\\WA-FKM}  & \NLcell{C-Random-\\GD-FKM} & \NLcell{C-KM++-\\GD-FKM} \\"
			out.write(line + '\n')
			out.write('\\hline\n')
			for i, vs in enumerate(out_df.values):
				line = ' & '.join(vs) + '\t \\\\' + '\n'
				out.write(line)
				out.write('\hline\n')
				if i == 0:
					out.write('\hline\n')

		print(out_file)
		tot_cnt += 1

	return tot_cnt


if __name__ == '__main__':
	tot_cnt = main2()
	print()
	print(f'*** Total cases: {tot_cnt}')

