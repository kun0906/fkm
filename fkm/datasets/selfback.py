"""
	1. Download from https://archive.ics.uci.edu/ml/datasets/selfBACK

"""
import collections
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def selfback_diff_sigma_n(args, random_state=42):
	"""
	Parameters
	----------
	args
	random_state

	Returns
	-------

	"""
	n_clients = args['N_CLIENTS']
	dataset_detail = args['DATASET']['detail']  # 'nbaiot_user_percent_client:ratio_0.1'
	p1 = dataset_detail.split(':')
	ratio = float(p1[1].split('_')[1])

	p1_0 = p1[0].split('+')  # 'n1_100-sigma1_0.1, n2_1000-sigma2_0.2, n3_10000-sigma3_0.3
	p1_0_c1 = p1_0[0].split('-')
	n1 = int(p1_0_c1[0].split('_')[1])

	# tmp = p1_0_c1[1].split('_')
	# sigma1_0, sigma1_1 = float(tmp[1]), float(tmp[2])

	def get_xy(in_dir='datasets/SELFBACK/selfBACK/wt'):
		clients_train_x = []
		clients_train_y = []
		clients_test_x = []
		clients_test_y = []
		n_data_points = 0

		LABELS = set(['downstairs', 'jogging', 'lying', 'sitting', 'standing', 'upstairs', 'walkfast', 'walkmod', 'walkslow'])
		data = {}
		for file in sorted(os.listdir(in_dir)):
			if '.DS_Store' in file: continue
			label = file.split('_')[1]
			file = os.path.join(in_dir, file)
			df = pd.read_csv(file)
			X_ = df.values
			if label not in data:
				data[label] = []
			else:
				data[label].extend(X_)

		idx_label = 0
		for label, X_ in data.items():
			X_ = np.asarray(X_)
			if X_.shape[0] < 5000: continue
			y_ = np.ones((X_.shape[0],)) * idx_label
			X_train, X_, y_train, y_ = train_test_split(X_, y_, train_size=n1, shuffle=True,
			                                                    random_state=random_state)  # train set = 1-ratio

			clients_train_x.append(np.asarray(X_train))  # each client has one user's data
			clients_train_y.append(np.asarray(y_train))

			_, X_test, _, y_test = train_test_split(X_, y_, test_size=2, shuffle=True,
			                                                    random_state=random_state)  # train set = 1-ratio
			clients_test_x.append(np.asarray(X_test))  # each client has one user's data
			clients_test_y.append(np.asarray(y_test))

			idx_label += 1
			print(idx_label, label, X_.shape, collections.Counter(y_).items())

		return clients_train_x, clients_train_y, clients_test_x, clients_test_y

	clients_train_x, clients_train_y, clients_test_x, clients_test_y = get_xy()

	x = {'train': clients_train_x,
	     'test': clients_test_x}
	labels = {'train': clients_train_y,
	          'test': clients_test_y}

	print(f'n_train_clients: {len(clients_train_x)}, n_datapoints: {sum(len(vs) for vs in clients_train_y)}')
	print(f'n_test_clients: {len(clients_test_x)}, n_datapoints: {sum(len(vs) for vs in clients_test_y)}')
	return x, labels


if __name__ == '__main__':
	selfback_diff_sigma_n({'N_CLIENTS': 0, 'DATASET': {'detail': 'n1_200:ratio_0.1'}})
