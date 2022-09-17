"""
	1. Download from https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset+at+Different+Concentrations

"""
import collections
import os

import numpy as np
from sklearn.model_selection import train_test_split


def get_Xy(in_dir = ''):

	X = []
	y = []
	for file in sorted(os.listdir(in_dir)):
		if not file.endswith('.dat'): continue
		file = os.path.join(in_dir, file)
		with open(file, 'r') as f:
			line = f.readline()
			while line:
				line = line.split()
				x_ = [float(v.split(':')[1]) for v in line[1:]]
				y_ = int(line[0].split(';')[0])
				X.append(x_)
				y.append(y_)
				line = f.readline()

	# print(collections.Counter(y))

	return np.asarray(X), np.asarray(y)


def gassensor_diff_sigma_n(args, random_state=42):
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

	def get_xy(in_dir='datasets/GASSENSOR/driftdataset', n1=-1):
		clients_train_x = []
		clients_train_y = []
		clients_test_x = []
		clients_test_y = []

		n_data_points = 0
		dim = 0
		X, Y = get_Xy(in_dir)
		print(X.shape, collections.Counter(Y).items())
		# print(np.mean(X, axis=0), np.std(X, axis=0))

		for y in sorted(set(Y)):
			indices = np.where(Y==y)
			X_ = X[indices]
			y_ = Y[indices]
			if n1 == 0:
				n_train = X_.shape[0]-2
			X_train, X_test, y_train, y_test = train_test_split(X_, y_, train_size=n_train, shuffle=True,
			                                                    random_state=random_state)  # train set = 1-ratio

			clients_train_x.append(np.asarray(X_train))  # each client has one user's data
			clients_train_y.append(np.asarray(y_train))

			clients_test_x.append(np.asarray(X_test))  # each client has one user's data
			clients_test_y.append(np.asarray(y_test))

		return clients_train_x, clients_train_y, clients_test_x, clients_test_y

	clients_train_x, clients_train_y, clients_test_x, clients_test_y = get_xy(n1=n1)

	x = {'train': clients_train_x,
	     'test': clients_test_x}
	labels = {'train': clients_train_y,
	          'test': clients_test_y}

	print(f'n_train_clients: {len(clients_train_x)}, n_datapoints: {sum(len(vs) for vs in clients_train_y)}')
	print(f'n_test_clients: {len(clients_test_x)}, n_datapoints: {sum(len(vs) for vs in clients_test_y)}')
	return x, labels


if __name__ == '__main__':
	gassensor_diff_sigma_n({'N_CLIENTS': 0, 'DATASET': {'detail': 'n1_200:ratio_0.1'}})
