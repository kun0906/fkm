"""
	1. Download from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download

"""
import collections
import copy
import os
import pickle

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split


def mnist_diff_sigma_n(args, random_state=42):
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

	in_dir='datasets/MNIST/mnist'
	IS_PCA = args['IS_PCA']
	if IS_PCA:
		out_file = 'Xy_PCA.data'
	else:
		out_file = 'Xy.dat'
	out_file = os.path.join(in_dir, out_file)

	if random_state == 0: # i_repeat * 10 = 0:
		if os.path.exists(out_file):
			os.remove(out_file)

	if IS_PCA:
		if not os.path.exists(out_file):
			file = os.path.join(in_dir, 'mnist_train.csv')
			df = pd.read_csv(file)
			y = df.label.values
			normalized_method = args['NORMALIZE_METHOD']
			if not normalized_method in ['std']:
				msg = f'is_pca: {IS_PCA}, however, NORMALIZE_METHOD: {normalized_method}'
				raise ValueError(msg)
			else:
				X = copy.deepcopy(df.iloc[:, 1:].values)
				std = sklearn.preprocessing.StandardScaler()
				std.fit(X)
				X = std.transform(X)

			pca = sklearn.decomposition.PCA(n_components=0.95)
			pca.fit(X)
			print(f'pca.explained_variance_ratio_:{pca.explained_variance_ratio_}')
			X_train = pca.transform(X)
			y_train = y

			file = os.path.join(in_dir, 'mnist_test.csv')
			df = pd.read_csv(file)
			X_test = pca.transform(df.iloc[:, 1:].values)
			y_test = df.label.values
			with open(out_file, 'wb') as f:
				pickle.dump((X_train, y_train, X_test, y_test), f)
		else:
			with open(out_file, 'rb') as f:
				X_train, y_train, X_test, y_test = pickle.load(f)
	else:
		file = os.path.join(in_dir, 'mnist_train.csv')
		df = pd.read_csv(file)
		X_train =  df.iloc[:, 1:].values
		y_train = df.label.values

		file = os.path.join(in_dir, 'mnist_test.csv')
		df = pd.read_csv(file)
		X_test = df.iloc[:, 1:].values
		y_test = df.label.values


	def get_xy():
		clients_train_x = []
		clients_train_y = []
		clients_test_x = []
		clients_test_y = []
		n_data_points = 0

		# print('original train:', collections.Counter(df.label))
		for y_i in sorted(set(y_train)):
			indices = np.where(y_train==y_i)
			X_train_, X_, y_train_, y_ = train_test_split(X_train[indices], y_train[indices], train_size=n1, shuffle=True,
			                                            random_state=random_state)  # train set = 1-ratio

			clients_train_x.append(X_train_)  # each client has one user's data
			clients_train_y.append(y_train_)

		for y_i in sorted(set(y_test)):
			indices = np.where(y_test == y_i)
			X_test_, X_, y_test_, y_ = train_test_split(X_test[indices], y_test[indices], train_size=2,
			                                              shuffle=True,
			                                              random_state=random_state)  # train set = 1-ratio

			clients_train_x.append(X_test_)  # each client has one user's data
			clients_train_y.append(y_test_)

			clients_test_x.append(X_test)  # each client has one user's data
			clients_test_y.append(y_test)

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
	mnist_diff_sigma_n({'N_CLIENTS': 0, 'IS_PCA':True, 'DATASET': {'detail': 'n1_200:ratio_0.1'}})
