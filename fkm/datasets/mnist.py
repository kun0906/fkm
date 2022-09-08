"""
	1. Download from https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download

"""
import collections
import copy
import os

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

	def get_xy(in_dir='datasets/MNIST/mnist'):
		clients_train_x = []
		clients_train_y = []
		clients_test_x = []
		clients_test_y = []
		n_data_points = 0

		file = os.path.join(in_dir, 'mnist_train.csv')
		df = pd.read_csv(file)

		is_pca = args['IS_PCA']
		if type(is_pca) != bool:
			msg = f'is_pca: {is_pca}'
			raise ValueError(msg)

		if is_pca:
			normalized_method = args['NORMALIZE_METHOD']
			if normalized_method in ['std']:
				X = copy.deepcopy(df.iloc[:, 1:].values)
				std = sklearn.preprocessing.StandardScaler()
				std.fit(X)
				pca = sklearn.decomposition.PCA(n_components=0.95)
				pca.fit(X)
				print(f'pca.explained_variance_ratio_:{pca.explained_variance_ratio_}')
			else:
				msg = f'is_pca: {is_pca}, however, NORMALIZE_METHOD: {normalized_method}'
				raise ValueError(msg)

		# print('original train:', collections.Counter(df.label))
		for y in sorted(set(df.label)):
			X_ = df[df.label == y].iloc[:, 1:].values
			y_ = df.label[df.label == y].values
			if is_pca:
				X_ = std.transform(X_)
				X_ = pca.transform(X_)
			X_train, X_, y_train, y_ = train_test_split(X_, y_, train_size=n1, shuffle=True,
			                                            random_state=random_state)  # train set = 1-ratio

			clients_train_x.append(X_train)  # each client has one user's data
			clients_train_y.append(y_train)

		file = os.path.join(in_dir, 'mnist_test.csv')
		df = pd.read_csv(file)
		# print('test:', collections.Counter(df.label))
		for y in sorted(set(df.label)):
			X_ = df[df.label == y].iloc[:, 1:].values
			y_ = df.label[df.label == y].values
			if is_pca:
				X_ = std.transform(X_)
				X_ = pca.transform(X_)
			_, X_test, _, y_test = train_test_split(X_, y_, test_size=2, shuffle=True,
			                                                    random_state=random_state)  # train set = 1-ratio

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
