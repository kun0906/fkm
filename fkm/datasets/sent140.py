"""

"""
import collections
import json
import os
import numpy as np
import sys

import sklearn
from sklearn.model_selection import train_test_split

from fkm.utils import utils_func


def get_size(obj, unit='GB'):
	"""
	# sys.getsizeof() returns the memory size of an object, not the disk size of the object.

	# sys.getsizeof() returns the memory size of an instance of the float type. In addition to the 8 bytes
	# used by the 64-bit IEEE representation of a float, additional memory is used for reference count,
	# a pointer to type information, etc.
	# e.g.l sys.getsizeof(-0.4) => 24

	:param obj:
	:return:
	"""

	def dfs(obj, method='len'):
		# if type(obj) is set(int, str, float, bool):
		# 	return sys.getsizeof(obj)
		s = 0
		if type(obj) == list:
			s += sum(dfs(v, method) for v in obj)
		elif type(obj) == dict:
			s += sum(dfs(k, method) + dfs(v, method) for k, v in obj.items())
		else:
			if method == 'len':
				s += len(obj) if type == str else sys.getsizeof(obj)
			else:
				s += sys.getsizeof(obj)
		return s

	s = dfs(obj, method='getsizeof')
	# s_len = dfs(obj, method='len')
	if unit == 'GB':
		d = 1024 ** 3
	elif unit == 'MB':
		d = 1024 ** 2
	elif unit == 'KB':  # KB
		d = 1024
	else:  # B
		d = 1

	return s / d


def get_word_vector(X, w2v, STOPWORDS, max_words=10, UNKNOWN=None):
	"""
	:return:
	"""
	X_new = []
	for row in X:  # filter words for each row
		ws = []
		for w in row[4].split():
			w = w.lower()
			if w not in STOPWORDS and w in w2v.keys():
				ws.append(w)
		if len(ws) > max_words:
			ws = ws[:max_words]
		elif len(ws) < max_words:
			ws = ws + [UNKNOWN] * (max_words - len(ws))  # fix the vector size
		else:
			pass
		row[4] = [v for w in ws for v in w2v[w]]  # flat a nested list
		X_new.append(row[4])

	return np.asarray(X_new)

def get_lengths(data_dir = 'train', STOPWORDS = None, w2v = None):
	lengths = []
	lengths_no_stopwords = []
	for i, json_file in enumerate(os.listdir(data_dir)):
		if ".DS_Store" in json_file: continue
		with open(os.path.join(data_dir, json_file), 'rb') as f:
			vs = json.load(f)
		print(i, json_file, len(vs['users']))
		# size = get_size(vs, 'MB')
		# print(f'Possible size: {size} MB', flush=True)
		for j, user in enumerate(vs['users']):
			tmp = vs['user_data'][user]
			if j % 10000 == 0: print(i, j, user, len(tmp['y']), len(tmp['x'][0]))
			for row in tmp['x']:  # filter words for each row
				lengths.append(len(row[4].split()))
				ws = []
				for w in row[4].split():
					w = w.lower()
					if w not in STOPWORDS and w in w2v.keys():
						ws.append(w)
				lengths_no_stopwords.append(len(ws))

	return lengths, lengths_no_stopwords


def sent140_user_percent(args={}, random_state=42):
	in_dir = 'datasets/SENT140'
	# data_file = args['data_file']
	# dataset_detail = args['DATASET']['detail']
	# data_file = os.path.join(in_dir, f'{dataset_detail}.dat')
	# if os.path.exists(data_file):
	# 	return utils_func.load(data_file)

	DATASET_SETTING = 'all_data_niid_5_keep_50_train_9'
	# cp /scratch/gpfs/ky8517/leaf-torch/data/femnist /scratch/gpfs/ky8517/fkm/datasets/femnist
	embs_file = 'datasets/SENT140/glove.6B.50d.txt'
	print(os.path.abspath(embs_file))
	print(f'Load {embs_file}...')
	with open(embs_file, 'r') as inf:
		lines = inf.readlines()
	size = get_size(lines, 'MB')
	print(f'Possible size: {size} MB', flush=True)
	w2v = {}
	for l in lines:
		l = l.split()
		w = l[0]
		vector = [float(v) for v in l[1:]]
		w2v[w] = vector
	# add unknow words
	UNKNOWN = "++++++++++++++++++++++++?????????????????????"
	w2v[UNKNOWN] = [0] * 50
	print('Load completed.')

	# find the max_words
	import nltk
	nltk.download('stopwords')
	from nltk.corpus import stopwords

	STOPWORDS = set(stopwords.words())  # speed up the check

	lengths, lengths_no_stopwords = get_lengths(os.path.join(in_dir, DATASET_SETTING, 'train'), STOPWORDS, w2v)
	print('lengths: ', np.quantile(lengths, q=[0.5, 0.75, 0.9, 0.99]))
	print('lengths_no_stopwords: ', np.quantile(lengths_no_stopwords, q=[0.5, 0.75, 0.9, 0.99]))
	max_words = int(np.ceil(np.quantile(lengths_no_stopwords, q=0.9)))
	print(f'max_words: {max_words}')
	# max_words = 15

	def get_xy(in_dir='train'):
		X_clients = []
		y_clients = []
		Y = []
		n_data_points = 0
		n_users = 0
		dim = 0
		seen = set()
		for i, json_file in enumerate(os.listdir(in_dir)):
			with open(os.path.join(in_dir, json_file), 'rb') as f:
				vs = json.load(f)
			n_users += len(vs['users'])
			# print(f'{n_users} users in {json_file}')
			for user in vs['users']:
				if user in seen: continue
				seen.add(user)
				tmp = vs['user_data'][user]
				if len(tmp['y']) < 200:
					x_ = tmp['x']
					y_ = tmp['y']
				else:
					x_, y_ = sklearn.utils.resample(tmp['x'], tmp['y'], replace=False, n_samples=200,
					                                random_state=random_state)
				x_ = get_word_vector(x_, w2v, STOPWORDS, max_words, UNKNOWN)
				dim = len(x_[0])
				n_data_points += len(x_)
				X_clients.append(np.asarray(x_)) # each client has one user's data
				y_clients.append(np.asarray(y_))
				Y.extend(list(y_))

		Y = collections.Counter(Y)
		print(f'Y: {Y.items()}')
		print(f'n_users: {n_users}, n_data_points: {n_data_points}, dim: {dim}')
		return X_clients, y_clients

	clients_train_x, clients_train_y = get_xy(os.path.join(in_dir, DATASET_SETTING,'train'))
	clients_test_x, clients_test_y = get_xy(os.path.join(in_dir, DATASET_SETTING, 'test'))

	x = {'train': clients_train_x,
	     'test': clients_test_x}
	labels = {'train': clients_train_y,
	          'test': clients_test_y}

	print(f'n_train_clients: {len(clients_train_x)}, n_datapoints: {sum(len(vs) for vs in clients_train_y)}')
	print(f'n_test_clients: {len(clients_test_x)}, n_datapoints: {sum(len(vs) for vs in clients_test_y)}')
	return x, labels


def sent140_diff_sigma_n(args, random_state=42):
	n_clients = args['N_CLIENTS']
	dataset_detail = args['DATASET']['detail']  # 'nbaiot_user_percent_client:ratio_0.1'
	p1 = dataset_detail.split(':')
	ratio = float(p1[1].split('_')[1])

	p1_0 = p1[0].split('+')  # 'n1_100-sigma1_0.1, n2_1000-sigma2_0.2, n3_10000-sigma3_0.3
	p1_0_c1 = p1_0[0].split('-')
	n1 = int(p1_0_c1[0].split('_')[1])
	# tmp = p1_0_c1[1].split('_')
	# sigma1_0, sigma1_1 = float(tmp[1]), float(tmp[2])

	in_dir = 'datasets/SENT140'
	# data_file = args['data_file']
	# dataset_detail = args['DATASET']['detail']
	# data_file = os.path.join(in_dir, f'{dataset_detail}.dat')
	# if os.path.exists(data_file):
	# 	return utils_func.load(data_file)

	DATASET_SETTING = 'all_data_niid_5_keep_50_train_9'
	# cp /scratch/gpfs/ky8517/leaf-torch/data/femnist /scratch/gpfs/ky8517/fkm/datasets/femnist
	embs_file = 'datasets/SENT140/glove.6B.50d.txt'
	print(os.path.abspath(embs_file))
	print(f'Load {embs_file}...')
	with open(embs_file, 'r') as inf:
		lines = inf.readlines()
	size = get_size(lines, 'MB')
	print(f'Possible size: {size} MB', flush=True)
	w2v = {}
	for l in lines:
		l = l.split()
		w = l[0]
		vector = [float(v) for v in l[1:]]
		w2v[w] = vector
	# add unknow words
	UNKNOWN = "++++++++++++++++++++++++?????????????????????"
	w2v[UNKNOWN] = [0] * 50
	print('Load completed.')

	# find the max_words
	import nltk
	nltk.download('stopwords')
	from nltk.corpus import stopwords

	STOPWORDS = set(stopwords.words())  # speed up the check

	lengths, lengths_no_stopwords = get_lengths(os.path.join(in_dir, DATASET_SETTING, 'train'), STOPWORDS, w2v)
	print('lengths: ', np.quantile(lengths, q=[0.5, 0.75, 0.9, 0.99]))
	print('lengths_no_stopwords: ', np.quantile(lengths_no_stopwords, q=[0.5, 0.75, 0.9, 0.99]))
	max_words = int(np.ceil(np.quantile(lengths_no_stopwords, q=0.9)))
	print(f'max_words: {max_words}')
	# max_words = 15

	def get_xy(in_dir='train'):
		X_clients = []
		y_clients = []
		Y = []
		n_data_points = 0
		n_users = 0
		dim = 0
		seen = set()
		users = []
		for i, json_file in enumerate(os.listdir(in_dir)): # only have one txt file
			if ".DS_Store" in json_file: continue
			try:
				with open(os.path.join(in_dir, json_file), 'rb') as f:
					vs = json.load(f)
			except Exception as e:
				print(f'open error: {json_file}')
				continue
			users.extend(vs['user_data'].values())

		# n_users = len(users)
		n_users_tmp = len(users)
		B = n_users_tmp // n_clients    # assume we have 20 clients, and each clients has 23 users in maximum
		x = users
		for i in range(n_clients):
			x, x_ = train_test_split(x, test_size=n1, shuffle=True,
	                                              random_state=random_state)  # train set = 1-ratio
			tmp_x = []
			tmp_y = []
			for usr in x_:
				x1 = usr['x']
				y1 = usr['y']
				x1 = get_word_vector(x1, w2v, STOPWORDS, max_words, UNKNOWN)
				tmp_x.extend(x1)
				tmp_y.extend(y1)
				dim = x1.shape[1]
			n_data_points += len(tmp_y)
			X_clients.append(np.asarray(tmp_x))  # each client has one user's data
			y_clients.append(np.asarray(tmp_y))
			Y.extend(list(tmp_y))
		Y = collections.Counter(Y)
		print(f'Y({len(Y.items())}): {Y.items()}')
		print(f'n_clients: {n_clients}, {n1} users per client, n_data_points: {n_data_points}, dim: {dim}')
		return X_clients, y_clients

	clients_train_x, clients_train_y = get_xy(os.path.join(in_dir, DATASET_SETTING,'train'))
	clients_test_x, clients_test_y = get_xy(os.path.join(in_dir, DATASET_SETTING, 'test'))

	x = {'train': clients_train_x,
	     'test': clients_test_x}
	labels = {'train': clients_train_y,
	          'test': clients_test_y}

	print(f'n_train_clients: {len(clients_train_x)}, n_datapoints: {sum(len(vs) for vs in clients_train_y)}')
	print(f'n_test_clients: {len(clients_test_x)}, n_datapoints: {sum(len(vs) for vs in clients_test_y)}')
	return x, labels

if __name__ == '__main__':
	sent140_user_percent()
