import collections
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def gaussian10_diff_sigma_n(args, random_state=42, **kwargs):
	"""
		10 Gaussians with same \sigma (0.3), but different \mu

		# Client 1 is in (-1, 0) with N = 5000
		mu = (0, 0)
		# Client 2  is in (1, 0) with N = 5000
		mu = (1, 0)

		# Client 3 to 6 with  N = 1000
		# mu = (5, 0)
		# mu = (0, 5)
		# mu = (-5, 0)
		# mu = (0, -5)

		mu = (5, 5)
		mu = (5, 5)
		mu = (-5, 5)
		mu = (5, -5)



		# Client 7 to 10 with  N = 500
		# mu = (10, 10)
		# mu = (-10, 10)
		# mu = (-10, -10)
		# mu = (10, -10)

		mu = (10, 10)
		mu = (11, 11)
		mu = (15, 15)
		mu = (16, 16)


	Parameters
	----------
	args
	random_state

	Returns
	-------

	"""
	# # 'n1_5000-sigma1_0.3_0.3+n2_1000-sigma2_0.3_0.3+n3_500-sigma3_0.3_0.3:ratio_0.1:diff_sigma_n'
	dataset_detail = args['DATASET']['detail']
	p1 = dataset_detail.split(':')
	ratio = float(p1[1].split('_')[1])

	p1_0 = p1[0].split('+')  # 'n1_100-sigma1_0.1, n2_1000-sigma2_0.2, n3_10000-sigma3_0.3
	p1_0_c1 = p1_0[0].split('-')
	n1 = int(p1_0_c1[0].split('_')[1])
	tmp = p1_0_c1[1].split('_')
	sigma1_0, sigma1_1 = float(tmp[1]), float(tmp[2])

	p1_0_c2 = p1_0[1].split('-')
	n2 = int(p1_0_c2[0].split('_')[1])
	tmp = p1_0_c2[1].split('_')
	sigma2_0, sigma2_1 = float(tmp[1]), float(tmp[2])

	p1_0_c3 = p1_0[2].split('-')
	n3 = int(p1_0_c3[0].split('_')[1])
	tmp = p1_0_c3[1].split('_')
	sigma3_0, sigma3_1 = float(tmp[1]), float(tmp[2])

	def get_xy():
		clients_train_x = []
		clients_train_y = []
		clients_test_x = []
		clients_test_y = []

		r = np.random.RandomState(random_state)
		# idx = 0
		# for pos in range(0, 10, 1):
		# 	mus = [pos, pos]
		# 	cov = np.asarray([[0.1, 0.1], [0.1, 0.1]])
		# 	tmp_x = r.multivariate_normal(mus, cov, size=n1)
		# 	tmp_y = np.asarray([idx] * n1)
		# 	idx += 1
		# 	clients_train_x.append(tmp_x)
		# 	clients_train_y.append(tmp_y)

		idx = 0
		pos = 5
		for mus in [[pos, 0],[0, pos],
		            [-pos, 0],
		           [0, -pos]]:

		# for mus in [[pos, 0], [-pos, 0], [pos*2, 0], [-pos*2, 0],
		#             [0, pos], [0, -pos], [0, pos*2], [0, -pos*2],
		#             [1, 0],
		#             [-1, 0]]:
		# for mus in [[pos, 0], [-pos, 0], [pos * 2, 0], [-pos * 2, 0],
		#             [0, pos], [0, -pos], [0, pos * 2], [0, -pos * 2],
		#             [1, 0],
		#             [-1, 0]]:

			cov = np.asarray([[sigma1_0, 0], [0., sigma1_0]])
			tmp_x = r.multivariate_normal(mus, cov, size=n1)
			tmp_y = np.asarray([idx] * n1)
			idx += 1
			clients_train_x.append(tmp_x)
			clients_train_y.append(tmp_y)

		v = 5
		for mus in [[v * pos, v * pos],
		            [-v * pos, v * pos],
		            [-v * pos, -v * pos],
		            [v * pos, -v * pos]]:

			cov = np.asarray([[sigma2_0, 0], [0., sigma2_0]])
			tmp_x = r.multivariate_normal(mus, cov, size=n2)
			tmp_y = np.asarray([idx] * n2)
			idx += 1
			clients_train_x.append(tmp_x)
			clients_train_y.append(tmp_y)

		for mus in [[-1, 0],
		            [1, 0],
		           ]:

			cov = np.asarray([[sigma3_0, 0], [0., sigma3_0]])
			tmp_x = r.multivariate_normal(mus, cov, size=n3)
			tmp_y = np.asarray([idx] * n3)
			idx += 1
			clients_train_x.append(tmp_x)
			clients_train_y.append(tmp_y)


		# idx = 2
		# # client 1 and 2
		# X1, y1 = [], []
		# position = 2
		# mus_lst = [[position, 0], [-position, 0]]
		# for mus in mus_lst:
		# 	cov = np.asarray([[sigma1_0, 0], [0, sigma1_1]])
		# 	tmp_x = r.multivariate_normal(mus, cov, size=n1)
		# 	tmp_y = np.asarray([idx] * n1)
		# 	idx += 1
		# 	clients_train_x.append(tmp_x)
		# 	clients_train_y.append(tmp_y)
		#
		# # client 3 to 6:
		# position = 3
		# # mus_lst = [[position, 0],
		# #            [0, position],
		# #            [-position, 0],
		# #            [0, -position],
		# #            ]
		# mus_lst = [[-position, 0],
		#            [-2 * position, position],
		#            [-3 * position, 2],
		#            [-2 * position-2, -position],
		#            ]
		# X2, y2 = [], []
		# for mus in mus_lst:
		# 	cov = np.asarray([[sigma2_0, 0], [0, sigma2_1]])
		# 	tmp_x = r.multivariate_normal(mus, cov, size=n2)
		# 	tmp_y = np.asarray([idx] * n2)
		# 	idx += 1
		# 	clients_train_x.append(tmp_x)
		# 	clients_train_y.append(tmp_y)

		# # client 7 to 10:
		# position = 3
		# mus_lst = [[position, 0],
		#            [2 * position, position],
		#            [3 * position, 2],
		#            [2 * position-2, -position],
		#            ]
		# # mus_lst = [[position, -position],
		# #            [position + 5, -position + 5],
		# #            [position + 10, -position + 10],
		# #            [position + 15, -position + 15],
		# #            ]
		# X3, y3 = [], []
		# for mus in mus_lst:
		# 	cov = np.asarray([[sigma3_0, 0], [0, sigma3_1]])
		# 	tmp_x = r.multivariate_normal(mus, cov, size=n3)
		# 	tmp_y = np.asarray([idx] * n3)
		# 	idx += 1
		# 	clients_train_x.append(tmp_x)
		# 	clients_train_y.append(tmp_y)

		if (ratio < 0) or (ratio > 1):
			raise ValueError
		elif ratio == 0:
			pass
		elif ('centralized' in args['ALGORITHM']['py_name']):
			# there should have not effect for centralized kmeans for different ratios.
			pass
		else:
			new_client_train_x = []
			new_client_train_y = []
			# P(ratio) = 0.1: draw 10% from the rest of all clusters.
			# i.e., client 1 has 90% cluster 1, 10% of the rest of all clusters (10%/9 per each rest cluster)
			n_classes = 10
			for i in range(n_classes):
				train_xi, test_xi, train_yi, test_yi = train_test_split(clients_train_x[i], clients_train_y[i],
				                                                        test_size=ratio, shuffle=True,
				                                                        random_state=random_state)  # train set = 1-ratio
				for j in range(n_classes):
					if i == j: continue
					train_xj, test_xj, train_yj, test_yj = train_test_split(clients_train_x[j], clients_train_y[j],
					                                                        test_size=ratio /(n_classes-1), shuffle=True,
					                                                        random_state=random_state)  # train set = 1-ratio
					train_xi = np.concatenate([train_xi, test_xj], axis=0)
					train_yi = np.concatenate([train_yi, test_yj])
				new_client_train_x.append(copy.deepcopy(train_xi))
				new_client_train_y.append(copy.deepcopy(train_yi))
				print(i, collections.Counter(train_yi))
			clients_train_x, clients_train_y = new_client_train_x, new_client_train_y
		# clients_test_x, clients_test_y = use the one when ratio = 0

		return clients_train_x, clients_train_y, clients_test_x, clients_test_y

	clients_train_x, clients_train_y, clients_test_x, clients_test_y = get_xy()
	x = {'train': clients_train_x,
	     'test': clients_test_x}
	labels = {'train': clients_train_y,
	          'test': clients_test_y}

	return x, labels

# gaussian10_diff_sigma_n(args)
