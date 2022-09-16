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

		r = np.random.RandomState(random_state)
		idx = 1
		# client 1 and 2
		X1, y1 = [], []
		position = 2
		mus_lst = [[5, position], [5, -position]]
		for mus in mus_lst:
			cov = np.asarray([[sigma1_0, 0], [0, sigma1_1]])
			X1.extend(r.multivariate_normal(mus, cov, size=n1))
			y1.extend(np.asarray([idx] * n1))
			idx += 1
		X1, y1 = np.asarray(X1), np.asarray(y1)

		# client 3 to 6:
		position = 5
		# mus_lst = [[position, 0],
		#            [0, position],
		#            [-position, 0],
		#            [0, -position],
		#            ]
		mus_lst = [[-position, 0],
		           [-2 * position, position],
		           [-3 * position, 0],
		           [-2 * position, -position],
		           ]
		X2, y2 = [], []
		for mus in mus_lst:
			cov = np.asarray([[sigma2_0, 0], [0, sigma2_1]])
			X2.extend(r.multivariate_normal(mus, cov, size=n2))
			y2.extend(np.asarray([idx] * n2))
			idx += 1
		X2, y2 = np.asarray(X2), np.asarray(y2)

		# client 7 to 10:
		position = 10
		# mus_lst = [[position, position],
		#            [-position, position],
		#            [-position, -position],
		#            [position, -position],
		#            ]
		mus_lst = [[position, -position],
		           [position + 2, -position + 2],
		           [position + 5, -position + 5],
		           [position + 7, -position + 7],
		           ]
		X3, y3 = [], []
		for mus in mus_lst:
			cov = np.asarray([[sigma3_0, 0], [0, sigma3_1]])
			X3.extend(r.multivariate_normal(mus, cov, size=n3))
			y3.extend(np.asarray([idx] * n3))
			idx += 1
		X3, y3 = np.asarray(X3), np.asarray(y3)

		return X1, y1, X2, y2, X3, y3

	X1, y1, X2, y2, X3, y3 = get_xy()
	if 2 * ratio <= 0 or 2 * ratio >= 1:
		pass
	else:
		# client 1: 90% cluster1, 10 % cluster2, 10 % cluster3
		# client 2: 10% cluster1, 90 % cluster2, 10 % cluster3
		# client 3: 10% cluster1, 10 % cluster2, 90 % cluster3
		train_x1, X1, train_y1, y1 = train_test_split(X1, y1, test_size=2 * ratio, shuffle=True,
		                                              random_state=random_state)  # train set = 1-ratio
		test_x11, test_x12, test_y11, test_y12 = train_test_split(X1, y1, test_size=0.5, shuffle=True,
		                                                          random_state=random_state)  # each test set = 50% of rest data

		train_x2, X2, train_y2, y2 = train_test_split(X2, y2, test_size=2 * ratio, shuffle=True,
		                                              random_state=random_state)
		test_x21, test_x22, test_y21, test_y22 = train_test_split(X2, y2, test_size=0.5, shuffle=True,
		                                                          random_state=random_state)

		train_x3, X3, train_y3, y3 = train_test_split(X3, y3, test_size=2 * ratio, shuffle=True,
		                                              random_state=random_state)
		test_x31, test_x32, test_y31, test_y32 = train_test_split(X3, y3, test_size=0.5, shuffle=True,
		                                                          random_state=random_state)

		X1 = np.concatenate([train_x1, test_x21, test_x31], axis=0)
		# y1 = np.concatenate([train_y1, test_y2], axis=0) # be careful of this
		y1 = np.concatenate([train_y1, test_y21, test_y31], axis=0)

		X2 = np.concatenate([test_x11, train_x2, test_x32], axis=0)
		# y2 = np.concatenate([test_y1, train_y2], axis=0)
		y2 = np.concatenate([test_y11, train_y2, test_y32], axis=0)

		X3 = np.concatenate([test_x12, test_x22, train_x3], axis=0)
		# y3 = np.ones((X3.shape[0],)) * 2
		y3 = np.concatenate([test_y12, test_y22, train_y3], axis=0)

	is_show = args['IS_SHOW']
	if is_show:
		# Plot init seeds along side sample data
		fig, ax = plt.subplots()
		# colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
		colors = ["r", "g", "b", "m", 'black']
		ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='$Group_1$')
		p = np.mean(X1, axis=0)
		ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="black", zorder=10)
		offset = 5
		# xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
		xytext = (p[0], p[1] - offset)
		# print(xytext)
		ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
		            ha='center', va='center',  # textcoords='offset points',
		            bbox=dict(facecolor='none', edgecolor='b', pad=1),
		            arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
		                            connectionstyle="angle3, angleA=90,angleB=0"))
		# angleA : starting angle of the path
		# angleB : ending angle of the path

		ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='$Group_2$')
		p = np.mean(X2, axis=0)
		ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="black", zorder=10)
		offset = 2
		# xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
		xytext = (p[0], p[1] - offset)
		ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
		            ha='center', va='center',  # textcoords='offset points', va='bottom',
		            bbox=dict(facecolor='none', edgecolor='red', pad=1),
		            arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
		                            connectionstyle="angle3, angleA=90,angleB=0"))

		ax.scatter(X3[:, 0], X3[:, 1], c=colors[2], marker="o", s=10, alpha=0.3, label='$Group_3$')
		p = np.mean(X3, axis=0)
		ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="black", zorder=10)
		offset = 5
		# xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
		xytext = (p[0] , p[1] - offset)
		ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
		            ha='center', va='center',  # textcoords='offset points', va='bottom',
		            bbox=dict(facecolor='none', edgecolor='red', pad=1),
		            arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
		                            connectionstyle="angle3, angleA=90,angleB=0"))

		ax.axvline(x=0, color='k', linestyle='--')
		ax.axhline(y=0, color='k', linestyle='--')
		ax.legend(loc='upper right', fontsize=13)
		if args['SHOW_TITLE']:
			plt.title(dataset_detail.replace(':', '\n'))

		if 'xlim' in kwargs:
			plt.xlim(kwargs['xlim'])
		else:
			plt.xlim([-20, 20])
		if 'ylim' in kwargs:
			plt.ylim(kwargs['ylim'])
		else:
			plt.ylim([-20, 20])

		fontsize = 13
		plt.xticks(fontsize=fontsize)
		plt.yticks(fontsize=fontsize)

		plt.tight_layout()
		if not os.path.exists(args['OUT_DIR']):
			os.makedirs(args['OUT_DIR'])
		# f = os.path.join(args['OUT_DIR'], dataset_detail + '.png')
		f = args['data_file'] + '.png'
		print(f)
		plt.savefig(f, dpi=600, bbox_inches='tight')
		plt.show()

	clients_train_x = []
	clients_train_y = []
	clients_test_x = []
	clients_test_y = []
	for i, (x, y) in enumerate([(X1, y1), (X2, y2), (X3, y3)]):

		if i == 0:
			# (X1, y1): share it with 2 clients
			# c1 has 30% data and c2 has 70% data
			c1_x, c2_x, c1_y, c2_y = train_test_split(x, y, test_size=0.7, shuffle=True,
			                                          random_state=random_state)
			for x_, y_ in [(c1_x, c1_y), (c2_x, c2_y)]:
				train_x, test_x, train_y, test_y = train_test_split(x_, y_, test_size=2, shuffle=True,
				                                                    random_state=random_state)
				clients_train_x.append(train_x)
				clients_train_y.append(train_y)
				clients_test_x.append(test_x)
				clients_test_y.append(test_y)
		else:
			# (X2, y2): share it with 4 clients
			# c1 has 30% data and c2 has 70% data
			train_size = int(x.shape[0] // 4)
			for j in range(4):
				if train_size != x.shape[0]:
					c_x, x, c_y, y = train_test_split(x, y, train_size=train_size, shuffle=True,
					                                  random_state=random_state)
				else:
					c_x, c_y, = x, y
				train_x, test_x, train_y, test_y = train_test_split(c_x, c_y, test_size=2, shuffle=True,
				                                                    random_state=random_state)
				clients_train_x.append(train_x)
				clients_train_y.append(train_y)
				clients_test_x.append(test_x)
				clients_test_y.append(test_y)

	x = {'train': clients_train_x,
	     'test': clients_test_x}
	labels = {'train': clients_train_y,
	          'test': clients_test_y}

	return x, labels

# gaussian10_diff_sigma_n(args)
