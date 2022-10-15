import os

import matplotlib.pyplot as plt
import numpy as np

from fkm.datasets.gaussian10 import gaussian10_diff_sigma_n
from fkm.datasets.gaussian3 import gaussian3_diff_sigma_n


def plot_3gaussians(X, labels, args, title=''):
	split = 'train'
	X1, y1 = X[split][0], labels[split][0]
	X2, y2 = X[split][1], labels[split][1]
	X3, y3 = X[split][2], labels[split][2]

	data_detail = args['DATASET']['detail']
	# Plot init seeds along side sample data
	fig, ax = plt.subplots()
	# colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
	colors = ["r", "g", "b", "m", 'black']
	ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.5, label='$G_{11}+G_{11}^{\'}$')
	p = np.mean(X1, axis=0)
	# ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
	# offset = 0.3
	# # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
	# xytext = (p[0] - offset, p[1] - offset)
	# # print(xytext)
	# ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
	#             ha='center', va='center',  # textcoords='offset points',
	#             bbox=dict(facecolor='none', edgecolor='b', pad=1),
	#             arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
	#                             connectionstyle="angle3, angleA=90,angleB=0"))
	# # angleA : starting angle of the path
	# # angleB : ending angle of the path

	ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.5, label='$G_{12}+G_{12}^{\'}$')
	p = np.mean(X2, axis=0)
	# ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
	# offset = 0.3
	# # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
	# xytext = (p[0] + offset, p[1] - offset)
	# ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
	#             ha='center', va='center',  # textcoords='offset points', va='bottom',
	#             bbox=dict(facecolor='none', edgecolor='red', pad=1),
	#             arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
	#                             connectionstyle="angle3, angleA=90,angleB=0"))

	ax.scatter(X3[:, 0], X3[:, 1], c=colors[2], marker="o", s=10, alpha=0.5, label='$G_{13}+G_{13}^{\'}$')
	p = np.mean(X3, axis=0)
	# ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
	# offset = 0.3
	# # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
	# xytext = (p[0] + offset, p[1] - offset)
	# ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
	#             ha='center', va='center',  # textcoords='offset points', va='bottom',
	#             bbox=dict(facecolor='none', edgecolor='red', pad=1),
	#             arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
	#                             connectionstyle="angle3, angleA=90,angleB=0"))

	ax.axvline(x=0, color='k', linestyle='--')
	ax.axhline(y=0, color='k', linestyle='--')
	ax.legend(loc='upper right')

	# plt.title(args['p1'].replace(':', '\n') + f':{title}')
	# # plt.xlim([-2, 15])
	# # plt.ylim([-2, 15])
	# plt.xlim([-6, 6])
	# plt.ylim([-6, 6])
	# # plt.xticks([])
	# # plt.yticks([])
	plt.tight_layout()
	if not os.path.exists(args['OUT_DIR']):
		os.makedirs(args['OUT_DIR'])
	f = os.path.join(args['OUT_DIR'], data_detail.replace('/', ':') + '.png')
	f = os.path.join(args['OUT_DIR'], '3gaussians.png')
	plt.savefig(f, dpi=600, bbox_inches='tight')
	plt.show()


def plot_10gaussians(X, labels, args, title=''):
	split = 'train'
	data_detail = args['DATASET']['detail']
	# Plot init seeds along side sample data
	fig, ax = plt.subplots()
	# colors: https://matplotlib.org/stable/tutorials/colors/colors.html
	# colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
	colors = ["r", "g", "b", "m", 'black', "c", 'y', 'tab:purple', 'tab:gray', 'tab:brown', 'tab:olive']
	for i in range(0, 4):
		X1, y1 = X[split][i], labels[split][i]
		ax.scatter(X1[:, 0], X1[:, 1], c=colors[i], marker="x", s=10, alpha=0.5, label=f'$G_{i+1}$')
		p = np.mean(X1, axis=0)
		# ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
		# offset = 0.3
		# # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
		# xytext = (p[0] - offset, p[1] - offset)
		# # print(xytext)
		# ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
		#             ha='center', va='center',  # textcoords='offset points',
		#             bbox=dict(facecolor='none', edgecolor='b', pad=1),
		#             arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
		#                             connectionstyle="angle3, angleA=90,angleB=0"))
		# # angleA : starting angle of the path
		# # angleB : ending angle of the path

	for i in range(4, 8):
		X2, y2 = X[split][i], labels[split][i]
		ax.scatter(X2[:, 0], X2[:, 1], c=colors[i], marker="o", s=10, alpha=0.5, label=f'$G_{i+1}$')
		p = np.mean(X2, axis=0)
		# ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
		# offset = 0.3
		# # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
		# xytext = (p[0] + offset, p[1] - offset)
		# ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
		#             ha='center', va='center',  # textcoords='offset points', va='bottom',
		#             bbox=dict(facecolor='none', edgecolor='red', pad=1),
		#             arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
		#                             connectionstyle="angle3, angleA=90,angleB=0"))

	for i in range(8, 10):
		X3, y3 = X[split][i], labels[split][i]
		label = '$G_{10}$' if i+1 == 10 else f'$G_{i+1}$'
		ax.scatter(X3[:, 0], X3[:, 1], c=colors[i], marker="o", s=10, alpha=0.5, label=label)
		p = np.mean(X3, axis=0)
		# ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
		# offset = 0.3
		# # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
		# xytext = (p[0] + offset, p[1] - offset)
		# ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
		#             ha='center', va='center',  # textcoords='offset points', va='bottom',
		#             bbox=dict(facecolor='none', edgecolor='red', pad=1),
		#             arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
		#                             connectionstyle="angle3, angleA=90,angleB=0"))

	ax.axvline(x=0, color='k', linestyle='--')
	ax.axhline(y=0, color='k', linestyle='--')
	# bbox (x, y, width, height)
	ax.legend(loc='upper right',bbox_to_anchor=(0.5, 0., 0.5, 0.8))

	# plt.title(args['p1'].replace(':', '\n') + f':{title}')
	# # plt.xlim([-2, 15])
	# # plt.ylim([-2, 15])
	plt.xlim([-30, 30])
	plt.ylim([-30, 30])
	# # plt.xticks([])
	# # plt.yticks([])
	plt.tight_layout()
	if not os.path.exists(args['OUT_DIR']):
		os.makedirs(args['OUT_DIR'])
	f = os.path.join(args['OUT_DIR'], data_detail.replace('/', ':') + '.png')
	f = os.path.join(args['OUT_DIR'], '10gaussians.png')
	plt.savefig(f, dpi=600, bbox_inches='tight')
	plt.show()


if __name__ == '__main__':
	# args = {'N_CLIENTS': 0, 'IS_PCA': 'CNN', 'IN_DIR': './datasets', 'NORMALIZE_METHOD': 'std',
	#         'OUT_DIR': 'out',
	#         'DATASET': {'name': 'MNIST',
	#                     'detail': 'n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_1.0_0.1:ratio_0.00:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0'}}
	# X, labels = gaussian3_diff_sigma_n(args, random_state=42)
	# plot_3gaussians(X, labels, args)

	args = {'N_CLIENTS': 0, 'IS_PCA': 'CNN', 'IN_DIR': './datasets', 'NORMALIZE_METHOD': 'std',
	        'OUT_DIR': 'out',
	        'DATASET': {'name': 'MNIST',
	                    'detail': 'n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_0.1_0.1:ratio_0.00:diff_sigma_n|std|PCA_False|M_10|K_10|REMOVE_OUTLIERS_False/SEED_DATA_0'}}
	X, labels = gaussian10_diff_sigma_n(args, random_state=42)
	plot_10gaussians(X, labels, args)
