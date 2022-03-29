
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
mean = [1, 0]
cov = [[0.1, 0], [0, 0.1]]  # diagonal covariance. Note that cov = variance = 0.1 = std ** 2 , however, std = 0.3
x, y = np.random.multivariate_normal(mean, cov, 5000).T
x2 = np.random.normal(1, 0.1, 5000).T
plt.plot(x, y, 'x')
# plt.axis('equal')
plt.show()


def gaussian_data(random_state = 42):
	mus = [-1, 0]
	cov = np.asarray([[0.1, 0], [0, 0.1]])
	n = 10000
	n_outliers = 0
	i = 0
	r = np.random.RandomState(random_state)
	X = r.multivariate_normal(mus, cov, size=n - n_outliers)
	y = np.asarray([f'C{i + 1}'] * X.shape[0])
	X1 = X

	mus = [1, 0]
	cov = np.asarray([[0.1, 0], [0, 0.1]])
	n = 10000
	n_outliers = 0
	i = 1
	r = np.random.RandomState(random_state)
	X = r.multivariate_normal(mus, cov, size=n - n_outliers)
	y = np.asarray([f'C{i + 1}'] * X.shape[0])
	X2 = X

	SHOW = True
	if SHOW:


		# Plot init seeds along side sample data
		fig, ax = plt.subplots()
		# colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
		colors = ["r", "g", "b", "m", 'black']
		ax.scatter(X1[:, 0], X1[:, 1], c=colors[0], marker="x", s=10, alpha=0.3, label='$G_1$')
		p = np.mean(X1, axis=0)
		ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
		offset = 2
		# xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
		xytext = (p[0] - offset, p[1] - offset)
		print(xytext)
		ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='b',
		            ha='center', va='center',  # textcoords='offset points',
		            bbox=dict(facecolor='none', edgecolor='b', pad=1),
		            arrowprops=dict(arrowstyle="->", color='b', shrinkA=1, lw=2,
		                            connectionstyle="angle3, angleA=90,angleB=0"))
		# angleA : starting angle of the path
		# angleB : ending angle of the path

		ax.scatter(X2[:, 0], X2[:, 1], c=colors[1], marker="o", s=10, alpha=0.3, label='$G_2$')
		p = np.mean(X2, axis=0)
		ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
		offset = 2
		# xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
		xytext = (p[0] + offset, p[1] - offset)
		ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
		            ha='center', va='center',  # textcoords='offset points', va='bottom',
		            bbox=dict(facecolor='none', edgecolor='red', pad=1),
		            arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
		                            connectionstyle="angle3, angleA=90,angleB=0"))
		#
		# ax.scatter(X3[:, 0], X3[:, 1], c=colors[2], marker="o", s=10, alpha=0.3, label='$G_3$')
		# p = np.mean(X3, axis=0)
		# ax.scatter(p[0], p[1], marker="x", s=150, linewidths=3, color="w", zorder=10)
		# offset = 0.3
		# # xytext = (p[0] + (offset / 2 if p[0] >= 0 else -offset), p[1] + (offset / 2 if p[1] >= 0 else -offset))
		# xytext = (p[0] + offset, p[1] - offset)
		# ax.annotate(f'({p[0]:.1f}, {p[1]:.1f})', xy=(p[0], p[1]), xytext=xytext, fontsize=15, color='r',
		#             ha='center', va='center',  # textcoords='offset points', va='bottom',
		#             bbox=dict(facecolor='none', edgecolor='red', pad=1),
		#             arrowprops=dict(arrowstyle="->", color='r', shrinkA=1, lw=2,
		#                             connectionstyle="angle3, angleA=90,angleB=0"))

		# show the noise n4
		# ax.scatter(noise[:, 0], noise[:, 1], c=colors[3], marker="o", s=10, alpha=0.3)
		# p = np.mean(X3, axis=0)
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
		ax.legend(loc='upper right', fontsize=13)
		# if True:
		# 	title = params['DATASET']['params']
		# 	plt.title(title[:30] + '\n' + title[30:])

		# if 'xlim' in kwargs:
		# 	plt.xlim(kwargs['xlim'])
		# else:
		# 	plt.xlim([-12, 12])
		# if 'ylim' in kwargs:
		# 	plt.ylim(kwargs['ylim'])
		# else:
		# 	plt.ylim([-12, 12])

		fontsize = 13
		plt.xticks(fontsize=fontsize)
		plt.yticks(fontsize=fontsize)

		plt.tight_layout()
		# if not os.path.exists(params['OUT_DIR']):
		# 	os.makedirs(params['OUT_DIR'])
		# f = os.path.join(params['OUT_DIR'], 'noise-sep.png')
		# print(f)
		# plt.savefig(f, dpi=600, bbox_inches='tight')
		plt.show()

	return

if __name__ == '__main__':
    gaussian_data()