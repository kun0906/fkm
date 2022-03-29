
import numpy as np

def exp_dim():
	a = np.asarray([[1, 2], [3,4], [5,6]])
	print(a)
	print(a.shape)

	a1 = np.expand_dims(a, axis=0)
	print(a1)
	print(a1.shape)

	a2 = np.expand_dims(a, axis=1)
	print(a2)
	print(a2.shape)

	a3 = a1-a2
	print(a3)
	print(a3.shape)

	a4 = np.sum(np.square(a3), axis=2)
	print(a4)
	print(a4.shape)

	b = np.asarray([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
	print(b)
	b1 = a4/b
	print(b1)

# exp_dim()


import matplotlib.pyplot as plt
colors = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
                  'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
                  'tab20c']

def plot2():
	X= range(100)
	X2 = range(100)
	y = [1, 2, 3, 4, 5, 6, 7, 9, 10, 20] * 20
	y = y[:100]
	mp = {v: i/(len(set(y))-1)* 100 for i, v in enumerate(sorted(set(y)))}
	v2label = {v:k for k, v in mp.items()}
	c = [mp[v] for v in y]
	print(c)
	plt.scatter(X, y, c = c, cmap = 'gnuplot2', alpha=0.5)
	for label, x, y in zip(y, X, X2):
		plt.annotate(label, xy=(x, y), xytext=(0.1, 0.1), fontsize=20, textcoords="offset points")
	ticks = sorted(mp.values())
	print(ticks)
	cbar = plt.colorbar(ticks = ticks, fraction=0.3, pad=0.2)
	# cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
	cbar.ax.set_yticklabels([v2label[v]for v in ticks], fontsize=50)  # horizontal colorbar

	plt.show()

def plot4():
	import matplotlib.pyplot as plt
	import matplotlib.colors
	import numpy as np

	colors = [plt.cm.tab20(0), plt.cm.tab20(1), plt.cm.tab20c(4),
	          plt.cm.tab20c(5), plt.cm.tab20c(6), plt.cm.tab20c(7)]

	cmap = matplotlib.colors.ListedColormap(colors)
	norm = matplotlib.colors.BoundaryNorm(np.arange(1, 8) - 0.5, len(colors))

	x = np.arange(0, 20)

	sc = plt.scatter(x, x, c=x, s=100, cmap='tab20')
	plt.colorbar(sc, ticks=x)

	plt.show()

# plot4()

def plot3():
	X = range(100)
	X2 = range(100)
	y = [1, 2, 3, 4, 5, 6, 7, 9, 10, 50, 100, 200] * 20
	y = y[:100]
	mp = {v:i for i, v in enumerate(sorted(set(y)))}
	v2label = {i: v for v, i in mp.items()}
	# c = [mp[v] for v in y]
	c = [mp[v] for v in y]
	print(c)
	sc = plt.scatter(X, y, c=c, cmap='tab20', alpha=0.5)
	for label, x_, y_ in zip(y, X, X2):
		plt.annotate(label, xy=(x_, y_), xytext=(0.1, 0.1), fontsize=20, textcoords="offset points")
	ticks = list(range(20))
	print(ticks)
	cbar = plt.colorbar(sc, ticks = ticks, fraction=0.3, pad=0.2)
	# ticks = cbar.ax.get_yticks()
	print(ticks)
	# cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
	cbar.ax.set_yticklabels([str(v2label[i]) if i in v2label else str(i) + 'a' for i in ticks], fontsize=10)  # horizontal colorbar

	plt.show()
plot3()
#
# def coloramp_demo():
# 	import matplotlib.pyplot as plt
# 	import numpy as np
# 	import matplotlib as mpl
# 	import matplotlib.pyplot as plt
# 	from matplotlib import cm
# 	from colorspacious import cspace_converter
#
# 	cmaps = {}
#
# 	gradient = np.linspace(0, 1, 256)
# 	gradient = np.vstack((gradient, gradient))
#
# 	def plot_color_gradients(category, cmap_list):
# 		# Create figure and adjust figure height to number of colormaps
# 		nrows = len(cmap_list)
# 		figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
# 		fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))
# 		fig.subplots_adjust(top=1 - 0.35 / figh, bottom=0.15 / figh,
# 		                    left=0.2, right=0.99)
# 		axs[0].set_title(f'{category} colormaps', fontsize=14)
#
# 		for ax, name in zip(axs, cmap_list):
# 			ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
# 			ax.text(-0.01, 0.5, name, va='center', ha='right', fontsize=10,
# 			        transform=ax.transAxes)
#
# 		# Turn off *all* ticks & spines, not just the ones with colormaps.
# 		for ax in axs:
# 			ax.set_axis_off()
# 		plt.show()
# 		# Save colormap list for later.
# 		cmaps[category] = cmap_list
#
# 	plot_color_gradients('Qualitative',
# 	                     ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
# 	                      'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b',
# 	                      'tab20c'])
#
# 	mpl.rcParams.update({'font.size': 12})
#
# 	# Number of colormap per subplot for particular cmap categories
# 	_DSUBS = {'Perceptually Uniform Sequential': 5, 'Sequential': 6,
# 	          'Sequential (2)': 6, 'Diverging': 6, 'Cyclic': 3,
# 	          'Qualitative': 4, 'Miscellaneous': 6}
#
# 	# Spacing between the colormaps of a subplot
# 	_DC = {'Perceptually Uniform Sequential': 1.4, 'Sequential': 0.7,
# 	       'Sequential (2)': 1.4, 'Diverging': 1.4, 'Cyclic': 1.4,
# 	       'Qualitative': 1.4, 'Miscellaneous': 1.4}
#
# 	# Indices to step through colormap
# 	x = np.linspace(0.0, 1.0, 100)
#
# 	# Do plot
# 	for cmap_category, cmap_list in cmaps.items():
#
# 		# Do subplots so that colormaps have enough space.
# 		# Default is 6 colormaps per subplot.
# 		dsub = _DSUBS.get(cmap_category, 6)
# 		nsubplots = int(np.ceil(len(cmap_list) / dsub))
#
# 		# squeeze=False to handle similarly the case of a single subplot
# 		fig, axs = plt.subplots(nrows=nsubplots, squeeze=False,
# 		                        figsize=(7, 2.6 * nsubplots))
#
# 		for i, ax in enumerate(axs.flat):
#
# 			locs = []  # locations for text labels
#
# 			for j, cmap in enumerate(cmap_list[i * dsub:(i + 1) * dsub]):
#
# 				# Get RGB values for colormap and convert the colormap in
# 				# CAM02-UCS colorspace.  lab[0, :, 0] is the lightness.
# 				rgb = cm.get_cmap(cmap)(x)[np.newaxis, :, :3]
# 				lab = cspace_converter("sRGB1", "CAM02-UCS")(rgb)
#
# 				# Plot colormap L values.  Do separately for each category
# 				# so each plot can be pretty.  To make scatter markers change
# 				# color along plot:
# 				# https://stackoverflow.com/q/8202605/
#
# 				if cmap_category == 'Sequential':
# 					# These colormaps all start at high lightness but we want them
# 					# reversed to look nice in the plot, so reverse the order.
# 					y_ = lab[0, ::-1, 0]
# 					c_ = x[::-1]
# 				else:
# 					y_ = lab[0, :, 0]
# 					c_ = x
#
# 				dc = _DC.get(cmap_category, 1.4)  # cmaps horizontal spacing
# 				ax.scatter(x + j * dc, y_, c=c_, cmap=cmap, s=300, linewidths=0.0)
#
# 				# Store locations for colormap labels
# 				if cmap_category in ('Perceptually Uniform Sequential',
# 				                     'Sequential'):
# 					locs.append(x[-1] + j * dc)
# 				elif cmap_category in ('Diverging', 'Qualitative', 'Cyclic',
# 				                       'Miscellaneous', 'Sequential (2)'):
# 					locs.append(x[int(x.size / 2.)] + j * dc)
#
# 			# Set up the axis limits:
# 			#   * the 1st subplot is used as a reference for the x-axis limits
# 			#   * lightness values goes from 0 to 100 (y-axis limits)
# 			ax.set_xlim(axs[0, 0].get_xlim())
# 			ax.set_ylim(0.0, 100.0)
#
# 			# Set up labels for colormaps
# 			ax.xaxis.set_ticks_position('top')
# 			ticker = mpl.ticker.FixedLocator(locs)
# 			ax.xaxis.set_major_locator(ticker)
# 			formatter = mpl.ticker.FixedFormatter(cmap_list[i * dsub:(i + 1) * dsub])
# 			ax.xaxis.set_major_formatter(formatter)
# 			ax.xaxis.set_tick_params(rotation=50)
# 			ax.set_ylabel('Lightness $L^*$', fontsize=12)
#
# 		ax.set_xlabel(cmap_category + ' colormaps', fontsize=14)
#
# 		fig.tight_layout(h_pad=0.0, pad=1.5)
# 		plt.show()
# coloramp_demo()
#
