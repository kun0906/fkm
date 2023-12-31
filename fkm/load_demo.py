import pickle

in_file = '/Users/kun/Downloads/SEED_DATA_440.dat'
with open(in_file, 'rb') as f:
    data = pickle.load(f)
# print(data)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# # Load the MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# # Flatten the images
# x_train = x_train.reshape(x_train.shape[0], -1)
# x_test = x_test.reshape(x_test.shape[0], -1)

# x_train = np.vstack(data[0]['train'])
# y_train = np.vstack(data[1]['train'])

x_train = []
y_train = []
for i in range(2):
    x_train.append(data[0]['train'][i][:5000])
    y_train.append(data[1]['train'][i][:5000])

x_train = np.asarray(x_train).reshape((-1, 115))
y_train = np.asarray(y_train).reshape((-1, ))
# Perform t-SNE
tsne = TSNE(n_components=2, perplexity=100, random_state=42)
X_tsne = tsne.fit_transform(x_train)

# Visualize the results
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train)
plt.show()