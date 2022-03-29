"""
    Stanford:
        1. For clients, we randomly select initialized centroids or use Kmeans++ to select initialized centroids.
        2. For server, we use the average of clients' centroids.

    # PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 Stanford_client_initialization.py -n '00' > a.txt &
    PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 Stanford_client_initialization.py --dataset 'FEMNIST' \
                --data_details '1client_multiwriters_1digit' --algorithm 'Centralized_random'
"""
# Email: Kun.bj@outlook.com
import argparse
import traceback
from pprint import pprint

import numpy as np
from sklearn.cluster import kmeans_plusplus

from fkm.cluster._federated_base import check_centroids
from fkm.cluster.centralized_kmeans import KMeans
from fkm.utils.utils_func import random_initialize_centroids
from fkm.utils.utils_stats import evaluate2, plot_progress


def compute_step_for_client(client_data, centroids):
	# compute distances
	# computationally efficient
	differences = np.expand_dims(client_data, axis=1) - np.expand_dims(centroids, axis=0)
	sq_dist = np.sum(np.square(differences), axis=2)

	# memory efficient
	# sq_dist = np.zeros((client_data.shape[0], self.n_clusters))
	# for i in range(self.n_clusters):
	#     sq_dist[:, i] = np.sum(np.square(client_data - centroids[i, :]), axis=1)

	# assign to cluster
	labels = np.argmin(sq_dist, axis=1)

	# update centroids
	centroid_updates = np.zeros_like(centroids)
	counts = np.zeros(centroids.shape[0])
	for i in range(centroids.shape[0]):
		mask = np.equal(labels, i)
		counts[i] = np.sum(mask)
		if counts[i] > 0:
			# # remove the outliers
			# df = client_data[mask] - centroids[i]
			# diffs = np.sum(np.abs(df), axis=1)
			# thres = np.quantile(diffs, q = 0.8)
			# mk = np.where(diffs < thres) # remove the outliers
			#
			# centroid_updates[i, :] = np.sum(client_data[mk] - centroids[i], axis=0)
			centroid_updates[i, :] = np.sum(client_data[mask] - centroids[i], axis=0)
	return centroid_updates, counts


class KMeansFederated(KMeans):
	def __init__(
			self,
			n_clusters,
			server_init_method='greedy',
			client_init_method='random',
			true_centroids=None,
			max_iter=300,
			tol=1e-4,
			distance_metric='euclidean',
			random_state=None,
			reassign_min=None,
			reassign_after=None,
			verbose=0,
			batch_size=None,
			sample_fraction=1.0,
			epochs_per_round=1,
			learning_rate=None,
			max_no_change=None,
			adaptive_lr=None,
			momentum=None,
			epoch_lr=1.0,
			params={},
	):
		super().__init__(
			n_clusters=n_clusters,
			# init_centroids=None,
			max_iter=max_iter,
			tol=tol,
			distance_metric=distance_metric,
			random_state=random_state,
			reassign_min=reassign_min,
			reassign_after=reassign_after,
			verbose=verbose
		)
		self.use_client_init_centroids = True
		self.batch_size = batch_size
		self.sample_fraction = sample_fraction
		self.epochs = epochs_per_round
		self.lr = learning_rate
		self.adaptive_lr = adaptive_lr
		self.max_no_change = max_no_change
		self.momentum_rate = momentum
		self.epoch_lr = epoch_lr
		self.server_init_method = server_init_method
		self.client_init_method = client_init_method
		self.true_centroids = true_centroids
		self.random_state = random_state
		self.params = params

	# def do_federated_round_single_step(self, clients_in_round, centroids):
	# 	# print(len(clients_in_round))
	# 	# print(clients_in_round[0].shape)
	# 	updates_sum = np.zeros((self.n_clusters, self.dim))
	# 	counts = np.zeros(self.n_clusters)
	# 	for client_data in clients_in_round:
	# 		client_updates_sum, client_counts = compute_step_for_client(
	# 			client_data=client_data,
	# 			centroids=centroids
	# 		)
	# 		# if self.epoch_lr is not None:
	# 		#     client_updates_sum = self.epoch_lr * client_updates_sum
	# 		updates_sum += client_updates_sum
	# 		counts += client_counts
	# 		if self.verbose >= 10:
	# 			print("client_counts: {}; client_updates: {}".format(client_counts, client_updates_sum))
	# 	return updates_sum, counts

	def do_federated_round(self, clients_in_round, centroids, iteration):  # kun's version
		updates_sum = np.zeros((self.n_clusters, self.dim))
		counts = np.zeros(self.n_clusters)
		KM_params = {'centroids': [], 'cluster_sizes': [], 'avg_dists': []}
		client_centroids_avg = np.zeros((self.n_clusters, self.dim))
		for i, client_data in enumerate(clients_in_round):
			if self.use_client_init_centroids:  # added by kun
				if self.client_init_method == 'kmeans++':
					# Calculate seeds from kmeans++
					client_centroids, indices = kmeans_plusplus(client_data, n_clusters=self.n_clusters,
					                                            random_state=self.random_state)
				elif self.client_init_method == 'true':  # true centroids
					client_centroids = self.true_centroids['train']
				else:
					# client_centroids = randomly_init_centroid(0, self.n_clusters + 1, self.dim, self.n_clusters, self.random_state)
					client_centroids = random_initialize_centroids(client_data, self.n_clusters, self.random_state)
			else:
				client_centroids = centroids
			if iteration == 0:
				_, client_counts = compute_step_for_client(
					client_data=client_data,
					centroids=client_centroids,
				)
				updates_sum += (client_centroids - centroids) * np.expand_dims(client_counts,
				                                                               axis=1)  # for the initial centroids
				counts += client_counts
				client_updates_sum = client_centroids - centroids
			else:
				for e in range(self.epochs):
					client_updates_sum, client_counts = compute_step_for_client(
						client_data=client_data,
						centroids=client_centroids
					)
					interim_updates = client_updates_sum / np.expand_dims(
						np.maximum(client_counts, np.ones_like(client_counts)), axis=1)
					# if self.epoch_lr is not None:
					# 	interim_updates = self.epoch_lr * interim_updates
					client_centroids = client_centroids + interim_updates
					if self.verbose >= 5:
						print(f'\t~client_{i}, epoch_{e}, client_centroids: {client_centroids}, '
						      f'client_counts: {client_counts}, given centroids: {centroids}')
				# print(client_centroids)
				client_updates_sum = (client_centroids - centroids) * np.expand_dims(client_counts, axis=1)
				updates_sum += client_updates_sum
				counts += client_counts  # each cluster's size (i.e., number of data points belongs to a cluster)

			if self.verbose >= 4 and iteration > 0:
				print("client_{}, client_counts: {}; client_updates_sum: {}, each_update: {}".format(i, client_counts,
				                                                                                     client_updates_sum,
				                                                                                     (client_centroids
				                                                                                      - centroids)))
		self.use_client_init_centroids = False
		if self.verbose >= 3:
			print(f'total update_sum: {updates_sum} and counts: {counts}')
		return updates_sum, counts, KM_params

	def fit(self, X_dict, y_dict, splits=None, record_at=None):
		X = X_dict['train']
		self.n_clients = len(X)
		self.dim = X[0].shape[1]
		# clients_per_round = max(1, int(self.sample_fraction * self.n_clients))
		# print(f'clients_per_round: {clients_per_round}')
		if self.n_clients > 3:
			if 0 < self.sample_fraction < 1:
				n_clients_per_round = max(1, int(self.sample_fraction * self.n_clients))
			else:
				print(f'Error: {self.sample_fraction}')
				return
		else:
			n_clients_per_round = self.n_clients
		# placehold for centroids
		n_consecutive = 0
		not_changed = 0
		overall_counts = np.zeros(self.n_clusters)
		# momentum = np.zeros_like(centroids)
		means_record = []
		stds_record = []
		to_reassign = np.zeros(self.n_clusters)

		self.training_iterations = self.max_iter
		self.history = []
		for iteration in range(0, self.max_iter):
			r = np.random.RandomState(iteration * max(1, self.random_state))
			if iteration % 10 == 0:  # only choose once. If choose everytime, the model won't converge.
				indices = r.choice(range(self.n_clients), size=n_clients_per_round,
				                   replace=False)  # without replacement and random
				clients_in_round = [X[j] for j in indices]
			else:
				# use previous client_in_round
				pass
			# print(clients_in_round)
			if self.verbose >= 2:
				print("round: {}".format(iteration))

			if iteration == 0:
				updates_sum, counts, _ = self.do_federated_round(
					clients_in_round=clients_in_round,
					centroids=np.zeros((self.n_clusters, self.dim)), iteration=iteration
				)
				self.use_client_init_centroids = False
				centroids = updates_sum / np.expand_dims(np.maximum(counts, np.ones_like(counts)),
				                                              axis=1)  # for the first iteration, we use the average of clients' centroids as the server's initialized centroids
				# make sure each centroid that has a point assigned to it at least
				# compute centroids on part of all clients' data
				self.centroids = check_centroids(centroids, clients_in_round)
				self.initial_centroids = self.centroids
				if self.verbose >= 3:
					print(f'initial_centroids: {self.centroids}')
				# testing after each iteration
				scores = evaluate2(
					kmeans=self,
					x=X_dict, y=y_dict,
					splits=splits,
					federated=True,
					verbose=self.verbose,
				)
				centroids_diff = {}
				for split in splits:
					centroids_diff[split] = self.centroids - self.true_centroids[split]
				centroids_update = np.zeros((self.n_clusters, self.dim))
				self.history.append({'iteration': iteration, 'centroids': self.centroids, 'scores': scores,
				                     'centroids_update': centroids_update, 'centroids_diff': centroids_diff})
				if self.verbose >= 3:
					for split in splits:
						print(f'{split}:')
						pprint(scores[split])
				continue

			# updates_sum, counts = self.do_federated_round_single_step(
			#     clients_in_round=clients_in_round,
			#     centroids=centroids,
			# )
			updates_sum, counts, _ = self.do_federated_round(
				clients_in_round=clients_in_round,
				centroids=self.centroids.copy(), iteration=iteration
			)

			# overall_counts += counts
			# np.maximum(counts, np.one_like(counts)) # if counts are 0s, the max are 1s
			centroids_update = updates_sum / np.expand_dims(np.maximum(counts, np.ones_like(counts)), axis=1)

			# if self.adaptive_lr:
			#     rel_counts = counts / np.maximum(overall_counts, np.ones_like(overall_counts))
			#     update_weights = np.minimum(self.adaptive_lr, rel_counts)
			#     centroids_update = centroids_update * np.expand_dims(update_weights, axis=1)
			#
			# if self.lr is not None:
			#     centroids_update = self.lr * centroids_update
			#
			# if self.momentum_rate is not None:
			#     momentum = self.momentum_rate * momentum + (1 - self.momentum_rate) * centroids_update
			#     centroids_update = momentum

			# np.sum(np.square(centroids - (centroids + centroid_updates)), axis=1)
			delta = np.sum(np.square(centroids_update))
			if self.verbose >= 2:
				print(f'iteration: {iteration}, np.sum(np.square(centroids_update)): {delta}')
			if delta < self.tol:
				if n_consecutive >= self.params['n_consecutive']:
					self.training_iterations = iteration
					# training finishes in advance
					break
				else:
					n_consecutive += 1
			else:
				n_consecutive = 0

			# at the first iteration, the centroids should use the mean of clients' centroids.
			self.centroids = self.centroids + centroids_update
			if self.verbose >= 3:
				print(f'server\'s centroids_update: {centroids_update} and n_points per cluster: {counts}')
				print(f'new centroids: {self.centroids}')
			# print(iteration, centroids)
			# changed = np.any(np.absolute(centroids_update) > self.tol)

			# if self.reassign_min is not None:
			#     for i in range(self.n_clusters):
			#         if counts[i] < (sum(counts) * self.reassign_min):
			#             to_reassign[i] += 1
			#         else:
			#             to_reassign[i] = 0
			#         if to_reassign[i] >= self.reassign_after:
			#             centroids[i] = randomly_init_centroid(0, self.n_clusters + 1, self.dim, 1, self.random_state)
			#             momentum[i] = np.zeros(self.dim)
			#             to_reassign[i] = 0
			#             changed = True
			#
			# if self.max_no_change is not None:
			#     not_changed += 1
			#     if changed:
			#         not_changed = 0
			#     if not_changed > self.max_no_change:
			#         break
			#
			# if record_at is not None and iteration in record_at:
			#     means, stds = record_state(centroids, np.concatenate(X, axis=0))
			#     means_record.append(means)
			#     stds_record.append(stds)

			# testing after each iteration
			scores = evaluate2(
				kmeans=self,
				x=X_dict, y=y_dict,
				splits=splits,
				federated=True,
				verbose=self.verbose,
			)
			centroids_diff = {}
			for split in splits:
				centroids_diff[split] = self.centroids - self.true_centroids[split]
			self.history.append({'iteration': iteration, 'centroids': self.centroids, 'scores': scores,
			                     'centroids_update': centroids_update, 'centroids_diff': centroids_diff})
			if self.verbose >= 3:
				for split in splits:
					print(f'{split}:')
					pprint(scores[split])
		if record_at is not None:
			#  NOTE: only for dummy data
			plot_progress(means_record, stds_record, record_at)


# self.cluster_centers_ = centroids
# return centroids, overall_counts
#
#
# if __name__ == "__main__":
# 	print(__file__)
# 	parser = argparse.ArgumentParser(description='Description of your program')
# 	# parser.add_argument('-C', '--config_file', help='A configuration file (yaml) that includes all parameters',
# 	#                     default='config.yaml')
# 	# parser.add_argument('-p', '--py_name', help='python file name', required=True)
# 	parser.add_argument('-S', '--dataset', help='dataset', default='NBAIOT')
# 	# parser.add_argument('-T', '--data_details', help='data details', default='n1_5000-sigma1_3+n2_5000-sigma2_3+n3_5000-sigma3_3+n4_5000-sigma4_3:ratio_0.0:diff_sigma_n')
# 	# parser.add_argument('-T', '--data_details', help='data details', default='n1_5000-sigma1_0.3_0.3+n2_5000-sigma2_0.3_0.3+n3_10000-sigma3_0.3_0.3+n4_0.1-sigma4_0.3_0.3:ratio_0.0:diff_sigma_n')
# 	parser.add_argument('-T', '--data_details', help='data details',
# 	                    default='nbaiot_user_percent')
# 	parser.add_argument('-M', '--algorithm', help='algorithm', default='Federated-Server_average-Client_kmeans++')
# 	parser.add_argument('-K', '--n_clusters', help='number of clusters', default=2)  # 9 or 11
# 	parser.add_argument('-C', '--n_clients', help='number of clients', default=2)
# 	# args = vars(parser.parse_args())
# 	args = parser.parse_args()
# 	pprint(args)
# 	p3 = __file__.split('/')[-1]
# 	params = get_experiment_params(p0=args.dataset, p1=args.data_details, p2=args.algorithm, p3=p3,
# 	                               n_clusters=int(args.n_clusters), n_clients=int(args.n_clients))
# 	pprint(params)
# 	try:
# 		run_clustering_federated(
# 			params,
# 			KMeansFederated,
# 			verbose=5 if args.dataset == 'FEMNIST' else 10,
# 		)
# 	except Exception as e:
# 		print(f'Error: {e}')
# 		traceback.print_exc()
