"""
Empty clusters of kmeans
https://user.ceng.metu.edu.tr/~tcan/ceng465_f1314/Schedule/KMeansEmpty.html
https://stackoverflow.com/questions/11075272/k-means-empty-cluster
https://stackoverflow.com/questions/70660989/how-does-k-means-work-when-initial-centroid-locations-far-away-from-the-data-are
https://github.com/tslearn-team/tslearn/issues/269

Possible solutions: for each iteration, the empty cluster could appear, so we should check and adjust the empty clusters in each iteration.
    1. For each old centroid, compute each point to its centroid's distance
       then find the farthest distance, and then assign the corresponding point to the empty cluster (for the rest empty cluster, do the same thing).
        sklearn: _relocate_empty_clusters_dense, https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bc/sklearn/cluster/_k_means_common.pyx)
    2. random select a point and assign it to the empty cluster
    3. discard the empty clusters() or rerun kmeans with new initial centroids with different seeds.

"""
import numpy as np


def check_centroids(centroids, clients_in_round):
	""" Assign the farthest point to the empty cluster.

	Parameters
	----------
	centroids
	clients_in_round

	Returns
	-------

	"""
	# make sure each centroid that has a point assigned to it at least
	# reference: https://github.com/klebenowm/cs540/blob/master/hw1/KMeans.java
	# https://datascience.stackexchange.com/questions/24380/in-k-means-what-happens-if-a-centroid-is-never-the-closest-to-any-point
	farthest_points = []
	centroids_info = [0 for _ in range(len(centroids))]
	for i_client, client_data in enumerate(clients_in_round):
		# for each centroid, find the farthest point.
		# computationally efficient
		differences = np.expand_dims(client_data, axis=1) - np.expand_dims(centroids, axis=0)
		sq_dist = np.sum(np.square(differences), axis=2)

		# assign to cluster
		labels = np.argmin(sq_dist, axis=1)

		# update centroids
		client_farthest_points = []
		for i_centroid in range(centroids.shape[0]):
			mask = np.equal(labels, i_centroid)
			cnt = sum(mask)
			if cnt > 0:
				tmp_sq_dist = sq_dist[mask][:, i_centroid]
				tmp_data = client_data[mask]
				tmp = [(i_client, i_centroid, d_, point_) for idx_, (d_, point_) in enumerate(zip(tmp_sq_dist, tmp_data))]
				client_farthest_points.extend(tmp)
			centroids_info[i_centroid] += cnt
		topk = sorted(client_farthest_points, key=lambda x: x[2], reverse=True)[:len(centroids)]
		farthest_points.extend(topk)

	print(f'before adjustment of valid centroids ({len([1 for cnt_ in centroids_info if cnt_ > 0 ])}), each centroid have the number of points', centroids_info)
	farthest_points = sorted(farthest_points, key=lambda x: x[2], reverse=True)   # only needs keep the top k data and send them back to the server
	for i, cnt in enumerate(centroids_info):
		if cnt == 0:
			i_client, i_centroid, d_, point_ = farthest_points.pop(0)
			while centroids_info[i_centroid] <= 1:
				i_client, i_centroid, d_, point_ = farthest_points.pop(0)
			centroids[i] = point_
			centroids_info[i_centroid] -= 1
			centroids_info[i] += 1
	print(f'After adjustment of centroids ({len([1 for cnt_ in centroids_info if cnt_ > 0 ])}), each centroid have the number of points', centroids_info)
	return centroids
