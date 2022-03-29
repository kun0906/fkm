import copy

import numpy as np
from sklearn.cluster import kmeans_plusplus

from fkm.datasets.gen_dummy import load_federated
from fkm.cluster._greedy_init import greedily_initialize
from fkm.cluster.centralized_kmeans import KMeans
from fkm.utils.utils_func import randomly_init_centroid, record_state,timer
from fkm.utils.utils_stats import evaluate, plot_stats, plot_progress


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
            centroid_updates[i, :] = np.sum(client_data[mask] - centroids[i], axis=0)
    return centroid_updates, counts


def compute_client_params(X, centroids):
    differences = np.expand_dims(X, axis=1) - np.expand_dims(centroids, axis=0)
    sq_dist = np.sum(np.square(differences), axis=2)

    # memory efficient
    # sq_dist = np.zeros((client_data.shape[0], self.n_clusters))
    # for i in range(self.n_clusters):
    #     sq_dist[:, i] = np.sum(np.square(client_data - centroids[i, :]), axis=1)

    # assign to cluster
    labels = np.argmin(sq_dist, axis=1)
    cluster_avg_dists = []
    cluster_sizes = []
    for i in range(centroids.shape[0]):
        mask = np.equal(labels, i)
        ni = np.sum(mask)
        if ni:
            avg_dist = np.sqrt(np.sum(np.square(X[mask] - centroids[i]))) / ni
        else:
            avg_dist = 0.0
        cluster_sizes.append(ni)
        cluster_avg_dists.append(avg_dist)

    return centroids, cluster_sizes, cluster_avg_dists


def update_server_centroids(KM_centroids, KM_ns, KM_ds, centroids, alpha):
    differences = np.expand_dims(KM_centroids, axis=1) - np.expand_dims(centroids, axis=0)
    sq_dist = np.sum(np.square(differences), axis=2)

    # memory efficient
    # sq_dist = np.zeros((client_data.shape[0], self.n_clusters))
    # for i in range(self.n_clusters):
    #     sq_dist[:, i] = np.sum(np.square(client_data - centroids[i, :]), axis=1)

    # assign to cluster
    labels = np.argmin(sq_dist, axis=1)
    n_clusters, n_dim = centroids.shape
    new_centroids = np.zeros((n_clusters, n_dim))
    # for i in range(centroids.shape[0]):
    #     mask = np.equal(labels, i)
    #     ni = np.sum(mask)
    #     if ni:
    #         diff = np.sum(KM_centroids[mask]-centroids[i]) / ni
    #     else:
    #         diff = 0
    #     new_centroids[i] = centroids[i] + alpha * diff

    # weighted update
    for i in range(centroids.shape[0]):
        mask = np.equal(labels, i)
        wi = KM_ns[mask]
        # ni = np.sum(mask)
        ni = np.sum(wi)
        if ni:
            diff = np.sum(wi[:, np.newaxis] * (KM_centroids[mask] - centroids[i]), axis=0) / ni
        else:
            diff = 0
        new_centroids[i] = centroids[i] + alpha * diff

    return new_centroids


class KMeansFederated(KMeans):
    def __init__(
            self,
            n_clusters,
            init_centroids='random',
            max_iter=300,
            tol=1e-4,
            distance_metric='euclidean',
            random_state=None,
            reassign_min=None,
            reassign_after=None,
            verbose=False,
            batch_size=None,
            sample_fraction=1.0,
            epochs_per_round=1,
            learning_rate=None,
            max_no_change=None,
            adaptive_lr=None,
            momentum=None,
            epoch_lr=1,
    ):
        super().__init__(
            n_clusters=n_clusters,
            init_centroids=init_centroids,
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

    def do_federated_round_single_step(self, clients_in_round, centroids):
        # print(len(clients_in_round))
        # print(clients_in_round[0].shape)
        updates_sum = np.zeros((self.n_clusters, self.n_dims))
        counts = np.zeros(self.n_clusters)
        for client_data in clients_in_round:
            client_updates_sum, client_counts = compute_step_for_client(
                client_data=client_data,
                centroids=centroids
            )
            # if self.epoch_lr is not None:
            #     client_updates_sum = self.epoch_lr * client_updates_sum
            updates_sum += client_updates_sum
            counts += client_counts
            if self.verbose:
                print("client_counts: {}; client_updates: {}".format(client_counts, client_updates_sum))
        return updates_sum, counts

    def do_federated_round(self, clients_in_round, centroids):
        updates_sum = np.zeros((self.n_clusters, self.n_dims))
        counts = np.zeros(self.n_clusters)
        KM_params = {'centroids': [], 'cluster_sizes': [], 'avg_dists': []}
        for client_data in clients_in_round:
            if self.use_client_init_centroids:  # added by kun
                # Calculate seeds from kmeans++
                client_centroids, indices = kmeans_plusplus(client_data, n_clusters=self.n_clusters,
                                                            random_state=self.random_state)
                centroids = client_centroids
            else:
                client_centroids = centroids
            for e in range(self.epochs):
                client_updates_sum, client_counts = compute_step_for_client(
                    client_data=client_data,
                    centroids=client_centroids
                )
                interim_updates = client_updates_sum / np.expand_dims(
                    np.maximum(client_counts, np.ones_like(client_counts)), axis=1)
                if self.epoch_lr is not None:
                    interim_updates = self.epoch_lr * interim_updates
                client_centroids = client_centroids + interim_updates
            updates_sum += (client_centroids - centroids) * np.expand_dims(client_counts, axis=1)
            counts += client_counts
            # compute client's params (such as centroids, cluster_sizes, and avg_dists)
            client_centroids, client_cluster_sizes, client_avg_dists = compute_client_params(client_data,
                                                                                             client_centroids)
            KM_params['centroids'].extend(list(client_centroids))
            KM_params['cluster_sizes'].extend(list(client_cluster_sizes))
            KM_params['avg_dists'].extend(list(client_avg_dists))
            if self.verbose:
                print("client_counts: {}; client_updates_sum: {}".format(client_counts, client_updates_sum))
        self.use_client_init_centroids = False

        KM_params['centroids'] = np.asarray(KM_params['centroids'])
        KM_params['cluster_sizes'] = np.asarray(KM_params['cluster_sizes'])
        KM_params['avg_dists'] = np.asarray(KM_params['avg_dists'])

        return updates_sum, counts, KM_params

    def fit(self, X, record_at=None):
        x = X
        self.num_clients = len(x)
        self.n_dims = x[0].shape[1]
        # clients_per_round = max(1, int(self.sample_fraction * self.num_clients))
        clients_per_round = 3
        # centroids = self.do_init_centroids()
        centroids = np.zeros((self.n_clusters, self.n_dims))
        self.n_consecutive = 0
        not_changed = 0
        overall_counts = np.zeros(self.n_clusters)
        momentum = np.zeros_like(centroids)
        means_record = []
        stds_record = []
        to_reassign = np.zeros(self.n_clusters)

        # while changed and round < self.max_iter:
        self.training_iterations = self.max_iter
        for iteration in range(1, 1 + self.max_iter):
            # clients_in_round = random.sample(x, clients_per_round)
            r = np.random.RandomState(iteration*max(1, self.random_state))
            clients_in_round = r.choice(np.asarray(x, dtype='object'), size=clients_per_round,
                                        replace=False)  # without replacement and random
            # print(clients_in_round)
            if self.verbose:
                print("round: {}".format(iteration))

            # updates_sum, counts = self.do_federated_round_single_step(
            #     clients_in_round=clients_in_round,
            #     centroids=centroids,
            # )
            updates_sum, counts, KM_params = self.do_federated_round(
                clients_in_round=clients_in_round,
                centroids=centroids,
            )
            if iteration == 1:  # for the first round, the server will select K centroids from K*M client's centroids.
                KM_centroids, KM_ns, KM_ds = KM_params['centroids'], KM_params['cluster_sizes'], KM_params['avg_dists']
                # 1. ignore n_points per each centroid
                # centroids, indices = kmeans_plusplus(KM_centroids, n_clusters=self.n_clusters, random_state=self.random_state)

                # 2. consider about cluster size: can't do this in the following way because the size of new_KM_centroids will be all clients'data
                # new_KM_centroids = []
                # for c, s in zip(KM_centroids, centroid_size):
                #     new_KM_centroids.extend([c] * int(s))
                # KM_centroids = np.asarray(new_KM_centroids)
                # centroids, indices = kmeans_plusplus(KM_centroids, n_clusters=self.n_clusters, random_state=self.random_state)

                # 3. weighted KMeans++
                # centroids, indices = weighted_kmeans_plusplus(KM_centroids, n_clusters=self.n_clusters, random_state=self.random_state)

                # 4. Greedy initialization
                centroids, indices = greedily_initialize(KM_centroids, KM_ns, KM_ds, self.n_clusters)
                self.initial_centroids = centroids
                continue
            # update server's params
            overall_counts += counts
            updates = updates_sum / np.expand_dims(np.maximum(counts, np.ones_like(counts)), axis=1)

            if self.adaptive_lr:
                rel_counts = counts / np.maximum(overall_counts, np.ones_like(overall_counts))
                update_weights = np.minimum(self.adaptive_lr, rel_counts)
                updates = updates * np.expand_dims(update_weights, axis=1)

            if self.lr is not None:
                updates = self.lr * updates

            if self.momentum_rate is not None:
                momentum = self.momentum_rate * momentum + (1 - self.momentum_rate) * updates
                updates = momentum

            # np.sum(np.square(centroids - (centroids + updates)), axis=1)
            if np.sum(np.square(updates)) < self.tol:
                if self.n_consecutive >= 5:
                    self.training_iterations = iteration
                    # training finishes in advance
                    break
                else:
                    self.n_consecutive += 1
            else:
                self.n_consecutive = 0

            centroids = centroids + updates
            # centroids = update_server_centroids(KM_centroids, KM_ns, KM_ds, centroids, alpha=self.lr)
            changed = np.any(np.absolute(updates) > self.tol)

            if self.reassign_min is not None:
                for i in range(self.n_clusters):
                    if counts[i] < (sum(counts) * self.reassign_min):
                        to_reassign[i] += 1
                    else:
                        to_reassign[i] = 0
                    if to_reassign[i] >= self.reassign_after:
                        centroids[i] = randomly_init_centroid(0, self.n_clusters + 1, self.n_dims, 1)
                        momentum[i] = np.zeros(self.n_dims)
                        to_reassign[i] = 0
                        changed = True

            if self.max_no_change is not None:
                not_changed += 1
                if changed:
                    not_changed = 0
                if not_changed > self.max_no_change:
                    break

            if record_at is not None and iteration in record_at:
                means, stds = record_state(centroids, np.concatenate(x, axis=0))
                means_record.append(means)
                stds_record.append(stds)
        if record_at is not None:
            #  NOTE: only for dummy data
            plot_progress(means_record, stds_record, record_at)

        self.cluster_centers_ = centroids
        return centroids, overall_counts

@timer
def run_clustering_federated(verbose=False, dummy=False):
    # settings
    # num_seeds = [10]
    np.random.seed(1234)  # set the global seed for numpy
    # num_seeds = [5]  # [30, 10, 10, 3, 3, 10], we run the experiments 5 times.
    REPEATS = 30
    # seeds = np.random.choice(range(100000), size=max(num_seeds), replace=False)   # used for repetitions: here is 30 repetitions
    seeds = [10 * v ** 2 for v in range(1, REPEATS + 1, 1)]
    print(f'REPEATS: {REPEATS}, {seeds}')
    # seeds = range(NUM_SEEDS)
    NUM_CLUSTERS = 2
    LIMIT_CSV = None
    EPOCHS = 1
    ROUNDS = 100
    use_metric = "davies_bouldin"  # "euclidean",  "davies_bouldin" "silhouette"
    # init_centroids = "random"  # "random "GMM"
    LR = 0.5
    LR_AD = None
    EPOCH_LR = 0.5
    MOMENTUM = None  # 0.5
    # RECORD = range(1, ROUNDS+1)
    RECORD = None
    REASSIGN = (0.01, 10)

    stats = {
        'train': {'avg': [], 'std': []},
        'test': {'avg': [], 'std': []},
    }
    # clients_per_round_fractions = [0.1, 0.3, 0.5, 0.70, 0.9, 1.00, "central"]
    # clients_per_round_fractions = [0.1]
    clients_per_round_fractions = [1.0, "central"]
    # clients_per_round_fractions = ["central"]
    # rhos = list(range(1, 100, 10)) + ['central']
    rhos = [1] + list(range(0, 30 + 1, 2))[1:]
    for grid_i, rho in enumerate(rhos):
        C = rho
        # num_s = num_seeds[grid_i] if len(num_seeds) > 1 else num_seeds[0] # why? not sure the reason
        # num_s = num_seeds[0]
        print(f'grid_i: {grid_i}, C: {C}')
        print("clients_per_round_fraction", C)
        splits = ['train', 'test']
        results = {}
        for split in splits:
            results[split] = []

        # only have one train and test set. However, given different seed, the initialization is different
        raw_x, _ = load_federated(
            limit_csv=LIMIT_CSV,
            verbose=verbose,
            seed=100,
            dummy=dummy,
            clusters=NUM_CLUSTERS,
            rho = rho
        )
        history = {'x': raw_x}
        res = []
        for s_i, SEED in enumerate(seeds):  # for repetitions to obtain average and std score.
            print(f'***{s_i}th repeat:')
            # np.random.seed(seed=SEED)
            if verbose:
                print("seed", SEED)
            # # for each SEED, there will be a new 'x'
            # x, _ = load_federated(
            #     limit_csv=LIMIT_CSV,
            #     verbose=verbose,
            #     seed=SEED,
            #     dummy=dummy,
            #     clusters=NUM_CLUSTERS,
            # )
            x = copy.deepcopy(raw_x)
            centroids = None
            # if dummy == 'dummy_2D':
            #     centroids = None
            # elif dummy == '':
            #     centroids = "random"
            # else:
            #     centroids = init_centroids_gmm(
            #         init_centroids=init_centroids,
            #         num_clusters=NUM_CLUSTERS,
            #         seed=SEED,
            #         dims=x['train'][0].shape[1],
            #         verbose=verbose,
            #     )
            if C == "central":
                # print(len(x['train']), x['train'][0].shape)
                for spl in splits:
                    x[spl] = np.concatenate(x[spl], axis=0)
                kmeans = KMeans(
                    n_clusters=NUM_CLUSTERS,
                    # batch_size=BATCH_SIZE,
                    init_centroids='kmeans++',
                    random_state=SEED,
                    max_iter=ROUNDS,
                    verbose=verbose,
                    reassign_min=REASSIGN[0],
                    reassign_after=REASSIGN[1],
                )
            else:
                kmeans = KMeansFederated(
                    n_clusters=NUM_CLUSTERS,
                    # batch_size=BATCH_SIZE,
                    sample_fraction=C,
                    epochs_per_round=EPOCHS,
                    max_iter=ROUNDS,
                    init_centroids='greedy',  # hard-code in the function
                    random_state=SEED,
                    learning_rate=LR,
                    adaptive_lr=LR_AD,
                    epoch_lr=EPOCH_LR,
                    momentum=MOMENTUM,
                    reassign_min=REASSIGN[0],
                    reassign_after=REASSIGN[1],
                )
            if verbose:
                print(vars(kmeans))
            kmeans.fit(
                X=x['train'],
                record_at=RECORD,
            )
            scores = evaluate(
                kmeans=kmeans,
                x=x,
                splits=splits,
                use_metric=use_metric,
                federated=(C != "central"),
                verbose=verbose,
            )
            res.append({'seed': SEED, 'initial_centroids': kmeans.initial_centroids,
                        'final_centroids': kmeans.cluster_centers_, 'training_iterations': kmeans.training_iterations,
                        'scores': scores})
            # print(grid_i, SEED, scores)
            for key, value in scores.items():
                results[key].append(value)
            if verbose:
                print(use_metric, scores)
        history['res'] = res
        # plot_centroids(history, title=f'Centralized KMeans with KMeans++ initialization' if type(C) == str
        # else f'Federated KMeans with greedy initialization')
        # get the average and std for each clients_per_round_fraction
        results_avg = {}
        for key, value in results.items():
            score_mean = np.around(np.mean(value), decimals=3)
            score_std = np.around(np.std(value), decimals=3)
            results_avg[key] = (score_mean, score_std)
            stats[key]['avg'].append(score_mean)
            stats[key]['std'].append(score_std)
        print(use_metric, results_avg)
    print('\n')
    print(stats)

    # plot_rhos(stats)
    plot_stats(
        stats,
        x_variable=rhos,
        x_variable_name="rhos",
        metric_name='WCSS (within-cluster sums of squares)' if use_metric == 'euclidean' else use_metric,
        title='Rhos'
    )


if __name__ == "__main__":
    # run_clustering(
    #     # verbose=True
    # )
    run_clustering_federated(
        dummy='dummy_2D_M3',
        verbose=False,
    )
