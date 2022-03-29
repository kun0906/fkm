import numpy as np

from fkm.datasets.gen_dummy import gen_data
from kmeans_sklearn import init_kmeans_sklearn
from kmeans_python_old import init_kmeans_python, KMeans, KMeansFederated
from utils_stats import evaluate, plot_stats
from utils_data import load, load_federated, init_centroids_gmm


def init_kmeans(implementation, **kwargs):
    if implementation == 'sklearn':
        return init_kmeans_sklearn(**kwargs)
    elif implementation == 'python':
        return init_kmeans_python(**kwargs)
    else:
        raise NotImplementedError


def run_clustering(verbose=False):
    # settings
    implementation = "python"
    NUM_SEEDS = 3
    seeds = np.random.choice(range(1000000), size=NUM_SEEDS, replace=False)
    # seeds = range(NUM_SEEDS)
    NUM_CLUSTERS = 10
    NUM_TRAIN = 1e5
    NUM_TEST = NUM_TRAIN
    BATCH_SIZE = None  # set to None for classic kmeans
    use_metric = "euclidean"  # "euclidean",  "davies_bouldin" "silhouette"
    init_centroids = "GMM"  # "random"
    stats = {
        'train': {'avg': [], 'std': []},
        'test': {'avg': [], 'std': []},
        'val': {'avg': [], 'std': []},
    }
    subsample_rates = [0.01, 0.1, None]
    for SUBSAMPLE in subsample_rates:
        print("Subsample", SUBSAMPLE)
        splits = ['train', 'test']
        if SUBSAMPLE is not None:
            # splits += ['val']
            pass
        results = {}
        for split in splits:
            results[split] = []
        for SEED in seeds:
            np.random.seed(seed=SEED)
            if verbose:
                print("seed", SEED)
            x = load(
                subsample_train_frac=SUBSAMPLE,
                num_train=NUM_TRAIN,
                num_test=NUM_TEST,
                verbose=verbose,
                seed=SEED,
            )
            centroids = init_centroids_gmm(
                init_centroids=init_centroids,
                num_clusters=NUM_CLUSTERS,
                seed=SEED,
                dims=x['train'].shape[1],
                verbose=verbose,
            )
            kmeans = init_kmeans(
                implementation=implementation,
                n_clusters=NUM_CLUSTERS,
                batch_size=BATCH_SIZE,
                init_centroids=centroids,
                seed=SEED,
            )
            kmeans.fit(
                X=x['train'],
            )
            scores = evaluate(
                kmeans=kmeans,
                x=x,
                splits=splits,
                use_metric=use_metric,
                verbose=verbose,
            )
            for key, value in scores.items():
                results[key].append(value)
            if verbose:
                print(use_metric, scores)
        results_avg = {}
        for key, value in results.items():
            score_mean = np.around(np.mean(value), decimals=3)
            score_std = np.around(np.std(value), decimals=3)
            results_avg[key] = (score_mean, score_std)
            stats[key]['avg'].append(score_mean)
            stats[key]['std'].append(score_std)
        print(use_metric, results_avg)
    print(stats)

    plot_stats(
        stats,
        x_variable=subsample_rates,
        x_variable_name="Train Subsample Fraction",
        metric_name=use_metric,
    )


def load_federated_dummy_2D(random_state=42, verbose=False, clients_per_cluster=10, clusters=5):
    # # assert dims == 1, "only one dimension implemented"
    # np.random.seed(seed)
    # x = {}
    # ids = {}
    # # data, means = create_dummy_data(clients_per_cluster=2*clients_per_cluster, clusters=clusters, verbose=verbose)
    # mid = clients_per_cluster * clusters
    # x["train"], ids["train"] = data[:mid], means[:mid]
    # x["test"], ids["test"] = data[mid:], means[mid:]
    # # print(len(x['train']), x['train'][0].shape)

    M = 25
    K = clusters
    # 1. Generate data
    is_show = False
    # generate train sets
    clients_train_x, clients_train_y = gen_data(n_clients=M, n_clusters=K, n_samples_per_cluster=1000, random_state=random_state, is_show=is_show)
    # generate test sets
    clients_test_x, clients_test_y = gen_data(n_clients=M, n_clusters=K, n_samples_per_cluster=100, random_state=random_state, is_show = is_show)
    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}
    return x, labels


def load_federated(limit_csv=None, verbose=False, seed=None, dummy='dummy_2D', clusters=None):
    if dummy == 'dummy_2D':   #
        return load_federated_dummy_2D(random_state=seed, verbose=verbose, clusters=clusters)



def run_clustering_federated(verbose=False, dummy=False):
    # settings
    # num_seeds = [10]
    np.random.seed(1234)  # set the global seed for numpy
    num_seeds = [5]  # [30, 10, 10, 3, 3, 10], we run the experiments 5 times.
    # seeds = np.random.choice(range(100000), size=max(num_seeds), replace=False)   # used for repetitions: here is 30 repetitions
    seeds = [100, 200, 300, 400, 500]
    print(seeds)
    # seeds = range(NUM_SEEDS)
    NUM_CLUSTERS = 5
    LIMIT_CSV = None
    EPOCHS = 5
    ROUNDS = 100
    use_metric = "davies_bouldin"  # "euclidean",  "davies_bouldin" "silhouette"
    init_centroids = "random"  # "random "GMM"
    LR = 0.5
    LR_AD = None
    EPOCH_LR = 0.5
    MOMENTUM = None # 0.5
    # RECORD = range(1, ROUNDS+1)
    RECORD = None
    REASSIGN = (0.01, 10)

    stats = {
        'train': {'avg': [], 'std': []},
        'test': {'avg': [], 'std': []},
    }
    clients_per_round_fractions = [0.1, 0.3, 0.5, 0.70, 0.9, 1.00, "central"]
    # clients_per_round_fractions = [0.1]
    # clients_per_round_fractions = [0.1, "central"]
    # clients_per_round_fractions = ["central"]
    for grid_i, C in enumerate(clients_per_round_fractions):
        # num_s = num_seeds[grid_i] if len(num_seeds) > 1 else num_seeds[0] # why? not sure the reason
        num_s = num_seeds[0]
        print(f'grid_i: {grid_i}, C: {C}, nums_s: {num_s}')
        print("clients_per_round_fraction", C)
        splits = ['train', 'test']
        results = {}
        for split in splits:
            results[split] = []
        for SEED in seeds[:num_s]: # for repetitions to obtain average and std score.
            # np.random.seed(seed=SEED)
            if verbose:
                print("seed", SEED)
            # for each SEED, there will be a new 'x'
            x, _ = load_federated(
                limit_csv=LIMIT_CSV,
                verbose=verbose,
                seed=SEED,
                dummy=dummy,
                clusters=NUM_CLUSTERS,
            )
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
                    init_centroids=centroids,
                    seed=SEED,
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
                    init_centroids=centroids,
                    seed=SEED,
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
            for key, value in scores.items():
                results[key].append(value)
            if verbose:
                print(use_metric, scores)
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

    plot_stats(
        stats,
        x_variable=clients_per_round_fractions,
        x_variable_name="Fraction of Clients per Round",
        metric_name=use_metric,
    )


if __name__ == "__main__":
    # run_clustering(
    #     # verbose=True
    # )
    run_clustering_federated(
        dummy='dummy_2D',
        verbose=False,
    )
