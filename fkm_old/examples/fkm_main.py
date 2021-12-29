""" Federated K-Means


"""
# Author: kun.bj@outlook.com
from fkm.clustering.federated_km import FKM
from fkm.clustering.kmeans import KMeans
from fkm.datasets.gen_data import gen_data
from fkm.utils import dump, timer

K = 5  # n_clusters
M = 25  # n_clients


def _main(random_state=42):
    history = {}
    # 1. Generate data
    # generate train sets
    clients_train = gen_data(n_clients=M, n_clusters=K, n_samples_per_cluster=1000, random_state=random_state)
    # generate test sets
    clients_test = gen_data(n_clients=M, n_clusters=K, n_samples_per_cluster=100, random_state=random_state)

    # # 2.1 Build, fit, and test a Federated K-Means
    # for frac_of_clients in [0.3, 0.5, 0.7, 0.9, 1.0]:
    # # for frac_of_clients in [0.9]:
    #     print('\n=======================================================================')
    #     print(f'The percent of clients who participate the training: {frac_of_clients}')
    #     fkm_old = FKM(n_clusters=K, n_rounds=100, frac_of_clients=frac_of_clients, random_state=random_state)
    #     fkm_old.fit(clients_train)
    #     fkm_old.test(clients_test)
    #     history[frac_of_clients] = fkm_old.scores

    # 2.2 Build, fit, and test a K-Means
    print('\n=======================================================================')
    print(f'KMeans')
    km = KMeans(n_clusters=K, random_state=random_state)
    km.fit(clients_train)
    km.test(clients_test)
    history['kmeans'] = km.scores

    return history

@timer
def main():
    random_states = [100, 200, 300, 400, 500]
    history = {}
    for random_state in random_states:
        print(f'*** Random State: {random_state}')
        history[random_state] = _main(random_state=random_state)

    print('\n=======================================================================')
    print(f'All results: ')
    out_file = 'results.dat'
    dump(history, out_file)
    print('\n'.join(f'{k:7.2f}: {vs}' if type(k) == int else f'{k:7}: {vs}' for k, vs in history.items()))


if __name__ == '__main__':
    main()
