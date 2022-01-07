"""
FEMNIST:
    n_writers: 3597
    n_classes: 62 (10 digits + 26+26 (letters: lowercase and uppercase))
    n_images: 805, 263
    n_images per writer: 226.83 (mean)	88.94 (std)
https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_plusplus.html#sphx-glr-auto-examples-cluster-plot-kmeans-plusplus-py

"""
import json
import os
from collections import Counter

import numpy as np
import sklearn.model_selection


def femnist(in_dir='datasets/femnist/all_data'):
    keys = []
    files = []
    for i, f in enumerate(os.listdir(in_dir)):
        # Opening JSON file
        f = os.path.join(in_dir, f)
        if i % 5 == 0:
            print(i, f)
        with open(f) as json_file:
            data = json.load(json_file)
            # res.update(data['user_dat']) # cannot load all data into memory
            keys.extend(data['users'])

        files.append(f)
        # if i >= 1: break
    print(f'Number of unique keys: ', len(np.unique(keys)), len(keys))
    return keys, files


def load_femnist_user_percent(params={}, random_state=42):
    keys, files = femnist(in_dir='datasets/femnist/all_data')

    # sample 10% users and then split them into train and test sets
    _, sampled_keys = \
        sklearn.model_selection.train_test_split(keys, test_size=params['user_percent'], shuffle=True,
                                                 random_state=random_state)

    # split train and test sets
    train_keys, test_keys = \
        sklearn.model_selection.train_test_split(sampled_keys, test_size=params['user_test_size'], shuffle=True,
                                                 random_state=random_state)
    is_crop_image = params['is_crop_image']
    image_shape = params['image_shape']
    def get_xy(train_keys, test_keys, files):
        clients_train_x = []
        clients_train_y = []

        clients_test_x = []
        clients_test_y = []
        for i, f in enumerate(files):
            with open(f) as json_file:
                res = json.load(json_file)

                for k in train_keys:
                    if k not in res['user_data'].keys(): continue
                    data = res['user_data'][k]
                    # only keep 0-9 digitals
                    # ab = [(v, l) for v, l in zip(data['x'], data['y']) if l in list(range(10))]
                    if is_crop_image:
                        ab = [(crop_image(v, image_shape), l) for v, l in zip(data['x'], data['y']) if l in list(range(10))]
                    else:
                        ab = [(v, l) for v, l in zip(data['x'], data['y']) if l in list(range(10))]
                    if len(ab) == 0: continue
                    a, b = zip(*ab)
                    clients_train_x.append(np.asarray(a))
                    clients_train_y.append(np.asarray(b))

                for k in test_keys:
                    if k not in res['user_data'].keys(): continue
                    data = res['user_data'][k]
                    # only keep 0-9 digitals
                    # ab = [(v, l) for v, l in zip(data['x'], data['y']) if l in list(range(10))]
                    if is_crop_image:
                        ab = [(crop_image(v, image_shape), l) for v, l in zip(data['x'], data['y']) if l in list(range(10))]
                    else:
                        ab = [(v, l) for v, l in zip(data['x'], data['y']) if l in list(range(10))]
                    if len(ab) == 0: continue
                    a, b = zip(*ab)
                    clients_test_x.append(np.asarray(a))
                    clients_test_y.append(np.asarray(b))

        return clients_train_x, clients_train_y, clients_test_x, clients_test_y

    clients_train_x, clients_train_y, clients_test_x, clients_test_y = get_xy(train_keys, test_keys, files)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    print(f'n_train_clients: {len(clients_train_x)}, n_datapoints: {sum(len(vs) for vs in clients_train_y)}')
    print(f'n_test_clients: {len(clients_test_x)}, n_datapoints: {sum(len(vs) for vs in clients_test_y)}')
    return x, labels


def crop_image(x, shape=(14, 14)):
    x = np.asarray(x).reshape((28, 28))
    h, w = shape
    d1 = (28 - h) // 2
    d2 = (28 - w) // 2
    return list(x[d1:d1 + h, d2:d2 + w].flatten())


def femnist_1client_1writer_multidigits(params={}, random_state=42):
    keys, files = femnist(in_dir='datasets/femnist/all_data')

    # sample 10% users and then split them into train and test sets
    _, sampled_keys = \
        sklearn.model_selection.train_test_split(keys, test_size=params['writer_ratio'], shuffle=True,
                                                 random_state=random_state)

    # split train and test sets
    train_keys, test_keys = sampled_keys, sampled_keys
    is_crop_image = params['is_crop_image']
    image_shape = params['image_shape']
    def get_xy(train_keys, test_keys, files):
        clients_train_x = []
        clients_train_y = []

        clients_test_x = []
        clients_test_y = []
        for i, f in enumerate(files):
            with open(f) as json_file:
                res = json.load(json_file)

                for k in train_keys:
                    if k not in res['user_data'].keys(): continue
                    data = res['user_data'][k]
                    # only keep 0-9 digitals
                    if is_crop_image:
                        ab = [(crop_image(v, image_shape), l) for v, l in zip(data['x'], data['y']) if l in list(range(10))]
                    else:
                        ab = [(v, l) for v, l in zip(data['x'], data['y']) if l in list(range(10))]
                    if len(ab) == 0: continue
                    a, b = zip(*ab)
                    train_x, test_x, train_y, test_y = \
                        sklearn.model_selection.train_test_split(a, b, test_size=params['data_ratio_per_writer'],
                                                                 shuffle=True,
                                                                 random_state=random_state)

                    clients_train_x.append(np.asarray(train_x))
                    clients_train_y.append(np.asarray(train_y))
                    clients_test_x.append(np.asarray(test_x))
                    clients_test_y.append(np.asarray(test_y))

        return clients_train_x, clients_train_y, clients_test_x, clients_test_y

    clients_train_x, clients_train_y, clients_test_x, clients_test_y = get_xy(train_keys, test_keys, files)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    # print(f'n_train_clients: {len(clients_train_x)}, n_datapoints: {sum(len(vs) for vs in clients_train_y)}')
    # print(f'n_test_clients: {len(clients_test_x)}, n_datapoints: {sum(len(vs) for vs in clients_test_y)}')
    return x, labels


def group_users(x, n):
    if len(x) < n:
        print(f'Error: {len(x)} < {n}')
        return
    clients = [[]] * n
    q, r = divmod(len(x), n)
    for i in range(n):
        if i == n - 1:
            clients[i] = x[i * q:]
        else:
            clients[i] = x[i * q: (i + 1) * q]
    return clients


def femnist_1client_multiwriters_multidigits(params={}, random_state=42):
    keys, files = femnist(in_dir='datasets/femnist/all_data')

    # sample 10% users and then split them into train and test sets
    _, sampled_keys = \
        sklearn.model_selection.train_test_split(keys, test_size=params['writer_ratio'], shuffle=True,
                                                 random_state=random_state)
    print(f'Number of sampled users: {len(sampled_keys)}')
    n_clients = params['n_clients'] if type(params['n_clients']) == int else 1

    clients = group_users(sampled_keys, n_clients)
    is_crop_image = params['is_crop_image']
    image_shape = params['image_shape']
    def get_xy(clients, files):
        clients_train_x = [[] for _ in range(n_clients)]
        clients_train_y = [[] for _ in range(n_clients)]

        clients_test_x = [[] for _ in range(n_clients)]
        clients_test_y = [[] for _ in range(n_clients)]
        for i, f in enumerate(files):
            with open(f) as json_file:
                res = json.load(json_file)

                for j, ks in enumerate(clients):  # each client has 'ks' users,
                    for k in ks:
                        if k not in res['user_data'].keys(): continue
                        data = res['user_data'][k]
                        # only keep 0-9 digitals
                        # ab = [(v, l) for v, l in zip(data['x'], data['y']) if l in list(range(10))]
                        if is_crop_image:
                            ab = [(crop_image(v, image_shape), l) for v, l in zip(data['x'], data['y']) if l in list(range(10))]
                        else:
                            ab = [(v, l) for v, l in zip(data['x'], data['y']) if l in list(range(10))]

                        if len(ab) == 0: continue
                        a, b = zip(*ab)
                        train_x, test_x, train_y, test_y = \
                            sklearn.model_selection.train_test_split(a, b, test_size=params['data_ratio_per_writer'],
                                                                     shuffle=True,
                                                                     random_state=random_state)

                        clients_train_x[j].extend(np.asarray(train_x))
                        clients_train_y[j].extend(np.asarray(train_y))
                        clients_test_x[j].extend(np.asarray(test_x))
                        clients_test_y[j].extend(np.asarray(test_y))

        for i in range(n_clients):
            clients_train_x[i] = np.asarray(clients_train_x[i])
            clients_train_y[i] = np.asarray(clients_train_y[i])
            clients_test_x[i] = np.asarray(clients_test_x[i])
            clients_test_y[i] = np.asarray(clients_test_y[i])
        return clients_train_x, clients_train_y, clients_test_x, clients_test_y

    clients_train_x, clients_train_y, clients_test_x, clients_test_y = get_xy(clients, files)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    # y_tmp = []
    # for vs in clients_train_y:
    #     y_tmp.extend(vs)
    # print(f'n_train_clients: {len(clients_train_x)}, n_datapoints: {sum(len(vs) for vs in clients_train_y)}, '
    #       f'cluster_size: {sorted(Counter(y_tmp).items(), key=lambda kv: kv[0], reverse=False)}')
    # y_tmp = []
    # for vs in clients_test_y:
    #     y_tmp.extend(vs)
    # print(f'n_test_clients: {len(clients_test_x)}, n_datapoints: {sum(len(vs) for vs in clients_test_y)},'
    #       f'cluster_size: {sorted(Counter(y_tmp).items(), key=lambda kv: kv[0], reverse=False)}')
    return x, labels


def femnist_1client_multiwriters_1digit(params={}, random_state=42):
    keys, files = femnist(in_dir='datasets/femnist/all_data')

    # sample 10% users and then split them into train and test sets
    _, sampled_keys = \
        sklearn.model_selection.train_test_split(keys, test_size=params['writer_ratio'], shuffle=True,
                                                 random_state=random_state)
    print(f'Number of sampled users: {len(sampled_keys)}')
    n_clients = params['n_clients']

    clients = group_users(sampled_keys, n_clients)
    is_crop_image = params['is_crop_image']
    image_shape = params['image_shape']
    def get_xy(clients, files):
        clients_train_x = [[] for _ in range(n_clients)]
        clients_train_y = [[] for _ in range(n_clients)]

        clients_test_x = [[] for _ in range(n_clients)]
        clients_test_y = [[] for _ in range(n_clients)]
        for i, f in enumerate(files):
            with open(f) as json_file:
                res = json.load(json_file)

                for j, ks in enumerate(clients):  # each client has 'ks' users,
                    for k in ks:
                        if k not in res['user_data'].keys(): continue
                        data = res['user_data'][k]
                        # only keep 0-9 digitals
                        # ab = [(v, l) for v, l in zip(data['x'], data['y']) if l in list(range(10))]
                        if is_crop_image:
                            ab = [(crop_image(v, image_shape), l) for v, l in zip(data['x'], data['y']) if l in list(range(10))]
                        else:
                            ab = [(v, l) for v, l in zip(data['x'], data['y']) if l in list(range(10))]
                        if len(ab) == 0: continue
                        for t in range(10):
                            ab_ = [(v, l) for v, l in ab if l == t]
                            if len(ab_) < 2: continue
                            a, b = zip(*ab_)
                            train_x, test_x, train_y, test_y = \
                                sklearn.model_selection.train_test_split(a, b,
                                                                         test_size=params['data_ratio_per_digit'],
                                                                         shuffle=True,
                                                                         random_state=random_state)

                            clients_train_x[t].extend(np.asarray(train_x))
                            clients_train_y[t].extend(np.asarray(train_y))
                            clients_test_x[t].extend(np.asarray(test_x))
                            clients_test_y[t].extend(np.asarray(test_y))

        for i in range(n_clients):
            clients_train_x[i] = np.asarray(clients_train_x[i])
            clients_train_y[i] = np.asarray(clients_train_y[i])
            clients_test_x[i] = np.asarray(clients_test_x[i])
            clients_test_y[i] = np.asarray(clients_test_y[i])
        return clients_train_x, clients_train_y, clients_test_x, clients_test_y

    clients_train_x, clients_train_y, clients_test_x, clients_test_y = get_xy(clients, files)

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    # y_tmp = []
    # for vs in clients_train_y:
    #     y_tmp.extend(vs)
    # print(f'n_train_clients: {len(clients_train_x)}, n_datapoints: {sum(len(vs) for vs in clients_train_y)}, '
    #       f'cluster_size: {sorted(Counter(y_tmp).items(), key=lambda kv: kv[0], reverse=False)}')
    # y_tmp = []
    # for vs in clients_test_y:
    #     y_tmp.extend(vs)
    # print(f'n_test_clients: {len(clients_test_x)}, n_datapoints: {sum(len(vs) for vs in clients_test_y)}, '
    #       f'cluster_size: {sorted(Counter(y_tmp).items(), key=lambda kv: kv[0], reverse=False)}')
    return x, labels
