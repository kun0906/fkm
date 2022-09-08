"""
FEMNIST:
    n_writers: 3597
    n_classes: 62 (10 digits + 26+26 (letters: lowercase and uppercase))
    n_images: 805, 263
    n_images per writer: 226.83 (mean)	88.94 (std)
https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_plusplus.html#sphx-glr-auto-examples-cluster-plot-kmeans-plusplus-py

    https://www.kaggle.com/datasets/crawford/emnist?select=emnist-byclass-train.csv


"""
import collections
import json
import os
import pickle
from collections import Counter

import numpy as np
import sklearn.model_selection
from sklearn.model_selection import train_test_split


def femnist_multiusers_per_client(args, random_state=42):
    # cp /scratch/gpfs/ky8517/leaf-torch/data/femnist /scratch/gpfs/ky8517/fkm/datasets/femnist
    in_dir='datasets/FEMNIST'
    # data_file = args['data_file']
    # dataset_detail = args['DATASET']['detail']
    # data_file = os.path.join(in_dir,  f'{dataset_detail}.dat')
    # if os.path.exists(data_file):
    #     return utils_func.load(data_file)

    def get_xy(in_dir = 'train'):
        X_clients = []
        y_clients = []
        n_users = 0
        n_data_points = 0
        seen = set()
        for i, json_file in enumerate(os.listdir(in_dir)):
            if ".DS_Store" in json_file: continue
            try:
                with open(os.path.join(in_dir, json_file), 'rb') as f:
                    vs = json.load(f)

                print(i, json_file, len(vs['users']), vs['num_samples'])
                n_users += len(vs['users'])
                for j, user in enumerate(vs['users']):
                    if user in seen: continue
                    seen.add(user)
                    tmp = vs['user_data'][user]
                    if len(tmp['y']) < 200:
                        x_ = tmp['x']
                        y_ = tmp['y']
                    else:
                        x_, y_ = sklearn.utils.resample(tmp['x'], tmp['y'], replace=False, n_samples=200, random_state=random_state)
                    dim = len(x_[0])
                    n_data_points += len(x_)
                    X_clients.append(np.asarray(x_))    # each client only has one user data
                    y_clients.append(np.asarray(y_))
            except Exception as e:
                print(f'open error: {json_file}')
                continue
        print(f'n_users: {n_users}, n_data_points: {n_data_points}, dim: {dim}')
        return X_clients, y_clients

    clients_train_x, clients_train_y = get_xy(os.path.join(in_dir, 'train'))
    clients_test_x, clients_test_y = get_xy(os.path.join(in_dir, 'test'))

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
    # utils_func.dump((x, labels), data_file)
    return x, labels


def femnist_diff_sigma_n(args, random_state=42):
    n_clients = args['N_CLIENTS']
    dataset_detail = args['DATASET']['detail']  # 'nbaiot_user_percent_client:ratio_0.1'
    p1 = dataset_detail.split(':')
    ratio = float(p1[1].split('_')[1])

    p1_0 = p1[0].split('+')  # 'n1_100-sigma1_0.1, n2_1000-sigma2_0.2, n3_10000-sigma3_0.3
    p1_0_c1 = p1_0[0].split('-')
    n1 = int(p1_0_c1[0].split('_')[1])
    # tmp = p1_0_c1[1].split('_')
    # sigma1_0, sigma1_1 = float(tmp[1]), float(tmp[2])

    # cp /scratch/gpfs/ky8517/leaf-torch/data/femnist /scratch/gpfs/ky8517/fkm/datasets/femnist
    in_dir='datasets/FEMNIST'
    # data_file = args['data_file']
    # dataset_detail = args['DATASET']['detail']
    # data_file = os.path.join(in_dir,  f'{dataset_detail}.dat')
    # if os.path.exists(data_file):
    #     return utils_func.load(data_file)

    def get_xy(in_dir = 'train'):
        X_clients = []
        y_clients = []
        Y = []
        n_users = 0
        n_data_points = 0
        seen = set()
        users = []
        for i, json_file in enumerate(os.listdir(in_dir)):
            if ".DS_Store" in json_file: continue
            try:
                with open(os.path.join(in_dir, json_file), 'rb') as f:
                    vs = json.load(f)
            except Exception as e:
                print(f'open error: {json_file}')
                continue
            users.extend(vs['user_data'].values())

        x = users
        dim = 0
        n_users = len(users)
        for i in range(n_clients):
            x, x_ = train_test_split(x, test_size=n1, shuffle=True,
                                     random_state=random_state)  # train set = 1-ratio
            tmp_x = []
            tmp_y = []
            for usr in x_:
                x1 = usr['x']
                y1 = usr['y']
                dim = len(x1[0])
                tmp_x.extend(x1)
                tmp_y.extend(y1)
            n_data_points += len(tmp_y)
            X_clients.append(np.asarray(tmp_x))  # each client has one user's data
            y_clients.append(np.asarray(tmp_y))
            Y.extend(list(tmp_y))
        Y = collections.Counter(Y)
        print(f'Y({len(Y.items())}): {Y.items()}')
        print(f'n_clients: {n_clients}, {n1} users per client, n_data_points: {n_data_points}, dim: {dim}')
        return X_clients, y_clients

    clients_train_x, clients_train_y = get_xy(os.path.join(in_dir, 'train'))
    clients_test_x, clients_test_y = get_xy(os.path.join(in_dir, 'test'))

    x = {'train': clients_train_x,
         'test': clients_test_x}
    labels = {'train': clients_train_y,
              'test': clients_test_y}

    return x, labels

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
            if k not in res['user_data'].keys(): continue
            with open(f) as json_file:
                res = json.load(json_file)

                for k in train_keys:
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


def preprocess_femnist(in_dir = 'datasets/FEMNIST/raw_data/by_write'):

    """
    https://www.kaggle.com/datasets/crawford/emnist?select=emnist-byclass-train.csv


    Reference:
        NIST Special Database 19 Handprinted Forms and Characters Database
        https://s3.amazonaws.com/nist-srd/SD19/1stEditionUserGuide.pdf

        Under "by_write/hsf_0/f0000_14/"
            digit, upper, lower, const and the prefixes “d”, “u”, “l” and “c”.

        f0000_14:
            The universal descripton of a writer on this database is of the form fyyyy_zz. For example the directory f3100_45
uniquely identifies writer 3100 and the second index, 45, identifies which of the 100 possible form templates was
used in the form image

    Creates .pkl files for:
    1. list of directories of every image in 'by_class'
    2. list of directories of every image in 'by_write'
    the hierarchal structure of the data is as follows:
    - by_class -> classes -> folders containing images -> images
    - by_write -> folders containing writers -> writer -> types of images -> images
    the directories written into the files are of the form 'raw_data/...'


    Parameters
    ----------
    in_dir

    Returns
    -------

    """

    import PIL

    writers = collections.defaultdict(list)
    cnt = 0
    for sub_dir in os.listdir(in_dir):
        d1 = os.path.join(in_dir, sub_dir)
        if not os.path.isdir(d1): continue
        for writer_name in os.listdir(d1):
            d2 = os.path.join(d1, writer_name)
            if not os.path.isdir(d2): continue
            for image_dir in os.listdir(d2):
                d3 = os.path.join(d2, image_dir)
                if not os.path.isdir(d3): continue
                for image in os.listdir(d3):
                    f = os.path.join(d3, image)
                    if '.png' not in f : continue
                    img = PIL.Image.open(f)
                    gray = img.convert('L')
                    size = (28, 28)
                    gray.thumbnail(size, PIL.Image.ANTIALIAS)
                    arr = np.asarray(gray).copy()
                    X = arr.flatten()
                    # vec = vec / 255  # scale all pixel values to between 0 and 1
                    y = ''
                    writers[writer_name].append((X, y))
                    cnt += 1
                    if cnt % 1000 == 0:
                        print(f'cnt: {cnt}')
                    break

    out_file = os.path.join(in_dir, 'writers.dat')
    with open(out_file, 'wb') as f:
        pickle.dump(writers, f)

    return writers



    # write_dir = os.path.join(parent_path, 'data', 'raw_data', 'by_write')
    # rel_write_dir = os.path.join('data', 'raw_data', 'by_write')
    # write_parts = os.listdir(write_dir)
    #
    # for write_part in write_parts:
    #     writers_dir = os.path.join(write_dir, write_part)
    #     rel_writers_dir = os.path.join(rel_write_dir, write_part)
    #     if os.path.isfile(writers_dir): continue
    #     writers = os.listdir(writers_dir)
    #
    #     for writer in writers:
    #         writer_dir = os.path.join(writers_dir, writer)
    #         rel_writer_dir = os.path.join(rel_writers_dir, writer)
    #         wtypes = os.listdir(writer_dir)
    #
    #         for wtype in wtypes:
    #             type_dir = os.path.join(writer_dir, wtype)
    #             rel_type_dir = os.path.join(rel_writer_dir, wtype)
    #             images = os.listdir(type_dir)
    #             image_dirs = [os.path.join(rel_type_dir, i) for i in images]
    #
    #             for image_dir in image_dirs:
    #                 write_files.append((writer, image_dir))
    #

    # writer_count = 0
    # for hsf in os.listdir(in_dir):
    #     hsf = os.path.join(in_dir, hsf)
    #     for
    #     for w in []:
    #         if w not in writers.keys():
    #             writers[w] = []
    #         else:
    #             writers.append(w)
    #         writer_count +=1
    #
    # print(f'total writes: {writer_count}')

        # size = 28, 28  # original image size is 128, 128
        # for (f, c) in l:
        #     file_path = os.path.join(parent_path, f)
        #     img = Image.open(file_path)
        #     gray = img.convert('L')
        #     gray.thumbnail(size, Image.ANTIALIAS)
        #     arr = np.asarray(gray).copy()
        #     vec = arr.flatten()
        #     vec = vec / 255  # scale all pixel values to between 0 and 1
        #     vec = vec.tolist()


            # with open(file_path, 'w') as outfile:
            #     json.dump(all_data, outfile)



if __name__ == '__main__':
    preprocess_femnist(in_dir = 'datasets/FEMNIST/raw_data')