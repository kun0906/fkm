import collections
import json
import os

import matplotlib.pyplot as plt
import sklearn

from fkm.datasets.femnist import femnist


def femnist_distribution(in_dir='', random_state=42, out_dir='', is_show=True):
    writer_keys, files = femnist(in_dir)

    # sample 10% users and then split them into train and test sets
    _, sampled_keys = \
        sklearn.model_selection.train_test_split(writer_keys, test_size=100, shuffle=True,
                                                 random_state=random_state)
    print(len(writer_keys), len(sampled_keys))
    ys = {}
    for i, f in enumerate(files):
        with open(f) as json_file:
            res = json.load(json_file)

            for k in sampled_keys:
                if k not in res['user_data'].keys(): continue
                data = res['user_data'][k]
                # only keep 0-9 digitals
                ab = [(v, l) for v, l in zip(data['x'], data['y']) if l in list(range(10))]
                if len(ab) == 0:
                    print(f'*Writer: {k} does not include 0-9.')
                    a, b = [], []
                else:
                    a, b = zip(*ab)
                if k not in ys.keys():
                    ys[k] = b[:]
                else:
                    ys[k] += b

    # data per user
    nrows, ncols = 10, 10
    fig, axes = plt.subplots(nrows, ncols, sharex=False, sharey=False, figsize=(15, 15))
    for i, k in enumerate(ys.keys()):
        r, c = divmod(i, ncols)
        # print(i, r, c, k)
        ax = axes[r, c]
        ys_ = ys[k]
        if len(ys_) == 0:
            x, y = [], []
        else:
            x, y = zip(*sorted(collections.Counter(ys_).items(), key=lambda kv: kv[0], reverse=False))
        ax.bar(x, y)
        ax.set_title(f'Writer:{k}')
    title = 'data per user'
    # fig.set_facecolor("black")
    fig.suptitle(title, fontsize=20)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{title}.png")
    tmp_dir = os.path.dirname(fig_path)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    if is_show:
        plt.show()


if __name__ == '__main__':
    femnist_distribution(in_dir='femnist/all_data', out_dir='.')
