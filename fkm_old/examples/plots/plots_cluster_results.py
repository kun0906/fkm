from sklearn.datasets import make_blobs

from fkm.datasets.gen_data import gen_data
import matplotlib.pyplot as plt
import numpy as np

from fkm.utils import load


def main():
    data = load('../results.dat')
    dbs = []
    sils = []
    fs = [0.3, 0.5, 0.7, 0.9, 1.0, 'kmeans']
    for f in fs:
        _dbs = []
        _sils = []
        for k, vs in data.items():
            db, sil = vs[f]['db'], vs[f]['sil']
            _dbs.append(db)
            _sils.append(sil)
        dbs.append((np.mean(_dbs), np.std(_dbs)))
        sils.append((np.mean(_sils), np.std(_sils)))
    print(f'dbs: {dbs}')
    print(f'sils: {sils}')

    # plot
    fig, ax = plt.subplots()
    dbs_means, dbs_std = zip(*dbs)
    xs = range(0, len(fs))
    capsize = 4
    ax.errorbar(xs, dbs_means, yerr=dbs_std, capsize = capsize, label='Davies-Bouldin Score')
    # sils_means, sils_std = zip(*sils)
    # ax.errorbar(xs, sils_means, yerr=sils_std, capsize=capsize, label='Silhouette Score')

    # ax.errorbar(xs, dbs_means, yerr=dbs_std, uplims=True, lolims=True, label='Davias-Bouldin Score')
    # sils_means, sils_std = zip(*sils)
    # ax.errorbar(xs, sils_means, yerr=sils_std,uplims=True, lolims=True, label='Silhouette Score')

    ax.set_xticklabels(['0']+[str(v) for v in fs] + [' '])
    plt.setp(ax.get_xticklabels(), fontsize=7)
    # plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=7)
    plt.legend(loc='upper right')

    # plt.xlim([0, 15])
    # plt.ylim([0, 15])
    # plt.xticks([])
    # plt.yticks([])
    plt.xlabel('Fraction of clients')
    plt.ylabel('Score')
    plt.savefig('synthetic_results.pdf', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()

