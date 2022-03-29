"""
    Centralized K-means

    PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 centralized_kmeans.py --dataset 'FEMNIST' \
                --data_details '1client_multiwriters_1digit' --algorithm 'Centralized_random'
"""
# Email: Kun.bj@outlook.com
import argparse
import traceback
from pprint import pprint
import numpy as np
from fkm.cluster.centralized_kmeans import KMeans
from fkm._main import run_clustering_federated
from fkm.experiment_cases import get_experiment_params

# These options determine the way floating point numbers, arrays and
# other NumPy objects are displayed.
# np.set_printoptions(precision=3, suppress=True)
np.set_printoptions(precision=3, suppress=True, formatter={'float': '{:20.3f}'.format}, edgeitems = 120, linewidth=100000)

if __name__ == '__main__':
    print(__file__)
    parser = argparse.ArgumentParser(description='Description of your program')
    # parser.add_argument('-C', '--config_file', help='A configuration file (yaml) that includes all parameters',
    #                     default='config.yaml')
    # parser.add_argument('-p', '--py_name', help='python file name', required=True)
    parser.add_argument('-S', '--dataset', help='dataset', default='NBAIOT')
    # parser.add_argument('-T', '--data_details', help='data details', default='n1_5000-sigma1_3+n2_5000-sigma2_3+n3_5000-sigma3_3+n4_5000-sigma4_3:ratio_0.0:diff_sigma_n')
    # parser.add_argument('-T', '--data_details', help='data details', default='n1_5000-sigma1_0.3_0.3+n2_5000-sigma2_0.3_0.3+n3_10000-sigma3_0.3_0.3+n4_0.1-sigma4_0.3_0.3:ratio_0.0:diff_sigma_n')
    parser.add_argument('-T', '--data_details', help='data details',
                        default='nbaiot_user_percent_client11')
    parser.add_argument('-M', '--algorithm', help='algorithm', default='Centralized_kmeans++')
    parser.add_argument('-K', '--n_clusters', help='number of clusters', default= 2)    # 9 or 11
    parser.add_argument('-C', '--n_clients', help='number of clients', default= 11)
    # args = vars(parser.parse_args())
    args = parser.parse_args()
    pprint(args)
    p3 = __file__.split('/')[-1]
    params = get_experiment_params(p0=args.dataset, p1=args.data_details, p2=args.algorithm, p3 =p3,
                                   n_clusters = int(args.n_clusters), n_clients=int(args.n_clients))
    pprint(params)
    try:
        run_clustering_federated(
            params,
            KMeans,
            verbose=5 if args.dataset == 'FEMNIST' else 10,
        )
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()
