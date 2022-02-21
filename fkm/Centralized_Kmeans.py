"""
    Stanford:
        1. For server, we use randomly select centroids from [0, 1] for each dimension.
        2. For clients, we use the server's centroids.
    PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 Centralized_Kmeans.py --dataset 'FEMNIST' \
                --data_details '1client_multiwriters_1digit' --algorithm 'Centralized_random'
"""
# Email: Kun.bj@outlook.com
import argparse
import traceback
from pprint import pprint

from fkm import _main
from fkm.experiment_cases import get_experiment_params

if __name__ == '__main__':
    print(__file__)
    parser = argparse.ArgumentParser(description='Description of your program')
    # parser.add_argument('-p', '--py_name', help='python file name', required=True)
    parser.add_argument('-S', '--dataset', help='dataset', default='2GAUSSIANS')
    # parser.add_argument('-T', '--data_details', help='data details', default='n1_5000-sigma1_3+n2_5000-sigma2_3+n3_5000-sigma3_3+n4_5000-sigma4_3:ratio_0.0:diff_sigma_n')
    parser.add_argument('-T', '--data_details', help='data details', default='n1_5000-sigma1_5+n2_5000-sigma2_5:ratio_0.0:diff_sigma_n')
    parser.add_argument('-M', '--algorithm', help='algorithm', default='Centralized_kmeans++')
    # args = vars(parser.parse_args())
    args = parser.parse_args()
    print(args)
    p3 = __file__.split('/')[-1].split('.')[0]
    params = get_experiment_params(p0=args.dataset, p1=args.data_details, p2=args.algorithm, p3 =p3)
    pprint(params)
    try:
        _main.run_clustering_federated(
            params,
            "",
            verbose=15,
        )
    except Exception as e:
        print(f'Error: {e}')
        traceback.print_exc()
