"""
    Centralized K-means


    PYTHONPATH='..' PYTHONUNBUFFERED=TRUE python3 centralized_kmeans.py --config_file 'config.yaml'
"""
# Email: Kun.bj@outlook.com
import traceback
from pprint import pprint

from fkm import _main, config

if __name__ == '__main__':
	print(__file__)
	params = config.parser()
	pprint(params, sort_dicts=False)

	try:
		_main.run_clustering_federated(
			params,
			"",
			verbose=15,
		)
	except Exception as e:
		print(f'Error: {e}')
		traceback.print_exc()
