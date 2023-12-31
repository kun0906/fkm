""" Run this main file for a single experiment

	Run instruction:
	$pwd
	$fkm/fkm
	$PYTHONPATH='..' python3 main_single.py
"""
# Email: kun.bj@outllok.com

from pprint import pprint

from fkm import config, _main
from fkm.vis import visualize


def main(config_file='config.yaml'):
	"""

	Parameters
	----------
	config_file

	Returns
	-------

	"""
	# Step 0: config the experiment
	args = config.parser(config_file)
	if args['VERBOSE'] >= 2:
		print(f'~~~ The template config {config_file}, which will be modified during the later experiment ~~~')
		pprint(args, sort_dicts=False)

	# Step 1: run cluster and get result
	history_file = _main.run_model(args)
	args['history_file'] = history_file

	# Step 2: visualize the result
	visual_file = visualize.visualize_data(args)
	args['visual_file'] = visual_file

	# # Step 3: dump the config
	# config.dump(args['config_file'][:-4] + 'out.yaml', args)

	return args


if __name__ == '__main__':

	# 3Gaussians
	p=0.10
	config_file1 = f'out/3GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_1.0_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0/centralized_kmeans/R_50|random|None|0.0001|std/config_file.yaml'
	config_file2 = f'out/3GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_1.0_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0/centralized_kmeans/R_50|kmeans++|None|0.0001|std/config_file.yaml'
	config_file3 = f'out/3GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_1.0_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_server_init_first/R_50|min_max|None|0.0001|std/config_file.yaml'
	config_file4 = f'out/3GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_1.0_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_client_init_first/R_50|average|random|0.0001|std/config_file.yaml'
	config_file5 = f'out/3GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_1.0_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_client_init_first/R_50|average|kmeans++|0.0001|std/config_file.yaml'
	config_file6 = f'out/3GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_1.0_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_greedy_kmeans/R_50|greedy|random|0.0001|std/config_file.yaml'
	config_file7 = f'out/3GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_1.0_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_greedy_kmeans/R_50|greedy|kmeans++|0.0001|std/config_file.yaml'
	# config_file = 'out/MNIST/n1_50-sigma1_0.1_0.1+n2_50-sigma2_0.1_0.1+n3_50-sigma3_1.0_0.1:ratio_0.00:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0/centralized_kmeans/R_50|random|None|0.0001|std/config_file.yaml'

	# 10Gaussians
	p = 0.10
	config_file1 = f'out/10GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_0.1_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_10|K_10|REMOVE_OUTLIERS_False/SEED_DATA_0/centralized_kmeans/R_50|random|None|0.0001|std/config_file.yaml'
	config_file2 = f'out/10GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_0.1_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_10|K_10|REMOVE_OUTLIERS_False/SEED_DATA_0/centralized_kmeans/R_50|kmeans++|None|0.0001|std/config_file.yaml'
	config_file3 = f'out/10GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_0.1_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_10|K_10|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_server_init_first/R_50|min_max|None|0.0001|std/config_file.yaml'
	config_file4 = f'out/10GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_0.1_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_10|K_10|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_client_init_first/R_50|average|random|0.0001|std/config_file.yaml'
	config_file5 = f'out/10GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_0.1_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_10|K_10|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_client_init_first/R_50|average|kmeans++|0.0001|std/config_file.yaml'
	config_file6 = f'out/10GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_0.1_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_10|K_10|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_greedy_kmeans/R_50|greedy|random|0.0001|std/config_file.yaml'
	config_file7 = f'out/10GAUSSIANS/n1_1000-sigma1_0.1_0.1+n2_1000-sigma2_0.1_0.1+n3_1000-sigma3_0.1_0.1:ratio_{p:.2f}:diff_sigma_n|std|PCA_False|M_10|K_10|REMOVE_OUTLIERS_False/SEED_DATA_0/federated_greedy_kmeans/R_50|greedy|kmeans++|0.0001|std/config_file.yaml'

	configs = [config_file1, config_file2, config_file3, config_file4, config_file5, config_file6, config_file7]
	for config_file in configs:
		args = main(config_file=config_file)
		pprint(args, sort_dicts=False)
