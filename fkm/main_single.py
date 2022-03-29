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
	args = main(config_file='config.yaml')
	pprint(args, sort_dicts=False)
