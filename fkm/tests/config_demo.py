""" Parse the config file

	# Use a dictionary to configure the project directly
	# instead of methods (e.g., ini and yaml) that ultimately require you to parse the configuration file into a dictionary.

"""
# Email: kun.bj@outlook.com

import argparse
import os.path
import traceback
from pprint import pprint

import ruamel.yaml as ryaml


def parser(config_file):
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('-C', '--config_file', help='A configuration file (yaml) that includes all parameters',
	                    default=('config.yaml' if not config_file else config_file))
	args_ = parser.parse_args()
	print(args_)
	# myyaml = ryaml.YAML()
	# with open(args_.config_file, "r") as stream:
	# 	try:
	# 		args = myyaml.load(stream)
	# 		args['config_file'] = args_.config_file
	# 	except ryaml.YAMLError as exc:
	# 		traceback.print_exc()
	# pprint(args, sort_dicts=False)

	return args_


parser('')