"""Use a dictionary to configure the project directly
	instead of methods (e.g., ini and yaml) that ultimately require you to parse the configuration file into a dictionary.
"""
import argparse
import configparser
from pprint import pprint

CONFIG = {
	# 1. random seed to reproduce the experiments
	'SEED': 42,

	# 2. project and output directory
	'ROOT_DIR': 'fkm',
	'OUT_DIR': 'fkm/results',

	# 3. verbose level
	'verbose': 10,

	# 'dataset'
	'dataset': {'name': '',
	            'others': ''},

	# algorithm
	'algorithm': {'name': '',
	              'others': ''},

}

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-C', '--config_file', help='A configuration file (yaml) that includes all parameters',
                    default='config.yaml')
# args = vars(parser.parse_args())  # choose as dict
args = parser.parse_args()
print(args)
#
# import yaml
#
# with open(args.config_file, "r") as stream:
# 	try:
# 		params = yaml.safe_load(stream)
# 	except yaml.YAMLError as exc:
# 		print(exc)
# pprint(params)
#
# with open('out_config.yaml', "w") as f:
# 	try:
# 		yaml.safe_dump(params, f, sort_keys=False)
# 	except yaml.YAMLError as exc:
# 		print(exc)
# # pprint(params)


import ruamel.yaml as ryaml
myyaml = ryaml.YAML()
with open(args.config_file, "r") as stream:
	try:
		params = myyaml.load(stream)
	except ryaml.YAMLError as exc:
		print(exc)
pprint(params, sort_dicts=False)

with open('out_config.yaml', "w") as f:
	try:
		myyaml.dump(params, f)
	except ryaml.YAMLError as exc:
		print(exc)
pprint(params, sort_dicts=False)


# Nested object is not easy to implement
# class MyParser(configparser.ConfigParser):
#
# 	def as_dict(self):
# 		d = dict(self._sections)
# 		for k in d:
# 			d[k] = dict(self._defaults, **d[k])
# 			d[k].pop('__name__', None)
# 		return d
#
#
# cfg = MyParser()
# cfg.read(args.algorithm_file)
# print(cfg)
# pprint(cfg.as_dict())
#
# #
# # cfg = ConfigParser()
# # cfg.read(args.algorithm_file)
# # print(cfg)
# # d = cfg.__dict__['_sections'].copy()
# # print(d)
