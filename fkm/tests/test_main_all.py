from fkm.main_all import get_datasets_config_lst


def test_get_datasets_config_lst():
	res = get_datasets_config_lst()
	print(res)
	assert type(res) == list
	assert type(res) == dict