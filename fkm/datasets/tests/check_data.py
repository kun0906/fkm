import pickle


def check():

	f1 = './datasets/3GAUSSIANS/n1_5000-sigma1_0.1_0.1+n2_5000-sigma2_0.1_0.1+n3_5000-sigma3_1.0_0.1:ratio_0.10:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_0.dat'
	f2 = './datasets/3GAUSSIANS/n1_5000-sigma1_0.1_0.1+n2_5000-sigma2_0.1_0.1+n3_5000-sigma3_1.0_0.1:ratio_0.10:diff_sigma_n|std|PCA_False|M_3|K_3|REMOVE_OUTLIERS_False/SEED_DATA_10.dat'

	with open(f1, 'rb') as f_:
		d1 = pickle.load(f_)

	with open(f2, 'rb') as f_:
		d2 = pickle.load(f_)

	print(d1, d2)


if __name__ =='__main__':
	check()