
#
# import matplotlib
# import matplotlib.pyplot as plt
# # matplotlib.use('module://backend_interagg')
# matplotlib.pyplot.set_loglevel('notset')
# plt.plot([1, 2, 3, 4])
# plt.show()
import json
from pprint import pprint


history_file = ''
results_avg = {"train": {"Iterations": [13.0, 0.0], "davies_bouldin": [0.6993036870437687, 0.0], "silhouette": [0.6515535702926448, 0.0], "euclidean": [80.62202164723637, 0.0], "n_clusters": [2.0, 0.0], "n_clusters_pred": [2.0, 0.0], "ari": [0.6347256136652784, 0.0], "ami": [0.6121150344208492, 0.0], "ch": [4261.534362916788, 0.0], "labels_true": [{"0": 4998, "1": 4998}], "labels_pred": [{"0": 6008, "1": 3988}]}, "test": {"Iterations": ["", ""], "davies_bouldin": [0.4399754812982135, 0.0], "silhouette": [0.7460154185166414, 0.0], "euclidean": [8.82935533953977, 0.0], "n_clusters": [2.0, 0.0], "n_clusters_pred": [2.0, 0.0], "ari": [1.0, 0.0], "ami": [1.0, 0.0], "ch": [14.581119853225282, 0.0], "labels_true": [{"0": 2, "1": 2}], "labels_pred": [{"1": 2, "0": 2}]}}
with open(history_file + '-results_avg.json', 'w') as file:
	file.write(json.dumps(results_avg, indent=None))  # use `json.loads` to do the reverse

with open(history_file + '-results_avg1.json', 'w') as file:
	pprint(results_avg, file)

with open(history_file + '-results_avg1.json', 'r') as file:
	data = json.load(file)

print(data)


