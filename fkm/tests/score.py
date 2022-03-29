


from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix

true = [0, 0, 1, 1]
pred = [0, 0, 0, 1]
# true = [0, 0, 1, 1, 0,0]
# pred = [0, 0, 0, 3, 1,1]
# print(contingency_matrix(true, pred, sparse=False))
# print(metrics.pair_confusion_matrix(true, pred))
print(metrics.rand_score(true, pred))
print(metrics.adjusted_rand_score(true, pred))
print(metrics.adjusted_mutual_info_score(true,pred))
print(metrics.v_measure_score(true, pred))
print(metrics.fowlkes_mallows_score(true, pred))
# print(metrics.davies_bouldin_score(true,pred))


#
# true = [0, 0, 1, 1]
# pred = [0, 0, 0, 1]
# res = metrics.adjusted_rand_score(true, pred)
# print(res)