import collections

from sklearn import metrics

N = 1000

X = [(i, i+2) for i in range(N)]
y = [0 if i%2 == 0 else 1 for i in range(N)]
y_pred = [0 if i < N-5 else 1 for i in range(N)]
print(collections.Counter(y), collections.Counter(y_pred))

ri = metrics.rand_score(y, y_pred)
print(ri)

ari = metrics.adjusted_rand_score(y, y_pred)
print(ari)

db = metrics.davies_bouldin_score(X, y_pred)
print(db)

sil = metrics.silhouette_score(X, y_pred)
print(sil)

