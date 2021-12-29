import collections

import sklearn


def metric_scores(X, y_pred, y_true=None):
    scores = {}
    print(sorted(collections.Counter(y_pred).items(), key=lambda v: v[0]))
    if not y_true:
        print(sorted(collections.Counter(y_true).items(), key=lambda v: v[0]))
        report = sklearn.metrics.classification_report(y_true, y_pred)
        print(report)
        scores['report'] = report
    # the smaller, the better
    # db = (Si+Sj) / Mij, where Mij = norm(Mi, Mj) is the separation between Ci and Cj
    db = sklearn.metrics.davies_bouldin_score(X, y_pred)
    print(sklearn.metrics.davies_bouldin_score(X, y_true))
    scores['db'] = db
    # sil = (bi - ai)/ max(bi, ai), the larger, the better
    sil = sklearn.metrics.silhouette_score(X, y_pred)
    print(sklearn.metrics.silhouette_score(X, y_true))
    scores['sil'] = sil

    return scores


