"""Classification metrics (minority class = 1) and placement measures of synthetic samples."""
from collections import Counter

import numpy as np
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.neighbors import NearestNeighbors

from .sona_full import nearest


def classification_metrics(y_true, y_pred, scores=None, prefix=""):
    m = {"f1": f1_score(y_true, y_pred, zero_division=0),
         "precision": precision_score(y_true, y_pred, zero_division=0),
         "recall": recall_score(y_true, y_pred, zero_division=0),
         "gmean": geometric_mean_score(y_true, y_pred),
         "bal_acc": balanced_accuracy_score(y_true, y_pred),
         "mcc": matthews_corrcoef(y_true, y_pred)}
    if scores is not None:
        m.update(pr_auc=average_precision_score(y_true, scores), roc_auc=roc_auc_score(y_true, scores))
    return {prefix + k: float(v) for k, v in m.items()}


def best_f1_threshold(y, scores):
    """Decision threshold that maximises the F1-score."""
    precision, recall, thr = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thr[np.nanargmax(f1[:-1])])


def n3(X, y):
    """Class-overlap measure N3 (Ho & Basu, 2002): leave-one-out error of the 1-NN classifier."""
    idx = NearestNeighbors(n_neighbors=2).fit(X).kneighbors(X, return_distance=False)
    nb = np.where(idx[:, 0] == np.arange(len(X)), idx[:, 1], idx[:, 0])   # nearest point other than itself
    return float(np.mean(y[nb] != y))


def synthetic_mask(X, y, X_res, y_res):
    """True for rows of the resampled set that are not original (row, label) pairs."""
    left = Counter(zip(map(bytes, np.ascontiguousarray(X, dtype=float)), y.tolist()))
    mask = np.ones(len(X_res), dtype=bool)
    for i, key in enumerate(zip(map(bytes, np.ascontiguousarray(X_res, dtype=float)), y_res.tolist())):
        if left[key] > 0:
            left[key] -= 1
            mask[i] = False
    return mask


def placement(X, y, X_res, y_res, k=5):
    """Where the synthetic samples lie relative to the original training points (lower = safer)."""
    S = X_res[synthetic_mask(X, y, X_res, y_res)]
    if len(S) == 0:
        return {}
    labels = y[NearestNeighbors(n_neighbors=k).fit(X).kneighbors(S, return_distance=False)]
    d_min, d_maj = nearest(X[y == 1], S)[0], nearest(X[y != 1], S)[0]
    return {"syn_nn_is_majority": float(np.mean(labels[:, 0] != 1)),
            "syn_knn_majority_frac": float(np.mean(labels != 1)),
            "syn_rel_pos_median": float(np.median(d_min / np.maximum(d_min + d_maj, 1e-12))),
            "syn_on_majority_point": float(np.mean(d_maj <= 1e-12)),
            "N3_after": n3(X_res, y_res)}
