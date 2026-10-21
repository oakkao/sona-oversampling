import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


def _reverse_nn(fit_X, query_X):
    nn = NearestNeighbors(algorithm="ball_tree").fit(fit_X)
    k = min(8, len(fit_X))
    while True:
        d, i = nn.kneighbors(query_X, n_neighbors=k)
        tied = d <= d[:, :1] + 1e-12
        if k == len(fit_X) or not tied[:, -1].any():
            return d[:, 0], np.bincount(i[tied], minlength=len(fit_X))
        k = min(2 * k, len(fit_X))


def _terminals(X_min, anchors, rng):
    with np.errstate(divide="ignore"):
        W = 1.0 / cdist(X_min, X_min)
    W[~np.isfinite(W)] = 0.0
    W[W.sum(axis=1) == 0] = 1.0          # every other minority point is a duplicate
    np.fill_diagonal(W, 0.0)
    P = W / W.sum(axis=1, keepdims=True)
    terminals = np.empty(len(anchors), dtype=int)
    for a in np.unique(anchors):
        idx = np.flatnonzero(anchors == a)
        terminals[idx] = rng.choice(len(X_min), size=len(idx), p=P[a])
    return terminals


def SONA(X, y, min_label, new_label=0, random_state=None):
    rng = np.random.default_rng(random_state)
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    X_min, X_maj = X[y == min_label], X[y != min_label]
    n_min, n_syn = len(X_min), len(X_maj) - len(X_min)
    if n_syn <= 0 or n_min < 2:
        return X, y

    # 1. NEA score of minority points and PEA score of majority points
    _, nea = _reverse_nn(X_min, X_maj)
    dist, pea = _reverse_nn(X_maj, X_min)

    # 2. Radius to the nearest safe majority point (PEA = 0)
    safe = pea == 0
    if safe.any():
        dist = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(X_maj[safe]).kneighbors(X_min)[0][:, 0]
    radius = dist / 2

    # 3. Anchors proportional, 1 / (1 + NEA)
    p = 1.0 / (1.0 + nea)
    anchors = rng.choice(n_min, size=n_syn, p=p / p.sum())
    terminals = _terminals(X_min, anchors, rng)

    # 4. Step toward the terminal, never longer than the anchor's radius
    v = X_min[terminals] - X_min[anchors]
    norm = np.linalg.norm(v, axis=1)
    u = np.divide(v, norm[:, None], out=np.zeros_like(v), where=norm[:, None] > 0)
    step = rng.random(n_syn) * np.minimum(radius[anchors], norm)
    X_syn = X_min[anchors] + step[:, None] * u

    y_syn = np.full(n_syn, min_label + new_label, dtype=y.dtype)
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])
