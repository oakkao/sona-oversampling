"""SONA with the two ablation switches and its internal values; same random draws as sona_oversampling.SONA."""
import numpy as np
from sklearn.neighbors import NearestNeighbors

from sona_oversampling.main import _reverse_nn, _terminals


def nearest(fit_X, query_X):
    """Distance and index of the nearest fit_X point for every query point."""
    d, i = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(fit_X).kneighbors(query_X)
    return d[:, 0], i[:, 0]


def sona_resample(X, y, min_label=1, random_state=None, use_nea=True, use_radius=True, return_info=False):
    """use_nea=False: uniform anchors (SONA w/o NEA). use_radius=False: no radius cap (SONA w/o DSS)."""
    rng = np.random.default_rng(random_state)
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    X_min, X_maj = X[y == min_label], X[y != min_label]
    n_min, n_syn = len(X_min), len(X_maj) - len(X_min)
    if n_syn <= 0 or n_min < 2:
        return (X, y, None) if return_info else (X, y)

    _, nea = _reverse_nn(X_min, X_maj)
    dist, pea = _reverse_nn(X_maj, X_min)
    safe = pea == 0
    if safe.any():
        dist = nearest(X_maj[safe], X_min)[0]
    R = dist / 2

    p = 1.0 / (1.0 + nea) if use_nea else np.ones(n_min)
    anchors = rng.choice(n_min, size=n_syn, p=p / p.sum())
    terminals = _terminals(X_min, anchors, rng)

    v = X_min[terminals] - X_min[anchors]
    norm = np.linalg.norm(v, axis=1)
    u = np.divide(v, norm[:, None], out=np.zeros_like(v), where=norm[:, None] > 0)
    step = rng.random(n_syn) * (np.minimum(R[anchors], norm) if use_radius else norm)
    X_syn = X_min[anchors] + step[:, None] * u

    X_out, y_out = np.vstack([X, X_syn]), np.concatenate([y, np.full(n_syn, min_label, dtype=y.dtype)])
    if not return_info:
        return X_out, y_out
    return X_out, y_out, dict(X_min=X_min, X_maj=X_maj, X_syn=X_syn, anchors=anchors, R=R, nea=nea, pea=pea, safe=safe)


def check_guarantee(info):
    """violations: synthetic samples at least as close to a safe majority point as to their anchor (theory: 0).
    closer_to_unsafe: share of synthetic samples closer to an unsafe majority point than to their anchor."""
    X_syn, X_maj, safe = info["X_syn"], info["X_maj"], info["safe"]
    d_anchor = np.linalg.norm(X_syn - info["X_min"][info["anchors"]], axis=1)
    dist = lambda mask: nearest(X_maj[mask], X_syn)[0] if mask.any() else np.full(len(X_syn), np.inf)
    return {"violations": int(np.sum(dist(safe) <= d_anchor)), "closer_to_unsafe": float(np.mean(dist(~safe) < d_anchor))}
