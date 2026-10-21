"""SONA, its ablations, the smote-variants baselines and TVAE behind one function."""
import logging
import random

import numpy as np
import pandas as pd
import smote_variants as sv

from .config import K_METHODS
from .sona_full import sona_resample

logging.getLogger("smote_variants").setLevel(logging.ERROR)   # smote-variants logs every call


def tvae(X, y, seed, epochs):
    """SDV TVAE trained on the minority rows; samples n_maj - n_min new rows."""
    import torch
    from sdv.metadata import Metadata
    from sdv.single_table import TVAESynthesizer
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    X_min = X[y == 1]
    n_syn = int((y != 1).sum() - len(X_min))
    cols = [f"f{i}" for i in range(X.shape[1])]
    metadata = Metadata.load_from_dict({"tables": {"minority": {"columns": {c: {"sdtype": "numerical"} for c in cols}}}})
    model = TVAESynthesizer(metadata, epochs=epochs, enforce_rounding=False, enable_gpu=torch.cuda.is_available())
    model.fit(pd.DataFrame(X_min, columns=cols))
    X_syn = model.sample(num_rows=n_syn)[cols].to_numpy(dtype=float)
    return np.vstack([X, X_syn]), np.concatenate([y, np.ones(n_syn, dtype=y.dtype)])


def oversample(method, X, y, k, seed, tvae_epochs=300):
    """Balance a standardised training set (minority label 1). k is used only by the K_METHODS baselines."""
    if method == "Origin":
        return X, y
    if method.startswith("SONA"):
        return sona_resample(X, y, 1, seed, use_nea=method != "SONA_noNEA", use_radius=method != "SONA_noDSS")
    if method == "TVAE":
        return tvae(X, y, seed, tvae_epochs)
    kw = {"n_neighbors": int(k)} if method in K_METHODS else {}
    Xr, yr = getattr(sv, method)(proportion=1.0, random_state=seed, **kw).sample(X, y)
    return np.asarray(Xr, dtype=float), np.asarray(yr).astype(y.dtype)
