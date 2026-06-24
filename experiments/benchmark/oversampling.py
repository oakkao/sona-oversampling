import warnings

import numpy as np
import pandas as pd
import smote_variants as sv
from sdv.single_table import TVAESynthesizer
from sdv.metadata import Metadata
from sona_oversampling import SONA


def tvae_oversample(X, y, min_label=1):
    X_df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    y_series = pd.Series(y, name="target")
    data = pd.concat([X_df, y_series], axis=1)

    maj_size = np.sum(y != min_label)
    min_size = np.sum(y == min_label)
    num_to_generate = maj_size - min_size

    if num_to_generate <= 0:
        return X, y

    minority_data = data[data["target"] == min_label].drop(columns=["target"])
    metadata = Metadata.detect_from_dataframe(data=minority_data, table_name="minority")
    tvae = TVAESynthesizer(metadata)
    tvae.fit(minority_data)
    synthetic_samples = tvae.sample(num_to_generate)

    X_resampled = np.vstack([X, synthetic_samples.values])
    y_resampled = np.hstack([y, np.repeat(min_label, num_to_generate)])
    return X_resampled, y_resampled


CUSTOM_OVERSAMPLER_REGISTRY = {
    "SONA": SONA,
    "tvae_oversample": tvae_oversample,
}


def oversampling_only(config, X_train, y_train):
    if config is None:
        return X_train, y_train

    method_name, params = config

    if method_name in CUSTOM_OVERSAMPLER_REGISTRY:
        return CUSTOM_OVERSAMPLER_REGISTRY[method_name](X_train, y_train, 1)

    if hasattr(sv, method_name):
        sampler = getattr(sv, method_name)(**params)
        return sampler.sample(X_train, y_train)

    warnings.warn(f"Oversampling method '{method_name}' not found - returning original data.")
    return X_train, y_train
