import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def compute_roc_auc(pipeline, X_test, y_test):
    clf = pipeline.named_steps["classifier"]
    if not hasattr(clf, "predict_proba"):
        return np.nan

    y_proba = pipeline.predict_proba(X_test)
    unique_classes = np.unique(y_test)

    if len(unique_classes) <= 1:
        return np.nan

    try:
        if len(unique_classes) == 2:
            pos_label = unique_classes[-1]
            pos_idx = np.where(clf.classes_ == pos_label)[0][0]
            return roc_auc_score(y_test, y_proba[:, pos_idx], average="weighted")
        else:
            if y_proba.ndim == 2 and y_proba.shape[1] == len(unique_classes):
                return roc_auc_score(
                    y_test,
                    y_proba,
                    multi_class="ovr",
                    average="weighted",
                    labels=unique_classes,
                )
            return np.nan
    except ValueError:
        return np.nan


def evaluate_once(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "roc_auc": compute_roc_auc(pipeline, X_test, y_test),
    }
