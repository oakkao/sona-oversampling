import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

from .config import (
    CLASSIFIER_NAMES,
    OVERSAMPLING_METHODS,
    METHODS_WITH_OS_N_NEIGHBORS,
    METHODS_SKIP_TUNING,
    RANDOM_STATE,
)
from .classifiers import get_classifier_params, get_classifier_instance
from .oversampling import oversampling_only

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective_original(trial, classifier_name, X_train, y_train):
    params = get_classifier_params(trial, classifier_name)
    classifier = get_classifier_instance(classifier_name, params)
    pipeline = Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        pipeline, X_train, y_train, cv=kf, scoring="f1_weighted", n_jobs=-1
    )
    return scores.mean()


def objective_joint(trial, X_train, y_train, method_name, classifier_name):
    os_params = {}
    if method_name in METHODS_WITH_OS_N_NEIGHBORS:
        os_params["n_neighbors"] = trial.suggest_int("os_n_neighbors", 1, 7)

    clf_params = get_classifier_params(trial, classifier_name)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_t, X_v = X_train[train_idx], X_train[val_idx]
        y_t, y_v = y_train[train_idx], y_train[val_idx]

        os_config = (method_name, os_params)
        X_res, y_res = oversampling_only(os_config, X_t, y_t)

        clf = get_classifier_instance(classifier_name, clf_params)
        pipe = Pipeline([("scaler", StandardScaler()), ("classifier", clf)])
        pipe.fit(X_res, y_res)

        preds = pipe.predict(X_v)
        scores.append(f1_score(y_v, preds, average="weighted"))
    return np.mean(scores)


def run_tuning(train_test_list, output_csv, n_trials=10):
    all_params_list = []

    for data_name, (X_train, X_test, y_train, y_test) in train_test_list:
        print(f"\nTuning dataset: {data_name}")

        for clf_name in CLASSIFIER_NAMES:
            print(f"  Origin | {clf_name}")
            study = optuna.create_study(direction="maximize")
            study.optimize(
                lambda trial, cn=clf_name, xt=X_train, yt=y_train: objective_original(
                    trial, cn, xt, yt
                ),
                n_trials=n_trials,
            )
            all_params_list.append(
                {
                    "dataset": data_name,
                    "oversampling": "Origin",
                    "classifier": clf_name,
                    "best_params": str(study.best_params),
                }
            )

        for method_config in OVERSAMPLING_METHODS:
            method_name = method_config[0]
            if method_name in METHODS_SKIP_TUNING:
                print(
                    f"  {method_name} | skipped (deep learning method, uses Origin params)"
                )
                continue
            for clf_name in CLASSIFIER_NAMES:
                print(f"  {method_name} | {clf_name}")
                study = optuna.create_study(direction="maximize")
                study.optimize(
                    lambda trial, mn=method_name, cn=clf_name, xt=X_train, yt=y_train: objective_joint(
                        trial, xt, yt, mn, cn
                    ),
                    n_trials=n_trials,
                )
                all_params_list.append(
                    {
                        "dataset": data_name,
                        "oversampling": method_name,
                        "classifier": clf_name,
                        "best_params": str(study.best_params),
                    }
                )

    best_param_df = pd.DataFrame(all_params_list)
    best_param_df.to_csv(output_csv, index=False)
    print(f"\nSaved {output_csv} ({len(best_param_df)} rows)")
    return best_param_df
