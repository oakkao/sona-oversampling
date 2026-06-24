import ast

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    CLASSIFIER_NAMES,
    OVERSAMPLING_METHODS,
    METHODS_SKIP_TUNING,
    N_REPEATS,
)
from .classifiers import get_classifier_instance
from .metrics import evaluate_once
from .oversampling import oversampling_only


def load_best_params(csv_path):
    df = pd.read_csv(csv_path)
    return {
        (row["dataset"], row["oversampling"], row["classifier"]): ast.literal_eval(
            row["best_params"]
        )
        for _, row in df.iterrows()
    }


def run_experiments(train_test_list, param_lookup, output_csv):
    all_methods = ["Origin"] + [m[0] for m in OVERSAMPLING_METHODS]
    evaluation_results = []

    for dataset_name, (X_train, X_test, y_train, y_test) in train_test_list:
        print(f"Dataset: {dataset_name}")

        for os_name in all_methods:
            print(f"  Oversampler: {os_name}")

            for clf_name in CLASSIFIER_NAMES:
                print(f"    Classifier: {clf_name}")

                lookup_os = "Origin" if os_name in METHODS_SKIP_TUNING else os_name
                all_params = param_lookup[(dataset_name, lookup_os, clf_name)]

                os_params = {}
                clf_params = {}
                for k, v in all_params.items():
                    if k.startswith("os_"):
                        os_params[k.replace("os_", "")] = v
                    else:
                        clf_params[k] = v

                metric_lists = {
                    m: []
                    for m in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
                }

                for _ in range(N_REPEATS):
                    if os_name == "Origin":
                        X_res, y_res = X_train, y_train
                    else:
                        os_config = (os_name, os_params)
                        X_res, y_res = oversampling_only(os_config, X_train, y_train)

                    clf = get_classifier_instance(clf_name, clf_params)
                    pipe = Pipeline(
                        [("scaler", StandardScaler()), ("classifier", clf)]
                    )
                    pipe.fit(X_res, y_res)

                    metrics = evaluate_once(pipe, X_test, y_test)
                    for k, v in metrics.items():
                        metric_lists[k].append(v)

                evaluation_results.append(
                    {
                        "dataset": dataset_name,
                        "classifier": clf_name,
                        "oversamping": os_name,
                        **metric_lists,
                    }
                )

    evaluation_df = pd.DataFrame(evaluation_results)
    evaluation_df.to_csv(output_csv)
    print(f"\nResults saved to {output_csv} ({len(evaluation_df)} rows)")
    return evaluation_df
