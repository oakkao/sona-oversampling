"""Benchmark data and nested cross-validation. Each result row is appended to results/<dataset>.jsonl at once,
so an interrupted run continues where it stopped."""
import json
import time

import numpy as np
import optuna
import pandas as pd
from imblearn.datasets import fetch_datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .classifiers import SEARCH_SPACES, make_classifier, decision_scores
from .config import CLASSIFIERS, K_METHODS
from .metrics import classification_metrics, best_f1_threshold, n3, placement
from .oversampling import oversample

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_benchmark():
    """The binary datasets of imbalanced-learn (minority = 1) and a table of their sizes."""
    data, meta = {}, []
    for name, b in fetch_datasets().items():
        X, y = np.asarray(b.data, dtype=float), (np.asarray(b.target) == 1).astype(int)
        data[name] = X, y
        meta.append({"dataset": name, "n": len(y), "d": X.shape[1], "n_min": int(y.sum()),
                     "IR": round((len(y) - y.sum()) / y.sum(), 2)})
    return data, pd.DataFrame(meta).sort_values("n").reset_index(drop=True)


def select_datasets(meta, cfg):
    if cfg.datasets:
        return [d for d in cfg.datasets if d in set(meta.dataset)]
    return meta.dataset[(meta.n <= cfg.max_samples) & (meta.d <= cfg.max_features)].tolist()


def read_jsonl(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass   # line cut short by an interrupted run
    return rows


def evaluate(cfg, ds, fold, method, clf, resampled, Xtr, ytr, Xte, yte, inner):
    """Tune the classifier (and k) on the inner folds, refit on the oversampled outer training fold, test."""
    t0 = time.time()
    seed = cfg.seed(ds, fold, clf)
    best = {"score": -np.inf}

    def objective(trial):
        params = SEARCH_SPACES[clf](trial)
        k = trial.suggest_categorical("os_k", cfg.k_grid) if method in K_METHODS else None
        oof, f1s = np.zeros(len(ytr)), []
        for i, (_, va) in enumerate(inner):
            Xr, yr = resampled(i, k)[:2]
            model = make_classifier(clf, params, seed).fit(Xr, yr)
            f1s.append(f1_score(ytr[va], model.predict(Xtr[va]), zero_division=0))
            oof[va] = decision_scores(model, Xtr[va])
        if np.mean(f1s) > best["score"]:
            best.update(score=np.mean(f1s), oof=oof)
        return np.mean(f1s)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=cfg.seed(ds, fold, method, clf, "tpe")))
    study.optimize(objective, n_trials=cfg.n_trials, catch=(Exception,))
    params = dict(study.best_params)
    k = params.pop("os_k", None)
    Xr, yr, os_status, os_time, place = resampled("outer", k)
    model = make_classifier(clf, params, seed).fit(Xr, yr)
    scores = decision_scores(model, Xte)
    thr = best_f1_threshold(ytr, best["oof"])
    return {"os_k": k, "best_params": params, "inner_cv_f1": float(study.best_value), "os_status": os_status,
            "os_time_s": os_time, "threshold": thr,
            **classification_metrics(yte, model.predict(Xte), scores),
            **classification_metrics(yte, (scores >= thr).astype(int), prefix="thr_"),
            **place,
            "elapsed_s": time.time() - t0}


def run_dataset(cfg, ds, X, y):
    """Run every (fold, method, classifier) row of one dataset that is not yet in its results file."""
    path = cfg.results_dir / f"{ds}.jsonl"
    done = {(r["fold"], r["method"], r["classifier"]) for r in (read_jsonl(path) if path.exists() else [])
            if r["row_status"] == "ok"}
    outer = StratifiedKFold(cfg.n_outer_folds, shuffle=True, random_state=cfg.seed(ds, "outer"))
    for fold, (tr, te) in enumerate(outer.split(X, y)):
        todo = [(m, c) for m in cfg.methods for c in CLASSIFIERS if (fold, m, c) not in done]
        if not todo:
            continue
        scaler = StandardScaler().fit(X[tr])
        Xtr, Xte, ytr, yte = scaler.transform(X[tr]), scaler.transform(X[te]), y[tr], y[te]
        inner = list(StratifiedKFold(cfg.n_inner_folds, shuffle=True,
                                     random_state=cfg.seed(ds, fold, "inner")).split(Xtr, ytr))
        n3_before = n3(Xtr, ytr)
        for method in dict.fromkeys(m for m, _ in todo):
            cache = {}

            def resampled(split, k):
                """Oversampled inner training fold `split` (or the whole outer training fold), computed once."""
                if (split, k) not in cache:
                    idx = slice(None) if split == "outer" else inner[split][0]
                    t0 = time.time()
                    try:
                        Xr, yr = oversample(method, Xtr[idx], ytr[idx], k, cfg.seed(ds, fold, split, method, k),
                                            cfg.tvae_epochs)
                        status = "ok"
                    except Exception as e:   # the classifier is then trained on the original fold
                        Xr, yr, status = Xtr[idx], ytr[idx], f"error: {type(e).__name__}: {str(e)[:150]}"
                    t = time.time() - t0
                    place = placement(Xtr, ytr, Xr, yr) if split == "outer" and method != "Origin" else {}
                    cache[split, k] = Xr, yr, status, t, place
                return cache[split, k]

            for clf in [c for m, c in todo if m == method]:
                row = {"dataset": ds, "fold": fold, "method": method, "classifier": clf, "N3_before": n3_before}
                try:
                    row.update(row_status="ok", **evaluate(cfg, ds, fold, method, clf, resampled,
                                                           Xtr, ytr, Xte, yte, inner))
                except Exception as e:
                    row.update(row_status="error", error=f"{type(e).__name__}: {str(e)[:300]}")
                with open(path, "a") as f:
                    f.write(json.dumps(row, default=float) + "\n")
                print(f"  {ds:16s} fold {fold}  {method:22s} {clf:4s}  F1={row.get('f1', np.nan):.3f}  "
                      f"[{row['row_status']}]", flush=True)
