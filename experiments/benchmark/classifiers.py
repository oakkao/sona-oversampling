"""Classifiers and their Optuna search spaces."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEARCH_SPACES = {
    "LR": lambda t: dict(C=t.suggest_float("C", 1e-4, 1e2, log=True),
                         solver=t.suggest_categorical("solver", ["liblinear", "lbfgs"])),
    "DT": lambda t: dict(max_depth=t.suggest_int("max_depth", 2, 20),
                         min_samples_split=t.suggest_int("min_samples_split", 2, 20),
                         min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 10),
                         criterion=t.suggest_categorical("criterion", ["gini", "entropy"])),
    "KNN": lambda t: dict(n_neighbors=t.suggest_int("n_neighbors", 1, 25),
                          weights=t.suggest_categorical("weights", ["uniform", "distance"]),
                          metric=t.suggest_categorical("metric", ["euclidean", "manhattan"])),
    "MLP": lambda t: dict(hidden_layer_sizes=t.suggest_categorical(
                              "hidden_layer_sizes", ["64", "128", "256", "64-32", "128-64", "256-128"]),
                          activation=t.suggest_categorical("activation", ["relu", "tanh"]),
                          alpha=t.suggest_float("alpha", 1e-5, 1e-1, log=True),
                          learning_rate_init=t.suggest_float("learning_rate_init", 1e-4, 1e-1, log=True)),
    "SVM": lambda t: dict(C=t.suggest_float("C", 1e-2, 1e2, log=True),
                          kernel=t.suggest_categorical("kernel", ["rbf", "linear", "poly"]),
                          gamma=t.suggest_categorical("gamma", ["scale", "auto"])),
    "RF": lambda t: dict(n_estimators=t.suggest_int("n_estimators", 50, 300),
                         max_depth=t.suggest_int("max_depth", 3, 30),
                         min_samples_split=t.suggest_int("min_samples_split", 2, 20),
                         max_features=t.suggest_categorical("max_features", ["sqrt", "log2"])),
    "XGB": lambda t: dict(n_estimators=t.suggest_int("n_estimators", 50, 300),
                          max_depth=t.suggest_int("max_depth", 2, 10),
                          learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
                          subsample=t.suggest_float("subsample", 0.6, 1.0),
                          colsample_bytree=t.suggest_float("colsample_bytree", 0.5, 1.0),
                          min_child_weight=t.suggest_int("min_child_weight", 1, 10)),
}


def make_classifier(clf, params, seed):
    p = dict(params)
    if clf == "LR":
        return LogisticRegression(**p, max_iter=5000, random_state=seed)
    if clf == "DT":
        return DecisionTreeClassifier(**p, random_state=seed)
    if clf == "KNN":
        return KNeighborsClassifier(**p, n_jobs=-1)
    if clf == "MLP":
        p["hidden_layer_sizes"] = tuple(int(h) for h in p.get("hidden_layer_sizes", "100").split("-"))
        return MLPClassifier(**p, max_iter=500, early_stopping=True, n_iter_no_change=15, random_state=seed)
    if clf == "SVM":
        return SVC(**p, cache_size=1000, max_iter=200_000, random_state=seed)
    if clf == "RF":
        return RandomForestClassifier(**p, n_jobs=-1, random_state=seed)
    if clf == "XGB":
        return XGBClassifier(**p, tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=seed, verbosity=0)
    raise ValueError(clf)


def decision_scores(model, X):
    """Probability of the minority class (label 1), or the SVM decision function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, list(model.classes_).index(1)]
    return model.decision_function(X)
