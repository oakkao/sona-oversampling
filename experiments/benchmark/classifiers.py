from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

CLASSIFIER_REGISTRY = {
    "Logistic regression": LogisticRegression,
    "XGBoost": XGBClassifier,
    "KNeighborsClassifier": KNeighborsClassifier,
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "MLPClassifier": MLPClassifier,
    "SVC": SVC,
    "RandomForestClassifier": RandomForestClassifier,
}

CLASSIFIER_SEARCH_SPACES = {
    "Logistic regression": {
        "C": ("float_log", 1e-4, 1e2),
        "solver": ("categorical", ["liblinear", "lbfgs"]),
    },
    "XGBoost": {
        "n_estimators": ("int", 50, 200),
        "max_depth": ("int", 3, 10),
        "learning_rate": ("float_log", 0.01, 0.3),
        "subsample": ("float", 0.6, 1.0),
    },
    "KNeighborsClassifier": {
        "n_neighbors": ("int", 3, 20),
        "weights": ("categorical", ["uniform", "distance"]),
        "metric": ("categorical", ["euclidean", "manhattan", "minkowski"]),
    },
    "DecisionTreeClassifier": {
        "max_depth": ("int", 3, 20),
        "min_samples_split": ("int", 2, 20),
        "criterion": ("categorical", ["gini", "entropy"]),
    },
    "MLPClassifier": {
        "hidden_layer_sizes": ("categorical", [(50,), (100,), (50, 50)]),
        "activation": ("categorical", ["tanh", "relu"]),
        "alpha": ("float_log", 1e-4, 1e-1),
        "learning_rate_init": ("float_log", 1e-3, 1e-1),
    },
    "SVC": {
        "C": ("float_log", 0.1, 100),
        "kernel": ("categorical", ["rbf", "linear", "poly"]),
        "gamma": ("categorical", ["scale", "auto"]),
    },
    "RandomForestClassifier": {
        "n_estimators": ("int", 50, 200),
        "max_depth": ("int", 3, 20),
        "min_samples_split": ("int", 2, 20),
    },
}

CLASSIFIER_FIXED_PARAMS = {
    "XGBoost": {"verbosity": 0},
    "MLPClassifier": {"max_iter": 500},
    "SVC": {"probability": True},
}


def suggest_param(trial, name, spec):
    kind = spec[0]
    if kind == "float":
        return trial.suggest_float(name, spec[1], spec[2])
    if kind == "float_log":
        return trial.suggest_float(name, spec[1], spec[2], log=True)
    if kind == "int":
        return trial.suggest_int(name, spec[1], spec[2])
    if kind == "categorical":
        return trial.suggest_categorical(name, spec[1])
    raise ValueError(f"Unknown param kind: {kind}")


def get_classifier_params(trial, classifier_name):
    space = CLASSIFIER_SEARCH_SPACES.get(classifier_name)
    if space is None:
        raise ValueError(f"No search space defined for '{classifier_name}'")
    return {k: suggest_param(trial, k, v) for k, v in space.items()}


def get_classifier_instance(classifier_name, params):
    cls = CLASSIFIER_REGISTRY.get(classifier_name)
    if cls is None:
        raise ValueError(f"Unknown classifier: {classifier_name}")
    merged = {**params, **CLASSIFIER_FIXED_PARAMS.get(classifier_name, {})}
    return cls(**merged)
