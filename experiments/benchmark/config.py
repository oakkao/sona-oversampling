DATASET_NAMES = [
    "pen_digits",
    "thyroid_sick",
    "libras_move",
    "car_eval_4",
    "wine_quality",
    "mammography",
    "optical_digits",
]

CLASSIFIER_NAMES = [
    "Logistic regression",
    "XGBoost",
    "KNeighborsClassifier",
    "DecisionTreeClassifier",
    "MLPClassifier",
    "SVC",
    "RandomForestClassifier",
]

OVERSAMPLING_METHODS = [
    ("SONA", {}),
    ("tvae_oversample", {}),
    ("SMOTE", {}),
    ("Borderline_SMOTE2", {}),
    ("Safe_Level_SMOTE", {}),
    ("polynom_fit_SMOTE_poly", {}),
    ("SMOTE_IPF", {}),
]

METHODS_WITH_OS_N_NEIGHBORS = {"SMOTE", "Borderline_SMOTE2", "Safe_Level_SMOTE", "SMOTE_IPF"}
METHODS_SKIP_TUNING = {"tvae_oversample"}

RANDOM_STATE = 42
N_REPEATS = 3

BEST_PARAM_CSV = "best_param.csv"
RESULTS_CSV = "results.csv"
