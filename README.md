# SONA

**Stochastic directional Oversampling using Negative Anomalous scores for imbalanced dataset**

SONA is a parameter-free oversampling method for imbalanced binary classification. Unlike SMOTE and its variants, SONA requires no k-neighbors parameter to tune. It uses ball-tree spatial indexing to identify border regions between classes and constrains synthetic sample generation within a safety radius, preventing new points from being placed in majority-class territory.

Published on PyPI as [`sona-oversampling`](https://pypi.org/project/sona-oversampling/).

## Installation

```bash
pip install sona-oversampling            # normal install
pip install --upgrade sona-oversampling  # or update if needed
```

## Requirements

**Core** (installed with the package):

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 2.0.2 | Array operations |
| scipy | >= 1.8.1 | Distance computations (cdist) |
| scikit-learn | >= 1.3.1 | NearestNeighbors (ball-tree indexing) |

**Benchmark experiments** (additional):

| Package | Purpose |
|---------|---------|
| imbalanced-learn | Benchmark dataset loading |
| smote-variants | Baseline SMOTE oversampling methods |
| optuna | Hyperparameter tuning |
| xgboost | XGBClassifier |
| sdv / ctgan | TVAE deep learning oversampling baseline |
| matplotlib | Visualization |

Python >= 3.9 required.

## Dataset Information

The benchmark experiments evaluate SONA against competing methods on 7 imbalanced datasets from the UCI Machine Learning Repository, loaded via `imblearn.datasets.fetch_datasets()`:

| Dataset | Samples | Features | Minority % | Reference |
|---------|---------|----------|-----------|-----------|
| pen_digits | 10,992 | 16 | 9.6% | Alpaydin & Alimoglu (1998) |
| thyroid_sick | 3,163 | 25 | 9.3% | Quinlan (1986) |
| libras_move | 360 | 90 | 6.7% | Dias et al. (2009) |
| car_eval_4 | 1,728 | 6 | 3.8% | Bohanec (1997) |
| wine_quality | 4,898 | 11 | 3.7% | Cortez et al. (2009) |
| mammography | 11,183 | 6 | 2.3% | Woods et al. (1993) |
| optical_digits | 5,620 | 62 | 9.9% | Alpaydin & Kaynak (1998) |

## Code Information

```
sona-oversampling/
  src/sona_oversampling/
    main.py                 # SONA algorithm implementation
  experiments/
    benchmark/              # Modular benchmark pipeline
      config.py             #   Constants: datasets, classifiers, methods
      data.py               #   Dataset loading from imblearn
      classifiers.py        #   Classifier registry and Optuna search spaces
      oversampling.py       #   TVAE + SONA + smote-variants dispatcher
      metrics.py            #   Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC)
      tuning.py             #   Optuna hyperparameter tuning loop
      experiment.py         #   Experiment loop with repeated evaluation
      run.py                #   Main entrypoint
    SONA_experiments.ipynb  # Notebook version of the benchmark
    best_param.csv          # Pre-tuned hyperparameters (343 entries)
  examples/                 # Visualization output images
```

## Usage Instructions

### Basic Usage

```python
from sona_oversampling import SONA

# X: feature matrix (n_samples, n_features)
# y: binary labels (0 = majority, 1 = minority)
X_resampled, y_resampled = SONA(X, y, min_label=1)
```

**API:**

`SONA(X, y, min_label, new_label=0)`

| Parameter | Type | Description |
|-----------|------|-------------|
| X | numpy.ndarray | Input feature matrix of shape (n_samples, n_features) |
| y | numpy.ndarray | Target labels |
| min_label | int | Label of the minority class |
| new_label | int | Offset added to min_label for synthetic samples (default 0) |

**Returns:** `(X_augmented, y_augmented)` — original data concatenated with synthetic minority samples.

### Visualization Example

```python
from sona_oversampling import SONA
from sklearn.datasets import make_circles, make_moons, make_blobs
import matplotlib.pyplot as plt

X_circles, y_circles = make_circles(n_samples=(500, 100), noise=0.05, random_state=42)
X_moons, y_moons = make_moons(n_samples=(500, 100), noise=0.1, random_state=42)
X_blobs, y_blobs = make_blobs(n_samples=[500, 50], cluster_std=0.5, centers=[[0, 0], [1, 1]], random_state=42)

synthetic_datasets = [
    ("Double circle", (X_circles, y_circles)),
    ("Blue-moons", (X_moons, y_moons)),
    ("Gaussian cluster", (X_blobs, y_blobs))
]

for name, (X_original, y_original) in synthetic_datasets:
    X_oversampled, y_oversampled = SONA(X_original, y_original, min_label=1, new_label=1)
    mask_maj = (y_oversampled == 0)
    mask_min_orig = (y_oversampled == 1)
    mask_min_syn = (y_oversampled == 2)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_oversampled[mask_maj, 0], X_oversampled[mask_maj, 1],
                c='grey', label='Majority Class (0)', alpha=0.5, s=20)
    plt.scatter(X_oversampled[mask_min_orig, 0], X_oversampled[mask_min_orig, 1],
                c='blue', label='Original Minority (1)', s=30, edgecolors='k')
    plt.scatter(X_oversampled[mask_min_syn, 0], X_oversampled[mask_min_syn, 1],
                c='red', label='Synthetic Samples (2)', marker='x', s=40, alpha=0.8)
    plt.title(f"SONA Oversampling: {name} Dataset", fontsize=14)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axis('equal')
    plt.show()
```

**Output:**

![Double circles](https://github.com/oakkao/sona-oversampling/blob/main/examples/SONA_circle.png?raw=true)
![Blue moons](https://github.com/oakkao/sona-oversampling/blob/main/examples/SONA_blue_moons.png?raw=true)
![Gaussian clusters](https://github.com/oakkao/sona-oversampling/blob/main/examples/SONA_gaussian_cluster.png?raw=true)

### Running the Benchmark

Install experiment dependencies and run the full pipeline:

```bash
pip install sona-oversampling imbalanced-learn smote-variants optuna xgboost ctgan sdv
cd sona-oversampling
python -m experiments.benchmark.run
```

This executes the complete pipeline: dataset loading, Optuna hyperparameter tuning, and repeated evaluation across all method-classifier combinations. Outputs:
- `best_param.csv` — 343 rows of tuned hyperparameters
- `results.csv` — 392 rows of experiment results (metrics as lists of repeated measurements)

## Methodology

The benchmark evaluates SONA against 6 baseline oversampling methods and no-oversampling (Origin) across 7 classifiers and 7 datasets.

**Oversampling methods compared:**

| Method | Type | Tunable Params |
|--------|------|----------------|
| SONA (proposed) | Spatial border-based | None (parameter-free) |
| TVAE | Deep learning (VAE) | None (skips tuning) |
| SMOTE | Neighbor-based | n_neighbors |
| Borderline-SMOTE2 | Neighbor-based | n_neighbors |
| Safe-Level-SMOTE | Neighbor-based | n_neighbors |
| Polynom-fit-SMOTE | Polynomial fitting | None |
| SMOTE-IPF | Neighbor-based + filtering | n_neighbors |

**Classifiers:** Logistic Regression, XGBoost, KNN, Decision Tree, MLP, SVC, Random Forest.

**Experimental procedure:**

1. **Data preparation** — Load 7 imbalanced datasets, apply 80/20 stratified train/test split, remap minority label to 1 and majority to 0.
2. **Hyperparameter tuning** — Use Optuna (10 trials per study):
   - *Origin:* Tune classifier hyperparameters on original (un-oversampled) training data using 5-fold stratified CV with weighted F1 scoring.
   - *SMOTE variants:* Jointly tune oversampler `n_neighbors` (1-7) and classifier hyperparameters using 3-fold stratified CV with oversampling applied inside each fold.
   - *SONA and Polynom-fit-SMOTE:* Tune classifier hyperparameters only (no oversampler parameters).
   - *TVAE:* Skipped — as a deep learning method, it uses Origin's classifier parameters directly.
3. **Evaluation** — For each (dataset, oversampling method, classifier) combination, repeat N times (default 21, configurable). Train on oversampled training set, evaluate on held-out test set. Metrics: accuracy, precision (weighted), recall (weighted), F1-score (weighted), ROC-AUC (weighted).

## Citations

If you use SONA or this benchmark in your research, please cite the relevant works.

**Oversampling methods:**

```bibtex
@article{smote2002,
  title={SMOTE: synthetic minority over-sampling technique},
  author={Chawla, Nitesh V and Bowyer, Kevin W and Hall, Lawrence O and Kegelmeyer, W Philip},
  journal={Journal of artificial intelligence research},
  volume={16},
  pages={321--357},
  year={2002}
}

@inproceedings{Borderline_SMOTE2005,
  title={Borderline-SMOTE: A New Over-Sampling Method in Imbalanced Data Sets Learning},
  author={Han, Hui and Wang, Wen-Yuan and Mao, Bing-Huan},
  booktitle={Advances in Intelligent Computing},
  pages={878--887},
  year={2005},
  publisher={Springer}
}

@inproceedings{safe_level_smote2009,
  title={Safe-level-SMOTE: Safe-level-synthetic minority over-sampling technique for handling the class imbalanced problem},
  author={Bunkhumpornpat, Chumphol and Sinapiromsaran, Krung and Lursinsap, Chidchanok},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={475--482},
  year={2009},
  organization={Springer}
}

@inproceedings{polynom_smote2008,
  title={New oversampling approaches based on polynomial fitting for imbalanced data sets},
  author={Gazzah, Sami and Amara, Najoua Essoukri Ben},
  booktitle={The Eighth IAPR International Workshop on Document Analysis Systems},
  pages={677--684},
  year={2008},
  organization={IEEE}
}

@article{smote_ipf2015,
  title={SMOTE-IPF: Addressing the noisy and borderline examples problem in imbalanced classification by a re-sampling method with filtering},
  author={S{\'a}ez, Jos{\'e} A and Luengo, Juli{\'a}n and Stefanowski, Jerzy and Herrera, Francisco},
  journal={Information Sciences},
  volume={291},
  pages={184--203},
  year={2015}
}

@article{TVAE2018,
  title={TVAE: Triplet-based variational autoencoder using metric learning},
  author={Ishfaq, Haque and Hoogi, Assaf and Rubin, Daniel},
  journal={arXiv preprint arXiv:1802.04403},
  year={2018}
}

@article{Smote_variants2019,
  title={Smote-variants: A python implementation of 85 minority oversampling techniques},
  author={Kov{\'a}cs, Gy{\"o}rgy},
  journal={Neurocomputing},
  volume={366},
  pages={352--354},
  year={2019},
  publisher={Elsevier}
}
```

**Datasets:**

```bibtex
@misc{UCI,
  author={Dua, Dheeru and Graff, Casey},
  title={{UCI} Machine Learning Repository},
  year={2017},
  howpublished={http://archive.ics.uci.edu/ml}
}

@misc{pendigits_1998,
  author={E. Alpaydin and Fevzi Alimoglu},
  title={Pen-Based Recognition of Handwritten Digits},
  year={1998},
  howpublished={UCI Machine Learning Repository},
  doi={10.24432/C5MG6K}
}

@misc{thyroid_1987,
  author={Quinlan, Ross},
  title={Thyroid Disease},
  year={1986},
  howpublished={UCI Machine Learning Repository},
  doi={10.24432/C5D010}
}

@misc{libras_2009,
  author={Daniela Dias and Sarajane Peres and Heloisa Bscaro},
  title={Libras Movement},
  year={2009},
  howpublished={UCI Machine Learning Repository},
  doi={10.24432/C5GC82}
}

@misc{car_1997,
  author={Marko Bohanec},
  title={Car Evaluation},
  year={1997},
  howpublished={UCI Machine Learning Repository},
  doi={10.24432/C5JP48}
}

@misc{wine_quality_2009,
  author={Paulo Cortez and A. Cerdeira and F. Almeida and T. Matos and J. Reis},
  title={Wine Quality},
  year={2009},
  howpublished={UCI Machine Learning Repository},
  doi={10.24432/C56S3T}
}

@misc{mammography_1993,
  title={Comparative evaluation of pattern recognition techniques for detection of microcalcifications},
  author={Woods, Kevin S and Solka, Jeffrey L and Priebe, Carey E and Doss, Chris C and Bowyer, Kevin W and Clarke, Laurence P},
  booktitle={Biomedical Image Processing and Biomedical Visualization},
  volume={1905},
  pages={841--852},
  year={1993},
  organization={SPIE}
}

@misc{optical_digits_1998,
  author={E. Alpaydin and C. Kaynak},
  title={Optical Recognition of Handwritten Digits},
  year={1998},
  howpublished={UCI Machine Learning Repository},
  doi={10.24432/C50P49}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

