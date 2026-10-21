# SONA

**Stochastic directional Oversampling using Negative Anomalous scores for imbalanced dataset**

SONA is a parameter-free oversampling method for imbalanced binary classification. It needs no `k`-neighbours parameter. Reverse nearest-neighbour counts decide which minority points act as anchors, and every synthetic sample is kept inside a radius around its anchor, so that it is strictly closer to its anchor than to any *safe* majority point.

Published on PyPI as [`sona-oversampling`](https://pypi.org/project/sona-oversampling/).


## Installation

```bash
pip install sona-oversampling            # normal install
pip install --upgrade sona-oversampling  # update to the latest version
```


## Requirements

**Core** (installed with the package): Python ≥ 3.9, numpy, scipy, scikit-learn.

**Benchmark experiments** (`pip install -e ".[experiments]"`, or the exact versions in `experiments/requirements.txt`): imbalanced-learn (datasets and G-mean), smote-variants (SMOTE-family baselines), sdv with torch (TVAE), optuna, xgboost, statsmodels (p-value adjustment), pandas and matplotlib.

## Usage

```python
from sklearn.preprocessing import StandardScaler
from sona_oversampling import SONA

X_scaled = StandardScaler().fit_transform(X_train)        # SONA is distance-based: scale first
X_res, y_res = SONA(X_scaled, y_train, min_label=1, random_state=0)
```

`SONA(X, y, min_label, new_label=0, random_state=None)`

| Parameter | Description |
|-----------|-------------|
| `X` | Feature matrix of shape (n_samples, n_features) |
| `y` | Labels; every label other than `min_label` is treated as majority |
| `min_label` | Label of the minority class |
| `new_label` | Added to `min_label` for the synthetic samples (default 0; e.g. `1` labels them `min_label + 1` for plotting) |
| `random_state` | Seed (int or numpy Generator) for reproducible output |

**Returns** `(X_res, y_res)`: the original data followed by `n_maj − n_min` synthetic minority samples, so both classes have the same size.

### Visualization example

`examples/visualize.py` draws SONA on three 2-D toy datasets (`new_label=1` labels the synthetic samples 2, so they can be plotted separately):

![Double circles](https://github.com/oakkao/sona-oversampling/blob/main/examples/SONA_circle.png?raw=true)
![Two moons](https://github.com/oakkao/sona-oversampling/blob/main/examples/SONA_two_moons.png?raw=true)
![Gaussian clusters](https://github.com/oakkao/sona-oversampling/blob/main/examples/SONA_gaussian_cluster.png?raw=true)

## Benchmark

### Datasets

All 23 datasets of the imbalanced-learn benchmark collection (Ding, 2011; Zenodo DOI [10.5281/zenodo.61452](https://doi.org/10.5281/zenodo.61452)) with at most 12,000 instances and 300 features. The rule was fixed before the experiments; it excludes `isolet` (617 features), `letter_img`, `webpage` and `protein_homo`. Datasets are loaded with `imblearn.datasets.fetch_datasets()`; minority = label 1.

| Dataset | n | d | N_min | IR | Source |
|---------|--:|--:|------:|---:|--------|
| ecoli | 336 | 7 | 35 | 8.60 | UCI |
| libras_move | 360 | 90 | 24 | 14.00 | UCI |
| arrhythmia | 452 | 278 | 25 | 17.08 | UCI |
| spectrometer | 531 | 93 | 45 | 10.80 | UCI |
| oil | 937 | 49 | 41 | 21.85 | Kubat et al. (1998) |
| solar_flare_m0 | 1389 | 32 | 68 | 19.43 | UCI |
| yeast_me2 | 1484 | 8 | 51 | 28.10 | UCI |
| car_eval_34 | 1728 | 21 | 134 | 11.90 | UCI |
| car_eval_4 | 1728 | 21 | 65 | 25.58 | UCI |
| us_crime | 1994 | 100 | 150 | 12.29 | UCI |
| scene | 2407 | 294 | 177 | 12.60 | LIBSVM (Boutell et al., 2004) |
| yeast_ml8 | 2417 | 103 | 178 | 12.58 | LIBSVM (Elisseeff & Weston, 2001) |
| ozone_level | 2536 | 72 | 73 | 33.74 | UCI |
| sick_euthyroid | 3163 | 42 | 293 | 9.80 | UCI |
| thyroid_sick | 3772 | 52 | 231 | 15.33 | UCI |
| abalone | 4177 | 10 | 391 | 9.68 | UCI |
| abalone_19 | 4177 | 10 | 32 | 129.53 | UCI |
| wine_quality | 4898 | 11 | 183 | 25.77 | UCI |
| optical_digits | 5620 | 64 | 554 | 9.14 | UCI |
| satimage | 6435 | 36 | 626 | 9.28 | UCI |
| coil_2000 | 9822 | 85 | 586 | 15.76 | UCI |
| pen_digits | 10992 | 16 | 1055 | 9.42 | UCI |
| mammography | 11183 | 6 | 260 | 42.01 | Woods et al. (1993) |

n: instances, d: features, N_min: minority instances, IR: imbalance ratio.

The 2-D study uses three synthetic datasets from scikit-learn (double circles, two moons, two Gaussian clusters), each with 425 majority and 75 minority points.

### Methods

| Method | Type | Tuned parameters |
|--------|------|------------------|
| SONA (proposed) | Reverse-NN anchors + bounded synthesis | none (parameter-free) |
| SMOTE | Interpolation | `k ∈ {1, 3, 5, 7}` |
| Borderline-SMOTE2 | Interpolation from borderline points | `k ∈ {1, 3, 5, 7}` |
| Safe-Level-SMOTE | Safe-level interpolation | `k ∈ {1, 3, 5, 7}` |
| Polynom-fit-SMOTE (`poly_2`) | Polynomial fitting | none |
| SMOTE-IPF | Interpolation + iterative partitioning filter | `k ∈ {1, 3, 5, 7}` |
| TVAE (SDV `TVAESynthesizer`, 300 epochs) | Tabular variational autoencoder | none |
| Origin | No oversampling | – |
| SONA w/o NEA, SONA w/o DSS | Ablations | none |

The SMOTE-family baselines come from smote-variants with balanced output and library defaults otherwise. **Classifiers:** Logistic Regression, Decision Tree, kNN, MLP, SVM, Random Forest and XGBoost.

### Protocol

1. **Outer evaluation.** Stratified 5-fold cross-validation. Features are standardised on each outer training fold; test folds are never oversampled or used for tuning.
2. **Nested tuning.** Inside every outer training fold, Optuna (seeded TPE, 20 trials) tunes the classifier and the oversampler's `k` together to maximise the mean F1-score of the minority class over a stratified 3-fold inner cross-validation; only the inner training folds are oversampled. Search spaces are listed in `classifiers.py`.
3. **Metrics** (minority class = positive): F1 (primary), precision, recall, G-mean, balanced accuracy, MCC, PR-AUC and ROC-AUC, and the F1-score at a decision threshold tuned on the inner out-of-fold predictions.
4. **Placement of synthetic samples:** share of synthetic samples whose nearest original point is a majority point, majority share among the 5 nearest points, relative position between the classes, share lying on a majority point, and the N3 overlap measure.
5. **Statistics** (datasets are the unit of analysis): average ranks, Friedman test, Nemenyi critical difference, and two-sided Wilcoxon signed-rank tests of SONA against each competitor with Holm and Benjamini–Hochberg adjustment.
6. **Seeds.** Every random component (folds, oversamplers, classifiers, Optuna) gets a deterministic seed derived from the base seed `20260707`.

### Running the benchmark

**Notebook.** Open `experiments/SONA_experiments.ipynb` (for example in Google Colab) and run all cells. It is self-contained (same code as `experiments/benchmark/`). Set `SMOKE_TEST = True` first for a quick check.

**Command line** (from the repository root):

```bash
pip install -e ".[experiments]"
python -m experiments.benchmark.run --smoke                  # quick check: 2 datasets, 2 folds, 3 trials
python -m experiments.benchmark.run                          # full study
python -m experiments.benchmark.run --shard 0 --n-shards 3   # one of three parallel sessions
python -m experiments.benchmark.run --analysis-only          # tables and figures from saved results
```

Other options: `--output DIR`, `--run-name NAME`, `--datasets ecoli libras_move`, `--no-tvae`, `--no-ablation`.

Each result row is appended at once to `sona_results/<run>/results/<dataset>.jsonl`, so an interrupted run can be restarted and finished rows are skipped. Tables (CSV) are written to `tables/` and figures (PDF and 600 dpi PNG) to `figures/`.

## Code structure

```
sona-oversampling/
  src/sona_oversampling/
    main.py               # SONA (the published package)
  tests/
    test_sona.py          # unit tests; also checks the package against sona_full.py
  examples/
    visualize.py          # 2-D example figures shown above
  experiments/
    SONA_experiments.ipynb  # self-contained benchmark notebook
    requirements.txt        # library versions used for the published results
    benchmark/
      config.py           # settings and seeds
      sona_full.py        # SONA with ablation switches and guarantee check (same output as the package)
      oversampling.py     # SONA, ablations, smote-variants baselines, TVAE
      classifiers.py      # classifiers and Optuna search spaces
      metrics.py          # performance metrics and placement measures
      experiment.py       # datasets and nested cross-validation with checkpointing
      analysis.py         # ranks, statistical tests, tables, guarantee check
      synthetic2d.py      # 2-D figures and repeated study
      worked_example.py   # figures of the three SONA stages
      run.py              # command-line entry point
```

## Citations

If you use SONA or this benchmark in your research, please cite the relevant works.

**Methods and software**

```bibtex
@article{smote2002,
  title={{SMOTE}: synthetic minority over-sampling technique},
  author={Chawla, Nitesh V and Bowyer, Kevin W and Hall, Lawrence O and Kegelmeyer, W Philip},
  journal={Journal of Artificial Intelligence Research},
  volume={16}, pages={321--357}, year={2002}
}

@inproceedings{Borderline_SMOTE2005,
  title={{Borderline-SMOTE}: A New Over-Sampling Method in Imbalanced Data Sets Learning},
  author={Han, Hui and Wang, Wen-Yuan and Mao, Bing-Huan},
  booktitle={Advances in Intelligent Computing}, pages={878--887}, year={2005}, publisher={Springer}
}

@inproceedings{safe_level_smote2009,
  title={{Safe-Level-SMOTE}: Safe-level-synthetic minority over-sampling technique for handling the class imbalanced problem},
  author={Bunkhumpornpat, Chumphol and Sinapiromsaran, Krung and Lursinsap, Chidchanok},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining}, pages={475--482}, year={2009}, organization={Springer}
}

@inproceedings{polynom_smote2008,
  title={New oversampling approaches based on polynomial fitting for imbalanced data sets},
  author={Gazzah, Sami and Amara, Najoua Essoukri Ben},
  booktitle={The Eighth IAPR International Workshop on Document Analysis Systems}, pages={677--684}, year={2008}, organization={IEEE}
}

@article{smote_ipf2015,
  title={{SMOTE-IPF}: Addressing the noisy and borderline examples problem in imbalanced classification by a re-sampling method with filtering},
  author={S{\'a}ez, Jos{\'e} A and Luengo, Juli{\'a}n and Stefanowski, Jerzy and Herrera, Francisco},
  journal={Information Sciences}, volume={291}, pages={184--203}, year={2015}
}

@inproceedings{xu2019modeling,
  title={Modeling tabular data using conditional {GAN}},
  author={Xu, Lei and Skoularidou, Maria and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan},
  booktitle={Advances in Neural Information Processing Systems}, volume={32}, year={2019}
}

@article{Smote_variants2019,
  title={Smote-variants: A {Python} implementation of 85 minority oversampling techniques},
  author={Kov{\'a}cs, Gy{\"o}rgy},
  journal={Neurocomputing}, volume={366}, pages={352--354}, year={2019}
}

@article{lemaitre2017imbalanced,
  title={Imbalanced-learn: A {Python} toolbox to tackle the curse of imbalanced datasets in machine learning},
  author={Lema{\^\i}tre, Guillaume and Nogueira, Fernando and Aridas, Christos K},
  journal={Journal of Machine Learning Research}, volume={18}, number={17}, pages={1--5}, year={2017}
}

@article{demsar2006statistical,
  title={Statistical comparisons of classifiers over multiple data sets},
  author={Dem{\v{s}}ar, Janez},
  journal={Journal of Machine Learning Research}, volume={7}, pages={1--30}, year={2006}
}
```

**Datasets**

```bibtex
@phdthesis{ding2011diversified,
  title={Diversified Ensemble Classifiers for Highly Imbalanced Data Learning and their Application in Bioinformatics},
  author={Ding, Zejin}, school={Georgia State University}, year={2011}
}

@misc{UCI,
  author={Dua, Dheeru and Graff, Casey},
  title={{UCI} Machine Learning Repository}, year={2017},
  howpublished={http://archive.ics.uci.edu/ml}
}

@article{kubat1998machine,
  title={Machine learning for the detection of oil spills in satellite radar images},
  author={Kubat, Miroslav and Holte, Robert C and Matwin, Stan},
  journal={Machine Learning}, volume={30}, number={2--3}, pages={195--215}, year={1998}
}

@inproceedings{mammography_1993,
  title={Comparative evaluation of pattern recognition techniques for detection of microcalcifications},
  author={Woods, Kevin S and Solka, Jeffrey L and Priebe, Carey E and Doss, Chris C and Bowyer, Kevin W and Clarke, Laurence P},
  booktitle={Biomedical Image Processing and Biomedical Visualization}, volume={1905}, pages={841--852}, year={1993}, organization={SPIE}
}

@article{chang2011libsvm,
  title={{LIBSVM}: A library for support vector machines},
  author={Chang, Chih-Chung and Lin, Chih-Jen},
  journal={ACM Transactions on Intelligent Systems and Technology}, volume={2}, number={3}, pages={27:1--27:27}, year={2011}
}

@article{boutell2004learning,
  title={Learning multi-label scene classification},
  author={Boutell, Matthew R and Luo, Jiebo and Shen, Xipeng and Brown, Christopher M},
  journal={Pattern Recognition}, volume={37}, number={9}, pages={1757--1771}, year={2004}
}

@inproceedings{elisseeff2001kernel,
  title={A kernel method for multi-labelled classification},
  author={Elisseeff, Andr{\'e} and Weston, Jason},
  booktitle={Advances in Neural Information Processing Systems}, volume={14}, pages={681--687}, year={2001}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
