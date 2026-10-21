"""Two-dimensional synthetic datasets: comparison figures and a repeated study on independent test sets."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_circles, make_moons

from .classifiers import make_classifier, decision_scores
from .config import CLASSIFIERS, METHOD_LABELS, SV_BASELINES
from .metrics import classification_metrics, synthetic_mask
from .oversampling import oversample

N_MAJ, N_MIN, TEST_FACTOR = 425, 75, 5
NAMES = ["Double circles", "Two moons", "Gaussian clusters"]


def make_2d(name, seed, factor=1):
    n = (N_MAJ * factor, N_MIN * factor)
    if name == "Double circles":
        return make_circles(n_samples=n, noise=0.05, factor=0.6, random_state=seed)
    if name == "Two moons":
        return make_moons(n_samples=n, noise=0.1, random_state=seed)
    return make_blobs(n_samples=list(n), centers=[[0, 0], [1.2, 1.2]], cluster_std=[0.5, 0.4], random_state=seed)


def methods_2d(cfg):
    return [m for m in SV_BASELINES + ["TVAE", "SONA"] if m in cfg.methods]


def run_2d(cfg, method, X, y, seed):
    try:
        return oversample(method, X, y, 5, seed, cfg.tvae_epochs)   # k = 5 for the SMOTE family
    except Exception as e:
        print(f"  {method} failed: {e}")
        return X, y


def figures(cfg):
    """One figure per dataset: the original data and the output of every oversampler."""
    for name in NAMES:
        X, y = make_2d(name, seed=0)
        panels = [("Original", X, y)] + [(METHOD_LABELS[m], *run_2d(cfg, m, X, y, cfg.seed(name, m, "fig")))
                                         for m in methods_2d(cfg)]
        nrow = -(-len(panels) // 4)
        fig, axes = plt.subplots(nrow, 4, figsize=(7.2, 2.05 * nrow), sharex=True, sharey=True)
        for ax, (title, Xr, yr) in zip(axes.ravel(), panels):
            ax.scatter(*X[y == 0].T, s=3, c="tab:blue", alpha=0.45, lw=0)
            ax.scatter(*X[y == 1].T, s=8, c="orange", edgecolors="k", linewidths=0.25)
            ax.scatter(*Xr[synthetic_mask(X, y, Xr, yr)].T, s=8, c="tab:red", marker="x", alpha=0.6, linewidths=0.6)
            ax.set(title=title, xticks=[], yticks=[])
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        handles = [plt.Line2D([], [], ls="", marker="o", color="tab:blue", label="majority"),
                   plt.Line2D([], [], ls="", marker="o", color="orange", markeredgecolor="k", label="minority (original)"),
                   plt.Line2D([], [], ls="", marker="x", color="tab:red", label="minority (synthetic)")]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.4, h_pad=0.6)
        fname = "synthetic_" + name.lower().replace(" ", "_")
        fig.savefig(cfg.fig_dir / f"{fname}.pdf", bbox_inches="tight")
        fig.savefig(cfg.fig_dir / f"{fname}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)


def study(cfg):
    """Every classifier with default hyperparameters, test set five times larger, n_synth_reps repetitions."""
    rows = []
    for name in NAMES:
        for rep in range(cfg.n_synth_reps):
            Xtr, ytr = make_2d(name, seed=rep)
            Xte, yte = make_2d(name, seed=100_000 + rep, factor=TEST_FACTOR)
            for method in ["Origin"] + methods_2d(cfg):
                Xr, yr = run_2d(cfg, method, Xtr, ytr, cfg.seed(name, rep, method))
                for clf in CLASSIFIERS:
                    model = make_classifier(clf, {}, cfg.seed(name, rep, clf)).fit(Xr, yr)
                    m = classification_metrics(yte, model.predict(Xte), decision_scores(model, Xte))
                    rows.append({"dataset": name, "rep": rep, "method": method, "classifier": clf, **m})
        print("done:", name)
    runs = pd.DataFrame(rows)
    runs.to_csv(cfg.tables_dir / "synthetic2d_all_runs.csv", index=False)
    perf = runs.groupby(["dataset", "method", "rep"])[["f1", "pr_auc"]].mean().groupby(["dataset", "method"])
    mean, std = perf.mean(), perf.std()
    tbl = (mean.map("{:.3f}".format) + " ± " + std.map("{:.3f}".format)).unstack(0)
    tbl = tbl.reindex(["Origin"] + methods_2d(cfg)).rename(index=METHOD_LABELS)
    tbl.to_csv(cfg.tables_dir / "synthetic2d_performance.csv")
    return tbl
