"""Tables and figures from the saved results; the unit of analysis is the dataset (Demšar, 2006)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, studentized_range, wilcoxon
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from .config import ABLATIONS, CLASSIFIERS, CLF_LABELS, METHOD_LABELS
from .experiment import read_jsonl
from .sona_full import check_guarantee, sona_resample

METRICS = ["f1", "mcc", "pr_auc", "roc_auc", "thr_f1", "gmean", "bal_acc", "recall", "precision"]
PLACEMENT = ["syn_nn_is_majority", "syn_knn_majority_frac", "syn_rel_pos_median", "syn_on_majority_point", "N3_after"]


def save(cfg, df, name):
    df.to_csv(cfg.tables_dir / f"{name}.csv")
    return df


def save_fig(cfg, fig, name):
    fig.savefig(cfg.fig_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(cfg.fig_dir / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def score_matrices(ok, metric, methods, n_folds):
    """{classifier: datasets x methods mean score} over datasets with every fold done, plus 'ALL' = mean over
    classifiers (when every classifier is complete)."""
    out = {}
    for clf in CLASSIFIERS:
        g = ok[(ok.classifier == clf) & ok.method.isin(methods)].groupby(["dataset", "method"])[metric]
        complete = (g.size().unstack().reindex(columns=methods) == n_folds).all(axis=1)
        b = g.mean().unstack().reindex(index=complete.index[complete], columns=methods).dropna()
        if len(b):
            out[clf] = b
    if len(out) == len(CLASSIFIERS):
        common = sorted(set.intersection(*(set(b.index) for b in out.values())))
        if common:
            out["ALL"] = sum(b.loc[common] for b in out.values()) / len(out)
    return out


def avg_ranks(b):
    return b.rank(axis=1, ascending=False).mean()


def nemenyi_cd(k, n):
    return studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * n))


def compare(b, ref="SONA"):
    """Two-sided Wilcoxon signed-rank test of ref against every other method, Holm and BH adjusted."""
    ranks, cd, rows = avg_ranks(b), nemenyi_cd(b.shape[1], len(b)), []
    for m in b.columns.drop(ref):
        d = b[ref] - b[m]
        rows.append({"vs": METHOD_LABELS[m], "N": len(b), "mean diff": d.mean(),
                     "W/T/L": f"{(d > 1e-9).sum()}/{(d.abs() <= 1e-9).sum()}/{(d < -1e-9).sum()}",
                     "rank diff": ranks[m] - ranks[ref], "> CD": abs(ranks[m] - ranks[ref]) > cd,
                     "p": wilcoxon(b[ref], b[m]).pvalue if (d.abs() > 1e-9).any() else 1.0})
    t = pd.DataFrame(rows).set_index("vs")
    t["p Holm"] = multipletests(t.p, method="holm")[1]
    t["p BH"] = multipletests(t.p, method="fdr_bh")[1]
    return t


def friedman(b):
    stat, p = friedmanchisquare(*b.T.values)
    return pd.Series({"N": len(b), "chi2": stat, "p": p, "CD": nemenyi_cd(b.shape[1], len(b))})


def rank_table(blocks, methods):
    t = pd.DataFrame({CLF_LABELS[c]: avg_ranks(b) for c, b in blocks.items()}).reindex(methods)
    t.loc["N datasets"] = [len(b) for b in blocks.values()]
    return t.rename(index=METHOD_LABELS)


def per_classifier(blocks, fn):
    return pd.concat({CLF_LABELS[c]: fn(b) for c, b in blocks.items() if len(b) >= 2}, names=["classifier"])


def cd_plot(cfg, b, name):
    """Average ranks with the band of methods within one critical difference of the best."""
    ranks = avg_ranks(b).sort_values()
    cd = nemenyi_cd(b.shape[1], len(b))
    fig, ax = plt.subplots(figsize=(4.8, 3.3))
    ax.axvspan(ranks.iloc[0], ranks.iloc[0] + cd, color="tab:blue", alpha=0.13, lw=0)
    ax.hlines(range(len(ranks)), 1, ranks.values, color="0.82", lw=1)
    ax.plot(ranks.values, range(len(ranks)), "o", color="black", ms=5)
    for i, v in enumerate(ranks.values):
        ax.annotate(f"{v:.2f}", (v, i), xytext=(6, -3.5), textcoords="offset points", fontsize=9)
    ax.set_yticks(range(len(ranks)), [METHOD_LABELS[m] for m in ranks.index])
    ax.set(xlim=(1, len(ranks)), ylim=(len(ranks) - 0.5, -1.1), xlabel="average rank of F1 (lower is better)")
    ax.text(ranks.iloc[0] + cd / 2, -0.85, f"within CD of the best (CD = {cd:.2f})", ha="center", va="bottom",
            fontsize=9, color="tab:blue")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_fig(cfg, fig, name)


def rank_vs_dimension(cfg, b, meta):
    """Rank of SONA on each dataset against the number of features and the imbalance ratio."""
    dd = meta.set_index("dataset").loc[b.index, ["d", "n", "IR"]]
    dd["SONA rank"] = b.rank(axis=1, ascending=False)["SONA"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(dd))
    for ax, col, label in [(axes[0], "d", "number of features (log scale)"),
                           (axes[1], "IR", "imbalance ratio (log scale)")]:
        rho, p = spearmanr(dd[col], dd["SONA rank"])
        ax.scatter(dd[col], dd["SONA rank"] + jitter, s=22, alpha=0.8)
        ax.set(xscale="log", xlabel=label, title=f"Spearman ρ = {rho:.2f} (p = {p:.2f})",
               yticks=range(1, b.shape[1] + 1))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("rank of SONA (1 = best)")
    fig.tight_layout()
    save_fig(cfg, fig, "sona_rank_vs_dimension")
    return dd


def run_analysis(cfg, meta):
    """Build every table and figure from results/*.jsonl; returns the main tables."""
    res = pd.DataFrame([r for p in sorted(cfg.results_dir.glob("*.jsonl")) for r in read_jsonl(p)])
    if res.empty:
        print("no results in", cfg.results_dir)
        return {}
    res = res.drop_duplicates(["dataset", "fold", "method", "classifier"], keep="last")
    ok = res[res.row_status == "ok"]
    failed = ok[~ok.os_status.str.startswith("ok")].groupby("method").size()
    print(f"{len(res)} rows, {len(ok)} ok; oversampler failures (trained on the original fold): {failed.to_dict()}")
    n, main, out = cfg.n_outer_folds, cfg.main_methods, {}

    f1 = score_matrices(ok, "f1", main, n)
    out["avg_rank_f1"] = save(cfg, rank_table(f1, main), "avg_rank_f1")
    out["wilcoxon_f1"] = save(cfg, per_classifier(f1, compare), "wilcoxon_f1")
    save(cfg, per_classifier(f1, friedman).unstack(), "friedman_f1")

    ranks, pvals = {}, {}
    for met in METRICS:
        b = score_matrices(ok, met, main, n).get("ALL")
        if b is not None and len(b) >= 2:
            ranks[met], pvals[met] = avg_ranks(b), compare(b)["p Holm"]
    out["avg_rank_all_metrics"] = save(cfg, pd.DataFrame(ranks).rename(index=METHOD_LABELS), "avg_rank_all_metrics")
    out["holm_p_all_metrics"] = save(cfg, pd.DataFrame(pvals), "holm_p_all_metrics")

    if "ALL" in f1 and len(f1["ALL"]) >= 2:
        cd_plot(cfg, f1["ALL"], "cd_f1_all")
        save(cfg, rank_vs_dimension(cfg, f1["ALL"], meta), "sona_rank_vs_dimension")

    if cfg.include_ablation:
        abl = score_matrices(ok, "f1", ["SONA"] + ABLATIONS, n)
        out["ablation_ranks"] = save(cfg, rank_table(abl, ["SONA"] + ABLATIONS), "ablation_ranks")
        save(cfg, per_classifier(abl, compare), "ablation_wilcoxon")

    once = ok.drop_duplicates(["dataset", "fold", "method", "os_k"])   # one row per oversampled training fold
    syn = once[once.method != "Origin"].dropna(subset=["syn_nn_is_majority"])
    out["placement"] = save(cfg, syn.groupby(["method", "dataset"])[PLACEMENT].mean().groupby("method").mean()
                            .reindex([m for m in cfg.methods if m != "Origin"]).rename(index=METHOD_LABELS),
                            "placement_metrics")
    save(cfg, once.groupby(["method", "dataset"]).os_time_s.mean().groupby("method").agg(["mean", "median", "max"])
         .rename(index=METHOD_LABELS), "oversampler_runtime_seconds")
    return out


def guarantee_check(cfg, data, datasets):
    """Apply SONA to every standardised dataset and count samples closer to a safe majority point than to the anchor."""
    rows = {}
    for ds in datasets:
        X, y = data[ds]
        info = sona_resample(StandardScaler().fit_transform(X), y, 1, cfg.seed(ds, "guarantee"), return_info=True)[2]
        rows[ds] = {"min_radius": info["R"].min(), **check_guarantee(info)}
    guar = save(cfg, pd.DataFrame.from_dict(rows, orient="index"), "sona_guarantee_check")
    print(f"guarantee violations: {int(guar.violations.sum())} (theory: 0)")
    return guar
