"""Command-line entry point.

python -m experiments.benchmark.run --smoke                 # quick end-to-end check
python -m experiments.benchmark.run                         # full study
python -m experiments.benchmark.run --shard 0 --n-shards 3  # one of three parallel sessions
python -m experiments.benchmark.run --analysis-only         # tables and figures from saved results
"""
import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from . import analysis, synthetic2d, worked_example
from .config import Config
from .experiment import load_benchmark, select_datasets, run_dataset


def main():
    p = argparse.ArgumentParser(description="SONA benchmark (nested cross-validation)")
    p.add_argument("--smoke", action="store_true", help="2 datasets, 2 folds, 3 trials")
    p.add_argument("--output", default="sona_results", help="output folder")
    p.add_argument("--run-name", help="subfolder of this run (default: main, or smoke)")
    p.add_argument("--datasets", nargs="+", help="run only these datasets")
    p.add_argument("--shard", type=int, default=0, help="index of this session, 0 .. n-shards - 1")
    p.add_argument("--n-shards", type=int, default=1, help="number of parallel sessions")
    p.add_argument("--no-tvae", action="store_true", help="skip TVAE (no SDV/torch needed)")
    p.add_argument("--no-ablation", action="store_true", help="skip SONA w/o NEA and SONA w/o DSS")
    p.add_argument("--analysis-only", action="store_true", help="skip the experiments")
    a = p.parse_args()

    kw = dict(output_root=a.output, include_tvae=not a.no_tvae, include_ablation=not a.no_ablation)
    kw.update({k: v for k, v in [("run_name", a.run_name), ("datasets", a.datasets)] if v})
    cfg = (Config.smoke(**kw) if a.smoke else Config(**kw)).prepare()

    data, meta = load_benchmark()
    selected = select_datasets(meta, cfg)
    meta[meta.dataset.isin(selected)].to_csv(cfg.tables_dir / "datasets.csv", index=False)
    if not a.analysis_only:
        mine = selected[a.shard::a.n_shards]
        print(f"{len(selected)} datasets selected; this session runs {mine}")
        for ds in mine:
            print(f"=== {ds} ===", flush=True)
            run_dataset(cfg, ds, *data[ds])
        if a.n_shards > 1:
            print("When every session has finished, run once with --analysis-only.")
            return

    tables = analysis.run_analysis(cfg, meta)
    if "avg_rank_f1" in tables:
        print("Average rank of the F1-score (lower is better):\n" + tables["avg_rank_f1"].round(2).to_string())
    analysis.guarantee_check(cfg, data, selected)
    synthetic2d.figures(cfg)
    synthetic2d.study(cfg)
    worked_example.figures(cfg.fig_dir)
    print(f"Tables: {cfg.tables_dir}\nFigures: {cfg.fig_dir}")


if __name__ == "__main__":
    main()
