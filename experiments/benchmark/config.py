"""Experiment settings. The defaults reproduce the paper; Config.smoke() is a quick end-to-end check."""
import json
import zlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

CLASSIFIERS = ["LR", "DT", "KNN", "MLP", "SVM", "RF", "XGB"]
SV_BASELINES = ["SMOTE", "Borderline_SMOTE2", "Safe_Level_SMOTE", "polynom_fit_SMOTE_poly", "SMOTE_IPF"]
K_METHODS = {"SMOTE", "Borderline_SMOTE2", "Safe_Level_SMOTE", "SMOTE_IPF"}   # baselines with a tuned k
ABLATIONS = ["SONA_noNEA", "SONA_noDSS"]

METHOD_LABELS = {
    "Origin": "Origin", "SONA": "SONA", "SMOTE": "SMOTE", "Borderline_SMOTE2": "Borderline-SMOTE2",
    "Safe_Level_SMOTE": "Safe-Level-SMOTE", "polynom_fit_SMOTE_poly": "Polynom-fit-SMOTE", "SMOTE_IPF": "SMOTE-IPF",
    "TVAE": "TVAE", "SONA_noNEA": "SONA w/o NEA", "SONA_noDSS": "SONA w/o DSS",
}
CLF_LABELS = {"LR": "Logistic Regression", "DT": "Decision Tree", "KNN": "kNN", "MLP": "MLP", "SVM": "SVM",
              "RF": "Random Forest", "XGB": "XGBoost", "ALL": "ALL"}


@dataclass
class Config:
    output_root: str = "sona_results"
    run_name: str = "main"
    datasets: Optional[List[str]] = None      # None: every dataset within max_samples and max_features
    include_tvae: bool = True
    include_ablation: bool = True
    base_seed: int = 20260707
    n_outer_folds: int = 5
    n_inner_folds: int = 3
    n_trials: int = 20
    k_grid: List[int] = field(default_factory=lambda: [1, 3, 5, 7])
    max_samples: int = 12_000
    max_features: int = 300
    tvae_epochs: int = 300
    n_synth_reps: int = 10

    @classmethod
    def smoke(cls, **kw):
        return cls(**{**dict(run_name="smoke", datasets=["ecoli", "libras_move"], n_outer_folds=2, n_inner_folds=2,
                             n_trials=3, tvae_epochs=20, n_synth_reps=2), **kw})

    @property
    def main_methods(self):
        return ["Origin", "SONA"] + SV_BASELINES + (["TVAE"] if self.include_tvae else [])

    @property
    def methods(self):
        return self.main_methods + (ABLATIONS if self.include_ablation else [])

    @property
    def run_dir(self):
        return Path(self.output_root) / self.run_name

    @property
    def results_dir(self):
        return self.run_dir / "results"

    @property
    def tables_dir(self):
        return self.run_dir / "tables"

    @property
    def fig_dir(self):
        return self.run_dir / "figures"

    def seed(self, *parts):
        """Deterministic seed from any identifiers (independent of Python's hash randomisation)."""
        return zlib.crc32("|".join(map(str, (self.base_seed, *parts))).encode()) & 0x7FFFFFFF

    def prepare(self):
        """Create the run folders; refuse to mix results of different protocols in one run folder."""
        for d in (self.results_dir, self.tables_dir, self.fig_dir):
            d.mkdir(parents=True, exist_ok=True)
        skip = {"output_root", "run_name", "datasets", "include_tvae", "include_ablation"}
        protocol = {k: v for k, v in asdict(self).items() if k not in skip}
        path = self.run_dir / "config.json"
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise RuntimeError(f"The settings differ from the run saved in {self.run_dir}; use another run name.")
        path.write_text(json.dumps(protocol, indent=2))
        return self
