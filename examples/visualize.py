"""Draw SONA on three 2-D toy datasets and save the images used in the README."""
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons

from sona_oversampling import SONA

OUT = Path(__file__).resolve().parent
datasets = [
    ("Double circles", "SONA_circle.png", make_circles(n_samples=(500, 100), noise=0.05, factor=0.6, random_state=42)),
    ("Two moons", "SONA_two_moons.png", make_moons(n_samples=(500, 100), noise=0.1, random_state=42)),
    ("Gaussian clusters", "SONA_gaussian_cluster.png",
     make_blobs(n_samples=[500, 100], centers=[[0, 0], [1.2, 1.2]], cluster_std=[0.5, 0.4], random_state=42)),
]

for name, fname, (X, y) in datasets:
    # new_label=1 labels the synthetic samples 2, so they can be drawn separately
    X_res, y_res = SONA(X, y, min_label=1, new_label=1, random_state=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(*X_res[y_res == 0].T, c="tab:blue", s=12, alpha=0.5, label="majority (0)")
    ax.scatter(*X_res[y_res == 1].T, c="orange", s=22, edgecolors="k", linewidths=0.4, label="minority (1)")
    ax.scatter(*X_res[y_res == 2].T, c="tab:red", marker="x", s=20, alpha=0.7, label="synthetic (2)")
    ax.set_title(f"SONA oversampling: {name}")
    ax.set_aspect("equal")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / fname)
