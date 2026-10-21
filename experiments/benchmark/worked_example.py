"""Figures of the three SONA stages on a small example (14 majority and 6 minority points)."""
import matplotlib.pyplot as plt
import numpy as np

from .sona_full import nearest, sona_resample

XMIN = np.array([[1.0, 1.0], [2.0, 1.6], [1.4, 2.6], [3.0, 2.3], [2.4, 3.3], [4.3, 1.1]])
XMAJ = np.array([[0.0, 0.0], [2.0, -0.2], [4.0, -0.3], [5.4, 2.0], [5.2, 3.6], [3.9, 3.5], [3.0, 4.7],
                 [1.0, 4.2], [-0.2, 3.0], [-0.6, 1.4], [3.5, 1.3], [0.5, 2.0], [2.6, 0.6], [5.6, 0.5]])
BLUE, RED = "#1f5fa8", "#c0392b"
OFF = {0: (6, -30), 1: (8, -10), 2: (-58, 6), 3: (12, -24), 4: (9, 2), 5: (-22, -36)}   # label offsets
BOX = dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85)


def figures(fig_dir):
    X = np.vstack([XMAJ, XMIN]); y = np.r_[np.zeros(len(XMAJ), int), np.ones(len(XMIN), int)]
    _, _, inf = sona_resample(X, y, 1, random_state=3, return_info=True)
    nea, pea, safe, R = inf["nea"], inf["pea"], inf["safe"], inf["R"]
    p_anchor = (1 / (1 + nea)) / (1 / (1 + nea)).sum()

    def base(ax, safe_style=False):
        if safe_style:
            ax.scatter(*XMAJ[safe].T, c=BLUE, s=45, zorder=3, label="safe majority (PEA = 0)")
            ax.scatter(*XMAJ[~safe].T, facecolors="white", edgecolors=BLUE, linewidths=1.5, s=45, zorder=3,
                       label="unsafe majority (PEA > 0)")
        else:
            ax.scatter(*XMAJ.T, c=BLUE, s=45, zorder=3, label="majority")
        ax.scatter(*XMIN.T, c=RED, marker="s", s=45, zorder=3, label="minority")
        ax.set(xlim=(-1.1, 6.2), ylim=(-0.9, 5.3)); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    def arrow(ax, src, dst):
        ax.annotate("", dst, src, arrowprops=dict(arrowstyle="-|>", color="0.45", lw=0.9, mutation_scale=9,
                                                  shrinkA=4, shrinkB=5), zorder=2)

    def finish(fig, ax, name, **legend_kw):
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False, **legend_kw)
        fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(fig_dir / f"{name}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

    # Stage 1: NEA scores and anchor probabilities
    fig, ax = plt.subplots(figsize=(4.9, 4.4))
    for j, i in enumerate(nearest(XMIN, XMAJ)[1]):
        arrow(ax, XMAJ[j], XMIN[i])
    base(ax)
    for i, p in enumerate(XMIN):
        ax.annotate(f"NEA = {nea[i]}\n$P$ = {p_anchor[i]:.2f}", p, xytext=OFF[i], textcoords="offset points",
                    color=RED, fontsize=10, linespacing=1.0, bbox=BOX, zorder=4)
    finish(fig, ax, "sona_step1_anchor")

    # Stage 2: PEA scores, safe majority subset and boundary constraint radii
    fig, ax = plt.subplots(figsize=(4.9, 4.4))
    for i, j in enumerate(nearest(XMAJ, XMIN)[1]):
        arrow(ax, XMIN[i], XMAJ[j])
    for i, c in enumerate(XMIN):
        ax.add_patch(plt.Circle(c, R[i], fill=False, ls="--", lw=1.0, color=RED, alpha=0.8))
    base(ax, safe_style=True)
    for j, p in enumerate(XMAJ):
        ax.annotate(f"{pea[j]}", p, xytext=(5, -13), textcoords="offset points", fontsize=10, color=BLUE,
                    bbox=BOX, zorder=4)
    finish(fig, ax, "sona_step2_radius")

    # Stage 3: one directional spherical synthesis step (anchor = minority point with the smallest radius)
    fig, ax = plt.subplots(figsize=(4.9, 4.4))
    a = int(np.argmin(R)); D = np.linalg.norm(XMIN - XMIN[a], axis=1)
    w = np.where(D > 0, 1 / np.where(D > 0, D, 1), 0); pt = w / w.sum()
    beyond = np.flatnonzero((D > R[a]) & (pt > 0)); t = int(beyond[np.argmax(pt[beyond])])
    ax.add_patch(plt.Circle(XMIN[a], R[a], fill=False, color=RED, lw=1.2))
    ax.plot(*np.c_[XMIN[a], XMIN[t]], color=RED, lw=1, ls=":")
    v = XMIN[t] - XMIN[a]; L = min(np.linalg.norm(v), R[a])
    seg = XMIN[a] + np.outer(np.linspace(0, L, 50), v / np.linalg.norm(v))
    ax.plot(*seg.T, color="#2e8b57", lw=4, alpha=0.7, solid_capstyle="butt", label=r"possible $\mathbf{x}_{syn}$",
            zorder=2)
    base(ax, safe_style=True)
    for i, p in enumerate(XMIN):
        if i == a:
            ax.annotate(r"anchor $\mathbf{x}_a$", p, xytext=(-40, -26), textcoords="offset points", fontsize=10,
                        fontweight="bold", bbox=BOX, zorder=4)
        else:
            lab = (r"terminal $\mathbf{x}_t$" + f"\n{pt[i]:.2f}") if i == t else f"{pt[i]:.2f}"
            ax.annotate(lab, p, xytext=(9, -6) if i == t else (6, 5), textcoords="offset points", fontsize=10,
                        linespacing=1.0, bbox=BOX, zorder=4)
    ax.annotate(r"$R_a$", XMIN[a] + np.array([-R[a] * 0.72, R[a] * 0.72]), xytext=(-14, 2),
                textcoords="offset points", fontsize=10, color=RED)
    finish(fig, ax, "sona_step3_dss", fontsize=9.5)
    return {"NEA": nea.tolist(), "PEA": pea.tolist(), "radii": np.round(R, 3).tolist()}
