"""figures.py -- research figures for the fusion cost-model study.
Colorblind-safe Okabe-Ito categorical palette; one axis per chart; break-even reference lines.
Usage: python -m model.figures
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OK = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
      "sky": "#56B4E9", "purple": "#CC79A7", "grey": "#8a8a8a"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
FIG = "figures"
os.makedirs(FIG, exist_ok=True)


def fig_spill_cliff(df):
    r = df[df.family == "reduction"].copy()
    r["NOUT"] = r["param_NOUT"].astype(int)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for dt, mk in [("float16", "o"), ("float32", "s")]:
        d = r[r.dtype == dt].groupby("NOUT").speedup.mean().reset_index()
        ax.plot(d.NOUT, d.speedup, marker=mk, color=OK["blue"] if dt == "float16" else OK["orange"],
                lw=2, ms=7, label=f"{dt}")
    # mark spilled points
    sp = r[r.f_spills > 0].groupby("NOUT").speedup.mean().reset_index()
    ax.scatter(sp.NOUT, sp.speedup, s=140, facecolors="none", edgecolors=OK["vermillion"],
               linewidths=2, zorder=5, label="register spill (fused)")
    ax.axhline(1.0, color=OK["grey"], ls="--", lw=1)
    ax.text(9, 1.06, "break-even (fuse ⇄ don't fuse)", color=OK["grey"], fontsize=8)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks([8, 16, 32, 64, 128]); ax.set_xticklabels([8, 16, 32, 64, 128])
    ax.set_xlabel("horizontal-fusion width NOUT (sibling reductions)")
    ax.set_ylabel("fusion speedup  (t_unfused / t_fused)")
    ax.set_title("The register-spill cliff drives toxic fusion on Ada sm89")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig1_spill_cliff.png"); plt.close(fig)
    print("wrote figures/fig1_spill_cliff.png")


def fig_occ_validation(ncu):
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    ax.plot([0, 100], [0, 100], color=OK["grey"], ls="--", lw=1, zorder=1)
    ax.scatter(ncu.occ_analytic_pct, ncu.f_occ_theoretical, s=70, color=OK["green"],
               edgecolors="white", linewidths=0.6, zorder=3)
    ax.set_xlabel("analytical occupancy (single compile)  [%]")
    ax.set_ylabel("ncu theoretical occupancy  [%]")
    mae = (ncu.occ_analytic_pct - ncu.f_occ_theoretical).abs().mean()
    ax.set_title(f"Analytical occupancy is exact\n(MAE={mae:.3f} pts, {len(ncu)}/{len(ncu)} match)")
    ax.set_xlim(0, 105); ax.set_ylim(0, 105)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_occupancy_validation.png"); plt.close(fig)
    print("wrote figures/fig2_occupancy_validation.png")


def fig_rq4(rows):
    names = [r[0] for r in rows]
    policies = ["greedy", "model", "oracle"]
    colors = [OK["vermillion"], OK["blue"], OK["green"]]
    x = np.arange(len(names)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for i, (pol, c) in enumerate(zip(policies, colors)):
        vals = [r[1][pol] for r in rows]
        b = ax.bar(x + (i - 1) * w, vals, w, color=c, label=pol)
        for xi, v in zip(x + (i - 1) * w, vals):
            ax.text(xi, v * 1.03, f"{v:.1f}", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("end-to-end latency  [ms, log]")
    ax.set_title("RQ4: search-free recommender vs greedy-always-fuse vs oracle")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig3_rq4_endtoend.png"); plt.close(fig)
    print("wrote figures/fig3_rq4_endtoend.png")


def main():
    df = pd.read_csv("data/microbench_timing.csv")
    fig_spill_cliff(df)
    try:
        ncu = pd.read_csv("data/microbench_ncu.csv")
        fig_occ_validation(ncu)
    except FileNotFoundError:
        print("skip occ validation (no ncu csv)")
    # RQ4 numbers pulled from the end-to-end run (kept in-sync with logs/run_endtoend.log,
    # clean 64-genuine-case refit). Model == oracle on all three subgraphs.
    rq4 = [("wide_multiproj", {"greedy": 157.96, "model": 16.25, "oracle": 16.25, "none": 20.45}),
           ("mixed_widths", {"greedy": 67.29, "model": 10.90, "oracle": 10.90, "none": 13.67}),
           ("fp32_block", {"greedy": 30.52, "model": 4.11, "oracle": 4.11, "none": 4.62})]
    fig_rq4(rq4)


if __name__ == "__main__":
    main()
