"""bootstrap_ci.py -- 95% CI on the Ada RQ1 decision quality (Bucket-1 task A2).

LOG-09 put CIs on the *timing* claims; this puts one on the *decision* metric. The point estimate
F1 comes from a single fit on N genuine cases; with N~64 a lone borderline case moves F1 a lot, so we
bootstrap: resample the genuine cases with replacement B times, recompute precision/recall/F1 each
time (constants held at the committed fit -- we are quantifying the sampling error of the METRIC, not
re-fitting per resample), and report the percentile CI. Also reports the held-out fold spread.

Usage: python -m model.bootstrap_ci [csv] [--B 1000]
"""
from __future__ import annotations
import argparse, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from model.costmodel import DeviceConstants
from model.fit import decisions, prf, nondegenerate


def load_k(path):
    return DeviceConstants(**json.load(open(path)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="data/microbench_timing.csv")
    ap.add_argument("--constants", default="model/ada_constants.json")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    dfe, n_deg = nondegenerate(df)
    k = load_k(a.constants)
    pred, true = decisions(dfe, k)
    point = prf(pred, true)
    print(f"[bootstrap] {a.csv}: {len(df)} rows, {n_deg} degenerate excluded, "
          f"{len(dfe)} genuine ({int(true.sum())} toxic)")
    print(f"[point]  precision={point['precision']:.3f} recall={point['recall']:.3f} "
          f"F1={point['f1']:.3f} acc={point['accuracy']:.3f}")

    rng = np.random.default_rng(a.seed)
    n = len(pred)
    f1s, precs, recs, accs = [], [], [], []
    for _ in range(a.B):
        idx = rng.integers(0, n, n)                 # resample cases with replacement
        m = prf(pred[idx], true[idx])
        f1s.append(m["f1"]); precs.append(m["precision"]); recs.append(m["recall"]); accs.append(m["accuracy"])

    def ci(v, label):
        v = np.array(v); lo, hi = np.percentile(v, [2.5, 97.5])
        print(f"[boot]   {label:9s} = {np.median(v):.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        return float(np.median(v)), float(lo), float(hi)

    print(f"[bootstrap] B={a.B} resamples of the {n} genuine cases:")
    f1m, f1lo, f1hi = ci(f1s, "F1")
    pm, plo, phi = ci(precs, "precision")
    rm, rlo, rhi = ci(recs, "recall")
    am, alo, ahi = ci(accs, "accuracy")

    # greedy baseline for reference (never predicts toxic -> F1 0 by construction)
    g = prf(np.zeros(n, dtype=int), true)
    print(f"[baseline] greedy-always-fuse F1={g['f1']:.3f} acc={g['accuracy']:.3f}")

    if a.out:
        pd.DataFrame([{
            "metric": mname, "point": point[mname], "boot_median": med, "ci_lo": lo, "ci_hi": hi,
            "B": a.B, "n_genuine": n, "n_toxic": int(true.sum()),
        } for mname, (med, lo, hi) in [("f1", (f1m, f1lo, f1hi)), ("precision", (pm, plo, phi)),
                                       ("recall", (rm, rlo, rhi)), ("accuracy", (am, alo, ahi))]
        ]).to_csv(a.out, index=False)
        print(f"[bootstrap] wrote {a.out}")


if __name__ == "__main__":
    main()
