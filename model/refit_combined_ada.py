"""refit_combined_ada.py -- Ada counterpart of the LOG-06 C500 reread validation (Bucket-1 A1).

Re-fits DeviceConstants on the combined Ada set (pointwise + reduction + GEMM) and reports the
baseline (no-reread) vs reread-aware model, overall and per-family, plus the leave-one-family-out
anti-overfit check LOG-06 ran on the C500.

Usage: python -m model.refit_combined_ada [csv]
"""
from __future__ import annotations
import argparse, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from model.fit import fit, decisions, prf, nondegenerate


def score(df, k, label):
    p, t = decisions(df, k)
    return prf(p, t)


def by_family(df, k):
    out = {}
    for fam in sorted(df.family.unique()):
        d = df[df.family == fam]
        p, t = decisions(d, k)
        m = prf(p, t)
        out[fam] = (m["f1"], m["recall"], int(t.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default="data/microbench_ada_combined.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    dfe, n_deg = nondegenerate(df)
    print(f"[combined-ada] {len(df)} rows, {n_deg} degenerate excluded, {len(dfe)} genuine "
          f"({int((dfe.beneficial==0).sum())} toxic)")
    print(f"  toxic by family: {dfe[dfe.beneficial==0].groupby('family').size().to_dict()}")

    # --- baseline: reread ablated back to 1 (the LOG-05 failure mode) ---
    base = dfe.copy(); base["f_reread"] = 1.0; base["u_reread"] = 1.0
    kb, _ = fit(base, restarts=4)
    mb = score(base, kb, "baseline")
    fb = by_family(base, kb)

    # --- fix: taxonomy-derived reread-aware spill traffic ---
    kf, _ = fit(dfe, restarts=4)
    mf = score(dfe, kf, "reread")
    ff = by_family(dfe, kf)

    print("\n=== reread ablation on Ada (mirror of LOG-06 Table 2) ===")
    print(f"{'model':34s} {'overall F1':>10s} {'overall R':>10s} | per-family F1/R (n_toxic)")
    for name, m, f in [("baseline (no reread)", mb, fb), ("fix (reread-aware)", mf, ff)]:
        fam = "  ".join(f"{k}={v[0]:.3f}/{v[1]:.3f}(n={v[2]})" for k, v in f.items())
        print(f"{name:34s} {m['f1']:10.3f} {m['recall']:10.3f} | {fam}")
    print(f"\n  gamma_spill: baseline={kb.gamma_spill:.5f}  reread-fit={kf.gamma_spill:.5f}")

    # --- leave-one-family-out: fit WITHOUT any GEMM, score held-out GEMM ---
    tr = dfe[dfe.family != "gemm_epilogue"]
    te = dfe[dfe.family == "gemm_epilogue"]
    klofo, _ = fit(tr, restarts=4)
    p, t = decisions(te, klofo)
    m_lofo = prf(p, t)
    te1 = te.copy(); te1["f_reread"] = 1.0
    p1, t1 = decisions(te1, klofo)
    m_cf = prf(p1, t1)
    print("\n=== leave-one-family-out (fit on non-GEMM only; the LOG-06 anti-overfit test) ===")
    print(f"  held-out GEMM (reread=2): F1={m_lofo['f1']:.3f} recall={m_lofo['recall']:.3f} "
          f"(TP={m_lofo['tp']} FP={m_lofo['fp']} FN={m_lofo['fn']} TN={m_lofo['tn']}, "
          f"n_toxic={int(t.sum())})")
    print(f"  counterfactual reread=1 : F1={m_cf['f1']:.3f} recall={m_cf['recall']:.3f}")
    # This NOTE must ALWAYS print: on Ada every GEMM row has f_spills=0, so the reread column
    # multiplies zero and the two arms are an ARITHMETIC IDENTITY, not a comparison. (An earlier
    # version guarded this on n_toxic==0, which is 2 -- so the caveat never printed. LOG-10 s2.)
    n_spill_gemm = int((dfe[dfe.family == "gemm_epilogue"]["f_spills"] > 0).sum())
    if n_spill_gemm == 0:
        print("  ⚠ VACUOUS BY CONSTRUCTION, NOT A CONTROL: all 16 Ada GEMM rows have f_spills=0, so")
        print("    spill_factor(spills x reread) multiplies ZERO -- both arms optimise a bit-identical")
        print("    objective. Identical F1/gamma is an arithmetic identity: it is NO evidence the reread")
        print("    fix is correct, and equally none that it is harmless (it has no path to any decision).")
        print("    A control must be able to fail; this one cannot. The fix's evidence is the C500 alone")
        print("    (LOG-06: held-out GEMM recall 1.0 w/ reread=2 vs 0.0 w/ reread=1).")
        print(f"  ⚠ The 'GEMM F1/recall={m_lofo['f1']:.3f}/{m_lofo['recall']:.3f} (n_toxic={int(t.sum())})' above is NOT a")
        print("    meaningful score: both 'toxic' labels are single-shot timing noise that 20-round CIs")
        print("    overturn (0.9904 CI [0.964,1.055] ambiguous; 1.0250 CI [1.025,1.029] BENEFICIAL), so")
        print("    those FNs are the model disagreeing with a wrong label. See LOG-10 s2.")

    json.dump(kf.as_dict(), open("model/ada_combined_constants.json", "w"), indent=2)
    print("\n[combined-ada] wrote model/ada_combined_constants.json")


if __name__ == "__main__":
    main()
