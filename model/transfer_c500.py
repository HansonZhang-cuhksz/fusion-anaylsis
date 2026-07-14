"""transfer_c500.py -- RQ3 cross-vendor transfer analysis: Ada sm89 -> MetaX C500.

Loads the Ada and C500 microbench timing datasets (same 72-case matrix), and:
  1. prints the cross-vendor reduction comparison (static spills / speedup / #toxic) and flags
     any DECISION-FLIP (a fusion beneficial on one vendor, toxic on the other);
  2. re-fits DeviceConstants on C500 (formulas frozen -- MODEL_SPEC section 7 -- constants only),
     saves model/c500_constants.json;
  3. reports RQ3 decision quality on C500 (re-parameterized model), plus a NO-adaptation baseline
     (Ada constants on C500 data) to show what re-parameterization does and does not buy;
  4. prints the documented decision-flip case with the model's per-device verdict.

Usage: python -m model.transfer_c500
"""
from __future__ import annotations
import json, warnings
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")
from model.fit import fit, decisions, prf, nondegenerate
from model.costmodel import DeviceConstants, decide

ADA_CSV, C500_CSV = "data/microbench_timing.csv", "data/microbench_timing_c500.csv"


def reduction_summary(df):
    return (df[df.family == "reduction"]
            .groupby(["param_NOUT", "dtype"])
            .agg(spill=("f_spills", "first"), sp=("speedup", "mean"),
                 tox=("beneficial", lambda s: int((s == 0).sum())), n=("beneficial", "size"))
            .reset_index().set_index(["param_NOUT", "dtype"]))


def main():
    ada, c500 = pd.read_csv(ADA_CSV), pd.read_csv(C500_CSV)
    ra, rc = reduction_summary(ada), reduction_summary(c500)

    print("=== Ada vs C500 reduction comparison (static spills / mean speedup / #toxic) ===")
    print(f"{'NOUT':>4} {'dt':>7} | {'Ada spill':>9} {'Ada sp':>6} {'tox':>5} | "
          f"{'C500 spill':>10} {'C500 sp':>7} {'tox':>5} | flip?")
    flips = []
    for idx in sorted(set(ra.index) & set(rc.index)):
        a, c = ra.loc[idx], rc.loc[idx]
        flip = (a.tox == 0 and c.tox == c.n) or (a.tox == a.n and c.tox == 0)
        if flip:
            flips.append(idx)
        tag = "  <== DECISION-FLIP" if flip else ("  (both toxic)" if a.tox and c.tox else "")
        print(f"{int(idx[0]):>4} {idx[1]:>7} | {int(a.spill):>9} {a.sp:>6.2f} {int(a.tox)}/{int(a.n)}  | "
              f"{int(c.spill):>10} {c.sp:>7.2f} {int(c.tox)}/{int(c.n)}  |{tag}")

    # ---- re-fit C500 constants (formulas frozen) ----
    kc, _ = fit(c500)
    kc.name = "metax_c500"
    json.dump(kc.as_dict(), open("model/c500_constants.json", "w"), indent=2)
    ka = DeviceConstants(**json.load(open("model/ada_constants.json")))
    print("\n=== C500 re-fit DeviceConstants (formulas unchanged; model/c500_constants.json) ===")
    print(f"  gamma_spill {ka.gamma_spill:.4f} -> {kc.gamma_spill:.4f} | "
          f"B_peak {ka.B_peak:.2e} -> {kc.B_peak:.2e} (~{kc.B_peak/1e12:.2f} TB/s) | "
          f"T_launch {kc.T_launch:.2e}")

    # ---- RQ3 decision quality on C500 ----
    c500e, nd = nondegenerate(c500)
    p, t = decisions(c500e, kc); m = prf(p, t)
    p2, t2 = decisions(c500e, ka); m2 = prf(p2, t2)
    print(f"\n=== RQ3: C500 decision quality ({len(c500e)} genuine cases; {nd} no-ops excluded) ===")
    print(f"  re-parameterized (C500 constants): P={m['precision']:.3f} R={m['recall']:.3f} "
          f"F1={m['f1']:.3f} acc={m['accuracy']:.3f}  (TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")
    print(f"  no-adaptation (Ada constants):     P={m2['precision']:.3f} R={m2['recall']:.3f} "
          f"F1={m2['f1']:.3f} acc={m2['accuracy']:.3f}")
    print("  -> the binary decision is spill-dominated, so both transfer; the FLIP is driven by the")
    print("     re-read hardware-specific spill count, not by re-fitting the constants.")

    # ---- the documented decision-flip ----
    print("\n=== DECISION-FLIP (same fusion, opposite verdict across vendors) ===")
    for idx in flips:
        nout, dt = int(idx[0]), idx[1]
        for name, df, k in [("Ada", ada, ka), ("C500", c500, kc)]:
            row = df[(df.family == "reduction") & (df.param_NOUT == nout) & (df.dtype == dt)].iloc[0]
            d = decide(row, k)
            verdict = "BENEFICIAL (fuse)" if d["pred_beneficial"] else "TOXIC (reject)"
            truth = "beneficial" if row.beneficial else "toxic"
            print(f"  redux NOUT={nout} {dt} on {name:5}: spills={int(row.f_spills):4d} -> "
                  f"model={verdict:18s} | measured {row.speedup:.2f} ({truth})")


if __name__ == "__main__":
    main()
