"""rq2.py -- RQ2 analysis on the ncu ground-truth: (a) validate the analytical occupancy model
against ncu theoretical occupancy, (b) attribution accuracy of the model's dominant-penalty vs the
profiled dominant penalty.

Usage: python -m model.rq2 data/microbench_ncu.csv
"""
from __future__ import annotations
import sys, json
import numpy as np
import pandas as pd
from model.costmodel import DeviceConstants, decide


def load_k(path="model/ada_constants.json"):
    try:
        return DeviceConstants(**json.load(open(path)))
    except FileNotFoundError:
        return DeviceConstants()


def main(csv):
    df = pd.read_csv(csv)
    print(f"[rq2] {len(df)} profiled cases")

    # ---- (a) occupancy-model validation: analytic vs ncu theoretical ----
    d = df.dropna(subset=["f_occ_theoretical"])
    err = (d["occ_analytic_pct"] - d["f_occ_theoretical"]).abs()
    print("\n=== RQ2a: analytical occupancy vs ncu THEORETICAL occupancy ===")
    print(f"  MAE = {err.mean():.3f} pct-points | max = {err.max():.3f} | "
          f"exact matches = {(err < 1.0).sum()}/{len(d)}")
    print(d[["op_pair", "dtype", "occ_analytic_pct", "f_occ_theoretical", "occ_val_abs_err"]]
          .to_string(index=False, max_rows=40))

    # ---- (b) attribution accuracy: model dominant penalty vs profiled dominant penalty ----
    k = load_k()
    # map model's fine labels to the ground-truth vocabulary {spill, layout, none}
    def model_dom(row):
        nout = row.get("param_NOUT", np.nan)
        nout = 1 if (nout is None or (isinstance(nout, float) and np.isnan(nout))) else int(nout)
        n_launch = max(1, nout // 16)
        r = {
            "flops": nout * row["R"] * row["C"] * 2,
            "bytes_fused": row["R"] * row["C"] * 2,
            "bytes_unfused": row["R"] * row["C"] * 2 * n_launch,
            "f_occ": row["f_occ_analytic"], "f_spills": row["f_spills"],
            "u_occ": row["u_occ_analytic"], "u_spills": 0,
            "n_launches_unfused": n_launch,
        }
        dd = decide(r, k)["dominant_penalty"]
        return "spill" if dd in ("spill", "occupancy") else dd  # collapse occupancy->spill branch

    df["model_dom"] = df.apply(model_dom, axis=1)
    # only score the toxic/degraded cases where a dominant penalty is defined
    scored = df[df["dominant_penalty"] != "none"].copy()
    if len(scored):
        match = (scored["model_dom"] == scored["dominant_penalty"]).mean()
        print("\n=== RQ2b: attribution accuracy (cases with a profiled dominant penalty) ===")
        print(f"  cases = {len(scored)} | model dominant == profiled dominant: {match*100:.1f}%")
        print(scored[["op_pair", "dtype", "f_spills", "f_spill_bytes", "f_bank_conf",
                      "dominant_penalty", "model_dom"]].to_string(index=False, max_rows=40))
    else:
        print("\n[rq2b] no toxic cases with a defined dominant penalty in this subset")

    # spill separability: do spilled kernels have orders-more local traffic?
    print("\n=== spill separability (P_occ ground truth) ===")
    g = df.groupby(df["f_spills"] > 0)["f_spill_bytes"].agg(["mean", "max", "count"])
    print(g.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/microbench_ncu.csv")
