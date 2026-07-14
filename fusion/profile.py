"""profile.py -- run the ncu ground-truth pass over a subset and build the profiled dataset.

For each subset case: profile the fused and unfused plans, aggregate concept metrics, derive the
ground-truth `dominant_penalty` (spill/occupancy vs layout) and validate the analytical occupancy
model against ncu's THEORETICAL occupancy. Ground truth is used only for validation (RQ2), never
inside the deployed decision.

Usage: python -m fusion.profile data/microbench_ncu.csv
"""
from __future__ import annotations
import sys, csv, time, traceback
from . import matrix
from .ncu import profile_plan, aggregate
from .kernels import pointwise, reduction
from .static import from_triton
from .hw import ADA_SM89
import torch

FAMILIES = {"pointwise": pointwise, "reduction": reduction}


def static_for(family, params):
    """Rebuild the case (with compile probe) to read single-compile static inputs of both plans."""
    p = dict(params); dt = getattr(torch, p.pop("dtype"))
    case = FAMILIES[family].make_case(dtype=dt, **p)
    sf = from_triton(case.fused_kernels[0])
    su = from_triton(case.unfused_kernels[-1])
    return case, sf, su


def dominant_penalty(fused_agg: dict) -> str:
    """Ground-truth attribution from the fused kernel's profile.
    spill (local-memory) traffic => P_occ(spill); high bank conflicts w/o spill => P_layout."""
    spill = fused_agg.get("sig_local_bytes", 0.0)
    dram = max(1.0, fused_agg.get("dram_bytes_total", 1.0))
    bank = fused_agg.get("sig_bank_conf", 0.0)
    # normalise: spill traffic relative to useful DRAM; bank conflicts relative to a scale.
    spill_ratio = spill / dram
    bank_ratio = bank / dram          # conflicts per useful byte (rough scale)
    if spill_ratio > 0.25:
        return "spill"
    if bank_ratio > 5.0:
        return "layout"
    return "none"


def main(out_csv: str):
    cases = list(matrix.ncu_subset())
    print(f"[profile] {len(cases)} cases x 2 plans via ncu -> {out_csv}", flush=True)
    rows, t0 = [], time.time()
    for i, (family, params) in enumerate(cases):
        try:
            case, sf, su = static_for(family, params)
            fk = aggregate(profile_plan(family, "fused", params))
            uk = aggregate(profile_plan(family, "unfused", params))
            occ_val_err = abs(sf.occupancy * 100 - fk.get("sig_occ_theoretical", 0.0))
            row = {
                "family": family, "op_pair": case.op_pair, "dtype": params["dtype"],
                "R": params["R"], "C": params["C"],
                **{f"param_{k}": v for k, v in params.items() if k not in ("R", "C", "dtype")},
                # static (single-compile)
                "f_regs": sf.n_regs, "f_spills": sf.n_spills, "f_occ_analytic": round(sf.occupancy, 4),
                "u_regs": su.n_regs, "u_occ_analytic": round(su.occupancy, 4),
                # ncu fused
                "f_occ_theoretical": fk.get("sig_occ_theoretical", 0.0),
                "f_occ_achieved": fk.get("sig_occ_achieved", 0.0),
                "f_regs_ncu": fk.get("sig_regs", 0.0),
                "f_spill_bytes": fk.get("sig_local_bytes", 0.0),
                "f_bank_conf": fk.get("sig_bank_conf", 0.0),
                "f_dram_bytes": fk.get("dram_bytes_total", 0.0),
                "f_dur_us": fk.get("dur_us_total", 0.0),
                # ncu unfused
                "u_spill_bytes": uk.get("local_bytes_total", 0.0),
                "u_bank_conf": uk.get("bank_conf_total", 0.0),
                "u_dram_bytes": uk.get("dram_bytes_total", 0.0),
                # occupancy-model validation
                "occ_analytic_pct": round(sf.occupancy * 100, 2),
                "occ_val_abs_err": round(occ_val_err, 2),
                # attribution ground truth
                "dominant_penalty": dominant_penalty(fk),
            }
            rows.append(row)
            print(f"[{i+1:2d}/{len(cases)}] {case.op_pair:20s} {params['dtype']:7s} "
                  f"occ(an={row['occ_analytic_pct']:.0f} ncu_th={row['f_occ_theoretical']:.0f}) "
                  f"spill={row['f_spill_bytes']:.2g} bank={row['f_bank_conf']:.2g} "
                  f"dom={row['dominant_penalty']}", flush=True)
        except Exception as e:
            print(f"[{i+1:2d}/{len(cases)}] SKIP {family}/{params}: {type(e).__name__}: {str(e)[:90]}",
                  flush=True)
            traceback.print_exc()
    if rows:
        cols = list(rows[0].keys())
        for r in rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in rows: w.writerow(r)
    print(f"[profile] wrote {len(rows)} rows in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/microbench_ncu.csv")
