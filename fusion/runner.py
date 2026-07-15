"""runner.py -- build every case in the matrix, check correctness, extract single-compile static
inputs, time fused vs unfused, and emit the timing+static dataset (one CSV row per case).

This is the cheap pass (no ncu): it produces the `beneficial?` ground-truth label and all static
features the search-free model is allowed to use. The expensive ncu ground truth (attribution +
occupancy validation) is added by profile.py over a subset.
"""
from __future__ import annotations
import sys, csv, time, traceback
import torch
from .kernels import pointwise, reduction
from .static import from_triton
from .timing import time_ms
from .hw import ADA_SM89
from . import matrix

FAMILIES = {"pointwise": pointwise, "reduction": reduction}


def build(family: str, params: dict):
    p = dict(params)
    dtype = getattr(torch, p.pop("dtype"))
    return FAMILIES[family].make_case(dtype=dtype, **params_wo_dtype(params)), dtype


def params_wo_dtype(params: dict) -> dict:
    return {k: v for k, v in params.items() if k != "dtype"}


def adaptive_iters(fn) -> tuple[int, int]:
    """Pick (warmup, iters) so the timing loop stays ~1-2s even for slow (spilling) kernels."""
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    if dt > 0.03:   return 5, 12
    if dt > 0.005:  return 8, 30
    return 12, 60


def run_case(family: str, params: dict) -> dict:
    # PROPHYLACTIC (LOG-10 s1): keep torch's caching allocator from growing its RESERVED footprint past
    # physical VRAM. When reserved oversubscribes physical (observed 7.8-8.1 GB on this 8 GB card),
    # WSL2/WDDM permits it rather than failing, and the context's local-memory backing store can be left
    # host-resident -- every spill then crosses PCIe, inflating SPILLING kernels only (non-spilling are
    # untouched in ratio) and fabricating a too-severe "spill cliff".
    # Honest scope: the trigger is oversubscription, NOT "free VRAM ~ 0" (free=0 with reserved BELOW
    # physical is harmless). empty_cache() cannot REPAIR an already-poisoned context, so this is
    # prevention, not a cure, and its efficacy is a design argument -- a direct A/B on a process that
    # never oversubscribed showed no change. See tooling/repro_vram_artifact.py.
    torch.cuda.empty_cache()
    case, dtype = build(family, params)
    diff = case.check()
    sf = from_triton(case.fused_kernels[0], name="fused")
    su = from_triton(case.unfused_kernels[-1], name="unfused_sig")  # signature unfused kernel
    w, it = adaptive_iters(case.run_fused)
    tf = time_ms(case.run_fused, warmup=w, iters=it)
    w, it = adaptive_iters(case.run_unfused)
    tu = time_ms(case.run_unfused, warmup=w, iters=it)
    beneficial = tf["ms"] < tu["ms"]
    ai = case.flops / max(1, case.bytes_moved_fused)
    return {
        "family": family, "op_pair": case.op_pair,
        "producer_class": case.producer_class, "consumer_class": case.consumer_class,
        "dtype": str(dtype).replace("torch.", ""),
        "R": case.shape.get("R"), "C": case.shape.get("C"),
        **{f"param_{k}": v for k, v in case.params.items()},
        # ---- single-compile static inputs (fused) ----
        "f_regs": sf.n_regs, "f_spills": sf.n_spills, "f_smem": sf.shared_bytes,
        "f_threads": sf.threads_per_block, "f_occ": round(sf.occupancy, 4),
        "f_occ_binder": sf.occ_binder, "f_spilled": int(sf.spilled),
        # ---- static inputs (unfused signature kernel) ----
        "u_regs": su.n_regs, "u_spills": su.n_spills, "u_smem": su.shared_bytes,
        "u_occ": round(su.occupancy, 4), "u_spilled": int(su.spilled),
        "n_launches_unfused": case.n_launches_unfused,
        # ---- analytic memory / intensity ----
        "bytes_fused": case.bytes_moved_fused, "bytes_unfused": case.bytes_moved_unfused,
        "flops": case.flops, "arith_intensity": round(ai, 4),
        # ---- measured ground truth (label) ----
        "t_fused_ms": round(tf["ms"], 5), "t_unfused_ms": round(tu["ms"], 5),
        "t_fused_min": round(tf["ms_min"], 5), "t_unfused_min": round(tu["ms_min"], 5),
        "speedup": round(tu["ms"] / tf["ms"], 4),
        "beneficial": int(beneficial),
        "maxdiff": f"{diff:.2e}",
    }


def main(out_csv: str):
    cases = list(matrix.all_cases())
    print(f"[runner] {len(cases)} cases -> {out_csv}", flush=True)
    rows, t_start = [], time.time()
    for i, (family, params) in enumerate(cases):
        try:
            row = run_case(family, params)
            rows.append(row)
            print(f"[{i+1:3d}/{len(cases)}] {row['op_pair']:22s} {row['dtype']:7s} "
                  f"R{row['R']}xC{row['C']} regs={row['f_regs']:3d} spill={row['f_spills']:4d} "
                  f"occ={row['f_occ']:.3f} sp={row['speedup']:5.2f}x ben={row['beneficial']}",
                  flush=True)
        except Exception as e:
            print(f"[{i+1:3d}/{len(cases)}] SKIP {family}/{params}: {type(e).__name__}: "
                  f"{str(e)[:100]}", flush=True)
            traceback.print_exc()
    if rows:
        keys = list({k for r in rows for k in r})
        # stable column order: put the common ones first
        head = ["family","op_pair","producer_class","consumer_class","dtype","R","C"]
        cols = head + [k for k in sorted(keys) if k not in head]
        with open(out_csv, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=cols)
            wtr.writeheader()
            for r in rows:
                wtr.writerow(r)
    print(f"[runner] wrote {len(rows)} rows in {time.time()-t_start:.1f}s", flush=True)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "data/microbench_timing.csv"
    main(out)
