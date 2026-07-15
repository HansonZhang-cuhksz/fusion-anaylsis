"""build_combined_ada.py -- assemble the combined ADA training set (reductions + pointwise + GEMM)
with the taxonomy-derived `f_reread` spill-traffic feature. Ada counterpart of
`model/build_combined_c500.py` (LOG-06), for the cross-vendor GEMM comparison (Bucket-1 task A1).

Difference from the C500 script: the unfused-GEMM static spill count is **derived from a real Ada
compile probe** (`make_case(compile_probe=True).unfused_kernels[0]`, i.e. the FUSE=0 specialization)
rather than hardcoded. The C500 script hardcodes a verified U_SPILL dict; those numbers are
C500-specific (64-wide wavefronts, 128K reg/CU) and would be wrong on Ada, so we read Ada's own
single-compile report — which is exactly the search-free input the deployed pass is allowed to use.

Usage: python -m model.build_combined_ada     (needs the GPU: it compiles each GEMM config once)
"""
from __future__ import annotations
import warnings
import pandas as pd, numpy as np, torch
warnings.filterwarnings("ignore")
from fusion.kernels import gemm_epilogue as G
from fusion.kernels.base import spill_reread, CONTRACTION, POINTWISE
from fusion.static import from_triton

RED_CSV, GEMM_CSV, OUT = ("data/microbench_timing.csv",
                          "data/microbench_gemm_ada.csv",
                          "data/microbench_ada_combined.csv")


def _probe_unfused(M, N, K, BM, BN, dt) -> tuple[int, int, float]:
    """Compile the FUSE=0 (unfused GEMM) specialization on Ada and read its static report."""
    case = G.make_case(M=M, N=N, K=K, dtype=getattr(torch, dt), BM=BM, BN=BN, BK=32,
                       compile_probe=True)
    su = from_triton(case.unfused_kernels[0])
    del case
    torch.cuda.empty_cache()
    return su.n_regs, su.n_spills, round(su.occupancy, 4)


def gemm_rows() -> pd.DataFrame:
    g = pd.read_csv(GEMM_CSV)
    rows, cache = [], {}
    for _, r in g.iterrows():
        M, N, K, BM, BN, dt = int(r.M), int(r.N), int(r.K), int(r.BM), int(r.BN), r["dtype"]
        key = (M, N, K, BM, BN, dt)
        if key not in cache:
            cache[key] = _probe_unfused(*key)
        u_regs, u_spills, u_occ = cache[key]
        isz = 2 if dt == "float16" else 4
        rows.append({
            "family": "gemm_epilogue", "op_pair": f"gemm_epi_{M}x{N}x{K}",
            "producer_class": CONTRACTION, "consumer_class": POINTWISE, "dtype": dt,
            "R": M, "C": N, "param_NOUT": np.nan, "param_M": M, "param_N": N, "param_K": K,
            "f_regs": int(r.f_regs), "f_spills": int(r.f_spills), "f_occ": float(r.f_occ),
            "u_regs": u_regs, "u_spills": u_spills, "u_occ": u_occ,
            "n_launches_unfused": 2,
            "bytes_fused": (M * K + K * N) * isz + N * 4 + M * N * 4,
            "bytes_unfused": (M * K + K * N) * isz + N * 4 + M * N * 4 + 2 * M * N * 4,
            "flops": 2 * M * N * K,
            "t_fused_ms": float(r.t_fused_ms), "t_unfused_ms": float(r.t_unfused_ms),
            "speedup": float(r.speedup), "beneficial": int(r.beneficial),
            "f_reread": spill_reread(CONTRACTION, POINTWISE), "u_reread": 1,
        })
    return pd.DataFrame(rows)


def main():
    red = pd.read_csv(RED_CSV)
    red["f_reread"] = [spill_reread(p, c) for p, c in zip(red.producer_class, red.consumer_class)]
    red["u_reread"] = 1
    gem = gemm_rows()
    combined = pd.concat([red, gem], ignore_index=True)
    combined.to_csv(OUT, index=False)
    print(f"[combined-ada] {len(red)} reduction/pointwise + {len(gem)} GEMM = {len(combined)} rows -> {OUT}")
    print("  f_reread by family:")
    print(combined.groupby("family")["f_reread"].agg(["first", "size"]).to_string())
    print(f"  toxic by family: {combined[combined.beneficial==0].groupby('family').size().to_dict()}")
    print("\n  Ada GEMM static (derived from this machine's compile probe):")
    print(gem[["param_M", "param_N", "param_K", "dtype", "f_regs", "f_spills",
               "u_regs", "u_spills", "speedup", "beneficial"]].to_string(index=False))


if __name__ == "__main__":
    main()
