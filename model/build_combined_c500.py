"""build_combined_c500.py -- assemble the combined C500 training set (reductions + pointwise + GEMM)
with the taxonomy-derived `f_reread` spill-traffic feature, for the re-fit that fixes the GEMM blind
spot (LOG-05 / TODO G2).

The reduction+pointwise rows come from data/microbench_timing_c500.csv (full schema). The GEMM rows
are assembled from data/microbench_gemm_c500.csv (timing + fused static) plus:
  - u_spills: the unfused-GEMM static spill count (deterministic ptxas; verified in the root-cause
    compile pass, LOG-05): (128x128,fp16)=118, (128x128,fp32)=234, else 0;
  - bytes/flops/n_launches from the GEMM-epilogue Case formulas;
  - f_reread = spill_reread(CONTRACTION, POINTWISE) = 2 (epilogue re-reads the spilled accumulator).

Usage: python -m model.build_combined_c500
"""
from __future__ import annotations
import pandas as pd, numpy as np
from fusion.kernels.base import spill_reread, CONTRACTION, POINTWISE

RED_CSV, GEMM_CSV, OUT = ("data/microbench_timing_c500.csv",
                          "data/microbench_gemm_c500.csv",
                          "data/microbench_c500_combined.csv")
# unfused-GEMM (FUSE=0) spill count, from the verified compile pass (ptxas-deterministic).
U_SPILL = {(64, 64, "float16"): 0, (64, 64, "float32"): 0,
           (128, 128, "float16"): 118, (128, 128, "float32"): 234}


def gemm_rows():
    g = pd.read_csv(GEMM_CSV)
    rows = []
    for _, r in g.iterrows():
        M, N, K, BM, BN, dt = int(r.M), int(r.N), int(r.K), int(r.BM), int(r.BN), r["dtype"]
        isz = 2 if dt == "float16" else 4
        rows.append({
            "family": "gemm_epilogue", "op_pair": f"gemm_epi_{M}x{N}x{K}",
            "producer_class": CONTRACTION, "consumer_class": POINTWISE, "dtype": dt,
            "R": M, "C": N, "param_NOUT": np.nan, "param_M": M, "param_N": N, "param_K": K,
            "f_regs": int(r.f_regs), "f_spills": int(r.f_spills), "f_occ": float(r.f_occ),
            "u_regs": int(r.f_regs), "u_spills": U_SPILL[(BM, BN, dt)], "u_occ": float(r.f_occ),
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
    print(f"[combined] {len(red)} reduction/pointwise + {len(gem)} GEMM = {len(combined)} rows -> {OUT}")
    print("  f_reread distribution by family:")
    print(combined.groupby("family")["f_reread"].agg(["first", "size"]).to_string())
    print(f"  toxic by family: "
          f"{combined[combined.beneficial==0].groupby('family').size().to_dict()}")


if __name__ == "__main__":
    main()
