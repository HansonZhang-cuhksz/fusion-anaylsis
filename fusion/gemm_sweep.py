"""gemm_sweep.py -- sweep the CONTRACTION (GEMM-epilogue) family on one GPU.

Probes the compute-bound / occupancy regime (TODO G2/G4): vary shape (memory- vs compute-bound via K)
and tile size (BM,BN -> register footprint -> occupancy, WITHOUT spilling) to see whether epilogue
fusion is ever net-toxic for a non-spill reason. Emits timing + single-compile static inputs.

Run one slice on one device:
  MACA_VISIBLE_DEVICES=<g> CUDA_VISIBLE_DEVICES=<g> FUSION_HW=c500 \
      python -m fusion.gemm_sweep --configs 0,1,2,3 --out data/gemm_slice_<g>.csv
"""
from __future__ import annotations
import sys, csv, argparse, warnings
import torch
warnings.filterwarnings("ignore")
from fusion.kernels import gemm_epilogue as G
from fusion.static import from_triton
from fusion.timing import time_ms

# (M, N, K) x (BM, BN) x dtype ; K small=memory-bound, large=compute-bound; big tile=low occupancy.
_SHAPES = [(2048, 2048, 512), (2048, 2048, 2048), (4096, 4096, 512), (1024, 1024, 4096)]
_TILES = [(64, 64), (128, 128)]
_DTYPES = ["float16", "float32"]
CONFIGS = [(M, N, K, BM, BN, dt) for (M, N, K) in _SHAPES for (BM, BN) in _TILES for dt in _DTYPES]


def run_one(cfg):
    M, N, K, BM, BN, dtn = cfg
    # Release torch's cached blocks first: if free VRAM hits ~0, a SPILLING kernel's local-memory
    # backing store falls back to host memory and the fused time inflates ~10x (LOG-10). Ada's GEMM
    # tiles do not spill, but the C500's 128x128 tiles do (f_spills=205) -- keep every device honest.
    torch.cuda.empty_cache()
    case = G.make_case(M=M, N=N, K=K, dtype=getattr(torch, dtn), BM=BM, BN=BN, BK=32)
    diff = case.check()
    sf = from_triton(case.fused_kernels[0])
    tf = time_ms(case.run_fused, warmup=5, iters=20)["ms"]
    tu = time_ms(case.run_unfused, warmup=5, iters=20)["ms"]
    return {"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "dtype": dtn,
            "f_regs": sf.n_regs, "f_spills": sf.n_spills, "f_occ": round(sf.occupancy, 3),
            "t_fused_ms": round(tf, 4), "t_unfused_ms": round(tu, 4),
            "speedup": round(tu / tf, 3), "beneficial": int(tf < tu), "maxdiff": f"{diff:.2g}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    idx = range(len(CONFIGS)) if a.configs == "all" else [int(x) for x in a.configs.split(",")]
    rows = []
    for i in idx:
        try:
            r = run_one(CONFIGS[i]); r["idx"] = i; rows.append(r)
            print(f"[{i}] {r['dtype']:7s} M{r['M']}N{r['N']}K{r['K']} BM{r['BM']}BN{r['BN']}: "
                  f"regs={r['f_regs']} spill={r['f_spills']} occ={r['f_occ']} "
                  f"sp={r['speedup']}x ben={r['beneficial']}", flush=True)
        except Exception as e:
            print(f"[{i}] SKIP {CONFIGS[i]}: {type(e).__name__}: {str(e)[:90]}", flush=True)
    if rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
