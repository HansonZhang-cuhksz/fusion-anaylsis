"""inductor_baseline.py -- real-compiler (torch.compile / Inductor) fusion baseline on C500 (G4).

The prior microbenchmarks use hand-written Triton kernels vs a greedy/oracle baseline. This adds the
REAL compiler baseline the proposal names: does a production fusing compiler (Inductor) actually help,
and in which regime? We sweep two real subgraphs from memory-bound to compute-bound and measure the
fusion benefit (eager multi-kernel vs torch.compile single fused kernel):

  - `chain`  : a memory-bound pointwise+residual chain (gelu -> affine -> gated -> +residual). Inductor
               fuses the whole chain into ONE kernel; eager launches ~5. Fusion should WIN here (saves
               HBM round-trips) -- the regime the search-free model flags beneficial.
  - `mlp`    : a real transformer MLP-FFN (Linear -> GELU -> Linear). GEMM-dominated; the epilogue-fusion
               win is small (the roofline says compute-bound -> fusion ~1x), which the model predicts.

Run one slice on one device:
  MACA_VISIBLE_DEVICES=<g> CUDA_VISIBLE_DEVICES=<g> python -m fusion.inductor_baseline --configs 0,1 --out data/x.csv
"""
from __future__ import annotations
import sys, csv, time, argparse, warnings
import torch, torch.nn.functional as F
warnings.filterwarnings("ignore")


def chain(x, a, b, res):
    y = F.gelu(x)
    y = y * a + b
    y = y * torch.sigmoid(y)
    return y + res


def mlp(x, w1, b1, w2, b2):
    return F.linear(F.gelu(F.linear(x, w1, b1)), w2, b2)


# (kind, M, N, K)  K used only by mlp (hidden dim); memory-bound chain -> compute-bound mlp.
CONFIGS = [("chain", 2048, 2048, 0), ("chain", 4096, 4096, 0), ("chain", 8192, 8192, 0),
           ("chain", 8192, 2048, 0), ("mlp", 2048, 1024, 1024), ("mlp", 2048, 1024, 4096),
           ("mlp", 4096, 1024, 4096), ("mlp", 2048, 1024, 16384)]


def _time(fn, it=100):
    for _ in range(20):
        fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(it):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / it * 1e3


def run_one(cfg):
    kind, M, N, K = cfg
    dt = torch.float16
    if kind == "chain":
        x = torch.randn(M, N, device="cuda", dtype=dt)
        a = torch.randn(N, device="cuda", dtype=dt); b = torch.randn(N, device="cuda", dtype=dt)
        res = torch.randn(M, N, device="cuda", dtype=dt)
        eager = lambda: chain(x, a, b, res)
        cc = torch.compile(chain); comp = lambda: cc(x, a, b, res)
        bytes_rw = 3 * M * N * 2   # ~read x+res, write y (memory-bound estimate)
        flops = 8 * M * N          # a few pointwise ops
    else:
        H = N  # model dim
        x = torch.randn(M, H, device="cuda", dtype=dt)
        w1 = torch.randn(K, H, device="cuda", dtype=dt); b1 = torch.randn(K, device="cuda", dtype=dt)
        w2 = torch.randn(H, K, device="cuda", dtype=dt); b2 = torch.randn(H, device="cuda", dtype=dt)
        eager = lambda: mlp(x, w1, b1, w2, b2)
        cc = torch.compile(mlp); comp = lambda: cc(x, w1, b1, w2, b2)
        bytes_rw = (M * H + H * K + K * H + M * K) * 2
        flops = 2 * M * H * K + 2 * M * K * H
    ref = eager(); out = comp(); torch.cuda.synchronize()
    err = (out - ref).abs().max().item()
    te, tc = _time(eager), _time(comp)
    ai = flops / max(1, bytes_rw)   # arithmetic intensity (low=memory-bound, high=compute-bound)
    return {"kind": kind, "M": M, "N": N, "K": K, "arith_intensity": round(ai, 2),
            "maxerr": round(err, 4), "eager_ms": round(te, 4), "compiled_ms": round(tc, 4),
            "fusion_speedup": round(te / tc, 3), "regime": "memory" if ai < 5 else "compute"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--configs", default="all"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    idx = range(len(CONFIGS)) if a.configs == "all" else [int(x) for x in a.configs.split(",")]
    rows = []
    for i in idx:
        try:
            r = run_one(CONFIGS[i]); r["idx"] = i; rows.append(r)
            print(f"[{i}] {r['kind']:6s} M{r['M']}N{r['N']}K{r['K']} AI={r['arith_intensity']:.1f}({r['regime']}): "
                  f"eager={r['eager_ms']:.3f} compiled={r['compiled_ms']:.3f} fusion_speedup={r['fusion_speedup']}x", flush=True)
        except Exception as e:
            print(f"[{i}] SKIP {CONFIGS[i]}: {type(e).__name__}: {str(e)[:100]}", flush=True)
    if rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
