"""inductor_predict.py -- can the search-free model predict a REAL compiler's fusion outcomes? (G4++)

We hook `triton.compile` to capture the single-compile static report (n_regs / n_spills) of the fused
Triton kernel that torch.compile/Inductor actually generates for an elementwise subgraph, feed it +
the plan's analytic bytes to the model, and compare the MODEL's predicted fusion speedup to Inductor's
MEASURED speedup (eager multi-kernel vs the fused kernel). This closes the LOG-07 caveat: the model
reads a production compiler's own kernels and predicts whether its fusion pays off.

Elementwise chains only (Inductor emits ONE fused Triton kernel; the fusion IS a real memory-traffic
decision). GEMM-dominated blocks route to the vendor GEMM, so the "fused kernel" isn't Inductor's to
decide -- excluded here (see LOG-07).

Usage: MACA_VISIBLE_DEVICES=<g> FUSION_HW=c500 python -m fusion.inductor_predict
"""
from __future__ import annotations
import json, warnings
import torch, torch.nn.functional as F, triton
warnings.filterwarnings("ignore")
from fusion.static import from_triton
from fusion.timing import time_ms
from model.costmodel import DeviceConstants, predict_time

_CAUGHT = []
_ORIG = triton.compile
def _hook(*a, **k):
    ck = _ORIG(*a, **k); _CAUGHT.append(ck); return ck
triton.compile = _hook

K = DeviceConstants(**json.load(open("model/c500_combined_constants.json")))


def chain(x, a, b, res):          # ~5 pointwise ops: gelu, mul, add, sigmoid-gate, add
    y = F.gelu(x); y = y * a + b; y = y * torch.sigmoid(y); return y + res
N_OPS = 5


def run(M, N):
    dt = torch.float16
    x = torch.randn(M, N, device="cuda", dtype=dt); res = torch.randn(M, N, device="cuda", dtype=dt)
    a = torch.randn(N, device="cuda", dtype=dt); b = torch.randn(N, device="cuda", dtype=dt)
    _CAUGHT.clear()
    cc = torch.compile(chain, dynamic=False)
    for _ in range(3):
        cc(x, a, b, res)
    torch.cuda.synchronize()
    cands = [c for c in _CAUGHT if getattr(c, "n_regs", None) is not None]
    assert cands, f"no Inductor kernel captured with n_regs (len(_CAUGHT)={len(_CAUGHT)})"
    fused = max(cands, key=lambda c: c.n_regs)              # the main fused compute kernel
    si = from_triton(fused)                                 # model reads Inductor's own kernel
    # analytic plan bytes: fused reads x+res(+a,b) writes 1; unfused eager = N_OPS round-trips
    mn = M * N * 2
    bytes_fused = 3 * mn
    bytes_unfused = 2 * N_OPS * mn
    flops = 8 * M * N
    tf = predict_time(flops, bytes_fused, si.occupancy, si.n_spills, 1, K)["t"]
    tu = predict_time(flops, bytes_unfused, si.occupancy, 0, N_OPS, K)["t"]
    model_speedup = tu / tf
    # measured (eager vs compiled)
    me = time_ms(lambda: chain(x, a, b, res), warmup=20, iters=100)["ms"]
    mc = time_ms(lambda: cc(x, a, b, res), warmup=20, iters=100)["ms"]
    meas_speedup = me / mc
    return dict(M=M, N=N, kernel=si.name, f_regs=si.n_regs, f_spills=si.n_spills,
                f_occ=round(si.occupancy, 3), model_speedup=round(model_speedup, 2),
                meas_speedup=round(meas_speedup, 2),
                agree="beneficial" if (model_speedup > 1) == (meas_speedup > 1) else "DISAGREE")


def main():
    print(f"{'MxN':>12} | {'Inductor kernel':40s} | regs sp occ | model_sp meas_sp agree")
    for (M, N) in [(1024, 1024), (2048, 2048), (4096, 4096), (8192, 8192), (8192, 2048)]:
        r = run(M, N)
        print(f"{M}x{N:>6} | {r['kernel']:40s} | {r['f_regs']:4d} {r['f_spills']:2d} {r['f_occ']:.2f} | "
              f"{r['model_speedup']:>7.2f}x {r['meas_speedup']:>6.2f}x  {r['agree']}", flush=True)


if __name__ == "__main__":
    main()
