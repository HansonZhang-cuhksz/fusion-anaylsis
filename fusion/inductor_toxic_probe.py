"""inductor_toxic_probe.py -- hunt for a DISCRIMINATING real-compiler case (G4 / G2).

The prior Inductor-prediction (LOG-07 §6) was non-discriminating: elementwise fusions ~always win.
A real compiler CAN fuse net-harmfully via **recomputation**: when a fused kernel recomputes an
expensive shared producer for each consumer, its FLOPs blow up. Inductor's `realize_reads_threshold`
controls when it materializes vs recomputes; forcing recompute makes fusion toxic in the compute-bound
regime. Crucially this toxicity is caught by the model's ROOFLINE compute term (graph-derivable FLOPs),
NOT the spill signal -- so it is also the first *non-spill* toxicity the model catches (G2).

Mechanisms (--mechanism):
  recompute : expensive producer (2*D ops) feeding `fanout` consumers, recompute forced -> compute blow-up.
  deep      : deep single-live-value chain (control: Inductor keeps 1 live value -> no spill, stays benign).

Usage: MACA_VISIBLE_DEVICES=<g> FUSION_HW=c500 python -m fusion.inductor_toxic_probe --mechanism recompute
"""
from __future__ import annotations
import json, argparse, warnings
import torch, triton
warnings.filterwarnings("ignore")
import torch._inductor.config as ind
from fusion.static import from_triton
from fusion.timing import time_ms
from model.costmodel import DeviceConstants, predict_time

_C = []; _O = triton.compile
triton.compile = lambda *a, **k: (_C.append(r := _O(*a, **k)) or r)
K = DeviceConstants(**json.load(open("model/c500_combined_constants.json")))


def producer(y, D):
    for _ in range(D):
        y = torch.sin(y) * 1.1 + torch.cos(y) * 0.9
    return y


def recompute_sg(x, D, fanout):
    y = producer(x, D)
    acc = y * 1.0
    for i in range(2, fanout + 1):
        acc = acc + y * float(i)          # fanout consumers of the expensive y
    return acc


def deep_sg(x, D, fanout):
    return producer(x, D)                 # single live value


def measure(fn):
    _C.clear()
    cc = torch.compile(fn, dynamic=False)
    for _ in range(3):
        cc()
    torch.cuda.synchronize()
    cands = [c for c in _C if getattr(c, "n_regs", None) is not None]
    fused = max(cands, key=lambda c: c.n_regs) if cands else None
    te = time_ms(fn, warmup=20, iters=100)["ms"]
    tc = time_ms(cc, warmup=20, iters=100)["ms"]
    return fused, te, tc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism", default="recompute", choices=["recompute", "deep"])
    ap.add_argument("--M", type=int, default=2048); ap.add_argument("--N", type=int, default=2048)
    a = ap.parse_args()
    ind.realize_reads_threshold = 10_000_000   # force recompute (never materialize the shared producer)
    ind.realize_opcount_threshold = 10_000_000
    build = {"recompute": recompute_sg, "deep": deep_sg}[a.mechanism]
    M, N = a.M, a.N
    print(f"mechanism={a.mechanism} M{M}xN{N} (realize thresholds forced high)")
    print(f"{'D':>4} {'fanout':>6} | fused_regs sp | eager_ms comp_ms  speedup verdict | model_pred model_sp")
    for (D, fanout) in [(8, 8), (32, 8), (32, 16), (64, 16), (128, 8)]:
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        fn = (lambda x=x, D=D, fo=fanout: build(x, D, fo))
        fused, te, tc = measure(fn)
        si = from_triton(fused) if fused else None
        # roofline inputs: fused recomputes producer `fanout` times; eager computes it once.
        mn = M * N
        prod_flops = 2 * D * mn            # ~2 ops/step
        flops_fused = fanout * prod_flops if a.mechanism == "recompute" else prod_flops
        flops_unfused = prod_flops + fanout * mn
        bytes_f = 2 * mn * 2; bytes_u = (2 + fanout) * mn * 2
        occ = si.occupancy if si else 0.5; sp = si.n_spills if si else 0
        tf = predict_time(flops_fused, bytes_f, occ, sp, 1, K)["t"]
        tu = predict_time(flops_unfused, bytes_u, occ, 0, 1 + fanout, K)["t"]
        model_sp = tu / tf
        toxic = tc > te
        print(f"{D:>4} {fanout:>6} | {si.n_regs if si else '?':>10} {si.n_spills if si else '?':>2} | "
              f"{te:>7.3f} {tc:>7.3f} {te/tc:>6.2f}x {'TOXIC' if toxic else 'benef':6s} | "
              f"{'TOXIC' if model_sp<1 else 'benef':6s} {model_sp:>6.2f}x  "
              f"{'<-- MATCH' if (model_sp<1)==toxic else 'MISS'}", flush=True)


if __name__ == "__main__":
    main()
