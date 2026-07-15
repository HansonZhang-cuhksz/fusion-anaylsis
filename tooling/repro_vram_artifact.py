"""repro_vram_artifact.py -- reproduce (and bound) the Ada VRAM-oversubscription timing artifact.

Committed so the LOG-10 evidence is reproducible from the repo rather than prose.

WHAT THIS SHOWS (Ada, 8 GB, WSL2):
  When torch's caching allocator's RESERVED footprint oversubscribes physical VRAM, WSL2/WDDM permits
  it instead of failing, and the CUDA context's local-memory backing store can be left host-resident.
  Every spill access then crosses PCIe, inflating SPILLING kernels only. Non-spilling kernels are
  untouched in ratio.

WHAT IT ALSO SHOWS (the honest negatives -- do not re-derive these the hard way):
  * free VRAM ~ 0 is NOT the trigger: with reserved BELOW physical, free=0.000GB is harmless.
  * empty_cache() does NOT repair an already-poisoned context (it is prophylactic only).
  * a fresh spilling kernel first-loaded under free=0 is healthy => the "first load under starvation"
    rule is false; damage follows a LARGE-spill launch growing the context-wide local-memory store.

Usage:  source tooling/env.sh && python tooling/repro_vram_artifact.py [--stage all|oversub|freezero]
"""
from __future__ import annotations
import argparse, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, __file__.rsplit("/tooling/", 1)[0])
import torch
from fusion.kernels import reduction, pointwise
from fusion.timing import time_ms

TARGET = (2048, 2048, 64, "float32")     # 310 spills; healthy ~0.89
BIG = (2048, 2048, 128, "float32")       # 1868 spills; healthy ~0.12


def mem():
    f, t = torch.cuda.mem_get_info()
    return (f"free={f/1e9:5.3f}GB reserved={torch.cuda.memory_reserved()/1e9:5.2f}GB "
            f"physical={t/1e9:4.2f}GB")


def measure(tag, cfg=TARGET):
    R, C, N, dt = cfg
    case = reduction.make_case(R, C, N, getattr(torch, dt), GS=16)
    tf = time_ms(case.run_fused, warmup=8, iters=25)["ms"]
    tu = time_ms(case.run_unfused, warmup=8, iters=25)["ms"]
    print(f"  {tag:46s} t_fused={tf:9.3f} sp={tu/tf:.4f} | {mem()}", flush=True)
    del case
    return tu / tf


def hoard_until_oversubscribed(target_gb=8.2):
    """Grow torch's RESERVED footprint past physical VRAM (WSL2 permits it), then free to the cache."""
    keep = []
    while torch.cuda.memory_reserved() / 1e9 < target_gb:
        try:
            t = torch.empty(int(0.5e9 // 4), dtype=torch.float32, device="cuda")
            t.zero_()            # dirty it so it is really backed
            keep.append(t)
        except RuntimeError:
            break
    print(f"  hoarded: {mem()}", flush=True)
    del keep                     # -> returned to torch's CACHE, not to the driver
    print(f"  after del (cache retains): {mem()}", flush=True)


def stage_freezero():
    """NEGATIVE control: free VRAM -> 0 via the real pre-fix path, reserved stays BELOW physical."""
    print("\n[freezero] replay the pre-fix path (48 pointwise cases, NO empty_cache)")
    import itertools
    for (R, C), K, dt in itertools.product([(2048, 2048), (4096, 4096), (8192, 2048), (4096, 8192)],
                                           [1, 2, 4, 8, 16, 32], ["float16", "float32"]):
        c = pointwise.make_case(R=R, C=C, K=K, dtype=getattr(torch, dt))
        c.run_fused(); c.run_unfused(); del c
    torch.cuda.synchronize()
    print(f"  pointwise family done: {mem()}", flush=True)
    sp = measure("free~0 but reserved < physical")
    print(f"  => {sp:.4f}. EXPECTED ~0.89 (HEALTHY): free VRAM ~0 is NOT the trigger.")


def stage_oversub():
    """POSITIVE: reserved > physical, then run the LARGE-spill kernel; then show empty_cache cannot heal."""
    print("\n[oversub] grow reserved past physical, then run the 1868-spill kernel")
    healthy = measure("baseline (clean)")
    hoard_until_oversubscribed()
    fresh = measure("fresh N64 first-loaded while oversubscribed")
    print(f"  => {fresh:.4f}. If ~0.9, the 'first load under starvation' rule is FALSE.")
    big = measure("N128 (1868 spills) while oversubscribed", BIG)
    print(f"  => N128 sp={big:.4f} (healthy ~0.12). This is the artifact.")
    after = measure("N64 again, AFTER the big-spill launch")
    torch.cuda.empty_cache()
    healed = measure("N64 after empty_cache() (free restored)")
    print(f"\n  baseline={healthy:.4f} fresh={fresh:.4f} after_big={after:.4f} post_empty_cache={healed:.4f}")
    print("  => if post_empty_cache stays inflated, the fix CANNOT repair a poisoned context;")
    print("     it is prophylactic only (keep reserved from ever exceeding physical).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "oversub", "freezero"])
    a = ap.parse_args()
    print(f"device: {torch.cuda.get_device_name(0)} | {mem()}")
    if a.stage in ("all", "freezero"):
        stage_freezero()
    if a.stage in ("all", "oversub"):
        stage_oversub()
