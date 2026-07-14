"""timing.py -- robust GPU latency measurement (CUDA events, warmup, median of repeats).

Timing includes launch overhead by construction: the unfused runner issues multiple launches, so
its measured latency carries the extra launch + round-trip cost the cost model accounts for.
"""
from __future__ import annotations
import torch
from statistics import median


def time_ms(fn, warmup: int = 15, iters: int = 60, flush_l2: bool = True) -> dict:
    """Median kernel latency in ms over `iters`, after `warmup`. Optionally flush L2 between iters
    so we measure cold-cache HBM behavior (relevant to fusion's memory-traffic argument)."""
    # a buffer bigger than L2 (32 MB) to evict it
    flush = torch.empty(int(64 * 1024 * 1024 // 4), dtype=torch.float32, device="cuda") if flush_l2 else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        if flush is not None:
            flush.zero_()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return {"ms": median(samples), "ms_min": samples[0],
            "ms_p10": samples[max(0, len(samples) // 10)], "n": iters}
