"""matrix.py -- enumeration of the microbench sweep (the op-pair x shape x dtype x knob grid).

Each entry is (family, params) where params is a kwargs dict for that family's make_case (plus a
"dtype" string). Kept declarative so the runner and the ncu profiler iterate the exact same set.
"""
from __future__ import annotations
import itertools

# --- Family P: pointwise chain (easy-win control) --------------------------------
_P_SHAPES = [(2048, 2048), (4096, 4096), (8192, 2048), (4096, 8192)]
_P_K = [1, 2, 4, 8, 16, 32]
_P_DTYPE = ["float16", "float32"]

# --- Family R: sibling reductions (P_occ / spill knob) ---------------------------
_R_SHAPES = [(1024, 1024), (2048, 2048), (4096, 1024), (2048, 4096)]
# NOUT must be strictly greater than the unfused group size GS, else the unfused plan is a single
# launch identical to the fused kernel (n_launches_unfused==1) -- a degenerate no-op whose
# fuse/don't-fuse label is pure timing noise. With GS=16 that means NOUT>=32. (NOUT=8,16 were
# dropped for exactly this reason; see REVIEW_FINDINGS_TODO item 7.) Powers of 2 only (Triton
# tl.arange width must be pow2).
_R_NOUT = [32, 64, 128]
_R_DTYPE = ["float16", "float32"]
_R_GS = 16


def pointwise_cases():
    for (R, C), K, dt in itertools.product(_P_SHAPES, _P_K, _P_DTYPE):
        yield ("pointwise", {"R": R, "C": C, "K": K, "dtype": dt})


def reduction_cases():
    for (R, C), NOUT, dt in itertools.product(_R_SHAPES, _R_NOUT, _R_DTYPE):
        gs = _R_GS if NOUT % _R_GS == 0 else 8
        yield ("reduction", {"R": R, "C": C, "NOUT": NOUT, "GS": gs, "dtype": dt})


def all_cases():
    yield from pointwise_cases()
    yield from reduction_cases()


# A smaller, high-signal subset for the (slow) ncu ground-truth pass.
def ncu_subset():
    for (R, C) in [(2048, 2048), (4096, 1024)]:
        for K in [2, 8, 32]:
            yield ("pointwise", {"R": R, "C": C, "K": K, "dtype": "float16"})
    for (R, C) in [(2048, 2048), (1024, 1024)]:
        for NOUT in [16, 32, 64, 128]:      # pow2 only (Triton tl.arange width)
            for dt in ["float16", "float32"]:
                yield ("reduction", {"R": R, "C": C, "NOUT": NOUT, "GS": 16, "dtype": dt})


if __name__ == "__main__":
    cs = list(all_cases())
    print(f"total cases: {len(cs)}  (pointwise={len(list(pointwise_cases()))}, "
          f"reduction={len(list(reduction_cases()))})")
    print(f"ncu subset: {len(list(ncu_subset()))}")
