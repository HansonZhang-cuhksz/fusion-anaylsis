"""pointwise.py -- Family P: elementwise producer->consumer chain.

The "easy-win" control. A depth-K chain of elementwise ops on X[R,C]:
  fused   = one kernel, load X once, apply K steps, store Y   (2 HBM round trips)
  unfused = K kernels, each loads+stores (K HBM round trips)
Fusion removes (K-1) round trips. Memory-bound => fusion should almost always win, and register
pressure stays low so occupancy stays high. This family tests that the model does NOT over-reject.
"""
from __future__ import annotations
import torch, triton, triton.language as tl
from triton.language.extra import libdevice
from .base import Case, POINTWISE

# one chain step: t = tanh(t*s + b)  (nonlinear, bounded in (-1,1) so K can be swept freely)
@triton.jit
def _chain_step(t, s, b):
    ts = (t * s + b).to(tl.float32)
    return libdevice.tanh(ts).to(t.dtype)


@triton.jit
def _fused_chain(x_ptr, y_ptr, n, K: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    t = tl.load(x_ptr + off, mask=m)
    for i in tl.static_range(K):
        t = _chain_step(t, 1.0009765625, 0.01)  # constants ~1 to stay numerically tame
    tl.store(y_ptr + off, t, mask=m)


@triton.jit
def _one_step(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    t = tl.load(x_ptr + off, mask=m)
    t = _chain_step(t, 1.0009765625, 0.01)
    tl.store(y_ptr + off, t, mask=m)


def make_case(R: int, C: int, K: int, dtype=torch.float16, BLOCK: int = 1024,
              num_warps: int = 4, compile_probe: bool = True) -> Case:
    n = R * C
    grid = (triton.cdiv(n, BLOCK),)
    x = torch.randn((R, C), device="cuda", dtype=dtype).reshape(-1)
    y = torch.empty_like(x)
    bufs = [x] + [torch.empty_like(x) for _ in range(K)]  # ping-pong buffers for unfused

    def run_fused():
        _fused_chain[grid](x, y, n, K=K, BLOCK=BLOCK, num_warps=num_warps)
        return y.reshape(R, C)

    def run_unfused():
        cur = x
        for i in range(K):
            nxt = bufs[i + 1]
            _one_step[grid](cur, nxt, n, BLOCK=BLOCK, num_warps=num_warps)
            cur = nxt
        return cur.reshape(R, C)

    # compile (single-compile static inputs). Skipped for the ncu worker so profiling only sees
    # the plan's own launches (compile_probe=False).
    fk = uk = None
    if compile_probe:
        fk = _fused_chain[grid](x, y, n, K=K, BLOCK=BLOCK, num_warps=num_warps)
        uk = _one_step[grid](x, bufs[1], n, BLOCK=BLOCK, num_warps=num_warps)

    itemsize = x.element_size()
    return Case(
        family="pointwise", op_pair=f"pointwise_chain_k{K}",
        producer_class=POINTWISE, consumer_class=POINTWISE,
        dtype=dtype, shape={"R": R, "C": C}, params={"K": K, "BLOCK": BLOCK, "num_warps": num_warps},
        run_fused=run_fused, run_unfused=run_unfused,
        fused_kernels=[fk] if fk else [], unfused_kernels=[uk] if uk else [],
        tile={"BLOCK": BLOCK}, layout_compatible=True,
        n_launches_unfused=K,
        bytes_moved_fused=2 * n * itemsize,          # read X + write Y
        bytes_moved_unfused=2 * K * n * itemsize,    # K round trips
        flops=K * n * 5,                             # ~5 flops/step
    )
