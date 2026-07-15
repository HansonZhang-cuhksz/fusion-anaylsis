"""gemm_epilogue.py -- Family G: CONTRACTION (GEMM) with a fusable pointwise epilogue.

The canonical real fusion and a *compute-bound* regime (unlike the memory/spill-bound Family R):
    C = relu(A @ B + bias)                 A[M,K] @ B[K,N] (+ bias[N]) -> C[M,N]

  fused   = ONE Triton GEMM kernel that applies bias+relu in-register on the accumulator, writes C.
  unfused = GEMM writes the raw product T=A@B to HBM, then a separate pointwise kernel reads T and
            applies relu(T+bias) -> C. (2 kernels; extra M*N write + M*N read round-trip.)

Fusion saves the M*N output round-trip. Unlike Family R this uses tensor cores (`tl.dot`) and does NOT
spill at the tile sizes swept -> it probes whether the model handles the *compute-bound / occupancy*
regime (TODO G2/G4), where any toxicity must come from occupancy/tiling, not the register-spill cliff.
"""
from __future__ import annotations
import torch, triton, triton.language as tl
from .base import Case, POINTWISE, CONTRACTION


@triton.jit
def _gemm_epi(a_ptr, b_ptr, bias_ptr, c_ptr, M, N, K,
              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, FUSE: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    if FUSE:
        bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
        acc = tl.maximum(acc + bias[None, :], 0.0)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def _epilogue(t_ptr, bias_ptr, c_ptr, M, N, BM: tl.constexpr, BN: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    m = (rm[:, None] < M) & (rn[None, :] < N)
    t = tl.load(t_ptr + rm[:, None] * N + rn[None, :], mask=m, other=0.0)
    bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], tl.maximum(t + bias[None, :], 0.0), mask=m)


def make_case(M: int, N: int, K: int, dtype=torch.float16, BM: int = 64, BN: int = 64, BK: int = 32,
              num_warps: int = 4, compile_probe: bool = True) -> Case:
    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b = torch.randn((K, N), device="cuda", dtype=dtype) * (K ** -0.5)
    bias = torch.randn((N,), device="cuda", dtype=torch.float32)
    c = torch.empty((M, N), device="cuda", dtype=torch.float32)
    t = torch.empty((M, N), device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))

    def run_fused():
        _gemm_epi[grid](a, b, bias, c, M, N, K, BM, BN, BK, 1, num_warps=num_warps)
        return c

    def run_unfused():
        _gemm_epi[grid](a, b, bias, t, M, N, K, BM, BN, BK, 0, num_warps=num_warps)  # T = A@B
        _epilogue[grid](t, bias, c, M, N, BM, BN)                                    # relu(T+bias)
        return c

    fk = uk = None
    if compile_probe:
        fk = _gemm_epi[grid](a, b, bias, c, M, N, K, BM, BN, BK, 1, num_warps=num_warps)
        uk = _gemm_epi[grid](a, b, bias, t, M, N, K, BM, BN, BK, 0, num_warps=num_warps)

    isz = a.element_size()
    return Case(
        family="gemm_epilogue", op_pair=f"gemm_epi_{M}x{N}x{K}",
        producer_class=CONTRACTION, consumer_class=POINTWISE,
        dtype=dtype, shape={"R": M, "C": N},   # reuse R/C slots (M,N) so runner/CV code works
        params={"M": M, "N": N, "K": K, "BM": BM, "BN": BN, "BK": BK, "num_warps": num_warps},
        run_fused=run_fused, run_unfused=run_unfused,
        fused_kernels=[fk] if fk else [], unfused_kernels=[uk] if uk else [],
        tile={"BM": BM, "BN": BN, "BK": BK}, layout_compatible=True, n_launches_unfused=2,
        # fused: read A,B,bias + write C.   unfused: + write T + read T (the saved round-trip).
        bytes_moved_fused=(M * K + K * N) * isz + N * 4 + M * N * 4,
        bytes_moved_unfused=(M * K + K * N) * isz + N * 4 + M * N * 4 + 2 * M * N * 4,
        flops=2 * M * N * K,
    )
