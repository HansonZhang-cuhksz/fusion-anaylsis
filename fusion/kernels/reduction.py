"""reduction.py -- Family R: horizontal (sibling) fusion of NOUT reductions sharing an input.

NOUT independent output projections of the same input X[R,C]:
    O[r, j] = sum_c gelu(X[r,c]) * W[j,c]     for j = 0..NOUT-1

  fused   = ONE kernel computes all NOUT outputs, holding a wide [BLOCK_R, NOUT] fp32 accumulator.
            Reads X once. As NOUT grows, register pressure rises and the kernel SPILLS to local
            memory (occupancy cliff). This is the P_occ / spill knob.
  unfused = split the NOUT outputs into groups of GS; each group is a separate kernel with a
            narrow [BLOCK_R, GS] accumulator (low registers, no spill) that RE-READS X.

Fusion trades (NOUT/GS - 1) extra reads of X (memory saved) against the wide kernel's spill /
occupancy penalty. Small NOUT -> fusion wins; large NOUT -> the spill cliff can make it toxic.
This is exactly the horizontal-fusion decision Inductor/XLA make for sibling consumers.
"""
from __future__ import annotations
import torch, triton, triton.language as tl
from triton.language.extra import libdevice
from .base import Case, POINTWISE, REDUCTION


@triton.jit
def _sibling_redux(x_ptr, w_ptr, o_ptr, R, C, j0, WIDTH: tl.constexpr, NOUT_TOTAL: tl.constexpr,
                   BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    """Compute WIDTH output columns [j0, j0+WIDTH) for a row-tile. gelu(X) @ W[j0:j0+WIDTH]^T
    via a broadcast reduction (no tensor cores) so registers scale with WIDTH."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < R
    ns = j0 + tl.arange(0, WIDTH)
    acc = tl.zeros([BLOCK_R, WIDTH], dtype=tl.float32)
    for c0 in range(0, C, BLOCK_C):
        cols = c0 + tl.arange(0, BLOCK_C)
        cmask = cols < C
        x = tl.load(x_ptr + rows[:, None] * C + cols[None, :],
                    mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        t = 0.5 * x * (1.0 + libdevice.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))  # gelu
        w = tl.load(w_ptr + ns[:, None] * C + cols[None, :],
                    mask=cmask[None, :], other=0.0).to(tl.float32)         # [WIDTH, BLOCK_C]
        acc += tl.sum(t[:, None, :] * w[None, :, :], axis=2)              # [BLOCK_R, WIDTH]
    tl.store(o_ptr + rows[:, None] * NOUT_TOTAL + ns[None, :], acc,
             mask=rmask[:, None])


def make_case(R: int, C: int, NOUT: int, dtype=torch.float16, GS: int = 16,
              BLOCK_R: int = 32, BLOCK_C: int = 32, num_warps: int = 4,
              compile_probe: bool = True) -> Case:
    assert NOUT % GS == 0, "NOUT must be a multiple of the unfused group size GS"
    x = torch.randn((R, C), device="cuda", dtype=dtype)
    w = torch.randn((NOUT, C), device="cuda", dtype=dtype) * (C ** -0.5)
    o = torch.empty((R, NOUT), device="cuda", dtype=torch.float32)
    grid_r = (triton.cdiv(R, BLOCK_R),)
    n_groups = NOUT // GS

    def run_fused():
        _sibling_redux[grid_r](x, w, o, R, C, 0, WIDTH=NOUT, NOUT_TOTAL=NOUT,
                               BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C, num_warps=num_warps)
        return o

    def run_unfused():
        for g in range(n_groups):
            _sibling_redux[grid_r](x, w, o, R, C, g * GS, WIDTH=GS, NOUT_TOTAL=NOUT,
                                   BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C, num_warps=num_warps)
        return o

    fk = uk = None
    if compile_probe:
        fk = _sibling_redux[grid_r](x, w, o, R, C, 0, WIDTH=NOUT, NOUT_TOTAL=NOUT,
                                    BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C, num_warps=num_warps)
        uk = _sibling_redux[grid_r](x, w, o, R, C, 0, WIDTH=GS, NOUT_TOTAL=NOUT,
                                    BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C, num_warps=num_warps)

    isz = x.element_size()
    n = R * C
    return Case(
        family="reduction", op_pair=f"sibling_redux_n{NOUT}",
        producer_class=POINTWISE, consumer_class=REDUCTION,
        dtype=dtype, shape={"R": R, "C": C},
        params={"NOUT": NOUT, "GS": GS, "BLOCK_R": BLOCK_R, "BLOCK_C": BLOCK_C,
                "num_warps": num_warps},
        run_fused=run_fused, run_unfused=run_unfused,
        fused_kernels=[fk] if fk else [], unfused_kernels=[uk] if uk else [],
        tile={"BLOCK_R": BLOCK_R, "BLOCK_C": BLOCK_C, "WIDTH": NOUT},
        layout_compatible=True, n_launches_unfused=n_groups,
        # fused: read X once + read W + write O
        bytes_moved_fused=n * isz + NOUT * C * isz + R * NOUT * 4,
        # unfused: read X n_groups times + read W + write O
        bytes_moved_unfused=n_groups * n * isz + NOUT * C * isz + R * NOUT * 4,
        flops=R * C * NOUT * 2,
    )
