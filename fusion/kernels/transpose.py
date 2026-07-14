"""transpose.py -- Family T: permute(transpose) producer -> elementwise consumer (P_layout knob).

Y[C,R] = relu(s * X[R,C]^T + b).  The consumer wants X in transposed layout; reconciling the
producer's row-major layout with the consumer's is the P_layout cost.

  fused   = ONE kernel reads X with a transposed (strided) access pattern and writes Y. Saves the
            intermediate round trip but the strided reads are UNCOALESCED -> extra memory
            transactions (the layout penalty). Occupancy is fine and there are no spills, so the
            harm is attributable to layout, NOT P_occ.
  unfused = tiled transpose (Triton stages via shared memory -> coalesced global) writes X^T to
            HBM, then a coalesced elementwise epilogue. Two kernels, one extra round trip, but all
            memory access is coalesced.

Fusion trades one coalesced round trip against uncoalesced reads. Toxic when the uncoalescing cost
exceeds the saved round trip (small matrices that fit in L2). P_layout-dominant by construction.
"""
from __future__ import annotations
import torch, triton, triton.language as tl
from .base import Case, PERMUTE, POINTWISE


@triton.jit
def _fused_strided(x_ptr, y_ptr, R, C, s, b, BR: tl.constexpr, BC: tl.constexpr):
    # Y tile [BC, BR]: element [i,j] = relu(s*X[rr[j], cc[i]]+b). Read X strided (stride C on rr).
    pc = tl.program_id(0); pr = tl.program_id(1)
    cc = pc * BC + tl.arange(0, BC)     # C index (Y row / X col)
    rr = pr * BR + tl.arange(0, BR)     # R index (Y col / X row)
    x = tl.load(x_ptr + rr[None, :] * C + cc[:, None],
                mask=(cc < C)[:, None] & (rr < R)[None, :], other=0.0).to(tl.float32)
    y = tl.maximum(s * x + b, 0.0)
    tl.store(y_ptr + cc[:, None] * R + rr[None, :], y,
             mask=(cc < C)[:, None] & (rr < R)[None, :])


@triton.jit
def _trans_tile(x_ptr, xt_ptr, R, C, BR: tl.constexpr, BC: tl.constexpr):
    # coalesced tiled transpose: read X[BR,BC] coalesced, tl.trans in shared, write X^T coalesced.
    pr = tl.program_id(0); pc = tl.program_id(1)
    rr = pr * BR + tl.arange(0, BR)
    cc = pc * BC + tl.arange(0, BC)
    x = tl.load(x_ptr + rr[:, None] * C + cc[None, :],
                mask=(rr < R)[:, None] & (cc < C)[None, :], other=0.0)
    xt = tl.trans(x)                    # [BC, BR]
    tl.store(xt_ptr + cc[:, None] * R + rr[None, :], xt,
             mask=(cc < C)[:, None] & (rr < R)[None, :])


@triton.jit
def _epilogue(xt_ptr, y_ptr, n, s, b, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    v = tl.load(xt_ptr + off, mask=m).to(tl.float32)
    tl.store(y_ptr + off, tl.maximum(s * v + b, 0.0), mask=m)


def make_case(R: int, C: int, dtype=torch.float16, BR: int = 32, BC: int = 32,
              num_warps: int = 4, compile_probe: bool = True) -> Case:
    s, b = 1.0, 0.1
    x = torch.randn((R, C), device="cuda", dtype=dtype)
    y = torch.empty((C, R), device="cuda", dtype=torch.float32)
    xt = torch.empty((C, R), device="cuda", dtype=dtype)
    n = R * C
    gy = (triton.cdiv(C, BC), triton.cdiv(R, BR))
    gt = (triton.cdiv(R, BR), triton.cdiv(C, BC))
    ge = (triton.cdiv(n, 1024),)

    def run_fused():
        _fused_strided[gy](x, y, R, C, s, b, BR=BR, BC=BC, num_warps=num_warps)
        return y

    def run_unfused():
        _trans_tile[gt](x, xt, R, C, BR=BR, BC=BC, num_warps=num_warps)
        _epilogue[ge](xt, y, n, s, b, BLOCK=1024)
        return y

    fk = uk = None
    if compile_probe:
        fk = _fused_strided[gy](x, y, R, C, s, b, BR=BR, BC=BC, num_warps=num_warps)
        uk = _trans_tile[gt](x, xt, R, C, BR=BR, BC=BC, num_warps=num_warps)

    isz = x.element_size()
    return Case(
        family="transpose", op_pair=f"transpose_relu_{R}x{C}",
        producer_class=PERMUTE, consumer_class=POINTWISE,
        dtype=dtype, shape={"R": R, "C": C},
        params={"BR": BR, "BC": BC, "num_warps": num_warps},
        run_fused=run_fused, run_unfused=run_unfused,
        fused_kernels=[fk] if fk else [], unfused_kernels=[uk] if uk else [],
        tile={"BR": BR, "BC": BC}, layout_compatible=False, n_launches_unfused=2,
        bytes_moved_fused=2 * n * isz,               # read X + write Y (ideal; strided inflates it)
        bytes_moved_unfused=2 * n * isz + 2 * n * isz,  # transpose rd+wr + epilogue rd+wr
        flops=n * 2,
    )
