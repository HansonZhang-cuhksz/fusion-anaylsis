"""static.py -- extract the *single-compile* static inputs from a compiled Triton kernel.

These are the only quantities the deployed (search-free) pass is allowed to read: they come
from one codegen/compile of the fused candidate, never from a run or an autotune. Mirrors what
PyTorch Inductor already has after it lowers a fusion group to a Triton kernel.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from .hw import HardwareModel, ADA_SM89, default_hw


@dataclass
class StaticInputs:
    name: str
    n_regs: int            # registers / thread (ptxas-reported, read from compile)
    n_spills: int          # spill count (0 == no local-mem spill); >0 == P_occ spill cliff
    shared_bytes: int      # dynamic+static shared memory / block
    num_warps: int
    threads_per_block: int
    # derived analytical occupancy (from hw model) --------------------------------
    occupancy: float = 0.0
    blocks_per_sm: int = 0
    occ_binder: str = ""   # which resource caps occupancy: regs/smem/warps/threads/blocks
    spilled: bool = False

    def as_row(self, prefix: str = "") -> dict:
        d = asdict(self)
        return {f"{prefix}{k}": v for k, v in d.items()}


def from_triton(compiled, hw: HardwareModel = None, name: str = "") -> StaticInputs:
    """Build StaticInputs from a Triton CompiledKernel + a HardwareModel.
    hw defaults to default_hw() (env FUSION_HW: ada|c500; ada when unset)."""
    hw = hw or default_hw()
    n_regs = int(compiled.n_regs)
    n_spills = int(compiled.n_spills)
    shared = int(compiled.metadata.shared)
    num_warps = int(compiled.metadata.num_warps)
    tpb = num_warps * hw.warp_size
    occ = hw.occupancy(n_regs, shared, tpb)
    return StaticInputs(
        name=name or getattr(compiled.metadata, "name", "kernel"),
        n_regs=n_regs, n_spills=n_spills, shared_bytes=shared,
        num_warps=num_warps, threads_per_block=tpb,
        occupancy=occ["occupancy"], blocks_per_sm=occ["blocks_per_sm"],
        occ_binder=occ["binder"], spilled=(n_spills > 0),
    )


def from_ptxas(n_regs: int, spill_bytes: int, shared_bytes: int, threads_per_block: int,
               hw: HardwareModel = ADA_SM89, name: str = "cuda_kernel") -> StaticInputs:
    """Build StaticInputs from ptxas -v numbers (raw-CUDA path)."""
    occ = hw.occupancy(n_regs, shared_bytes, threads_per_block)
    return StaticInputs(
        name=name, n_regs=n_regs, n_spills=(1 if spill_bytes > 0 else 0),
        shared_bytes=shared_bytes, num_warps=threads_per_block // hw.warp_size,
        threads_per_block=threads_per_block,
        occupancy=occ["occupancy"], blocks_per_sm=occ["blocks_per_sm"],
        occ_binder=occ["binder"], spilled=(spill_bytes > 0),
    )
