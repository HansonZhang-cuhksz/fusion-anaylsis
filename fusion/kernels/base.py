"""base.py -- common interface for a fused-vs-unfused microbench case.

Each Case knows how to (1) run the FUSED implementation, (2) run the UNFUSED (multi-kernel)
implementation computing identical math, (3) hand back the compiled Triton kernels so we can read
their single-compile static inputs, and (4) describe its taxonomy Phi(v) metadata.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any
import torch


# Taxonomy topological classes (Phi.C) from PROPOSAL 5.1
POINTWISE = "pointwise"
BROADCAST = "broadcast"
REDUCTION = "reduction"
CONTRACTION = "contraction"
PERMUTE = "permute"


def spill_reread(producer_class: str, consumer_class: str) -> int:
    """Taxonomy-derived spill re-read multiplier (makes Phi(v) load-bearing; see LOG-05).

    A full-tile pointwise *epilogue* fused over a *spilling* producer (e.g. GEMM+bias+act) re-reads
    the spilled accumulator (store + reload) at runtime, so its spill TRAFFIC ~ 2x the static spill
    instruction COUNT. Producer=CONTRACTION, consumer=POINTWISE captures this; other fusions
    (reduction, pointwise chain) write the accumulator once -> 1."""
    return 2 if (producer_class == CONTRACTION and consumer_class == POINTWISE) else 1


@dataclass
class Case:
    family: str                 # "pointwise" | "reduction" | "transpose" | "gemm_epilogue"
    op_pair: str                # human label, e.g. "pointwise_chain_k8"
    producer_class: str         # taxonomy class of producer
    consumer_class: str         # taxonomy class of consumer
    dtype: torch.dtype
    shape: dict                 # problem dims, e.g. {"R":4096,"C":4096}
    params: dict                # fusion knobs, e.g. {"K":8,"NOUT":16,"BLOCK":1024,"PAD":1}
    # callables
    run_fused: Callable[[], torch.Tensor]
    run_unfused: Callable[[], torch.Tensor]
    # compiled Triton kernels for static-input extraction
    fused_kernels: list = field(default_factory=list)     # list of CompiledKernel
    unfused_kernels: list = field(default_factory=list)
    # taxonomy tile/layout descriptors (Phi.T, Phi.L)
    tile: dict = field(default_factory=dict)              # e.g. {"BLOCK_R":128,"BLOCK_C":128}
    layout_compatible: bool = True   # whether producer tile/layout matches consumer's preferred
    n_launches_unfused: int = 2      # number of kernel launches in the unfused plan
    bytes_moved_fused: int = 0       # analytic HBM bytes (min_in+min_out) for the fused plan
    bytes_moved_unfused: int = 0     # analytic HBM bytes for the unfused plan
    flops: int = 0                   # total useful work (for roofline / arithmetic intensity)

    def check(self, atol=2e-2, rtol=2e-2) -> float:
        """Assert fused and unfused compute the same thing; return max abs diff."""
        a = self.run_fused().float()
        b = self.run_unfused().float()
        diff = (a - b).abs().max().item()
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        if not ok:
            raise AssertionError(f"{self.op_pair}: fused/unfused mismatch, max|d|={diff:.4g}")
        return diff
