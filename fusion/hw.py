"""hw.py -- sm89 (Ada) hardware model + analytical occupancy calculator.

This is the interpretable backbone of P_occ. Everything here is a *static* function of
per-kernel resource usage (registers/thread, shared-mem bytes, threads/block) -- exactly the
quantities Triton reports from a single compile (`k.n_regs`, `k.metadata.shared`, num_warps*32)
or that ptxas -v prints. No profiling, no autotuning.

The register-file model is deliberately parameterised (see HardwareModel) so the same code
re-parameterises to MetaX C500 later: C500 has a split scalar/vector register file and a hard
4 KB/thread private-memory cap, represented by `split_regfile` and `spill_cap_bytes`.
"""
from __future__ import annotations
from dataclasses import dataclass
import math


def _round_up(x: int, unit: int) -> int:
    return ((x + unit - 1) // unit) * unit


@dataclass(frozen=True)
class HardwareModel:
    name: str
    sm_count: int
    max_warps_per_sm: int
    max_threads_per_sm: int
    max_blocks_per_sm: int
    regs_per_sm: int
    max_regs_per_thread: int
    reg_alloc_unit: int          # registers allocated per warp, rounded up to this (sm89: 256)
    warp_alloc_granularity: int  # warps/block rounded up to this for reg allocation (sm89: 4)
    smem_per_sm: int             # bytes usable for blocks
    smem_alloc_unit: int         # bytes (sm89: 128)
    warp_size: int = 32
    # --- vendor-portability knobs (NVIDIA defaults; overridden for C500 later) ---
    split_regfile: bool = False  # C500: separate scalar(ST)/vector(MT) files -> True
    spill_cap_bytes: int | None = None  # C500: hard 4096 B/thread private cap; None = soft (NVIDIA)

    # ---- analytical occupancy ---------------------------------------------------
    def blocks_per_sm(self, regs_per_thread: int, smem_bytes: int, threads_per_block: int) -> dict:
        """Active thread-blocks per SM, limited by registers / shared-mem / warp&block caps.

        Returns a dict with the binding limiter so the model can *explain* an occupancy loss.
        Mirrors the CUDA Occupancy Calculator (per-warp register allocation, cc 8.x granularities).
        """
        warps_per_block = math.ceil(threads_per_block / self.warp_size)

        # --- block-count cap & warp cap ---
        lim_warps = self.max_warps_per_sm // warps_per_block if warps_per_block else self.max_blocks_per_sm
        lim_blocks = self.max_blocks_per_sm
        lim_threads = self.max_threads_per_sm // threads_per_block if threads_per_block else self.max_blocks_per_sm

        # --- register cap (per-warp allocation, rounded to reg_alloc_unit, warps rounded to gran) ---
        if regs_per_thread <= 0:
            lim_regs = self.max_blocks_per_sm
        elif regs_per_thread > self.max_regs_per_thread:
            lim_regs = 0  # cannot launch (would spill/refuse); caller treats as hard fail
        else:
            regs_per_warp = _round_up(regs_per_thread * self.warp_size, self.reg_alloc_unit)
            warps_ru = _round_up(warps_per_block, self.warp_alloc_granularity)
            regs_per_block = regs_per_warp * warps_ru
            lim_regs = self.regs_per_sm // regs_per_block if regs_per_block else self.max_blocks_per_sm

        # --- shared-memory cap ---
        if smem_bytes <= 0:
            lim_smem = self.max_blocks_per_sm
        else:
            smem_ru = _round_up(smem_bytes, self.smem_alloc_unit)
            lim_smem = self.smem_per_sm // smem_ru if smem_ru else self.max_blocks_per_sm

        limits = {"regs": lim_regs, "smem": lim_smem, "warps": lim_warps,
                  "threads": lim_threads, "blocks": lim_blocks}
        blocks = max(0, min(limits.values()))
        binder = min(limits, key=limits.get)
        return {"blocks_per_sm": blocks, "warps_per_block": warps_per_block,
                "binder": binder, "limits": limits}

    def occupancy(self, regs_per_thread: int, smem_bytes: int, threads_per_block: int) -> dict:
        """Theoretical occupancy = active_warps / max_warps_per_sm, plus the binding limiter."""
        b = self.blocks_per_sm(regs_per_thread, smem_bytes, threads_per_block)
        active_warps = b["blocks_per_sm"] * b["warps_per_block"]
        occ = active_warps / self.max_warps_per_sm
        return {"occupancy": occ, "active_warps": active_warps,
                "blocks_per_sm": b["blocks_per_sm"], "binder": b["binder"], "limits": b["limits"]}


# The primary target of this project.
ADA_SM89 = HardwareModel(
    name="RTX4060-Laptop-sm89",
    sm_count=24,
    max_warps_per_sm=48,
    max_threads_per_sm=1536,
    max_blocks_per_sm=24,
    regs_per_sm=65536,
    max_regs_per_thread=255,
    reg_alloc_unit=256,
    warp_alloc_granularity=4,
    smem_per_sm=102400,      # 100 KB
    smem_alloc_unit=128,
    warp_size=32,
    split_regfile=False,
    spill_cap_bytes=None,    # NVIDIA: spilling is a soft cost (local-mem traffic), not a hard fail
)

# MetaX C500 (Phase 4 cross-vendor transfer). Constants from torch.cuda.get_device_properties +
# `cucc -resource-usage` on this machine (env `fusion`, MACA 3.7.0), 2026-07-14.
# Verified: multi_processor_count=104, max_threads_per_multi_processor=2048, warp_size=**64**,
# regs_per_multiprocessor=131072, shared_memory_per_multiprocessor=65536, compute cap 8.0.
# KEY vendor differences vs Ada sm89: 64-thread wavefronts (not 32); 2x register file;
# split ST/MT register file; hard 4 KB/thread private (spill) cap.
# NOTE: the occupancy-granularity fields (reg_alloc_unit, warp_alloc_granularity, max_blocks_per_sm,
# smem_alloc_unit, max_regs_per_thread) are not exposed by the driver; the values below are
# physically-motivated estimates to be CALIBRATED against MCPTI-measured occupancy (waves). Since
# the smooth occupancy term is inert on Ada (spills dominate), P_occ precision is secondary; the
# decision-critical constant is the spill behavior (4 KB cap).
METAX_C500 = HardwareModel(
    name="MetaX-C500",
    sm_count=104,
    max_warps_per_sm=32,          # 2048 threads / 64-wide wave
    max_threads_per_sm=2048,
    max_blocks_per_sm=32,         # estimate (sm80-like); calibrate
    regs_per_sm=131072,           # MT (vector) register file / CU
    max_regs_per_thread=256,      # Triton reports up to 256 MTregs; treat as cap (calibrate)
    reg_alloc_unit=512,           # estimate: 64-wide wave x 8-reg granularity (Ada is 32x8=256)
    warp_alloc_granularity=1,     # estimate; calibrate
    smem_per_sm=65536,            # 64 KB / CU
    smem_alloc_unit=128,          # estimate
    warp_size=64,                 # <-- 64-thread wavefronts (the big divergence from NVIDIA)
    split_regfile=True,           # ST + MT register files (n_regs read as MTregisters)
    spill_cap_bytes=4096,         # hard 4 KB/thread private-mem cap -> mcErrorMemoryValueTooLarge
)
METAX_C500_STUB = METAX_C500   # back-compat alias


def default_hw(name: str | None = None):
    """Select the HardwareModel for the current run. `name` or env FUSION_HW in {ada, c500};
    defaults to Ada (preserves the committed NVIDIA behavior when unset)."""
    import os
    key = (name or os.environ.get("FUSION_HW", "ada")).lower()
    return METAX_C500 if key in ("c500", "metax", "mx") else ADA_SM89


if __name__ == "__main__":
    hw = ADA_SM89
    # sanity: a 256-thread block using 32 regs, no smem -> should hit the 48-warp cap (occ 1.0)
    for regs in (32, 64, 96, 128, 168, 200, 255, 256):
        for smem in (0, 16384, 49152):
            o = hw.occupancy(regs, smem, 256)
            print(f"regs={regs:3d} smem={smem:6d} -> occ={o['occupancy']:.3f} "
                  f"blocks/SM={o['blocks_per_sm']:2d} binder={o['binder']}")
        print()
