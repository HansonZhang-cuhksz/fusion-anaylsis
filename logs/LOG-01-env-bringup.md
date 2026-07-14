# LOG-01 — Ada (sm89) environment bring-up  [Phase 1]

Date: 2026-07-14 · Machine: RTX 4060 Laptop GPU (Ada, sm89), WSL2 · Session: Ada primary

## Summary
Phase 0/1 environment validation is **complete and green**. The entire NVIDIA research
toolchain lives inside the conda env **`profiling`** (not on the base PATH). All static-input
and profiling capabilities the cost model needs are confirmed working — including the two
usual gotchas (WSL2 ncu counters, CUDA-12.8/GCC-14 header clash), both resolved.

## Machine / toolchain (verified)
| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 **Laptop** GPU |
| Arch | Ada, **sm89** (compute_cap 8.9) |
| SMs | **24** |
| Driver / CUDA UMD | 610.74 / 13.3 (WSL2) |
| VRAM | 8188 MiB (~8 GB — small; size microbenches to fit) |
| conda env | `profiling` → `/home/shuhan/miniconda3/envs/profiling` |
| torch / triton | 2.7.0+cu128 / 3.3.0 |
| nvcc | 12.8 (V12.8.93) |
| Nsight Compute (ncu) | 2025.1.1 at `$profiling/nsight-compute-2025.1.1/ncu` |

### sm89 hardware constants (from `torch.cuda.get_device_properties`)
- max_threads_per_sm = 1536  → **48 warps/SM** max
- regs_per_sm = 65536, max regs/thread = 255 (arch cap)
- shared_mem_per_sm = 102400 B (100 KB); per-block default 49152 B, opt-in 101376 B (99 KB)
- warp_size = 32, L2 = 32 MB, SMs = 24
- (arch, not in torch props) max_threads_per_block=1024, max_blocks_per_sm=24,
  register alloc granularity = 256 regs/warp, warp alloc granularity = 4, smem alloc unit = 128 B

## Two gotchas — RESOLVED
1. **CUDA 12.8 nvcc vs system GCC 14 headers**: raw `nvcc` fails with a `noexcept` clash on
   `cospi/sinpi` (`/usr/include/.../mathcalls.h`). **Fix:** pin the conda GCC 11.2 as host
   compiler → `nvcc -ccbin $profiling/bin/x86_64-conda-linux-gnu-gcc`. Baked into `tooling/env.sh`
   as `$NVCC_CCBIN` / `$NVCC`.
2. **WSL2 ncu counters** (HANDOFF §2 flagged this as the usual Ada blocker): **works**.
   `ncu --target-processes all` returns real counter values (occupancy, regs, bank conflicts,
   DRAM, tensor-op util). No `NVreg_RestrictProfilingToAdminUsers` change was needed on this box.
   Earlier "No kernels were profiled" was a self-inflicted `--launch-skip` > launch-count bug, not a
   platform limit.

## Counter map validated (concept → NVIDIA metric, all returning values)
| Concept (model term) | NVIDIA metric | verified value (fp16 2048³ gemm) |
|---|---|---|
| Registers/thread (static) | `ptxas -v "Used N registers"` **or** Triton `k.n_regs` | 234 (ncu) / matches |
| Register spills | `ptxas -v` spill stores/loads; Triton `k.n_spills` | 0 |
| Achieved occupancy (P_occ GT) | `sm__warps_active.avg.pct_of_peak_sustained_active` | 16.08 % |
| Bank conflicts (P_layout GT) | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | 5,799 |
| DRAM traffic | `dram__bytes.sum` | 28.68 MB |
| Tensor-Core util | `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active` | 49.25 % |

**Key enabler for "search-free":** Triton exposes `kernel.n_regs`, `kernel.n_spills`,
`kernel.metadata.shared` from a **single compile** — exactly the static P_occ input the deployed
pass is allowed to read (no run, no autotune). ptxas -v is the raw-CUDA equivalent. ncu is used
for **fitting/validation only**, never in the deployed decision.

## Deliverables committed this phase
- `tooling/env.sh` — `source` to put nvcc/ncu on PATH + pin host compiler + export sm89 constants.
- `tooling/check_profiling_stack_nvidia.sh` — NVIDIA analogue of the MetaX `check_profiling_stack.sh`;
  probes all 6 concept→metric mappings end to end. Current run: **PASS=11 WARN=0 FAIL=0**.

## Next
Phase 1 cont'd: build the analytical sm89 occupancy model + static-input extractor, then the
fused/unfused microbench kernel matrix, then automate ncu → vendor-neutral dataset. See LOG-02.
