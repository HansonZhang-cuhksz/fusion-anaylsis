# LOG-04 — MetaX C500 cross-vendor transfer  [Phase 4]

Date: 2026-07-14 · Machine: MetaX C500 ×4 (MACA 3.7.0), env `fusion` · Session: MetaX (this machine)

Phase 4 = the core-novelty transfer (PROPOSAL §8 / TODO G1): re-parameterize the frozen model to the
MetaX C500, validate transfer (RQ3), and find ≥1 **decision-flip** (a fusion safe on Ada, toxic/illegal
on C500). Formulas are frozen; only `DeviceConstants` + `HardwareModel` change (MODEL_SPEC §7).

## 1. C500 hardware constants (verified this machine)
From `torch.cuda.get_device_properties` + `cucc -resource-usage`, GPU 1 (idle target):
| field | C500 | Ada sm89 | note |
|---|---|---|---|
| compute units (SM/CU) | **104** | 24 | |
| max threads / CU | 2048 | 1536 | |
| **wavefront (warp) size** | **64** | 32 | **the big divergence** — 64-thread waves |
| max waves / CU | 32 (2048/64) | 48 | |
| register file / CU | **131072** | 65536 | 2× Ada |
| shared mem / CU | 65536 (64 KB) | 102400 (100 KB) | |
| L2 | 8 MB | 32 MB | |
| VRAM | ~64 GB | 8 GB | |
| register file | **split ST + MT** | unified | Triton `n_regs` = MTregisters (vector) |
| spill cap | **hard 4 KB/thread** | soft (local-mem traffic) | `mcErrorMemoryValueTooLarge` on overflow |
| compute cap (cu-bridge) | 8.0 | 8.9 | reports sm80-like |

`HardwareModel METAX_C500` added to `fusion/hw.py`; selected via env `FUSION_HW=c500`
(`default_hw()`), wired through `fusion/static.py`. Occupancy-granularity fields
(`reg_alloc_unit`, `warp_alloc_granularity`, `max_blocks_per_sm`, `smem_alloc_unit`,
`max_regs_per_thread`) are physically-motivated **estimates pending MCPTI calibration** — but the
smooth occupancy term is inert (spills dominate, per Ada), so P_occ precision is secondary.

## 2. Pipeline works on C500 (the make-or-break)
- The Triton microbench kernels **compile and run correctly** on C500 (`case.check()` max|Δ|=0.0).
- Triton exposes the single-compile static inputs on MetaX: `k.n_regs`, `k.n_spills`,
  `metadata.shared`, `metadata.num_warps` — the search-free premise transfers unchanged.
- Same fixed 72-case matrix (`fusion/matrix.py`) → directly comparable to the Ada dataset.

## 3. Early cross-vendor signal (before full dataset)
Same kernel `sibling_redux NOUT=64 fp16 R=C=2048`: **C500 spills 524 vs Ada 300** — the tighter/wider
64-wide-wave register pressure makes C500 spill more per case. This is the mechanism the decision-flip
thesis predicts (sharper spill cliff on C500). NOUT=128 may cross the hard 4 KB cap → **launch
failure** (a fusion that merely spills-but-runs on Ada becoming *illegal* on C500) = the cleanest flip.

## 4. Cross-vendor results (full 72-case C500 matrix; `data/microbench_timing_c500.csv`, 0 skips)
Reproduce: `python -m model.transfer_c500`. Reduction comparison (mean over 4 shapes):

| NOUT | dtype | Ada spill | Ada speedup | Ada | C500 spill | C500 speedup | C500 | |
|---|---|---|---|---|---|---|---|---|
| 32 | fp16 | 0 | 1.05 | ✓ benef | 100 | 1.08 | ✓ benef | — |
| **32** | **fp32** | **0** | **1.04** | **✓ benef** | **100** | **0.67** | **✗ toxic** | **⇐ DECISION-FLIP** |
| 64 | fp16 | 300 | 0.08 | ✗ | 524 | 0.46 | ✗ | both toxic |
| 64 | fp32 | 310 | 0.07 | ✗ | 524 | 0.30 | ✗ | both toxic |
| 128 | fp16 | 1986 | 0.01 | ✗ | 844 | 0.40 | ✗ | both toxic |
| 128 | fp32 | 1868 | 0.01 | ✗ | 848 | 0.26 | ✗ | both toxic |

**The decision-flip (the paper's core result).** `sibling_redux NOUT=32 fp32` is **beneficial on Ada
(0 spills, 1.04×) but toxic on C500 (100 spills, 0.67×)** — consistent across all 4 shapes. Mechanism:
the C500's **64-wide wavefront** doubles register pressure per wave, so the fused NOUT=32 kernel spills
on C500 where it fits in registers on Ada's 32-wide waves. The **same model, fed each device's compile
report, flips its verdict**: BENEFICIAL on Ada (reads 0 spills) → TOXIC on C500 (reads 100 spills). No
model change — just the re-read single-compile static input.

Note the *shape* of the cliff also inverts at the extreme: Ada's NOUT=128 is far more toxic (0.01×,
ptxas caps regs at 40 and spills 1986) than C500's (0.40×, 844 spills) — C500's 2× register file
absorbs more before catastrophe, so its cliff is *earlier but shallower*.

## 5. RQ3 transfer accuracy + honest limitations
Re-fit `DeviceConstants` on C500 (formulas frozen; `model/c500_constants.json`):
`gamma_spill 0.0807→0.0068`, **`B_peak 0.15→1.05 TB/s`** (physically correct for C500 HBM — a good
sanity check), `T_launch 1.6e-5`.

- **RQ3 decision quality on C500 (64 genuine cases):** P=0.833 R=**1.000** F1=**0.909** acc=0.938
  (TP=20 FP=4 FN=0 TN=40). The model transfers by re-parameterization.
- **Honest nuance 1 — re-parameterization is *not* what makes it transfer.** Applying the *Ada*
  constants to C500 data gives the **identical** F1=0.909: the binary decision is spill-dominated, so
  the flip is driven by the **re-read hardware-specific spill count**, not by re-fitting constants.
  (Constants still matter for absolute-time / roofline prediction, not for this binary decision.)
- **Honest nuance 2 — a same-hardware blind spot.** The 4 C500 false positives are `NOUT=32 fp16`
  (100 spills, *beneficial* 1.08×): identical spill count to the toxic `NOUT=32 fp32` (100 spills,
  0.67×), so the spill-only model flags both toxic. Spill *count* alone doesn't separate them — the
  memory-saving-vs-compute trade-off is dtype-dependent (ties to gap G2/G3: the model needs a
  non-spill signal).
- **Recall is 1.000 on C500** (vs 0.941 on Ada): the Ada FN (a non-spill toxic case) does not recur
  here because on C500 that case spills — the very mechanism of the flip.

## 7. RQ2-on-C500 — MCPTI ground truth (attribution validated on hardware)
The `mcProfiler` CLI value-dump is **confirmed unimplemented** in MACA 3.7.0 (configures counters
fine, but `MctxStreamProfilerCountDataGet` returns grpc_status 12 UNIMPLEMENTED). So I drove **MCPTI
directly via ctypes** — the working recipe (now in `fusion/mcpti_profile.py`, reproduce
`FUSION_HW=c500 MACA_VISIBLE_DEVICES=1 python -m fusion.mcpti_profile`):
- `libmcruntime.so :: mcCtxGetCurrent` → the MCcontext (after torch inits it);
- `libmcpti.so` Event API (CUPTI-compatible), `SetEventCollectionMode(CONTINUOUS)`;
- events are **CUPTI-legacy names**: `local_load`/`local_store` (spill/private), `global_load`/`store`
  (DRAM), `shared_ld`/`st_bank_conflict` (layout) — *not* the `AP_PERF_*` names the CLI shows;
- counters are **free-running (cumulative)** → per-kernel value = **read-before / read-after delta**.

Result (`data/microbench_c500_mcpti.csv`, 12 reduction cases at 2 shapes):
| NOUT | spills | local(spill) inst | DRAM inst | dominant |
|---|---|---|---|---|
| 32 | 100 | 1.5M | 33K | spill |
| 64 | 524 | 15.3M | 33K | spill |
| 128 | 844 | 26.4M | 50K | spill |

- **Local (spill) traffic scales monotonically with the static `n_spills`** and **dominates DRAM by
  ~307× on average** — the static single-compile spill count is grounded in real hardware private-memory
  traffic. Attribution ground truth = **spill on 12/12**.
- **RQ2b-on-C500: model dominant-penalty == profiled dominant on 12/12** (the C500 analogue of Ada's
  8/8). The interpretable spill attribution holds cross-vendor, on hardware.
- (`shared_*_bank_conflict` is nonzero — the `tl.sum` reduction uses shared memory — but local ≫ bank,
  so `spill` correctly dominates.)

## 8. Status / next
- [x] Env (`fusion`); C500 `HardwareModel`; pipeline verified; full C500 matrix; comparison table;
      C500 re-fit; RQ3; **decision-flip** documented; **RQ2 MCPTI attribution ground truth (12/12)**.
      → **G1 DoD met** (transfer accuracy + explained decision-flip + hardware-validated attribution).
- [ ] Achieved-occupancy on C500 needs the MCPTI **Metric** API (no raw `waves` event; the Event API
      exposes cycles/`sm_cta_launched` only) — deferred; low value (occupancy term is inert).
- [ ] Calibrate C500 occupancy granularities (secondary).
- [ ] Fold C500 results into `MODEL_SPEC`/`PROPOSAL` + a cross-vendor generalization table.
