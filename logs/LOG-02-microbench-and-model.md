# LOG-02 — Microbench matrix, cost model, RQ1 results  [Phase 1–2]

Date: 2026-07-14 · Machine: RTX 4060 Laptop (Ada sm89), WSL2 · env: `profiling`

## 1. What was built
A complete search-free fusion-decision pipeline, all under `fusion/` + `model/`:

- `fusion/hw.py` — sm89 analytical occupancy model (validated to match the CUDA Occupancy
  Calculator: 32 regs→100%, 64→66.7%, 96→33.3%, 255→16.7%, >255→hard-fail). Register-file model is
  **parameterised** (split_regfile, spill_cap_bytes) so it re-parameterises to MetaX C500.
- `fusion/static.py` — reads the **single-compile** static inputs (Triton `n_regs`, `n_spills`,
  `metadata.shared`, `num_warps`) → analytical occupancy + binding resource.
- `fusion/kernels/` — three fused/unfused microbench families (below).
- `fusion/timing.py` — CUDA-event timing, L2-flush, adaptive iters.
- `fusion/runner.py` — builds the whole matrix → `data/microbench_timing.csv` (label + static features).
- `fusion/ncu.py` + `ncu_worker.py` + `profile.py` — automate Nsight Compute into **vendor-neutral
  concept** columns → `data/microbench_ncu.csv` (attribution + occupancy-validation ground truth).
- `model/costmodel.py` — the interpretable degradation model; `model/fit.py` — fit + RQ1/RQ4 metrics.

## 2. Microbench families (op-pair taxonomy)
| Family | producer→consumer | fusion knob | intended mechanism |
|---|---|---|---|
| **P** pointwise chain | pointwise→pointwise | chain depth K | easy-win control (should stay beneficial) |
| **R** sibling reductions | pointwise→reduction (horizontal fusion of NOUT projections) | NOUT (width) | **P_occ / register-spill cliff** |
| **T** transpose→elementwise | permute→pointwise | size; (CUDA) shared PAD | **P_layout / bank-conflict + uncoalescing** |

Each family provides fused and unfused implementations computing identical math (verified `allclose`,
max|Δ|=0), so the fused-vs-unfused latency comparison is fair.

## 3. Dataset (`data/microbench_timing.csv`, 88 rows; 72 genuine + 16 degenerate no-ops)
Families P (48) + R (40) = 88 raw rows. **16 R-rows (NOUT∈{8,16}) are degenerate no-ops** (with
GS=16 the "unfused" plan is a single launch identical to the fused kernel), excluded from decision
scoring ⇒ **72 genuine cases (52 beneficial / 20 toxic)**. Powers-of-two NOUT only (Triton
`tl.arange` width must be pow2 — this is why NOUT∈{48,96} were skipped).

### Headline finding — the spill cliff drives toxicity (clean beneficial→toxic crossover)
sibling-reduction, fp16, R=C=2048, GS=16:
| NOUT | f_regs | f_spills | analytic occ | speedup | beneficial |
|---|---|---|---|---|---|
| 8  | 128 | 0 | 0.33 | 1.19× | ✓ |
| 16 | 128 | 0 | 0.33 | 0.93× | ~ |
| 32 | 255 | 0 | 0.17 | 1.09× | ✓ |
| 64 | 255 | **300** | 0.17 | **0.06×** | ✗ toxic |
| 128| 40  | **1986** | 1.00(nominal) | **0.005×** | ✗ toxic |

- **Every one of the 16 spilled fused kernels is toxic** (speedup < 0.1). Spills are a near-perfect
  static toxicity signal. **fp16 and fp32 both first spill at the same NOUT=64** in this sweep (fp16
  300 / fp32 310 spills at the cliff); fp32 spills marginally more per case but does **not** cross at
  a smaller NOUT.
- **NOUT=8 and 16 are degenerate no-ops** (with GS=16 the unfused plan is a single launch identical
  to the fused kernel), so their speedups above (1.19×, 0.93×) are timing noise, not fusion
  decisions — they are excluded from the decision metrics (see `REVIEW_FINDINGS_TODO` item 7; matrix
  now starts at NOUT=32). The genuine crossover is **NOUT=32 (beneficial, no spill) → NOUT=64
  (toxic, spills)**.
- **The NOUT=128 nuance:** ptxas caps registers at 40 and spills massively (1986), so the
  *register-based* occupancy reads a misleading 1.00 — yet it is the most toxic case (200× slower).
  ⇒ **the spill count must override register-derived occupancy.** The model encodes exactly this
  (spill_factor multiplies efficiency independent of the occupancy term).

## 4. Cost model & RQ1 (`model/fit.py`)
Model (faithful to PROPOSAL §5.2, extended so the memory/latency-bound regime is representable):

    eta_fused = min(eta_u,eta_v)·P_occ·P_layout ,   with P_occ = lam(occ)·spill_factor(spills)
    T_plan = max( F/(C_peak·eff), M/(B_peak·eff) ) + L·T_launch      (eff = the eta above)

Fitted on Ada (combined ratio+abs-time objective; occ_knee physically bounded to [0.08,0.35] since
memory-bound Ada saturates HBM by ~⅓ occupancy):
- **B_peak ≈ 152–254 GB/s** (physically correct for a 4060 laptop, 128-bit GDDR6) — a good sanity check.
- occ_knee → floor (0.08): occupancy above ~8% does not drive toxicity here; **spills do**.

### RQ1 — toxic-fusion decision quality (positive class = "don't fuse")
| | precision | recall | F1 | acc |
|---|---|---|---|---|
| **cost model (pooled shape-CV, 72 genuine cases)** | 0.833 | **1.000** | **0.909** | 0.944 |
| greedy-always-fuse baseline | 0.000 | 0.000 | 0.000 | 0.722 |

- **Recall = 1.0**: catches *all* toxic fusions (0 false negatives — never keeps a toxic fusion).
- The 4 false positives are pointwise K=1 cases (speedup 1.00–1.02) where the label is genuinely
  marginal — a conservative, safe failure mode for a compiler pass.
- **Caveat on this "held-out-shape" CV:** it holds out (R,C), but spills — the toxicity driver — are
  *identical* across shapes, so pooled CV == in-sample; it tests robustness to matrix size, not to
  the register/spill regime. The honest held-out-decision-variable CVs (LOG-03): leave-one-NOUT-out
  F1=1.00, leave-one-dtype-out F1=0.91 (recall stays 1.00 in both).

### Ablations (which interpretable term matters)
| ablation | F1 | reading |
|---|---|---|
| full model | 0.91 | — |
| **drop spill term** | **0.55** | spills are the dominant toxic signal |
| drop smooth-occupancy (occ=1) | 0.91 | the smooth P_occ term is negligible on Ada |

⇒ On Ada, the interpretable signal that matters is the **spill discontinuity**, not the smooth
occupancy curve — matching the proposal's emphasis on the spill cliff.

## 5. P_layout finding (Family T) — fusion is robust on Ada
- Triton strided/uncoalesced transpose→relu: fusion **beneficial at every size tested**
  (1.03–1.76×), even at X=134 MB ≫ 32 MB L2. Coalescing hardware + the saved HBM round trip win.
- Raw-CUDA shared-tile transpose (via `torch load_inline`, `-ccbin` conda gcc-11): bank conflicts
  (PAD 0 vs 1) slow the fused kernel **1.16–1.44×**, but fusion **still beats** the unfused 2-round-trip
  plan. Bank-conflict penalty (~1.3×) < round-trip saving (~2×).

**Scientific conclusion:** on Ada sm89 the *only decision-flipping* toxic mechanism in this sweep is
the **register-spill cliff (P_occ)**. Layout penalties degrade fused kernels but do not overturn the
round-trip savings. This is an honest negative result for P_layout on Ada — and it sharpens the
cross-vendor thesis: the MetaX C500's hard 4 KB/thread spill cap makes the P_occ cliff *closer*, and
its different memory subsystem may let P_layout actually flip decisions (the decision-flip target).

## 6. Gotchas resolved
- `tl.math.tanh` removed in Triton 3.3 → `triton.language.extra.libdevice.tanh` (needs fp32 arg).
- Triton bans list comprehensions AND `list.append` in kernels → 2D accumulator tiles instead.
- ncu profiled setup/curand kernels → filter with `-k regex:(...)` + `compile_probe=False` build path
  so ncu sees only the plan's own launches.
- `torch load_inline` + CUDA 12.8 vs GCC-14 → pass `-ccbin <conda gcc-11>` in `extra_cuda_cflags`.
- **ncu duration units:** `gpu__time_duration.sum` with `--print-units base` is in **nanoseconds**
  (the CSV column is `f_dur_ns`; it was mislabeled `f_dur_us`). This column is informational only —
  it is **not** consumed by the cost model. It also diverges from the event-timed latency for the
  extreme-spill case (ncu ~71 ms kernel duration vs ~1265 ms event-timed at NOUT=128, R=C=2048),
  because ncu reports isolated kernel time under clock-controlled replay, not the free-running loop;
  root-causing the gap would need Ada re-profiling and is not needed (duration is unused).

## 7. Next (this session)
- Finish ncu ground-truth pass → RQ2 attribution accuracy + occupancy-model validation (running).
- Add the CUDA bank-conflict layout cases to RQ2 (layout-attribution branch).
- Phase 3: offline recommender + end-to-end on real subgraphs (attention/MLP) → RQ4.
- Freeze model spec (`model/ada_constants.json` + formulas + schema) for the MetaX transfer.
