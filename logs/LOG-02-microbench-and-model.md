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

## 3. Dataset (`data/microbench_timing.csv`, 72 rows; 64 genuine + 8 degenerate no-ops)
Families P (48) + R (24) = 72 raw rows. **8 P-rows (pointwise K=1) are degenerate no-ops** (a single
elementwise op is not a fusion; `n_launches_unfused=1`, so the "unfused" plan is a single launch
identical to the fused kernel), excluded from decision scoring ⇒ **64 genuine cases (48 beneficial /
16 toxic)**. Reduction now sweeps NOUT∈{32,64,128} only (the degenerate NOUT∈{8,16} rows removed at
source); pointwise sweeps K∈{1,2,4,8,16,32}. Powers-of-two NOUT only (Triton `tl.arange` width must
be pow2 — this is why NOUT∈{48,96} were skipped).

### Headline finding — the spill cliff drives toxicity (clean beneficial→toxic crossover)
sibling-reduction, fp16, R=C=2048, GS=16 (regenerated 72-row dataset; matrix starts at NOUT=32):
| NOUT | f_regs | f_spills | analytic occ | speedup | beneficial |
|---|---|---|---|---|---|
| 32 | 255 | 0 | 0.17 | 1.01× | ✓ |
| 64 | 255 | **300** | 0.17 | **0.94×** | ✗ toxic |
| 128| 40  | **1986** | 1.00(nominal) | **0.13×** | ✗ toxic |

> **⚠ These numbers are the CORRECTED ones.** The magnitudes originally logged here (0.06× / 0.008×)
> were a **VRAM-starvation measurement artifact**, not physics — see `logs/LOG-10-ada-bucket1.md` §1.
> When torch's caching allocator grew until its *reserved* footprint oversubscribed the 8 GB card,
> WSL2/WDDM left the CUDA context's local-memory backing store host-resident, so every spill access
> crossed PCIe. This inflated the fused time of **spilling kernels only**: all 16 spilling rows moved
> one-directionally between the two runs (speedup ×9.7–×51), while the 56 non-spilling rows moved
> **bidirectionally** (×0.54–×1.44) — ordinary run-to-run variation, not the artifact. **Among the 64
> genuine cases exactly one label changed** (reduction fp16 R=C=1024 NOUT=32: 0.906→1.081). Note that
> row has **zero spills**, so its flip is *not* attributable to the artifact — it is a borderline case
> re-rolling across the noise band. The NOUT=32 entry above likewise moved 1.29→1.01 on a zero-spill
> row: also run-to-run variation, not the artifact. The cliff's *shape* (beneficial → toxic at the
> spill onset) is unchanged; only its depth was overstated.

- **Every one of the 16 spilled fused kernels is toxic, and every one of the 48 non-spilling genuine
  cases is beneficial** (NOUT∈{64,128} × 2 dtypes × 4 shapes). Spills remain a perfect static toxicity
  signal on this benchmark — but the *margin* is regime-dependent, which the artifact had hidden:
  NOUT=64 is a **mild** 0.88–0.98× slowdown, NOUT=128 a severe 0.09–0.29×. **fp16 and fp32 both first
  spill at the same NOUT=64** (fp16 300 / fp32 310 spills at the cliff); fp32 spills marginally more
  per case but does **not** cross at a smaller NOUT. The genuine crossover is **NOUT=32 (beneficial, no
  spill) → NOUT=64 (toxic, spills)**.
- **Honest caveat on the NOUT=64 labels:** at 0.88–0.98× these sit close enough to 1.0 that a
  single-shot `t_fused < t_unfused` threshold is noise-prone (the fp16 bucket contains a member at
  0.982, and has no CI). Only redux N64 fp32 is CI-backed (median 0.893, 95% CI [0.893, 0.898],
  20 rounds — `data/timing_ci_ada.csv`). Treat NOUT=64/fp16 "toxic" as provisional.
- **Degenerate no-ops excluded from decision scoring:** with GS≥NOUT the "unfused" plan is a single
  launch identical to the fused kernel, so its label is timing noise. On the clean 72-row dataset
  these are the **8 pointwise K=1 rows** (a single elementwise op is not a fusion, `n_launches=1`);
  the old NOUT∈{8,16} reduction no-ops are gone (matrix now starts at NOUT=32). `model/fit.py`
  excludes any `n_launches_unfused≤1` row (see `REVIEW_FINDINGS_TODO` item 7 + A1).
- **The NOUT=128 nuance:** ptxas caps registers at 40 and spills massively (1986), so the
  *register-based* occupancy reads a misleading 1.00 — yet it is still the most toxic case (**7.9×
  slower** at R=C=2048 fp16; 0.13× speedup). The artifact-era figure for this row was 125× slower.
  ⇒ **the spill count must override register-derived occupancy.** The model encodes exactly this
  (spill_factor multiplies efficiency independent of the occupancy term).

## 4. Cost model & RQ1 (`model/fit.py`)
Model (faithful to PROPOSAL §5.2, extended so the memory/latency-bound regime is representable):

    eta_fused = min(eta_u,eta_v)·P_occ·P_layout ,   with P_occ = lam(occ)·spill_factor(spills)
    T_plan = max( F/(C_peak·eff), M/(B_peak·eff) ) + L·T_launch      (eff = the eta above)

Fitted on Ada (combined ratio+abs-time objective; occ_knee physically bounded to [0.08,0.35] since
memory-bound Ada saturates HBM by ~⅓ occupancy). Refit on the corrected dataset
(`model/ada_constants.json`):
- **B_peak ≈ 1.81e11 B/s (181 GB/s)** — ~71% of the 4060 laptop's 256 GB/s theoretical peak
  (128-bit GDDR6 @ 16 Gbps), a plausible achieved bandwidth and a good sanity check.
- **T_launch ≈ 1.34e-4 s**; **gamma_spill ≈ 0.00567** (the artifact-era fit was ~14× larger at 0.0807
  — the inflated spill rows had demanded a much steeper spill penalty; RQ4's decisions are unchanged
  by the refit, see LOG-10).
- occ_knee → floor (0.08): occupancy above ~8% does not drive toxicity here; **spills do**.

### RQ1 — toxic-fusion decision quality (positive class = "don't fuse")
| | precision | recall | F1 | acc |
|---|---|---|---|---|
| **cost model (in-sample, 64 genuine cases)** | **1.000** | **1.000** | **1.000** | **1.000** |
| greedy-always-fuse baseline | 0.000 | 0.000 | 0.000 | 0.750 |

- **TP=16, FP=0, FN=0, TN=48.** The model reproduces every label. The old single false negative
  (sibling_redux NOUT=32, fp16, R=C=1024) is gone — but *because the measurement flipped, not because
  the model improved*: that row now reads 1.081 (beneficial) instead of 0.906, and it has zero spills,
  so it was always outside the spill-focused model's reach. Perfect recall here is partly luck of the
  re-run; the blind spot for **mild non-spill toxicity** is untested, not fixed.
- **⚠ F1=1.000 is the score of the dataset, not of the model.** After the degenerate filter the Ada
  benchmark is *perfectly separable by a single raw feature*: all 48 non-spilling genuine cases are
  beneficial and all 16 spilling ones are toxic, with zero exceptions. A **zero-parameter rule —
  "don't fuse iff the fused kernel spills" — scores exactly the same 1.000** in-sample. So this result
  carries **no evidence for the 5-parameter roofline model's structure**; the only baseline reported
  here (greedy-always-fuse) never predicts toxic and cannot discriminate the two. Ada RQ1 establishes
  only that **spilling is a sufficient toxicity indicator on this benchmark**.
- **The perfect score is load-bearing on the degenerate filter:** 7 of the 23 raw toxic labels sit in
  the 8 excluded pointwise K=1 rows (speedups 0.91–1.01, i.e. the noise band). Keeping them would cap
  recall at 16/23 = 0.696. The exclusion is principled (`n_launches_unfused≤1` = no fusion decision to
  make), but its magnitude should be stated.
- **Caveat on this "held-out-shape" CV:** it holds out (R,C), but spills — the toxicity driver — are
  *identical* across shapes, so pooled shape-CV ≈ in-sample; it tests robustness to matrix size, not
  to the register/spill regime. The honest held-out-decision-variable CVs (LOG-10 §3):
  leave-one-dtype-out F1=0.968 (recall 0.938) — the only real in-family generalization signal;
  **leave-one-NOUT-out F1=0.667 (recall 0.500)** — the new honest limitation.
- **What the NOUT fold actually shows (it is a fit defect, not an extrapolation limit):** holding out
  NOUT=128 collapses `gamma_spill` to the ~1e-8 floor — the fit abandons the spill term *entirely*,
  even though the genuinely toxic 300-spill NOUT=64 rows remain in training — and then misses all 8
  toxic 1986-spill cases. The reverse fold (hold out NOUT=64) learns gamma_spill=0.0067 and scores
  F1=1.000. So `gamma_spill` is **identifiable only from the extreme-spill regime**; this is a
  parameter-identifiability failure, not a too-small-but-fitted penalty. The zero-parameter spill rule
  scores F1=1.000 (recall 1.000) under the *same* protocol, i.e. it **strictly dominates the fitted
  model out-of-sample**. We therefore do **not** claim Ada RQ1 as evidence that the fitted model
  generalizes across spill regimes.
- **The RQ1 bootstrap CI is degenerate** and reported only for completeness: with predictions frozen at
  the in-sample fit and matching the labels elementwise, *every* resample returns F1=1.0, so
  "F1=1.000, 95% CI [1.000,1.000]" is a mathematical identity, not an estimate. It quantifies no
  sampling error and has no power to detect overfitting (that would need a refit inside each resample).

### Ablations (which interpretable term matters)
| ablation | F1 | reading |
|---|---|---|
| full model | 1.000 | — |
| **drop spill term** | **0.000** | spills are the *only* toxic signal the model has on Ada |
| drop smooth-occupancy (occ=1) | 1.000 | the smooth P_occ term is negligible on Ada |

⇒ On Ada, the interpretable signal that matters is the **spill discontinuity**, not the smooth
occupancy curve — matching the proposal's emphasis on the spill cliff. The ablation is now sharper
than before (drop-spill fell 0.545→0.000 on the corrected data): **without the spill term the model
collapses to greedy** — it predicts *no* toxic case at all, scoring identically to the always-fuse
baseline. This is the same fact as the separability caveat above, viewed from the other side: on Ada
the spill term is doing 100% of the discriminative work, and the remaining parameters are unevidenced
by this dataset. Whether the roofline structure earns its keep must be settled on the C500, where
spilling and toxicity come apart (a C500 fp16 128×128 GEMM spills 117 and is still beneficial).

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
  it is **not** consumed by the cost model.
- **The ncu/event-timing divergence was the artifact's early warning — and we misread it.** This log
  previously recorded a large unexplained gap for the extreme-spill case (ncu ~71 ms vs ~1265 ms
  event-timed at NOUT=128, R=C=2048) and waved it off as "ncu reports isolated kernel time under
  clock-controlled replay". That explanation was **wrong**: on the corrected data the same row is
  82.99 ms event-timed against ncu's 71.03 ms, and all four NOUT=128 rows now agree to **1.06–1.39×**
  (`data/microbench_ncu.csv` vs `data/microbench_timing.csv`). ncu runs the kernel in its own process
  and so never entered the oversubscribed regime — meaning the ~18× gap *was* the VRAM-starvation
  artifact, visible in-repo the whole time. This also makes ncu a useful **independent cross-check**:
  a measurement taken by a different tool, in a different process, corroborates the corrected numbers.
  Lesson: an unexplained order-of-magnitude disagreement between two instruments is a bug to chase,
  not a footnote to rationalize.

## 7. Next (this session)
- Finish ncu ground-truth pass → RQ2 attribution accuracy + occupancy-model validation (running).
- Add the CUDA bank-conflict layout cases to RQ2 (layout-attribution branch).
- Phase 3: offline recommender + end-to-end on real subgraphs (attention/MLP) → RQ4.
- Freeze model spec (`model/ada_constants.json` + formulas + schema) for the MetaX transfer.
