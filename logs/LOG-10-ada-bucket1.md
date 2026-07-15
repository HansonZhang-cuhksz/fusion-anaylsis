# LOG-10 — Ada Bucket-1: GEMM on Ada, Ada CIs, and a measurement artifact in the Ada dataset

Date: 2026-07-15 · Machine: RTX 4060 Laptop (Ada sm89), 8 GB, WSL2 · env `profiling`
Scope: `HANDOFF-ADA-BUCKET1.md` A1/A2/A3 — **plus** a data-integrity bug found while doing A2.

> **Provenance note.** An earlier draft of this log made four claims that an adversarial verification
> pass then **refuted** (the 64-wide-wavefront mechanism; "root cause proven"; the `empty_cache()` fix
> being "verified"; Ada as a "negative control"). They are retracted below and the retractions are the
> authoritative version. `RESULTS.md` and `model/MODEL_SPEC.md` carry the same corrected text.

---

## 0. TL;DR
- **A1 (GEMM on Ada): a second decision-flip.** Ada never spills on the GEMM-epilogue family
  (`f_spills=0` on all 16 configs) and no configuration is significantly toxic; the *same* fp32 128×128
  tile spills **205** and is **toxic on C500** (0.824, CI [0.821, 0.827]).
- **A2 (Ada CIs):** both flips significant on both sides — *subject to a CI caveat (§3) that our
  intervals bound short-term noise only*.
- **A3 (sm80): BLOCKED** — no sm80 GPU reachable (only the RTX 4060). User confirmed.
- **⚠ Found a measurement artifact** that inflated the Ada spill cliff. **The corrected dataset is
  trustworthy** (independently reproduced end-to-end); the *causal explanation* is corroborated but
  **not proven**, and the fix is **prophylactic, not demonstrated**. Corrected, Ada RQ1 is perfect —
  but see §3: the benchmark is **trivially separable**, which is the more important finding.
- **C500 results are unaffected.**

---

## 1. ⚠ The artifact (Ada-only)

### Symptom
`data/microbench_timing.csv` said `redux N64 fp32 R=C=2048` → t_fused **55.9 ms**, speedup **0.055**.
Re-measuring the same config gave **4.09 ms / 0.892**. Three independent methods agree on ~0.89: an
isolated probe, the runner's own `run_case` in isolation, and `fusion/timing_ci.py` (20 rounds,
median 0.893, CI [0.893, 0.898]). Pre-fix means: **0.077** (NOUT=64) and **0.008** (NOUT=128).

### Mechanism — corroborated, NOT proven
torch's caching allocator retains freed blocks until its **reserved** footprint *oversubscribes*
physical VRAM (observed 7.82–8.08 GB reserved on an 8 GB card with ~6.9 GB available). WSL2/WDDM
permits the oversubscription instead of failing, and the CUDA context's local-memory backing store can
be left **host-resident**, so every spill access crosses PCIe. This inflates **spilling kernels only**;
non-spilling kernels are untouched *in ratio* (1.040 vs 1.043 healthy). Reproduced on demand: at
8.08 GB reserved / 0.00 GB free, `redux N128 fp32 2048²` measures **754 ms (0.0097×)** vs a healthy
**78 ms (0.122×)**.

**What is NOT established** (the earlier draft claimed otherwise):
- **The trigger is oversubscription (reserved > physical), *not* "free VRAM ≈ 0".** Replaying the entire
  pre-fix path (48 pointwise cases, no `empty_cache`, **free 0.000 GB** but reserved 6.73 GB) reproduces
  the **corrected** values exactly (N64 fp32 2048² = 0.8939, t_f 5.362 ms). The earlier "PHASE1 vs
  PHASE3 free-VRAM" table is confounded (PHASE3 differs by free VRAM *and* by having run the whole
  pointwise family: thermal state, fragmentation, oversubscription). **"Root cause (proven)" was wrong.**
- **The "order matters / first-load under starvation" rule is falsified.** A fresh N64 kernel
  *first-loaded* under free 0.000 GB / reserved 8.08 GB is **healthy** (0.913). Damage appears only
  after a large-spill (N128) launch grows the context-wide local-memory store into sysmem, which then
  poisons subsequent spilling kernels.
- **Severity is spill-magnitude- and history-dependent, not a uniform ~10×.** A fresh oversubscribed
  repro damages only the 1868-spill NOUT=128 rows and leaves the 300-spill NOUT=64 rows healthy
  (0.913) — whereas the original run inflated NOUT=64 too (t_f 13.55 ms). Not fully characterized.
- No direct instrumentation of host/PCIe traffic for the local-memory store was performed.

### The fix is prophylactic — a design argument, not a measured result
`torch.cuda.empty_cache()` before building each case (`fusion/runner.py::run_case`,
`fusion/gemm_sweep.py::run_one`, `fusion/endtoend.py::time_layer_width`,
`model/recommender.py::score_grouping`). Honestly:
- Once a context's local-memory store is host-resident it does **not** migrate back — `empty_cache()`
  does **not repair** an already-poisoned process (N64 still 9.70 ms / 0.129× after `empty_cache`
  restored 6.36 GB free).
- In a direct A/B on one process the fix **changed nothing** (0.8939 unfixed vs 0.8937 fixed) because
  that replay never entered the oversubscribed regime. **No committed experiment shows the fix changing
  any number.** It can only work by keeping the cache from *reaching* oversubscription.
- The matrix wall-clock drop (253.7 s → 52.1 s) is a **symptom** of the artifact being absent on the
  re-run, **not** evidence the fix caused the correction; an unfixed replay is equally fast today.

### Confidence in the corrected dataset is HIGH and independent of the causal story
All 24 reduction rows were independently reproduced end-to-end (n64/2048/fp32 0.8939 vs 0.8939;
n128/2048/fp32 0.1224 vs 0.1191; n32 rows 1.00–1.08). **Static columns** (`f_regs`, `f_spills`,
`f_smem`, `f_occ`, `u_*`, bytes, flops) are **byte-identical** before/after — ptxas is deterministic;
only timings moved. Of 72 rows just **4 labels** changed: 3 are the *excluded* degenerate pointwise-K=1
no-ops, and exactly **1 genuine case** — `sibling_redux_n32 fp16 R=C=1024`, 0.906 → **1.081** — which is
precisely the false negative earlier written up as the model's "honest blind spot". It was the artifact.
*(Caveat: non-spilling **absolute** times also moved run-to-run — 1024² rows ~1.75×, 2048² N32 ~1.4× —
a second, unexplained difference; both plans moved together so ratios/labels are unaffected.)*

### Not affected (checked, not assumed)
- **RQ2 / ncu ground truth.** `fusion/profile.py` spawns a **fresh process per plan**
  (`ncu python -m fusion.ncu_worker`, one case), so it never oversubscribes. No re-run needed.
- **RQ4 / endtoend.** The endtoend harness **never entered the starved regime** (pre-fix greedy(w128)
  = 151.4 ms is consistent with unstarved timing). The pre-fix RQ4 was **not** contaminated; the small
  shift (9.72 → 8.65×) is ordinary run-to-run variation. *The `empty_cache()` call added there is
  prophylactic hygiene only — an earlier code comment asserting the RQ4 speedup was "fabricated"
  without it was wrong and has been corrected.*
- **The Ada GEMM sweep** — Ada never spills, and the artifact only harms spilling kernels.
- **C500 (verified).** Its matrix and independent `timing_ci` agree (redux N32 fp32 0.6341 vs 0.6415;
  N64 fp32 0.2699 vs 0.271; deviations ≤1.6%). A large-VRAM datacenter card cannot reach an 8 GB card's
  oversubscription, and the C500 matrix allocates the same way without ever approaching its capacity.
  *Limits:* we cite matched points from 5 available; all 5 matrix values sit just outside the (very
  tight, within-process) CIs; and no direct free-VRAM measurement exists on the C500 (not reachable
  from this box). **Every C500 result stands.** The fix is committed on the shared paths as insurance.

---

## 2. A1 — CONTRACTION (GEMM-epilogue) on Ada → a second decision-flip

**Ada spills nowhere on this family.** Same kernel, same tile, same dtype:
| | Ada (sm89) | MetaX C500 |
|---|---|---|
| fp32 128×128 fused | 255 regs, **0 spills** | 256 regs, **205 spills** |
| fp32 128×128 speedup | **1.083**, CI [1.072, 1.104] beneficial | **0.824**, CI [0.821, 0.827] toxic |
| fp16 128×128 fused | 232 regs, 0 spills | 256 regs, 117 spills |
| 64×64 fp16 / fp32 regs | 84 / 96 | **134 / 168** |

⇒ **Handoff outcome (b):** the toxic-GEMM regime **exists on C500 but not on Ada**.

**Mechanism — corrected.** The real signal is that the **C500 toolchain allocates 1.6–1.75× more
registers per thread than ptxas for identical kernels** at tiles where *neither* device spills
(64×64 fp16 84→134; fp32 96→168). At 128×128 fp32 it saturates its 256-reg/thread cap and spills, while
Ada's allocation lands at 255 — just inside its own cap. **We do NOT attribute this to 64-wide
wavefronts** (the earlier draft did): the C500's register file per CU is doubled *in lockstep*
(131072 vs 65536, `fusion/hw.py`), so per-wave demand and per-CU capacity cancel — back-solving both
CSVs' `f_occ`, both devices hold **8** resident waves/warps on the reduction and **4** on the 128×128
GEMM. Wave width also predicts the **wrong direction** (the C500 spreads the same tile over 2× the
threads, which should *halve* per-thread demand). Separating **compiler register-allocation maturity**
from an ISA/accumulator-layout cause needs a per-wave register-demand measurement we do not have.

**Spilling is necessary but not sufficient** — a counter-example is in our own data: the C500's fp16
128×128 tile spills **117** and is still **beneficial** (1.011–1.064). And the reduction flip's dtype
dependence is unexplained by register state: at R=C=2048 NOUT=32 the C500's fp16 and fp32 rows have
*identical* `f_regs=256, f_spills=100, f_occ=0.25`, yet fp16 = 1.065 (beneficial) and fp32 = 0.634
(toxic). So "two families, one mechanism" is **not** established; what is established is the *outcome*
(a flip) in two families.

**No Ada GEMM configuration is significantly toxic.** Two of 16 rows carry single-shot toxic labels
(0.973, 0.957); on repeat the fp16 128×128 row is **beneficial** (1.0250, CI [1.025, 1.029]) and the
fp32 64×64 row is **inconclusive** (0.9904, CI [0.964, 1.055] — spans 1.0). We do **not** claim it is
benign, only that the single-shot label is unsupported.

**Does the reread fix hold on Ada? It is vacuous by construction — not a control.** The reread
multiplier attaches only to CONTRACTION→POINTWISE pairs (the 16 GEMM rows), and every one has
`f_spills=0` on Ada. Since the term is `spill_factor(spills × reread)`, the reread column **multiplies
zero**: the no-reread and reread-aware training sets give a **bit-identical objective** (loss delta 0.0
at every parameter setting) → identical fits (`gamma_spill=0.04881` both ways; overall F1 0.875 /
recall 0.778 both ways). This is an **arithmetic identity, not a measurement**: it is **no evidence**
that the fix is correct, and **equally none that it is harmless**, because on Ada the feature has no
path to affect any decision. **We therefore do not claim Ada as a negative control** (a control must be
able to fail). The fix's evidence remains the **C500 result alone** (LOG-06: held-out GEMM recall 1.0
with reread=2 vs 0.0 with reread=1) — validated only where the failure mode exists.
*Caveat:* the "GEMM F1/recall = 0.000 (n_toxic=2)" printed by `model/refit_combined_ada.py` is **not a
meaningful score** — both "toxic" labels are the noise the CIs overturn, so the 2 FNs are the model
disagreeing with a *wrong label*.

`model/build_combined_ada.py` mirrors `build_combined_c500.py` but **derives** `u_spills` from Ada's own
compile probe rather than hardcoding C500's numbers (which would be wrong for Ada).

---

## 3. A2 — Ada CIs + RQ1

**Both flips, both sides** (Ada 20 rounds; C500 = 4 GPUs × 20 rounds, LOG-09). `timing_ci.py` now takes
device-aware expectations (`--expect ada|c500`, default follows `$FUSION_HW`).
| claim | Ada median [CI] | Ada | C500 median [CI] | C500 | FLIP |
|---|---|---|---|---|---|
| redux_N32_fp16 | 1.011 [1.010, 1.013] | beneficial | 1.082 [1.077, 1.084] | beneficial | — |
| **redux_N32_fp32** | **1.040 [1.040, 1.041]** | **beneficial** | **0.642 [0.638, 0.645]** | **TOXIC** | ✅ |
| redux_N64_fp32 | 0.893 [0.893, 0.898] | TOXIC | 0.271 [0.270, 0.272] | TOXIC | — |
| gemm_128_fp16 | 1.164 [1.124, 1.190] | beneficial | 1.064 [1.059, 1.068] | beneficial | — |
| **gemm_128_fp32** | **1.083 [1.072, 1.104]** | **beneficial** | **0.824 [0.821, 0.827]** | **TOXIC** | ✅ |

**⚠ CI caveat (important).** These are within-process percentiles over min-of-N timings: they bound
**short-term noise only**. Two independent 20-round runs of the *identical* 2048×2048×512 fp32 BM128
config produced **non-overlapping** intervals ([1.111, 1.118] vs [1.072, 1.104]) — run-to-run variation
on this thermally-throttling laptop **exceeds the quoted CI width**. **Treat any verdict within ~5% of
1.0 as unresolved.** (The C500's [0.821, 0.827] is likewise a cross-GPU *envelope* of four per-GPU CIs.)
Both flips are far outside ±5%, so they survive this caveat; `redux_N32_fp16` (1.011) does not.

**RQ1 on the corrected dataset** (`python -m model.fit`):
| | precision | recall | F1 | acc |
|---|---|---|---|---|
| in-sample (64 genuine) | 1.000 | 1.000 | **1.000** | 1.000 |
| leave-one-dtype-out | 1.000 | 0.938 | 0.968 | 0.984 |
| leave-one-NOUT-out | 1.000 | 0.500 | **0.667** | 0.667 |
| greedy-always-fuse | 0.000 | 0.000 | 0.000 | 0.750 |
Bootstrap (`model/bootstrap_ci.py`, B=1000): F1 = 1.000, CI [1.000, 1.000] — **degenerate**: with
predictions frozen at the in-sample fit and pred==true on every case, every resample is 1.0 by
construction. It measures nothing about generalization.

**⚠ The decisive caveat — the benchmark is trivially separable.** After the degenerate filter, Ada's
64 genuine cases are perfectly separated by the single **raw** feature `f_spills > 0`: 48/48 beneficial
at 0 spills, 16/16 toxic at >0 spills, **zero exceptions**. A **zero-parameter rule — "don't fuse iff
the fused kernel spills" — also scores F1 = 1.000**, and scores **leave-one-NOUT-out F1 = 1.000 /
recall = 1.000** where the fitted 5-parameter model gets **0.667 / 0.500**. On this dataset the fitted
model is **tied in-sample and strictly dominated out-of-fold** by the trivial rule. F1 = 1.000 is
therefore **not evidence for the cost model** — it is evidence that the Ada benchmark is too easy to
discriminate models. (The model's value must come from cases the trivial rule cannot handle: the C500's
*spilling-but-beneficial* fp16 128×128 tile, and the roofline/reread terms — i.e. from C500 data.)

**And the leave-one-NOUT-out story is a fit failure, not extrapolation.** Holding out NOUT=128, the fit
drives `gamma_spill → 8.12e-09` (collapses to the floor — it **abandons the spill term entirely**), so
the earlier "gamma fit on the 300-spill regime under-predicts the 1986-spill regime" explanation is
**mechanically wrong**. `gamma_spill` is identifiable only from the extreme-spill regime.

---

## 4. A3 — sm80: BLOCKED
`nvidia-smi` exposes one GPU: `NVIDIA GeForce RTX 4060 Laptop GPU, 8.9`. No Ampere sm80 reachable; the
handoff gates A3 on reachability. Not attempted (user-confirmed). `RESULTS.md` previously said "Ampere
sm80 is available but not yet used" → corrected to *not reachable from the Ada machine*. The 3-way table
remains open.

---

## 5. Files
- **New data:** `data/microbench_gemm_ada.csv`, `data/microbench_ada_combined.csv`,
  `data/timing_ci_ada.csv`, `data/rq1_bootstrap_ada.csv`.
- **Regenerated (post-fix):** `data/microbench_timing.csv`, `model/ada_constants.json`,
  `logs/run_timing_matrix.log`, `logs/run_endtoend.log`, `figures/*`.
- **New code:** `model/build_combined_ada.py`, `model/refit_combined_ada.py`, `model/bootstrap_ci.py`,
  `tooling/repro_vram_artifact.py` (the artifact repro, so the evidence is reproducible from the repo).
- **Changed:** `empty_cache()` in `fusion/runner.py`, `fusion/gemm_sweep.py`, `fusion/endtoend.py`,
  `model/recommender.py`; device-aware expectations in `fusion/timing_ci.py`.

## 6. For the MetaX session
1. **No action needed on C500 data** — verified unaffected. The fix is committed on shared paths anyway.
2. **The GEMM flip is real and bilateral** — fold `gemm_128_fp32` in alongside `redux_N32_fp32`. But do
   **not** repeat the 64-wide-wavefront story (refuted, §2); the honest statement is a **compiler
   register-allocation** difference (1.6–1.75× more regs/thread) that pushes the C500 over its cap.
3. **The strongest Ada contribution is the flip, not RQ1.** Ada RQ1 = 1.000 is trivially separable
   (§3) — the cost model's discriminating evidence must come from C500 (spilling-but-beneficial fp16
   tile; reread). Consider designing a *non-trivially-separable* Ada benchmark if RQ1 is to carry weight.
4. **Spilling is necessary but not sufficient** (C500 fp16 128×128 spills 117 and is beneficial; the
   NOUT=32 dtype flip has identical register state). The "spill ⇒ toxic" story needs this qualifier.
