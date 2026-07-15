# LOG-03 — Results (RQ1–RQ4), layout study, end-to-end  [Phases 2–3 complete]

Date: 2026-07-14 · Machine: RTX 4060 Laptop (Ada sm89), WSL2 · env: `profiling`

> **[2026-07-15] NUMBERS REGENERATED — read `logs/LOG-10-ada-bucket1.md` first.** The original
> `data/microbench_timing.csv` behind this log carried a measurement artifact that inflated the fused
> time of *spilling* kernels only (9–41×), when torch's caching allocator grew until its *reserved*
> footprint oversubscribed physical VRAM and the CUDA context's local-memory backing store was left
> host-resident over PCIe. The dataset, `model/ada_constants.json` and `figures/` have been
> regenerated; **`gamma_spill` is now 0.00567** (was 0.0807) and `B_peak` 180.6 GB/s (was 151).
> All RQ1/RQ4 numbers below are the post-fix values. RQ2 (ncu occupancy/attribution) and the layout
> branch are unaffected (no spilling kernels involved). What the artifact *did* change is recorded
> inline; what it did **not** change (RQ4's conclusion) is flagged as such rather than as a correction.

## Scope delivered this session
Phases 1–3 of `PROPOSAL.md §8` are complete on Ada: env bring-up, microbench matrix + ncu dataset,
cost-model formalization + fit, the interpretable attribution, the offline recommender, and
end-to-end evaluation on real-ish subgraphs. A **frozen model spec** (`model/MODEL_SPEC.md`) is ready
for the MetaX C500 transfer (Phase 4).

## Headline numbers
| RQ | question | result |
|---|---|---|
| **RQ1** | predict fusion toxicity from static inputs? | **in-sample P=1.000 / R=1.000 / F1=1.000** (TP=16 FP=0 FN=0 TN=48, on 64 genuine cases; 8 degenerate no-ops excluded); F1 **0.941** shape-CV / **0.968** leave-one-dtype-out / **0.667** leave-one-NOUT-out; greedy F1=0.00. **⚠ The in-sample 1.000 is the score of the *dataset*, not of the model** — the benchmark is perfectly spill-separable, so a zero-parameter rule matches it exactly and beats the fit out-of-sample (see RQ1) |
| **RQ2a** | analytical occupancy == ncu *theoretical* occupancy? | **MAE=0.000, 22/22 exact** (reproduces the CUDA occupancy calculator by construction). *Achieved* occupancy differs 21.7 pts mean / 88 max — deliberately not predicted; the spill term captures the real harm |
| **RQ2b** | attribute the cause (spill vs layout)? | **12/12 = 100%**: 8/8 spill-dominated + 4/4 layout-dominated |
| **RQ4** | utility vs greedy / oracle? | model **5.8–8.7× faster than greedy** (8.65× is a constructed worst-case for greedy), **= oracle (1.00×) on all 3 subgraphs**, at ~0 timing cost. Holds in *both* the pre- and post-fix regimes — a robustness result, not a correction |

## RQ1 — search-free toxic-fusion decision
- Cost model (`model/costmodel.py`) fit on Ada (`model/ada_constants.json`): **B_peak≈180.6 GB/s**
  (was 151 GB/s pre-fix; still an achievable-bandwidth figure of the right order for a 4060 laptop),
  occ_knee→floor, **gamma_spill≈0.00567** (the ~14× drop from the pre-fix 0.0807 is the artifact
  leaving the data: the fit no longer has to explain 10×-inflated spilling kernels).
- **precision 1.000 / recall 1.000 / F1 1.000 / acc 1.000** (TP=16, FP=0, FN=0, TN=48) on the
  **64 genuine cases** (8 degenerate no-op rows — pointwise K=1 where n_launches_unfused=1 makes the
  "unfused" plan a single launch identical to the fused kernel — excluded from decision scoring;
  their labels are timing noise, not fusion decisions). **The filter is load-bearing:** 7 of those 8
  rows carry toxic labels, so it removes 7 of the 23 raw toxic labels (23→16). Keeping them would cap
  recall at 16/23=0.696. Their speedups (0.912/0.969/0.988/0.995/0.996/0.997/0.999/1.005) sit in the
  noise band — though the 0.912 row is an 8.8% slowdown, which is not *purely* noise.
- **The perfect in-sample score does NOT validate the cost model's structure.** After the degenerate
  filter this benchmark is trivially separable by one raw feature: **all 48 non-spilling cases are
  beneficial and all 16 spilling cases are toxic, zero exceptions.** A zero-parameter rule — *"don't
  fuse iff the fused kernel spills"* — therefore also scores **F1=1.000 (TP=16 FP=0 FN=0 TN=48)**,
  matching the 5-parameter model exactly. The only baseline reported (greedy-always-fuse, F1=0.000)
  never predicts toxic and cannot discriminate between the cost model and the one-line rule.
  ⇒ Ada RQ1 establishes only that **spilling is a sufficient toxicity indicator on this benchmark**;
  it is no evidence for the roofline model's added parameters. The fp16↔fp32 transfer
  (leave-one-dtype-out F1=0.968) is the only in-family generalization signal here.
- **Some of the 16 TPs rest on single-shot labels.** `fusion/runner.py` labels toxic by a bare
  `t_fused < t_unfused` threshold at exactly 1.0, with no noise band. Of the 16, the 8 NOUT=128 rows
  (0.09–0.29) and the fp32 NOUT=64 rows are safe — `data/timing_ci_ada.csv` confirms redux_N64_fp32 at
  median 0.893, 95% CI [0.893, 0.898], 20 rounds, sig. But **redux_N64_fp16 has no CI row at all**, and
  its weakest member is at **0.9818** — a 1.8% "slowdown" that LOG-10's own GEMM experience says can
  flip under repeat measurement (Ada single-shots of 0.957/0.973 became *ambiguous* and *beneficial*
  under 20-round CIs). Treat verdicts within ~5% of 1.0 as unresolved; that TP is not yet established.
- The pooled held-out-**shape** CV (P=0.889 / R=1.000 / F1=0.941) is ~in-sample: spills, the toxicity
  driver, are *identical* across (R,C) shapes, so it tests robustness to matrix size, not to the
  register/spill regime.
- **Honest held-out-decision-variable CV** (folds on the variables that actually move spills):
  leave-one-dtype-out **F1=0.968 / precision=1.000 / recall=0.938** (fp16 held-out F1=0.933, fp32
  held-out 1.000); leave-one-NOUT-out **F1=0.667 / precision=1.000 / recall=0.500**. (Reproduce:
  `python -m model.fit data/microbench_timing.csv`.)
- **The leave-one-NOUT-out=0.667 is a defect of the parametric fit, not of the problem** — and it is
  *not* an extrapolation/calibration-range limitation. Holding out NOUT=128 **collapses gamma_spill to
  8.1e-09**, i.e. the floor: the fit abandons the spill term entirely, even though the 8 genuinely
  toxic 300-spill NOUT=64 rows (speedup 0.940/0.897) remain in training. It then misses all 8 toxic
  1986-spill cases (fold acc=0.000). The reverse fold (hold out NOUT=64) learns gamma_spill=0.0067 and
  scores F1=1.000. ⇒ this is a **parameter-identifiability failure** — only the extreme-spill regime
  carries gradient for gamma_spill — not a too-small-but-fitted gamma that could be extrapolated.
  Damningly, the zero-parameter spill rule scores **F1=1.000 / recall=1.000** under the same
  leave-one-NOUT-out protocol: it **strictly dominates the fitted model out-of-sample**. We therefore
  do *not* claim Ada RQ1 as evidence that the fitted model generalizes across spill regimes.
  (Reporting nit: `fit.py` prints `held-out NOUT=32 (n=8): F1=0.000 recall=0.000` — that fold contains
  **zero** toxic cases, so recall is undefined and the model actually scores acc=1.000, 8/8 TN. The
  fold is vacuous, not a failure; the pooled 0.667 is computed correctly by concatenating predictions.)
- **Ablations** isolate the interpretable driver:
  | model | F1 | reading |
  |---|---|---|
  | full | 1.000 | — |
  | drop spill term | **0.000** | the spill term is the *only* toxic signal — without it the model never predicts toxic at all |
  | drop smooth-occupancy | 1.000 | the smooth P_occ term is negligible on Ada |
  ⇒ the decisive signal is the **register-spill discontinuity**, exactly the term PROPOSAL §5.2
  emphasises. `occ_knee` fitting to its floor is the model *telling us* occupancy≥8% doesn't gate these.
  Read together with the zero-parameter baseline above, the honest summary is: **on Ada the cost model
  is, decision-wise, an expensive re-derivation of `f_spills > 0`.** Its extra structure is untested
  here and must be justified (if at all) on the C500, where spilling is *not* sufficient for toxicity.

## RQ2 — interpretability (attribution validated against the profiler)
- **Occupancy model reproduces the calculator** (`fig2`): analytical sm89 occupancy == ncu
  *theoretical* occupancy on all 22 kernels (100↔100, 33↔33, 17↔17; even the deceptive 100% at
  NOUT=128 where ptxas caps regs at 40 and spills 1986× — both agree, and the *spill* feature flags
  the harm). This validates the calculator re-implementation, **not** achieved occupancy: measured
  *achieved* occupancy differs by 21.7 pts mean / 88 max (deliberately not modelled — the spill term
  handles the real degradation, which is why the deceptive NOUT=128 case is still caught).
- **Attribution ground truth is cleanly separable:** spilled kernels move mean 1.0e10 B of local
  (spill) traffic; non-spilled move exactly 0. Toxic reductions → `dominant=spill` (8/8).
- **Layout branch** (`fusion/cuda_layout.py`, raw CUDA via `load_inline`): a 32×32 shared-tile
  transpose with a PAD knob. PAD0 vs PAD1 have **identical registers/occupancy/spills** — only bank
  conflicts differ (100× more), giving a **1.26–1.40× slowdown** (median 1.32×) purely from layout.
  `beta_layout` fit from this microbench (=0.327); the model then attributes **4/4** of these to `layout`.
  ⇒ the model distinguishes the two microarchitectural causes, not just detects toxicity.

## RQ4 — end-to-end utility (`fusion/endtoend.py`)
Offline recommender (`model/recommender.py`) chooses horizontal-fusion width from single-compile
static resource reports (no timing). Policies compared per subgraph:
| subgraph | model vs greedy | model / oracle | search cost |
|---|---|---|---|
| wide_multiproj (fp16) | **8.65×** | 1.00× | 9 compiles vs 9 timed runs |
| mixed_widths (fp16) | **5.81×** | 1.00× | 14 compiles vs 14 timed runs |
| fp32_block | **7.42×** | 1.00× | 9 compiles vs 9 timed runs |
- The recommender **matches the timed oracle exactly (1.00×) on all three subgraphs** while
  greedy-always-fuse is 5.8–8.7× slower (it over-fuses wide layers into a spilling kernel).
  TOTALS ms — wide: none=25.248 greedy=192.140 model=oracle=22.215; mixed: none=17.481 greedy=85.484
  model=oracle=14.703; fp32: none=5.977 greedy=38.946 model=oracle=5.252.
- **RQ4 was NOT affected by the VRAM/local-memory artifact** — these are a post-fix *re-measurement*,
  not a correction. The end-to-end harness allocates few enough buffers that it never entered the
  oversubscribed regime (pre-fix greedy(w128)=151.4 ms sits with the *unstarved* matrix value ~173 ms,
  not the starved ~3142 ms). Post-fix times rose by a uniform ~1.22–1.37× across spilling and
  non-spilling kernels alike, with **no ~10× deflation of the spilling greedy kernel**; the shift from
  the pre-fix 9.72/6.17/7.43 is run-to-run drift that largely cancels in the ratio. ⇒ RQ4's conclusion
  holds in **both** regimes — a robustness result. It is additionally robust to the 14× refit of
  gamma_spill (0.0807 → 0.00567), which left every fusion decision identical.
- **On the earlier "3.03× off oracle on fp32" limitation:** that was a (separate, earlier) MEASUREMENT
  ARTIFACT — the old run mis-measured the model's width-32 fp32 kernel at ~6 ms; clean re-runs measure
  it at ~1.3 ms and the model matches oracle. (fp32 wide-reduction latencies are somewhat run-to-run
  noisy; w16/w32 are near-equivalent there.) NO RQ4 width decision flipped between the runs, and the
  numbers above have since been re-measured again after the LOG-10 fix — still 1.00× oracle.

## Key scientific finding (frames the cross-vendor thesis)
On Ada sm89 the **only decision-flipping** toxic-fusion mechanism in this sweep is the
**register-spill cliff (P_occ)**. Layout penalties (bank conflicts up to 1.6×, uncoalesced transpose
reads) *degrade* fused kernels but never overturn the HBM round-trip savings — fusion stays a win.
This sharpens Phase 4: the MetaX C500's **hard 4 KB/thread private cap** converts the soft spill cost
into a *launch failure*, so a fusion that merely spills-but-runs on Ada can be **illegal** on C500 —
the predicted decision-flip. (See `model/MODEL_SPEC.md §7`.)

## Artifacts produced
- Code: `fusion/` (hw, static, kernels/{pointwise,reduction,transpose}, timing, runner, ncu*,
  profile*, endtoend, cuda_layout) + `model/` (costmodel, fit, recommender, rq2, figures, MODEL_SPEC).
- Data: `data/microbench_timing.csv` (72 rows; 64 genuine cases scored + 8 degenerate no-ops excluded), `data/microbench_ncu.csv` (22), `data/cuda_layout.csv` (4).
- Figures: `figures/fig1_spill_cliff.png`, `fig2_occupancy_validation.png`, `fig3_rq4_endtoend.png`.
- Frozen spec + fitted constants: `model/MODEL_SPEC.md`, `model/ada_constants.json`.
- Logs: `logs/run_timing_matrix.log`, `run_ncu_profile.log`, `run_endtoend.log`, `run_layout.log`.
- **Post-hoc:** `logs/LOG-10-ada-bucket1.md` — the VRAM/local-memory measurement artifact, its
  reproduction, the `empty_cache()` prophylaxis, and the regeneration of the dataset/constants/figures
  that all RQ1/RQ4 numbers in this log now reflect. `data/timing_ci_ada.csv` — 20-round CIs for 5
  headline claims (does **not** cover redux_N64_fp16; see RQ1).

## Reproduce
```
source tooling/env.sh
python -m fusion.runner  data/microbench_timing.csv      # timing+static dataset
python -m fusion.profile data/microbench_ncu.csv         # ncu ground truth (slow)
python -m model.fit      data/microbench_timing.csv      # RQ1 + fitted constants
python -m model.rq2      data/microbench_ncu.csv         # RQ2 occupancy + attribution
python -m fusion.profile_layout data/cuda_layout.csv     # RQ2 layout branch + beta_layout
python -m fusion.endtoend                                # RQ4
python -m model.figures                                  # figures
```

## Open items for a follow-up session
- Optional: a GEMM-epilogue family (activation→GEMM) to broaden the taxonomy beyond reductions.
- Optional: wire the recommender into a real Inductor scheduler hook (currently the offline-recommender
  fallback, which PROPOSAL §5.4 sanctions).
- Phase 4 (MetaX, separate session): re-parameterise constants, hunt the 4 KB-spill decision-flip.
