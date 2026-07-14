# LOG-03 — Results (RQ1–RQ4), layout study, end-to-end  [Phases 2–3 complete]

Date: 2026-07-14 · Machine: RTX 4060 Laptop (Ada sm89), WSL2 · env: `profiling`

## Scope delivered this session
Phases 1–3 of `PROPOSAL.md §8` are complete on Ada: env bring-up, microbench matrix + ncu dataset,
cost-model formalization + fit, the interpretable attribution, the offline recommender, and
end-to-end evaluation on real-ish subgraphs. A **frozen model spec** (`model/MODEL_SPEC.md`) is ready
for the MetaX C500 transfer (Phase 4).

## Headline numbers
| RQ | question | result |
|---|---|---|
| **RQ1** | predict fusion toxicity from static inputs? | held-out-shape CV **F1=0.84, recall=1.00, acc=0.88**; greedy baseline F1=0.00 |
| **RQ2a** | analytical occupancy vs profiler? | **MAE = 0.000 pts, 22/22 exact** vs ncu theoretical occupancy |
| **RQ2b** | attribute the cause (spill vs layout)? | **12/12 = 100%**: 8/8 spill-dominated + 4/4 layout-dominated |
| **RQ4** | utility vs greedy / oracle? | model **2.8–9.85× faster than greedy**, **= oracle (1.00×) on fp16**, at ~0 timing cost |

## RQ1 — search-free toxic-fusion decision
- Cost model (`model/costmodel.py`) fit on Ada (`model/ada_constants.json`): **B_peak≈152 GB/s**
  (physically correct for a 4060 laptop), occ_knee→floor, gamma_spill≈0.089.
- **precision 0.725 / recall 1.000 / F1 0.841 / acc 0.875** (in-sample == pooled held-out-shape CV).
- **Recall = 1.0**: never keeps a toxic fusion (0 false negatives). The 11 false positives are all
  near-break-even (measured speedup ∈ [0.93,1.09]) — a safe, conservative failure mode.
- **Ablations** isolate the interpretable driver:
  | model | F1 | reading |
  |---|---|---|
  | full | 0.84 | — |
  | drop spill term | **0.49** | spills are the dominant toxic signal |
  | drop smooth-occupancy | 0.84 | the smooth P_occ term is negligible on Ada |
  ⇒ the decisive signal is the **register-spill discontinuity**, exactly the term PROPOSAL §5.2
  emphasises. `occ_knee` fitting to its floor is the model *telling us* occupancy≥8% doesn't gate these.

## RQ2 — interpretability (validated against the profiler)
- **Occupancy model exact** (`fig2`): analytical sm89 occupancy == ncu theoretical occupancy on all
  22 profiled kernels (100↔100, 33↔33, 17↔17; even the deceptive 100% at NOUT=128 where ptxas caps
  regs at 40 and spills 1986× — both agree, and the *spill* feature is what flags the harm).
- **Attribution ground truth is cleanly separable:** spilled kernels move mean 1.0e10 B of local
  (spill) traffic; non-spilled move exactly 0. Toxic reductions → `dominant=spill` (8/8).
- **Layout branch** (`fusion/cuda_layout.py`, raw CUDA via `load_inline`): a 32×32 shared-tile
  transpose with a PAD knob. PAD0 vs PAD1 have **identical registers/occupancy/spills** — only bank
  conflicts differ (100× more), giving a **1.27–1.58× slowdown** purely from layout. `beta_layout`
  fit from this microbench (=0.406); the model then attributes **4/4** of these to `layout`.
  ⇒ the model distinguishes the two microarchitectural causes, not just detects toxicity.

## RQ4 — end-to-end utility (`fusion/endtoend.py`)
Offline recommender (`model/recommender.py`) chooses horizontal-fusion width from single-compile
static resource reports (no timing). Policies compared per subgraph:
| subgraph | model vs greedy | model / oracle | search cost |
|---|---|---|---|
| wide_multiproj (fp16) | **9.85×** | 1.00× | 9 compiles vs 9 timed runs |
| mixed_widths (fp16) | **6.61×** | 1.00× | 14 compiles vs 14 timed runs |
| fp32_block | 2.80× | 3.03× | 9 compiles vs 9 timed runs |
- On fp16 the recommender **matches the timed oracle exactly** while greedy-always-fuse is up to ~10×
  slower (it over-fuses wide layers into a spilling kernel).
- **Honest limitation:** on `fp32_block` the recommender is 3× off oracle — it over-fuses width-32
  fp32 groups that are *occupancy/compute-bound without spilling*, which the spill-focused model
  (occ_knee at floor) under-penalises. Still 2.8× faster than greedy. This is the same trade-off the
  RQ1 ablation exposed: discounting smooth occupancy maximises the binary decision but costs some
  fine-grained fp32 width tuning. Documented, not hidden.

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
- Data: `data/microbench_timing.csv` (88 rows), `data/microbench_ncu.csv` (22), `data/cuda_layout.csv` (4).
- Figures: `figures/fig1_spill_cliff.png`, `fig2_occupancy_validation.png`, `fig3_rq4_endtoend.png`.
- Frozen spec + fitted constants: `model/MODEL_SPEC.md`, `model/ada_constants.json`.
- Logs: `logs/run_timing_matrix.log`, `run_ncu_profile.log`, `run_endtoend.log`, `run_layout.log`.

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
