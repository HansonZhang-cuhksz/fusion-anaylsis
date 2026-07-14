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
| **RQ1** | predict fusion toxicity from static inputs? | **in-sample P=1.000 / R=0.941 / F1=0.970** (catches 16/17 toxic; 1 borderline no-spill miss); F1 **0.865** shape-CV / **0.829** leave-one-NOUT-out / **0.970** leave-one-dtype-out (on 64 genuine cases; 8 degenerate no-ops excluded); greedy F1=0.00 |
| **RQ2a** | analytical occupancy == ncu *theoretical* occupancy? | **MAE=0.000, 22/22 exact** (reproduces the CUDA occupancy calculator by construction). *Achieved* occupancy differs 21.7 pts mean / 88 max — deliberately not predicted; the spill term captures the real harm |
| **RQ2b** | attribute the cause (spill vs layout)? | **12/12 = 100%**: 8/8 spill-dominated + 4/4 layout-dominated |
| **RQ4** | utility vs greedy / oracle? | model **6.2–9.7× faster than greedy** (9.72× is a constructed worst-case for greedy), **= oracle (1.00×) on all 3 subgraphs**, at ~0 timing cost |

## RQ1 — search-free toxic-fusion decision
- Cost model (`model/costmodel.py`) fit on Ada (`model/ada_constants.json`): **B_peak≈151 GB/s**
  (physically correct for a 4060 laptop), occ_knee→floor, gamma_spill≈0.081.
- **precision 1.000 / recall 0.941 / F1 0.970 / acc 0.984** (TP=16, FP=0, FN=1, TN=47) on the
  **64 genuine cases** (8 degenerate no-op rows — pointwise K=1 where n_launches_unfused=1 makes the
  "unfused" plan a single launch identical to the fused kernel — excluded from decision scoring;
  their labels are timing noise, not fusion decisions). The pooled held-out-**shape** CV
  (P=0.800 / R=0.941 / F1=0.865) is ~in-sample: spills, the toxicity driver, are *identical* across
  (R,C) shapes, so it tests robustness to matrix size, not to the register/spill regime.
- **Honest held-out-decision-variable CV** (folds on the variables that actually move spills):
  leave-one-NOUT-out **F1=0.829 / precision=0.708 / recall=1.000** (NOUT=32 F1=0.222 — it over-flags
  the hard borderline boundary — while NOUT=64 & 128 F1=1.000); leave-one-dtype-out **F1=0.970 /
  recall=0.941** (fp16 held-out 0.941, fp32 held-out 1.000). (Reproduce:
  `python -m model.fit data/microbench_timing.csv`.)
- **Recall = 0.941** in-sample: catches 16/17 toxic fusions. The single false negative is a
  sibling_redux NOUT=32, fp16, R=C=1024 case (measured speedup 0.906, spills=0) — a borderline ~9%
  slowdown with NO spill signal, which the spill-focused model cannot catch: its honest blind spot
  for mild non-spill toxicity. Precision is now perfect (0 false positives).
- **Ablations** isolate the interpretable driver:
  | model | F1 | reading |
  |---|---|---|
  | full | 0.97 | — |
  | drop spill term | **0.545** | spills are the dominant toxic signal |
  | drop smooth-occupancy | 0.97 | the smooth P_occ term is negligible on Ada |
  ⇒ the decisive signal is the **register-spill discontinuity**, exactly the term PROPOSAL §5.2
  emphasises. `occ_knee` fitting to its floor is the model *telling us* occupancy≥8% doesn't gate these.

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
| wide_multiproj (fp16) | **9.72×** | 1.00× | 9 compiles vs 9 timed runs |
| mixed_widths (fp16) | **6.17×** | 1.00× | 14 compiles vs 14 timed runs |
| fp32_block | **7.43×** | 1.00× | 9 compiles vs 9 timed runs |
- The recommender **matches the timed oracle exactly (1.00×) on all three subgraphs** while
  greedy-always-fuse is 6.2–9.7× slower (it over-fuses wide layers into a spilling kernel).
- **On the earlier "3.03× off oracle on fp32" limitation:** that was a MEASUREMENT ARTIFACT — the old
  run mis-measured the model's width-32 fp32 kernel at ~6 ms; clean re-runs measure it at ~1.3 ms and
  the model matches oracle. (fp32 wide-reduction latencies are somewhat run-to-run noisy; w16/w32 are
  near-equivalent there.) NO RQ4 width decision flipped between the runs.

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
