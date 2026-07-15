# LOG-06 — Reread-aware spill-traffic feature: implemented, re-fit, validated (G2/G3/G7)

Date: 2026-07-15 · Machine: MetaX C500, env `fusion` · Resolves the LOG-05 GEMM blind spot.

## 1. Implementation (still search-free — a better *static* signal, no profiling at decision time)
- **`model/costmodel.py`**: the spill term now uses spill *traffic* = `spill_factor(spills × reread)`;
  `reread` threaded through `plan_efficiency` / `predict_time` / `decide` (default 1 → backward-compatible).
- **`model/fit.py`**: `predict_times_row` passes `f_reread` / `u_reread`.
- **`fusion/kernels/base.py`**: `spill_reread(producer, consumer) = 2` for **CONTRACTION producer +
  POINTWISE consumer** (a full-tile epilogue re-reads the spilled accumulator: store + reload), else 1.
  This is **taxonomy-derived from Φ(v)** — the feature that makes the taxonomy load-bearing (**G7**).
- **`model/build_combined_c500.py`**: assembles the combined C500 training set
  (`data/microbench_c500_combined.csv`): 72 reduction/pointwise (`f_reread=1`) + 16 GEMM (`f_reread=2`).

## 2. Result — re-fit on the combined set (80 genuine cases; `scratchpad/refit_validate.py`)
| model | overall F1 | overall R | reduction F1 / R | **GEMM F1 / R** |
|---|---|---|---|---|
| baseline (no reread — the LOG-05 failure) | 0.826 | 0.792 | 0.905 / 0.950 | **0.000 / 0.000** |
| **fix (reread-aware spill-traffic)** | **0.852** | **0.958** | 0.905 / 0.950 | **0.667 / 1.000** |

- **GEMM recall 0 → 1.000** — catches all 4 toxic GEMM fusions the spill *count* missed.
- **Reductions unchanged** (F1=0.905, R=0.950) — the fix does not disturb the family it already handled;
  gamma_spill barely moves (0.0092 → 0.0078).
- Overall recall 0.79 → 0.96 (catches 23/24 toxic), F1 0.826 → 0.852.
- **Honest residual:** GEMM precision 0.500 — the flat ×2 over-rejects the 4 fp16 big-tile *beneficial*
  cases (spill 117×2 looks toxic-ish). Recall (the safety-critical direction for a rejection pass) is
  solid; separating fp16-benign from fp32-toxic at the same effective spill needs a dtype/compute-aware
  refinement (future).

**Anti-overfit — independently verified (workflow `wf_17a52568-2b6`, 4/4 passed).** Leave-one-family-out:
fit `DeviceConstants` on the non-GEMM rows ONLY (the model never sees a GEMM, never sees reread=2),
then score the held-out GEMM rows (reread=2) → **GEMM recall = 1.0** (catches all 4 toxic). Decisive
counterfactual: the *same* held-out fit with GEMM reread forced back to 1 → **recall = 0.0**. So the
generalization is carried by the taxonomy-derived reread feature, not by anything memorized — a
publication-grade result. Verifiers also confirmed: Ada byte-identical under reread=1; reduction
decisions byte-identical baseline-vs-fit (Δ=0); combined CSV `f_reread`/`u_spills` fingerprints correct.

## 3. Backward compatibility
- **Ada RQ1 unchanged: F1=0.970, R=0.941** (no GEMM on Ada; `reread` defaults to 1). The committed
  Ada model and the C500 reduction-only decision-flip result are unaffected (reduction F1 identical).
- Constants for the all-family C500 model: `model/c500_combined_constants.json` (gamma_spill 0.0078).
  The reduction-only `c500_constants.json` (transfer/decision-flip in `transfer_c500.py`) still holds.

## 4. Status
Closes the concrete G2/G3 failure from LOG-05 (spill-count wrong sign on GEMM) with a principled,
taxonomy-derived, still-search-free fix; makes Φ(v) load-bearing (**G7**). Remaining: refine GEMM
precision (dtype/compute-aware), and (G4) add BROADCAST / softmax-LayerNorm / a real subgraph, and run
the GEMM family on Ada for the cross-vendor GEMM comparison.
