# Review Findings — TODO

Findings from the cross-check of the Ada-session implementation against `PROPOSAL.md`
(review run 2026-07-14 on the MetaX machine). Each item: what's wrong, where, and the fix.
Severity: **[A] concrete inconsistency** · **[B] overclaim to soften** · **[C] minor/cosmetic**.

The core results are sound and reproduce (RQ1 F1=0.841/recall=1.000 re-run exactly; RQ2 occupancy
22/22 and attribution 8/8 spill + 4/4 layout; static regs == profiled regs; spill cliff real —
all 16 spilled kernels toxic). These TODOs are about consistency and framing, not the central result.

---

## [A] Concrete inconsistencies — FIXED & VERIFIED (2026-07-14)

- [x] **1. `beta_layout` mismatch: spec says 0.406, committed constants say 0.001.**  ✅ DONE
  `model/MODEL_SPEC.md:56` lists `beta_layout = 0.406`, but `model/ada_constants.json` had
  `beta_layout = 0.001` (dataclass default). Root cause: `model/fit.py` fits only
  `{C_peak,B_peak,T_launch,occ_knee,gamma_spill}` and rewrites the whole JSON via `k.as_dict()`,
  resetting `beta_layout`; `fusion/profile_layout.py:112` writes 0.406 back — so whichever script
  runs last wins, and `fit` ran last in the committed state. Effect: the "frozen constants" that
  travel to the C500 session carry an **inert layout term**.
  **Fix:** (a) make `fit.py` preserve `beta_layout` on write; (b) set `ada_constants.json` to the
  data-derived value (median of `(slowdown-1)/bank_conf_per_elem` over `data/cuda_layout.csv` =
  0.40561, rounds to 0.406).
  **DONE (verified):** `model/fit.py` now merges the previous `beta_layout` into the written JSON
  (`(FileNotFoundError, json.JSONDecodeError)`-guarded); `model/ada_constants.json` now holds
  `0.40561037308731873`. A live `model.fit` re-run confirmed the value is preserved on disk (no
  reset to 0.001) and RQ1 is unchanged (P=0.725 R=1.000 F1=0.841 acc=0.875). *Caveat:* the console
  "fitted DeviceConstants" block still prints `beta_layout: 0.001` (it echoes the fit's own `k`,
  which doesn't calibrate layout) — the value *saved to disk* is correct; cosmetic only.

- [x] **2. `MODEL_SPEC.md §6` RQ4 numbers are stale/self-contradictory.**  ✅ DONE
  `model/MODEL_SPEC.md:75` said "up to **8.6×** faster than greedy, **within ~1.1× of oracle**."
  Actual committed results (`logs/run_endtoend.log`, `logs/LOG-03`): **9.85×** vs greedy; oracle
  ratios 1.00× / 1.00× / **3.03×** — the fp32 subgraph flatly contradicts "within ~1.1× of oracle."
  **Fix:** replace with 9.85× vs greedy, = oracle on fp16, 3.03× off oracle on fp32_block.
  **DONE (verified):** `model/MODEL_SPEC.md §6` now reads "up to 9.85× faster than greedy; matches
  the timed oracle exactly (1.00×) on the fp16 subgraphs; 3.03× off oracle on the fp32 subgraph."
  Confirmed consistent with `logs/run_endtoend.log` (9.85× / 1.000× / 3.029×) and grep confirms the
  strings "8.6×" and "1.1× of oracle" no longer appear anywhere in the file.

---

## [B] Overclaims to soften — FIXED & VERIFIED (2026-07-14)

*None of these required an Ada GPU run — all are doc rewordings, plus item 4 is backed by a real
held-out-decision-variable CV computed from the committed `data/microbench_timing.csv`.*

- [x] **3. RQ2a "occupancy validated vs profiler, MAE=0.000" is near-tautological.**  ✅ DONE
  `model/rq2.py:27` compares analytic occupancy to ncu **theoretical** occupancy — a calculator
  output `fusion/hw.py` reproduces by construction, so 22/22 exact is expected, not evidence. The
  **achieved** (measured) occupancy in `data/microbench_ncu.csv` differs by **21.7 pct-pts mean /
  88 pct-pts max** (NOUT=128: theoretical 100% vs achieved ~12–22%). This is fine for the model
  (it's *why* the spill term exists), but reword `LOG-03`/`MODEL_SPEC §2,§6` to "reproduces the
  occupancy calculator; achieved occupancy is deliberately not predicted (the spill term captures
  the real degradation)."
  **DONE:** reworded in `logs/LOG-03` (headline table + RQ2 section), `model/MODEL_SPEC.md` §2 header
  (+ HTML caveat) and §6 RQ2a bullet, and `PROPOSAL.md` §8 (Phase-1/2 line + RQ2a headline). Every
  occurrence now says "reproduces ncu *theoretical* occupancy (the CUDA occupancy calculator)" and
  states the 21.7/88 achieved gap.

- [x] **4. "Held-out-shape CV" does not test generalization of the decision.**  ✅ DONE
  `f_spills`/`f_regs` — the variables that drive the label — are **identical across all four (R,C)
  shapes** the CV folds on (spills depend on NOUT/dtype/tile, not runtime dims), so each fold trains
  on the exact spill signatures it tests; pooled CV == in-sample to the digit (0.725/1.000/0.841),
  and per-fold F1 swings 0.00–0.93. Reword the RQ1 generalization claim in `LOG-02/03`, `MODEL_SPEC
  §6`, `PROPOSAL §6`: the CV demonstrates robustness across matrix sizes, **not** across unseen
  register/spill regimes. Consider adding a leave-one-NOUT-out (or leave-one-dtype-out) CV that
  actually holds out the decision-driving variable.
  **DONE (backed by real data):** added leave-one-NOUT-out + leave-one-dtype-out CV to `model/fit.py`
  and re-ran (pure data, no GPU). Results: **recall=1.00 in every scheme**; leave-one-NOUT-out
  **F1=0.877** (spill signal extrapolates), leave-one-dtype-out **F1=0.753 / precision=0.604** (the
  honest fp16↔fp32 transfer cost the shape-CV hid). Reworded `LOG-02`, `LOG-03`, `MODEL_SPEC §6`,
  `PROPOSAL §8` to report all three CV schemes and flag the shape-CV as ~in-sample.

- [x] **5. "Flagship NVIDIA GPU" is a wrong fact.**  ✅ DONE
  `PROPOSAL.md §0` thesis says the model "transfers from a flagship NVIDIA GPU to a domestic
  accelerator." The device (`logs/LOG-01`) is an **RTX 4060 Laptop GPU** — entry-level Ada, 8 GB,
  24 SMs, ~152 GB/s effective. Correct arch (sm89), not "flagship." Reword to "a consumer Ada GPU."
  **DONE:** `PROPOSAL.md §0` thesis now reads "a consumer NVIDIA Ada GPU (RTX 4060)". No "flagship"
  string remains in any doc.

- [x] **6. RQ4 9.85× headline is a constructed worst-case for greedy.**  ✅ DONE
  The `endtoend.py` subgraphs are deliberately built with wide layers greedy over-fuses (comments
  say so). Legitimate demonstration, but add a caveat that 9.85× is not a typical-workload number,
  and keep the disclosed fp32_block limitation (model 3.03× off oracle) visible in the headline.
  **DONE:** added the "constructed worst-case for greedy" caveat and the "3.03× off oracle on fp32"
  limit to the RQ4 headline in `logs/LOG-03`, `model/MODEL_SPEC.md §6`, and `PROPOSAL.md §8`.

---

## [C] Minor / cosmetic — FIXED & VERIFIED (2026-07-14)

*None needed an Ada GPU run for the core fix. One Ada-dependent sub-item for #7 is flagged below.*

- [x] **7. Degenerate no-op rows (fused == unfused).**  ✅ DONE
  In `fusion/matrix.py` NOUT=8 gets GS=8, so `run_fused` (WIDTH=8) and `run_unfused` (one group,
  WIDTH=8) compile to the *identical* kernel — the "fusion" is a no-op and its beneficial/toxic
  label is pure timing noise (8 of 88 rows). Drop NOUT=8 from Family R, or exclude these rows from
  the metrics, or annotate them as degenerate.
  **CORRECTION during fix:** it is **16 rows, not 8** — both NOUT=8 *and* NOUT=16 have
  `n_launches_unfused=1` (with GS=16, NOUT≤GS ⇒ unfused is a single launch identical to fused).
  **9 of the 29 "toxic" labels were noise from these no-ops.**
  **DONE (no GPU):** (a) `fusion/matrix.py` `_R_NOUT=[32,64,128]` (GS=16 unchanged, so a future Ada
  regen reproduces the 72 genuine rows byte-identically — `python -m fusion.matrix` now = 72 cases);
  (b) `model/fit.py` gained `nondegenerate()` and excludes these from decision scoring while keeping
  the roofline fit on full data, so the **committed constants stay byte-identical** (no cascade into
  the Ada-only endtoend/rq2/figures logs — verified). Honest numbers *improve*: in-sample F1
  0.84→**0.909**, leave-one-NOUT-out 0.88→**1.00**, leave-one-dtype-out 0.75→**0.909**, recall still
  1.000 (4 FPs, all pointwise K=1 near-break-even). Docs updated in LOG-02/03, MODEL_SPEC, PROPOSAL.
  Independently recomputed from git-HEAD artifacts: F1=0.909, TP=20/FP=4/FN=0/TN=48. **5/5 verifiers passed.**
  **⏳ Ada-dependent remainder (flagged, skipped per instruction):** fully *regenerating* the cleaned
  dataset from scratch — and, if one chooses to re-fit constants on the 72 rows (gamma_spill
  0.0887→0.0834, T_launch etc.), re-running `endtoend`/`rq2`/figures to match — needs the Ada GPU.
  The no-GPU-equivalent (exclude-in-analysis + fix-source + keep-constants-stable) is done.

- [x] **8. `f_dur_us` column is mislabeled and internally inconsistent.**  ✅ DONE
  Values in `data/microbench_ncu.csv` look like **nanoseconds** (pointwise k2 = 73632 → 0.074 ms),
  and for the extreme-spill case the ncu duration (~71 ms) is ~18× smaller than the timing CSV's
  `t_fused_ms` (1265 ms). Column is unused in any result, so harmless, but fix the unit label
  (`_ns` or convert) in `fusion/profile.py`/`fusion/ncu.py` and note the ncu-vs-event-timing gap.
  **DONE (no GPU):** confirmed `gpu__time_duration.sum` under `--print-units base` is nanoseconds;
  renamed `duration_us`→`duration_ns` / `dur_ns_total` (`fusion/ncu.py`), `f_dur_us`→`f_dur_ns`
  (`fusion/profile.py` + committed CSV header, values untouched), and the `MODEL_SPEC §5` schema.
  Documented the ncu-vs-event-timing gap as informational-only in `LOG-02 §6`. Verified: zero code
  references to the old names; CSV values unchanged (pointwise k2 = 73632.0 ns). *Root-causing the
  18× gap would need Ada re-profiling and is unnecessary (column is unused).*

- [x] **9. `LOG-02` claim "fp32 crosses the spill cliff earlier than fp16" is unsupported.**  ✅ DONE
  The data shows both fp16 and fp32 first spill at the **same** NOUT=64 (fp16 300, fp32 310).
  Correct the sentence in `logs/LOG-02-microbench-and-model.md §3`.
  **DONE:** reworded to "fp16 and fp32 both first spill at the same NOUT=64 (fp16 300 / fp32 310);
  fp32 spills marginally more but does not cross at a smaller NOUT," and annotated the degenerate
  NOUT=8/16 rows in that table. Verified the old phrase is gone.

---

## [ADA] Independent deferred task — regenerate the clean dataset & re-fit on Ada  ⏳ OPEN

*Transfer target: the **Ada machine's** Claude Code session. This is the GPU-only remainder of review
item 7; the no-GPU parts (source fix in `fusion/matrix.py`, decision-scoring exclusion in
`model/fit.py`, all doc rewrites) are already DONE and committed. Nothing here blocks the MetaX/C500
phase — it can be done any time on Ada.*

- [x] **A1. Regenerate the degenerate-free dataset on Ada and re-fit the deployed constants.**

  **Why this exists.** Review item 7 found 16 degenerate no-op rows (NOUT∈{8,16}: with GS=16 the
  "unfused" plan is a single launch identical to the fused kernel; 9 were noise-labeled toxic). On
  the MetaX review machine we could only (a) fix the source so it never recurs and (b) exclude those
  rows from the *reported* metrics. We could **not** regenerate the dataset (needs Ada) nor re-fit
  the deployed constants without invalidating the Ada-produced `endtoend`/`rq2`/figures artifacts.
  This task closes that loop.

  **State you inherit (already committed):**
  - `fusion/matrix.py`: `_R_NOUT=[32,64,128]`, `_R_GS=16` ⇒ `python -m fusion.matrix` prints
    "total cases: 72 (pointwise=48, reduction=24)".
  - `model/fit.py`: has `nondegenerate()`, fits on full data but scores decisions on the genuine
    subset (prints "excluding N degenerate no-op rows").
  - `model/ada_constants.json`: **[A1 DONE] now the clean 72-row-fit constants** (gamma_spill=0.0807,
    B_peak=1.51e11, T_launch=4.43e-5, beta_layout=0.327); the pre-A1 88-row fit was gamma_spill=0.0887,
    B_peak=1.524e11, T_launch=8.16e-5, beta_layout=0.406. `data/*.csv`, `logs/run_*.log`,
    `figures/*.png` were regenerated on the clean 72-row dataset with the new constants.

  **Steps (Ada machine, env `profiling`; `source tooling/env.sh` first):**
  1. `python -m fusion.runner data/microbench_timing.csv` → expect **72 rows** (no NOUT∈{8,16}).
     Sanity: `f_regs`/`f_spills` for NOUT 32/64/128 must match the current committed rows exactly
     (deterministic ptxas); only `t_*_ms`/`speedup` re-measure.
  2. `python -m model.fit data/microbench_timing.csv` → now 0 degenerate rows; fit is on the clean
     72. Expect a modest constant shift (~**gamma_spill≈0.083, B_peak≈1.57e11, T_launch≈4.1e-5**) and
     RQ1 ≈ **F1 0.909 / recall 1.000**, leave-one-NOUT-out **1.00**, leave-one-dtype-out **≈0.91**.
     This OVERWRITES `ada_constants.json` with the clean-refit constants.
  3. `python -m fusion.profile_layout data/cuda_layout.csv` (run AFTER step 2) → re-derives
     `beta_layout≈0.406` and re-persists it so the layout constant + attribution stay consistent.
  4. Re-run the now-stale downstream Ada artifacts with the new constants:
     `python -m fusion.profile data/microbench_ncu.csv`; `python -m model.rq2 data/microbench_ncu.csv`;
     `python -m fusion.endtoend > logs/run_endtoend.log`; `python -m model.figures`.
     **Verify the ~6% gamma_spill shift flips NO RQ4 width decision** (recommender still ≥ greedy and
     ≈ oracle on fp16).
  5. (Optional) drop NOUT=16 from `matrix.ncu_subset()` too — it is used only for occupancy
     *validation*, not fusion decisions, so keeping it is fine; if dropped, the ncu row count changes
     from 22 (update docs accordingly).
  6. Reconcile docs to the regenerated numbers: `MODEL_SPEC §4` constants table (→ clean-refit
     values), and any moved figures in `LOG-02`/`LOG-03`/`MODEL_SPEC §6`/`PROPOSAL §8` (row counts
     are already 72; the RQ1 F1≈0.909 story should hold). Then mark item 7's "Ada-dependent
     remainder" resolved.

  **DoD:** `microbench_timing.csv` = 72 genuine rows; `ada_constants.json`, `run_endtoend.log`,
  `microbench_ncu.csv`, `figures/*` all regenerated from the clean dataset with mutually consistent
  constants; docs match; recall still 1.000; no RQ4 width decision flipped.

  **Lighter alternative (not recommended):** keep the committed 88-row-fit constants and only
  regenerate `microbench_timing.csv` to 72 rows — but then the constants no longer correspond to a
  fit on the committed dataset, so the full re-fit path above is cleaner.

  **DONE (Ada session 2026-07-14):** regenerated `microbench_timing.csv` to **72 rows** (static
  regs/spills matched the committed rows exactly, **0 mismatches**); re-fit the deployed constants
  (gamma_spill 0.0887→**0.0807**, T_launch 8.16e-5→**4.43e-5**, B_peak **1.51e11**, C_peak 4.18e11).
  **GENERALIZED** `nondegenerate()` from reduction-only to `n_launches_unfused<=1` (ANY family) — it
  now excludes the **8 pointwise K=1 no-ops** the regen exposed (reduction has no degenerates on the
  clean data). Re-ran `profile_layout` (beta_layout 0.406→**0.327**, 4/4 layout), `rq2` (occupancy
  22/22, attribution 12/12 — unchanged), `endtoend` (**model=oracle on all 3 subgraphs** — the old
  fp32 3.03× artifact is gone; **NO RQ4 width decision flipped**), and figures. **NEW headline:** RQ1
  in-sample **F1=0.970 / P=1.000 / R=0.941** (1 FN — a borderline no-spill NOUT=32 case, speedup 0.91);
  leave-one-NOUT-out **F1=0.829**, leave-one-dtype-out **F1=0.970**; RQ2 **12/12**; RQ4
  **model=oracle on all 3**, **6.2–9.7× vs greedy**. **Recall is no longer a blanket 1.000.** Item 7's
  "Ada-dependent remainder" is now resolved.

---

*Note:* RQ4 and the achieved-occupancy figures could only be checked against the committed logs on
the review machine (MetaX C500), not independently re-executed on Ada. Everything pure-data (RQ1
fit, RQ2 occupancy + attribution, `beta_layout`, dataset integrity) was reproduced by re-running.
The one open item requiring Ada is **[ADA] task A1** above.
