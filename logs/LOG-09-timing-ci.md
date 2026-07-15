# LOG-09 — Statistical rigor: timing confidence intervals (G5)

Date: 2026-07-15 · Machine: MetaX C500 ×4, env `fusion`.

## Method
`fusion/timing_ci.py`: for each decisive fuse/don't-fuse claim, **20 independent rounds** (each a
min-of-50-iters CUDA-event timing), report the fused-vs-unfused **speedup median + 95% percentile CI**,
and flag whether the CI **excludes 1.0** (beneficial/toxic is statistically significant, not noise).
Run on **all 4 C500 GPUs** for cross-device reproducibility → `data/timing_ci_c500.csv` (5 claims × 4 GPUs).

## Result — every key claim is significant AND reproducible across 4 GPUs
| claim | expected | median (range over 4 GPUs) | 95% CI (union) | verdict | sig on |
|---|---|---|---|---|---|
| redux N32 fp16 | beneficial | 1.079–1.083 | [1.077, 1.084] | beneficial | **4/4** |
| **redux N32 fp32 (decision-flip)** | **toxic** | **0.641–0.643** | **[0.638, 0.645]** | **TOXIC** | **4/4** |
| redux N64 fp32 | toxic | 0.271 | [0.270, 0.272] | TOXIC | **4/4** |
| gemm 128×128 fp16 | beneficial | 1.063–1.064 | [1.059, 1.068] | beneficial | **4/4** |
| **gemm 128×128 fp32 (GEMM toxicity)** | **toxic** | **0.824–0.825** | **[0.821, 0.827]** | **TOXIC** | **4/4** |

## Takeaways
- **The headline results are statistically robust.** The cross-vendor **decision-flip** (C500 side)
  measures 0.641–0.643 on four *independent* GPUs with a CI of [0.638, 0.645] — far from 1.0. The
  **GEMM toxicity** is 0.824–0.825, CI [0.821, 0.827]. Both are toxic with overwhelming significance.
- **Even the marginal-looking beneficial cases are significant:** redux/gemm fp16 sit at ~1.06–1.08×
  with CIs that still exclude 1.0 — the beneficial verdicts are not noise either.
- **Cross-device reproducibility is tight** (medians agree to ~3 decimals across 4 GPUs), so the C500
  min-of-N timing is highly stable — this is the discipline that would have caught the earlier Ada
  fp32 measurement artifact.
- Directly upgrades the RESULTS.md "point-estimate timings" limitation. (The Ada-side claims — RQ1 F1
  and the Ada half of the decision-flip — still warrant the same CI pass on the Ada machine.)
