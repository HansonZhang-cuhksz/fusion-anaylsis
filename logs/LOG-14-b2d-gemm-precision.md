# LOG-14 — B2d: refine the residual GEMM precision (0.5)? → HONEST NEGATIVE

Date: 2026-07-15 · Machine: MetaX C500, env `fusion` · Bucket-2 item B2d. Investigated whether the
GEMM-family precision (0.500, F1 0.667 — the fused fp16 128×128 GEMM is over-rejected) can be lifted by
a **dtype-aware spill-traffic term**, without overfitting and with a defensible mechanism. Adjudicated
by a 3-agent workflow (`wf_ec5a4d64`) + inline cross-checks. **Conclusion: it cannot be fixed honestly
with a search-free static model; keep the flat `reread=2` and disclose.**

## The regime (data/microbench_c500_combined.csv; the 8 spilling GEMM-128 rows)
| dtype | f_spills | u_spills | true | model pred | pred speedup |
|---|---|---|---|---|---|
| fp16 128×128 | 117 | 118 | **beneficial** (1.01–1.06) | TOXIC ✗ | ~0.69 |
| fp32 128×128 | 205 | 234 | toxic (0.78–0.83) | TOXIC ✓ | ~0.68 |

The model predicts ~0.68 for **all** of them: the flat `reread=2` (spill signal = `spill_factor(spills·reread)`)
saturates, so the 117-vs-205 gap barely moves the verdict. Recall is 1.0 (all toxic caught); the cost is
**precision 0.5** (the 4 fp16 cases are false positives). This was already the known state at LOG-05 §6.

## What "works" — and why it is a hack, not a fix
A dtype-switched reread (`reread_gemm_fp16_1`: fp16-GEMM reread 2→1) **robustly** lifts gemm precision
0.500→1.000, F1 0.667→1.000, overall 0.873→**0.941**, stable across 8 seeds, generalizes
leave-(gemm,fp16)-out **8/8**, zero collateral (reduction/pointwise unchanged, 0 non-GEMM flips), works
at both `fp16_compute_mult` settings, gamma stays sane. *As a metric, it is a genuine, non-overfitting
improvement.* But it **misattributes the mechanism** and is physically false:

1. **The reread models re-reading the spilled ACCUMULATOR, which is fp32 in BOTH dtypes**
   (`gemm_epilogue.py:25` `acc=tl.zeros(...,tl.float32)`; c/t are `torch.float32` for fp16 and fp32
   alike). There is no mechanism by which the fp16 epilogue re-reads *less accumulator*.
2. **MCPTI hardware data refutes `reread=1` for fp16 (LOG-05 §5).** The fused kernel does **more** local
   (spill) traffic than the unfused for *both* dtypes: fp16 **461K vs 433K (+6.5%)**, fp32 **950K vs 833K
   (+14%)**. So the data-implied fused reread is ~**1.07** (fp16) and ~**1.14–1.30** (fp32) — neither is
   2.0, and **fp16 is > 1, not 1**. Setting fp16 reread=1 *zeroes a real +6.5% effect using the dtype
   label*, purely to push 4 cases below threshold.
3. **The true discriminator is a RUNTIME quantity, not a static input.** What separates fp16-benign from
   fp32-toxic is the *added* fused-vs-unfused local traffic (+28K fp16 vs +117K fp32) — measurable only by
   profiling. The **static** spill counts have the **wrong sign** (`f_spills ≤ u_spills` for both:
   117≤118, 205≤234), i.e. the single-compile signal says the fused GEMM spills *no more* than the
   unfused. A search-free static model has no honest handle on this.
4. **The regime is a default-launch-config artifact anyway (LOG-12/13).** At `num_warps=8` these 4 fp16
   cases don't spill at all (0 spills → trivially classified). Spending a physically-false dtype term to
   rescue 4 config-contingent cases is doubly unjustified.

## Candidates evaluated (restarts=12, fp16_mult=2.0; leave-one-out for overfitting)
| candidate | overall F1 | gemm P / F1 | redux F1 | LOO(gemm,fp16) | verdict |
|---|---|---|---|---|---|
| baseline (`reread=2`) | 0.873 | 0.500 / 0.667 | 0.930 | 4/8 | shipped |
| `reread_gemm_fp16_1` | 0.941 | 1.000 / 1.000 | 0.930 | 8/8 | robust **but** dtype-label hack (MCPTI-refuted) |
| `reread_x_isz` (uniform ×isz) | 1.000 | 1.000 / 1.000 | 1.000 | 8/8 | *additionally* hacks the byte-identical reduction twins — overfit |
| `spill_bytes_isz` (honest: scale both plans) | 0.906 | **0.500** / 0.667 | 0.976 | 4/8 | the physically-honest scaling **does NOT** fix GEMM (preserves the wrong sign) |

The telling row is the last: scaling the spill *volume* by element size on **both** plans (the honest
reading of "dtype-aware spill traffic") leaves gemm precision at 0.500 — because the sign problem is in
the fused/unfused *difference*, which is runtime-only. Only the dtype-*asymmetric* hacks move it.

## Decision & disclosure
- **Do NOT integrate** the dtype-switched reread. Keep the flat `reread=2` (`fusion/kernels/base.py`
  unchanged). The precision-0.5 is a **safe precision loss** (recall stays 1.0 — the model never fuses a
  toxic GEMM; it only over-rejects 4 beneficial fp16 cases), not a missed-toxic hazard.
- **Honest limitation (disclosed in RESULTS.md):** separating fp16-benign from fp32-toxic
  epilogue-into-spilling-GEMM needs the *runtime* fused-vs-unfused local-traffic delta (or a
  compute-serialization model), which is not a single-compile static input — a genuine boundary of the
  search-free approach, not a tuning gap. A dtype-label reread reaches the metric but not the mechanism.
- This mirrors #1 / B0o1: the discriminating cases here are the same fp16 spill-but-beneficial family,
  and the honest verdict is the same — the static model can rule out spilling configs but cannot, without
  runtime profiling, correctly order the borderline fp16 cases.

## Repro
`scratchpad/b2d_candidates.py` (candidate sweep), `b2d_baseline.py` (the 8-row table),
`b2d_overfit_check.py` (leave-(reduction,fp16)-out). MCPTI numbers: LOG-05 §5. Config-artifact: LOG-12/13.
