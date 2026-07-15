# LOG-11 — #1: can the cost model beat the trivial `spill>0 ⇒ toxic` rule? (working log)

Date: 2026-07-15 · Machine: MetaX C500, env `fusion` · From the Ada adversarial review (LOG-10 §3).
Pure-data / modeling task (no new GPU runs); C500 combined dataset (`data/microbench_c500_combined.csv`,
80 genuine cases, 24 toxic). **Guard: only 8 discriminating cases on a tiny dataset → overfitting is the
enemy; success = beating the trivial rule OUT-OF-SAMPLE (leave-one-out), not in-sample.**

## Step 1 — Diagnosis (baseline: the model currently LOSES to the trivial rule)
| model | in-sample F1 | leave-one-(family,dtype)-out F1 |
|---|---|---|
| trivial rule (`f_spills>0 ⇒ toxic`) | **0.857** | **0.857** |
| fitted cost model (`c500_combined_constants`) | 0.852 | **0.250** |

⇒ the cost model is **strictly worse than the one-line spill rule** on C500 — slightly worse in-sample,
catastrophically worse out-of-fold. This is the honest starting point for #1.

**The 8 discriminating cases** (spill>0 but *beneficial* — where the trivial rule is wrong and any real
cost model must win): **all fp16** — reduction NOUT=32 (spill 100, ~1.06–1.10×) ×4 shapes, and GEMM fp16
128×128 (spill 117, ~1.01–1.06×) ×4 shapes. Their **toxic fp32 twins have identical static inputs**
(f_regs=256, f_spills=100/117, f_occ) — so the ONLY search-free signal that can separate them is the
**roofline's dtype dependence** (bytes ∝ element size; flops; possibly dtype-dependent compute peak,
since fp16 arithmetic throughput ≈ 2× fp32). Spill *count* and occupancy are identical and cannot help.

## Step 2 — Candidate refinement + TWO method errors I made and corrected
Candidate B = dtype-aware compute (fp16 arithmetic ≈ 2× fp32), injected as effective flops
(`fp16 flops ÷ 2`), re-fit, evaluated. **I initially drew a wrong conclusion twice, then corrected it:**
1. **Bug #1 (scripting):** an intermediate leave-one-*shape*-out run appeared to show the baseline
   getting 7/8 discriminating cases right (F1 0.936). Scripting bug — a clean per-case reconciliation
   shows **1/8**.
2. **Bug #2 (under-converged fits):** at `restarts=3` the leave-one-out fits are numerically noisy
   (Nelder-Mead swings 0.82–0.94 across seeds; one seed collapsed to 0.25). I wrongly read that as
   "the refinement overfits / the model loses out-of-fold." An adversarial verifier (`wf_7845332b`)
   caught it; at **`restarts=12` the numbers are rock-stable across seeds** and tell the opposite story.

## Step 3 — Converged results (restarts=12, seed-stable; confirmed by me + 2 independent verifiers)
| model | in-sample F1 | leave-one-dtype-out (seeds 0/1/2) | discriminating-correct |
|---|---|---|---|
| trivial rule `f_spills>0` | 0.857 | 0.857 (parameter-free) | 0/8 |
| baseline cost model | 0.852 | **0.873** (stable) | 1/8 |
| + dtype-aware compute (B) | 0.873 | **0.906** (stable) | 1/8 |

## Step 4 — Honest conclusion (corrected): a QUALIFIED positive, not a negative
- **The cost model does beat the trivial `spill>0 ⇒ toxic` rule OUT-OF-FOLD** — baseline 0.873, and
  **0.906** with the dtype-aware compute term, vs the trivial rule's 0.857 (stable across seeds). So it
  is **not** merely a spill detector on C500: it generalizes slightly better than the spill heuristic,
  and the **dtype-compute refinement genuinely helps** (0.873 → 0.906) — my earlier "overfits" was wrong.
- **BUT the win is marginal and does not solve the hard cases.** In-sample the baseline is *worse* than
  trivial (0.852 < 0.857); both models still get **7 of the 8** discriminating (spill-but-beneficial
  fp16) cases **wrong**, rescuing only the mildest one (reduction N32 R=1024, 1.10×). The fp16-vs-fp32
  distinction at *identical* static inputs (256 regs, 100 spills, 0.25 occ) is not cracked; the
  out-of-fold gain comes mostly from generalization/fold structure, and the margin (~0.02–0.05 F1) is
  small on an 80-case / 2-fold test.
- **Net for the paper:** honestly, *"the interpretable cost model adds modest, converged, out-of-fold
  value over the spill heuristic, and a principled dtype-aware compute term improves it — but it does not
  solve the hardest spill-but-beneficial cases, which need a compute-serialization model and/or a larger
  dataset."* Not the clean "beats it decisively" of an earlier draft, nor the "it's just a spill
  detector" I wrongly wrote — the honest middle. (Refinement B validated but not yet integrated into
  `costmodel.py`; integrating it would cascade a re-fit of all C500 numbers — flagged, not done here.)
