# Toxic-Fusion Project — Open Gaps (bucketed)

**Bottom line (honest).** The core cross-vendor + spill-traffic story is coherent and its central
results are statistically backed (cross-device CIs, LOG-09). **Bucket 1 (hardware-gated) is now
resolved** — the GEMM family runs on both devices (a *second* decision-flip fell out of it), the Ada
CIs are committed, and sm80 is confirmed unreachable/blocked. What remains is Bucket 2 breadth (venue
upside, not required) and Bucket 3 genuine limitations (disclosed in `RESULTS.md`; two are inherent to
the current scope). **Defensible at the workshop / MLSys-short scale `RESULTS.md` claims.**

*Done work lives in the repo + git history; consolidated results in `RESULTS.md` (RQ1–RQ4 +
limitations), details in `logs/LOG-01…12`. Legend: `[x]` = done / resolved (blocked / parked /
disclosed-negative are marked as such); `[ ]` = genuinely open.*

---

## Bucket 0 — The substantive items surfaced by the Ada adversarial review (LOG-10). Working log: `logs/LOG-11..12`.
- [x] **#1 — Does the cost model beat the trivial `spill>0 ⇒ toxic` rule? → MARGINALLY YES out-of-fold,
      but it does NOT solve the hard cases (LOG-11).** Converged (restarts=12, seed-stable, verified):
      leave-one-dtype-out F1 **0.873** baseline / **0.906** with a dtype-aware compute term, vs trivial
      **0.857** — so the model adds modest out-of-fold value and the dtype refinement helps. BUT in-sample
      it's tied/worse (0.852) and both get **7/8 discriminating cases wrong** (fp16 spill-but-beneficial,
      static-identical to toxic fp32 twins). *(I made two method errors here — a scripting bug and
      under-converged `restarts=3` fits that wrongly read as "overfits/negative"; an adversarial verifier
      caught the second and I confirmed the correction.)* Outcome: honest claim = "modest out-of-fold gain,
      hard cases unsolved," not a decisive win nor "just a spill detector".
- [x] **Follow-up (b) — integrate the dtype-aware compute term. DONE (LOG-11 §5, commit 4368d3a).** Added
      `DeviceConstants.fp16_compute_mult` (fp16 ~2× FMA throughput; FIXED per-device, default off),
      threaded through `predict_time`/`decide`/`fit`; enabled on C500 (`c500_combined_constants.json`:
      2.0) → in-sample F1 0.852→**0.873**, LOO-dtype 0.873→**0.906**, decision-flip fp32-toxic 3/4→**4/4**,
      no regressions; left off on Ada (0 discriminating cases → mild out-of-fold regression 0.875→0.839).
      Reduction-only transfer (RQ3 0.909) and Inductor demo unaffected (verified).
- [ ] **Still open — crack the discriminating cases** *(disclosed limitation, RESULTS.md; deferred).* The
      integrated compute term still gets **7/8** C500 spill-but-beneficial fp16 cases wrong. A real fix
      needs a spill-as-**compute-serialization** model **and** a larger, non-trivially-separable dataset
      (separating 8 cases on the current data risks overfitting). Disclosed, not attempted.
- [x] **#2 — Why does the C500 toolchain allocate 1.6–1.75× more regs/thread? → IDENTIFIED (LOG-12).**
      Two additive causes: (a) **software-pipeline multi-buffering in registers** (~34 regs per
      `num_stages`; Triton default 3 → ~68 extra regs), and (b) a **general ~2× allocator-efficiency gap**
      vs ptxas (visible in the pipeline-free reduction too: C500 needs 2× the regs for *half* the
      per-thread accumulator work). **Ruled out:** wavefront width (refuted), scalar-register underuse
      (MACA uses ST: `26 MT + 16 ST`), accumulator layout (not MMA-specific). **Load-bearing
      consequence:** the decision-flip's C500 spill is a **default-launch-config artifact** — `num_warps=8`
      or `num_stages=1` eliminates it for BOTH families (GEMM 205→0, reduction 100→0). Reframes the flip
      as "default-config toxic on C500," not "fundamentally toxic." Disclosed in RESULTS.md + LOG-04.
- [ ] **#2 follow-up** *(untested; disclosed RESULTS.md L160 / LOG-12).* Time the 8-warp re-tuned
      fused-vs-unfused on C500 (does removing the spill flip the decision back to beneficial?). Not measured
      — all committed C500 timing is `num_warps=4`. Would extend the flip result to "the flip is tunable."

---

## Bucket 1 — Hardware-gated — RESOLVED. (`HANDOFF-ADA-BUCKET1.md`; done by the Ada session, commit `314055c`.)
- [x] **GEMM/CONTRACTION family on Ada — DONE.** `data/microbench_gemm_ada.csv` (16 rows, all
      `f_spills=0`; `logs/run_gemm_ada.log`, LOG-10 §2). The cross-vendor GEMM story is now bilateral, and
      it **exposed a SECOND decision-flip**: gemm 128² fp32 = beneficial on Ada (1.083, no spill) vs toxic
      on C500 (0.824, 205 spills). The reread fix is **vacuous on Ada** (multiplies zero spills — an
      arithmetic identity, honestly documented as *not* a control), so evidence for it stays C500-only.
- [x] **Ada-side timing CIs + RQ1 bootstrap — DONE.** `data/timing_ci_ada.csv`: `redux_N32_fp32` = 1.040,
      CI [1.040, 1.041], beneficial/sig/20-round — vs C500's 0.643 TOXIC → the flip is **CI-backed on both
      sides**. `data/rq1_bootstrap_ada.csv`: RQ1 F1 point 1.0, bootstrap CI [1.0, 1.0], B=1000 (degenerate
      because predictions are frozen — honestly disclosed, RESULTS.md).
- [x] **Ampere sm80 as a third hardware point — BLOCKED (not accomplished, not actionable here).** Only
      the RTX 4060 Laptop (sm89) is reachable; no sm80 GPU exists on either machine (LOG-10 §4,
      user-confirmed). No sm80 data/constants/3-way table exist. Disclosed as **blocked, not pending**
      (RESULTS.md L278–279). Recorded as resolved so it stops reading as open work.

## Bucket 2 — Breadth that raises the venue tier (not required for the core claims; genuinely open)
- [ ] **More op families.** BROADCAST is still an unused enum constant (`fusion/kernels/base.py`, zero
      usages); no softmax / LayerNorm or attention / FlashAttention kernel exists — only
      pointwise/reduction/gemm_epilogue are implemented. Unstarted.
- [ ] **Scale the dataset.** The fit CSVs (`microbench_{ada,c500}_combined.csv`) hold only
      pointwise/reduction/gemm_epilogue; no BROADCAST/softmax/LayerNorm/attention rows anywhere. Unstarted.
- [ ] **Compare against TVM / Welder** (beyond the done Inductor baseline). No artifact anywhere — grep
      hits only PROPOSAL related-work. Never attempted.
- [ ] **Refine the residual GEMM precision (0.5).** Flat reread ×2 over-rejects the fp16 big-tile.
      **The `fp16_compute_mult` integration (4368d3a) did NOT fix this** — GEMM precision/F1 stays
      0.500/0.667 (verified on the 16 GEMM rows), because the fp16 big-tile is **memory-bound**, so the
      *compute* term can't move a memory-dominated prediction. Needs a dtype-aware **spill/memory-traffic**
      term, not the compute term. Still open.

## Bucket 3 — Genuine limitations that qualify "defensible"
- [x] **Interpretability is narrow — DISCLOSED / scoped honestly (RESULTS.md L248–251, ablation
      L104–105).** Only the **spill** term ever flips a decision; P_occ and P_layout never do. The
      regime-hunt for a non-spilling P_occ/P_layout flip failed (GEMM sweep = spill-driven; recompute =
      beneficial), so the claim is scoped as "interpretable spill-traffic + roofline," not multi-cause.
      Resolution was to disclose — done.
- [ ] **No real compiler *integration*** *(disclosed, but the gap is real — G6).* RQ4 is an offline
      recommender (`model/recommender.py`) + an Inductor *comparison* (LOG-07), never wired into a
      scheduler with real end-to-end model latency. RESULTS.md frames it honestly as offline/synthetic;
      the substantive integration gap remains for any "compiler pass" claim.
- [x] **Discriminating real-compiler case — RESOLVED as an honest negative (LOG-08, commit `a4421c6`).** A
      well-tuned Inductor won't emit a toxic Triton fusion (forced recompute stayed beneficial 3.6×/11×);
      documented in `fusion/inductor_toxic_probe.py` + LOG-08 and disclosed in RESULTS.md. A cleaner future
      angle (intercept an autotuner's candidate spilling tiles) is noted but not required.

## Deferred — explicitly low value
- [x] **Achieved-occupancy on C500 via the MCPTI Metric API — PARKED (won't-do).** MACA 3.7.0's Event API
      has no raw `waves` event, and the occupancy term is **inert** (drop-P_occ ablation leaves F1
      unchanged at 1.000; RESULTS.md L104–105). Buys nothing; explicitly parked.
