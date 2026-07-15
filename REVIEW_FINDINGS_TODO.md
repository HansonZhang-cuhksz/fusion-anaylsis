# Toxic-Fusion Project — Open Gaps (bucketed)

**Bottom line (honest).** The core cross-vendor + spill-traffic story is coherent and its central
results are now statistically backed (cross-device CIs, LOG-09), **defensible at the workshop /
MLSys-short scale that `RESULTS.md` scopes — but it is not complete.** Three real limitations
(Bucket 3) would need work before a full conference paper; the hardware-gated items (Bucket 1) are
"return to the Ada/Ampere machine"; breadth (Bucket 2) is upside.

*Done work lives in the repo + git history; the consolidated results are in `RESULTS.md`
(RQ1–RQ4 + limitations), details in `logs/LOG-01…09`. Buckets 1 and 3 are what actually matter.*

---

## Bucket 0 — The two substantive open items surfaced by the Ada adversarial review (LOG-10) — DO HERE (C500)
Both are C500-side (do **not** need Ada); they are the sharpest current gaps. Working log: `logs/LOG-11-*`.
- [x] **#1 — Does the cost model beat the trivial `spill>0 ⇒ toxic` rule? → MARGINALLY YES out-of-fold,
      but it does NOT solve the hard cases (LOG-11).** Converged (restarts=12, seed-stable, verified):
      leave-one-dtype-out F1 **0.873** baseline / **0.906** with a dtype-aware compute term, vs trivial
      **0.857** — so the model adds modest out-of-fold value and the dtype refinement helps. BUT in-sample
      it's tied/worse (0.852) and both get **7/8 discriminating cases wrong** (fp16 spill-but-beneficial,
      static-identical to toxic fp32 twins). *(I made two method errors here — a scripting bug and
      under-converged `restarts=3` fits that wrongly read as "overfits/negative"; an adversarial verifier
      caught the second and I confirmed the correction.)* Outcome: honest claim = "modest out-of-fold gain,
      hard cases unsolved," not a decisive win nor "just a spill detector".
- [x] **Follow-up (b) — integrate the dtype-aware compute term. DONE (LOG-11 §5).** Added
      `DeviceConstants.fp16_compute_mult` (fp16 ~2× FMA throughput; FIXED per-device, default off),
      threaded through `predict_time`/`decide`/`fit`; enabled on C500 (`c500_combined_constants.json`:
      2.0) → in-sample F1 0.852→**0.873**, LOO-dtype 0.873→**0.906**, decision-flip fp32-toxic 3/4→**4/4**,
      no regressions; left off on Ada (0 discriminating cases → mild out-of-fold regression 0.875→0.839).
      Reduction-only transfer (RQ3 0.909) and Inductor demo unaffected (verified).
- [ ] **Still open — crack the discriminating cases.** The integrated term still gets **7/8** C500
      spill-but-beneficial fp16 cases wrong. A real fix needs a spill-as-**compute-serialization** model
      (so fp16's throughput actually separates the static-identical fp16/fp32 twins) **and** a larger,
      non-trivially-separable dataset. Deferred (separating 8 cases risks overfitting).
- [ ] **#2 — Why does the C500 toolchain allocate 1.6–1.75× more regs/thread than ptxas?** The flip
      mechanism is register-allocation divergence, not wavefront width (refuted, LOG-10 §2), but the
      *cause* (compiler maturity vs ISA/accumulator layout) is unidentified. C500-side investigation:
      the ST/MT split, tensor-core accumulator layout, SASS/PTX. (Ada reg counts already committed.)

---

## Bucket 1 — Hardware-gated (cannot be done in this MetaX session) → see `HANDOFF-ADA-BUCKET1.md`
Open only because the hardware isn't here, not because they're skipped. Instructions for the Ada
session are in **`HANDOFF-ADA-BUCKET1.md`**.
- [ ] **Run the GEMM/CONTRACTION family on Ada** (`fusion.gemm_sweep` → `data/microbench_gemm_ada.csv`);
      the cross-vendor GEMM story is currently **one-sided (C500-only)**. Check if the reread fix holds
      on Ada or exposes a second decision-flip.
- [ ] **Ada-side timing CIs** — the flip's Ada side (`redux_N32_fp32` should be significantly
      *beneficial* on Ada, making the decision-flip bulletproof on both sides) + a bootstrap CI on RQ1 F1.
- [ ] **Add the Ampere sm80** as a third hardware point (de-risks single-device overfit; a 3-way
      Ada/C500/sm80 generalization table). *Only if the sm80 GPU is reachable.*

## Bucket 2 — Breadth that raises the venue tier (not required for the core claims)
A workshop contribution stands without these; a top-tier conference paper would want them.
- [ ] More op families: the declared-but-unused **BROADCAST**; **softmax / LayerNorm** fusions (the
      "normalization-into-GEMM/attention" pattern); an **attention / FlashAttention-style** block.
- [ ] **Scale the dataset** — add op-pairs beyond reductions/pointwise/GEMM to the fit.
- [ ] Compare against **TVM / Welder** (beyond the Inductor baseline already done).
- [ ] Refine the residual GEMM precision (flat reread ×2 over-rejects fp16 big-tile ⇒ precision 0.5):
      a dtype/compute-aware spill-traffic estimate.

## Bucket 3 — Genuine limitations that qualify "defensible" (the ones to weigh)
- [ ] **Interpretability is narrow.** In every sweep only the **spill** term ever *flips* a decision;
      smooth occupancy (P_occ) and layout (P_layout) never do. So "interpretable *multi-cause*
      attribution" is really "interpretable spill-traffic + roofline." Either find a regime where P_occ
      or P_layout genuinely competes **without spilling** (tried: GEMM sweep = spill-driven; recompute =
      beneficial — both failed), or **scope the claim honestly** in the writeup. Disclosed in RESULTS.md.
- [ ] **No real compiler *integration*.** RQ4 is a real-compiler *comparison* (Inductor), not the
      recommender wired into a real scheduler with **real end-to-end model latency**. For a paper
      claiming "a compiler pass," this is a substantive gap (G6).
- [ ] **Discriminating real-compiler case is an honest negative** (LOG-08): a well-tuned Inductor
      wouldn't emit a toxic Triton fusion, so the model's discriminating power is shown only on
      controlled kernels. Cleaner future angle: intercept an autotuner's *candidate* (spilling) tiles,
      or run against the C500's native (non-Inductor) fuser.

## Deferred — explicitly low value
- [ ] Achieved-occupancy on C500 via the MCPTI **Metric** API (no raw `waves` event). Parked: the
      occupancy term is inert (spills dominate), so this buys little.
