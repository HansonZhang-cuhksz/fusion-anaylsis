# Toxic-Fusion Project — Open Gaps (bucketed)

**Bottom line (honest).** The core cross-vendor + spill-traffic story is coherent and its central
results are now statistically backed (cross-device CIs, LOG-09), **defensible at the workshop /
MLSys-short scale that `RESULTS.md` scopes — but it is not complete.** Three real limitations
(Bucket 3) would need work before a full conference paper; the hardware-gated items (Bucket 1) are
"return to the Ada/Ampere machine"; breadth (Bucket 2) is upside.

*Done work lives in the repo + git history; the consolidated results are in `RESULTS.md`
(RQ1–RQ4 + limitations), details in `logs/LOG-01…09`. Buckets 1 and 3 are what actually matter.*

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
