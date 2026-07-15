# Toxic-Fusion Project — Open Gaps toward Publication

*Completed work removed from this list per request (the actual work stays in the repo + git history):*
- *Review items 1–9 (MetaX review session) — commit `5a909ab`.*
- *Ada task A1 (regenerate clean dataset + re-fit constants on Ada) — commit `fdfce17`.*

This list now tracks **only the remaining gaps toward a publishable paper** (from the 2026-07-14
publication-gap review). Priority order ≈ blocking order.

**Current state (post-A1, Ada RTX 4060, 64 genuine cases):** RQ1 in-sample F1=0.970 (P=1.000,
R=0.941, 1 FN); RQ2 occupancy 22/22, attribution 12/12; RQ4 model=oracle on 3 *synthetic* subgraphs.
**Single consumer NVIDIA GPU; the C500 cross-vendor transfer — the core novelty — is not started.**

---

## G1 — [CORE NOVELTY] Cross-vendor transfer to MetaX C500 + decision-flip   ✅ DONE (occupancy-metric deferred)
**Why this is the paper.** The search-free + interpretable *outcome* is largely covered by prior work
(Welder, 2026 analytical models). The defensible differentiator is **cross-vendor transfer incl. a
domestic GPU + the first fusion characterization of the C500 + a documented decision-flip**.
Phase 4, **[MX] env `fusion`** (MACA 3.7.0, 4× C500). Full results: `logs/LOG-04-c500-transfer.md`;
reproduce: `python -m model.transfer_c500`.
- [x] C500 `HardwareModel` (`METAX_C500` in `fusion/hw.py`, env `FUSION_HW=c500`): 104 CUs, **64-wide
      waves**, 131072 regs/CU, split ST/MT RF, 4 KB spill cap.
- [x] Full 72-case matrix on C500 → `data/microbench_timing_c500.csv` (0 skips). Pipeline transfers
      unchanged (compile + correctness max|Δ|=0 + Triton static inputs).
- [x] Re-fit `DeviceConstants` for C500 (formulas frozen; `model/c500_constants.json`): gamma_spill
      0.0807→0.0068, **B_peak 0.15→1.05 TB/s** (physical sanity check passes).
- [x] **RQ3 transfer:** C500 decision F1=0.909, R=1.000 (64 genuine cases).
- [x] **≥1 decision-flip, documented + mechanistically explained:** `sibling_redux NOUT=32 fp32` =
      beneficial on Ada (0 spills, 1.04×) → **toxic on C500** (100 spills, 0.67×), consistent across
      all 4 shapes; the model flips its verdict via the re-read static spill count. Driven by the
      64-wide wavefront's register pressure.
- [x] **MCPTI ground truth on C500** (RQ2-on-C500): drove MCPTI directly via ctypes (the `mcProfiler`
      CLI value-dump is UNIMPLEMENTED — grpc 12); `fusion/mcpti_profile.py` → `data/microbench_c500_mcpti.csv`.
      **Attribution validated on hardware: dominant=spill 12/12** (local spill traffic scales with
      `n_spills`, dominates DRAM ~307×); **model attribution == profiled 12/12** (C500 analogue of Ada's 8/8).
- [x] Cross-vendor generalization table + folded C500 results into `MODEL_SPEC §7` and `PROPOSAL §8`
      (`model.transfer_c500` prints the table reproducibly).
- [ ] (deferred, low value) Achieved-occupancy on C500 via the MCPTI **Metric** API (no raw `waves`
      event); calibrate occupancy granularities vs measured waves. Occupancy term is inert, so parked.
**Honest caveat (feeds G2/G3):** the binary decision is spill-dominated, so Ada constants on C500 give
the *same* F1 — the transfer is carried by the re-read static inputs, not the re-fit constants; and the
4 C500 FPs are `NOUT=32 fp16` (same 100 spills as the toxic fp32) — spill *count* alone can't separate them.

## G2 — Model collapses to "did it spill?"  → now with a CONCRETE FAILURE CASE (see `logs/LOG-05`)
Ablations: drop-spill F1 **0.545** vs drop-occupancy **0.970** (occupancy inert); P_layout never flips
a decision on Ada. **New (C500 GEMM sweep):** even in the compute-bound GEMM regime, low occupancy
alone does NOT flip the decision (occ 0.125: fp16 spill-117 beneficial, fp32 spill-205 toxic) — spill
is the decisive signal across BOTH op families. **BUT the spill-count model then FAILS on GEMM's toxic
cases** (recall 0/4): the toxic fp32-128×128 configs have fused `f_spills=205` < unfused `u_spills=234`,
so the model predicts *beneficial* while the fusion is measured *toxic* — the static spill count has the
**wrong sign** (the fused epilogue re-reads spilled state; static count misses that runtime cost).
**VERIFIED + root-caused (LOG-05 §5–6):** toxicity is real (min-time 150 iters = 0.826, not an
artifact); MCPTI confirms fused local traffic **950K > unfused 833K** (epilogue re-reads spilled acc).
**Fix POC works:** spill-*traffic* = `f_spills × reread_mult` (2 for epilogue-into-spilling-producer,
from Φ(v)) flips recall **0→1.0** on GEMM. Flat ×2 costs precision (0.5), so →
- [x] **Implemented the reread-aware spill-traffic feature + re-fit on the combined dataset** (LOG-06):
      `spill_reread()` in `base.py` (taxonomy-derived), threaded through `costmodel.py`/`fit.py`
      (backward-compatible, default 1); combined C500 set via `build_combined_c500.py`. **Result:
      GEMM recall 0→1.000, reductions unchanged (F1=0.905), overall F1 0.826→0.852, Ada RQ1 unchanged
      (0.970).** Constants: `model/c500_combined_constants.json`. **Closes G7** (Φ(v) now load-bearing).
- [ ] Refine the residual: GEMM precision 0.5 (flat ×2 over-rejects fp16 big-tile) — a dtype/compute-
      aware reread or spill-traffic estimate to separate fp16-benign from fp32-toxic at equal spills.
- [ ] (still open) Find a regime where P_occ / P_layout genuinely *compete* without spilling at all —
      GEMM-epilogue didn't (its big round-trip saving makes it spill-or-beneficial); try **horizontal
      multi-GEMM fusion** (fuse-wide drops occupancy but saves little), the GEMM analog of Family R.

## G3 — Non-spill / spill-count-blind toxicity (Ada 1 FN + the new C500 GEMM cases)
Ada: a NOUT=32 reduction toxic (0.91×) with no spill. **C500 (new):** 4 GEMM fusions toxic while the
static spill count says beneficial (§G2). Same dangerous direction (model *keeps* a toxic fusion).
- [ ] Extend the model to catch these (couple with G2's non-spill signal), **OR** scope the claim to
      "spill-count-visible toxicity" and characterize this blind spot honestly in the writeup.

## G4 — Evaluation breadth (too thin for a conference)  → CONTRACTION done
- [x] **CONTRACTION (GEMM-epilogue)** implemented (`fusion/kernels/gemm_epilogue.py`) + characterized on
      C500 (16-config 4-GPU sweep → `data/microbench_gemm_c500.csv`; `tl.dot`/MMA works). Fusion
      beneficial 12/16, toxic 4/16 (fp32 big-tile). See `logs/LOG-05`.
- [ ] Still missing taxonomy classes: **BROADCAST**; **softmax / LayerNorm** fusions (the user's
      "normalization-into-GEMM/attention" PDF is a good anchor).
- [x] **Real transformer subgraphs + real-compiler baseline** (LOG-07): `torch.compile`/**Inductor
      works on the C500** (first demo); swept a memory-bound pointwise/residual chain + a compute-bound
      MLP-FFN (`fusion/inductor_baseline.py`, 4-GPU). **Result validates the model's thesis on a REAL
      compiler + REAL patterns:** fusion benefit is regime-governed — memory-bound **1.9–3.65×**,
      compute-bound **0.74–1.06× (2/4 MLPs toxic)**. Even Inductor over-fuses net-harmfully when
      compute-bound → motivates the interpretable pruning pass.
- [x] **Predict Inductor's own fusion outcomes per-kernel** (LOG-07 §6): hooked `triton.compile` to
      read the register/spill report of the fused Triton kernel Inductor emits; the model predicts its
      fusion beneficial 5/5 (sign) on elementwise chains — the model consumes REAL compiler output.
      Honest limits: capability demo (elementwise fusion ~always beneficial → not discriminating);
      compute-bound cases route to the vendor GEMM (not an Inductor Triton kernel to score).
- [ ] (deepen further) a *discriminating* real-compiler case (an Inductor Triton fusion that is
      actually toxic — e.g. force spilling), attention / FlashAttention-style block, TVM/Welder.
- [ ] Scale the dataset (add the GEMM family + more op-pairs to the fit, not just reductions/pointwise).
- [ ] Run the GEMM family on **Ada** too (needs the Ada machine) for the cross-vendor GEMM comparison.

## G5 — Single, noisy hardware point
- [ ] Add the **Ampere sm80** GPU (available) — cheap second NVIDIA point; de-risks single-device overfit
      and strengthens "transfer by re-parameterization."
- [ ] Report timing with **variance / CIs** and more iterations (the fp32 "3.03×→artifact" episode shows
      the laptop timing is noisy; RQ4 model=oracle currently rests on single-shot laptop timings).

## G6 — No real compiler integration
- [ ] Wire the recommender into an actual **Inductor/Triton scheduler hook** (currently the offline
      recommender — the sanctioned fallback).
- [ ] Report **real end-to-end model latency**, not synthetic-subgraph microbench sums.

## G7 — Make Φ(v) taxonomy substantive  ✅ DONE
- [x] The Φ(v) taxonomy is now **load-bearing**: `spill_reread(producer, consumer)` (`base.py`) derives
      the spill-traffic re-read multiplier from the topological classes (CONTRACTION→POINTWISE ⇒ ×2),
      and it fixes the GEMM blind spot (G2, LOG-06). The taxonomy is no longer decorative.

---
**Target venue (honest):** G1 + a second GPU (G5) + one real subgraph (G4) → **workshop paper or
MLSys short/poster**. A full MLSys/ASPLOS paper additionally needs G6 (real integration) + real
end-to-end speedups.
