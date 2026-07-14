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

## G1 — [CORE NOVELTY] Cross-vendor transfer to MetaX C500 + decision-flip   ✅ CORE DONE (polish remains)
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
- [ ] Achieved-occupancy on C500 needs the MCPTI **Metric** API (no raw `waves` event) — deferred,
      low value (occupancy term inert). Calibrate occupancy granularities vs measured waves (secondary).
- [ ] Fold C500 results into `MODEL_SPEC`/`PROPOSAL` + a cross-vendor generalization table.
**Honest caveat (feeds G2/G3):** the binary decision is spill-dominated, so Ada constants on C500 give
the *same* F1 — the transfer is carried by the re-read static inputs, not the re-fit constants; and the
4 C500 FPs are `NOUT=32 fp16` (same 100 spills as the toxic fp32) — spill *count* alone can't separate them.

## G2 — On Ada the model collapses to "did it spill?"
Ablations: drop-spill F1 **0.545** vs drop-occupancy **0.970** (occupancy term inert, occ_knee pinned
at floor); P_layout **never flips a decision** on Ada. So the interpretable occupancy-vs-layout
attribution reduces, on the only HW tested, to a single binary spill feature; the layout branch is an
isolated 4-row microbench that never affects a real decision.
- [ ] Demonstrate a regime where P_occ and P_layout genuinely compete / flip decisions (C500 hope, or
      richer ops) — otherwise "interpretable multi-cause attribution" is overstated.

## G3 — Non-spill-toxicity blind spot (recall 0.941, the 1 FN)
A NOUT=32 reduction is toxic (speedup 0.91) with **no spill** — the spill-focused model *keeps* it (the
dangerous FN direction for a rejection pass).
- [ ] Extend the model to catch occupancy/bandwidth-bound (non-spill) toxicity, **OR** explicitly scope
      the contribution to "spill-dominated toxicity" and characterize the blind spot.

## G4 — Evaluation breadth (too thin for a conference)
- [ ] Implement the declared-but-missing taxonomy classes: **CONTRACTION** (GEMM-epilogue) and
      **BROADCAST**; add **softmax / LayerNorm** fusions.
- [ ] Replace the 3 *synthetic* sibling-reduction "subgraphs" with **≥1 real transformer subgraph**
      (attention / MLP-FFN / LayerNorm+Linear) for RQ4.
- [ ] Compare against a **real compiler** baseline (torch.compile/Inductor default fusion; ideally
      TVM / Welder), not just greedy + oracle.
- [ ] Scale the dataset (currently 64 cases / 17 toxic; per-fold F1 swings 0.0–1.0; only 3 NOUT values).

## G5 — Single, noisy hardware point
- [ ] Add the **Ampere sm80** GPU (available) — cheap second NVIDIA point; de-risks single-device overfit
      and strengthens "transfer by re-parameterization."
- [ ] Report timing with **variance / CIs** and more iterations (the fp32 "3.03×→artifact" episode shows
      the laptop timing is noisy; RQ4 model=oracle currently rests on single-shot laptop timings).

## G6 — No real compiler integration
- [ ] Wire the recommender into an actual **Inductor/Triton scheduler hook** (currently the offline
      recommender — the sanctioned fallback).
- [ ] Report **real end-to-end model latency**, not synthetic-subgraph microbench sums.

## G7 — Minor / writeup
- [ ] The Φ(v) taxonomy is currently decorative (the model uses regs/spills/bytes/occupancy) — either
      use it substantively or de-emphasize it in the writeup.

---
**Target venue (honest):** G1 + a second GPU (G5) + one real subgraph (G4) → **workshop paper or
MLSys short/poster**. A full MLSys/ASPLOS paper additionally needs G6 (real integration) + real
end-to-end speedups.
