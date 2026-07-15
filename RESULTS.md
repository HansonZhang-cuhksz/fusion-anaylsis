# Detecting Toxic Operator Fusion — Consolidated Results

*Reviewer-ready synthesis of LOG-01…07 (2026-07). Reproduce pointers in each subsection; frozen model
in `model/MODEL_SPEC.md`; open items in `REVIEW_FINDINGS_TODO.md`.*

## Setup
- **Hardware.** NVIDIA **Ada** (RTX 4060 Laptop, sm89; 32-thread warps, 64K reg/SM, soft spill) and the
  domestic **MetaX C500** (MACA 3.7.0; **64-thread wavefronts**, 128K reg/CU, split ST/MT register file,
  **hard 4 KB/thread spill cap**). Ampere sm80 is available but not yet used.
- **Model (frozen; `MODEL_SPEC.md`).** A search-free interpretable degradation model
  `η_fused = min(η_u,η_v)·P_occ·P_layout`; every input comes from **one compile** (Triton `n_regs`,
  `n_spills`, shared mem, or `ptxas -v`). Deploy-time decision: prune the fusion iff `T_fused > T_unfused`
  on a per-device roofline. The spill term uses spill **traffic** = `n_spills × reread`, where `reread`
  is a **taxonomy-derived** multiplier (CONTRACTION→POINTWISE epilogue re-reads the spilled accumulator
  ⇒ 2, else 1). Profiling is used only to fit per-device constants / validate — never at decision time.
- **Op families.** P = pointwise chains, R = sibling reductions (the spill-cliff knob), T = transpose /
  bank-conflict (Ada), **G = GEMM-epilogue / CONTRACTION** (C500, tensor-core / compute-bound).

## RQ1 — Predictability (can static inputs predict fusion toxicity?)  ✅
- **Ada** (64 genuine cases; 16 degenerate no-ops excluded): in-sample **F1 0.970, precision 1.000,
  recall 0.941** (TP 16/FP 0/FN 1). Held-out CV — shape 0.865, leave-one-NOUT-out 0.829,
  leave-one-dtype-out 0.970. The single FN is a *non-spill* NOUT=32 toxicity (speedup 0.91) — an honest
  blind spot of a spill-focused model.
- **C500** combined (reduction + pointwise + GEMM, 80 cases): **F1 0.852, recall 0.958**.
- **Ablations** isolate the driver: drop the spill term → F1 0.55; drop smooth occupancy → F1 ≈ unchanged.
  The decisive signal is the **register-spill discontinuity**, not the smooth occupancy curve.
- *Reproduce:* `python -m model.fit data/microbench_timing.csv` (Ada);
  `python -m model.build_combined_c500 && …` (C500). LOG-02/03/06.

## RQ2 — Interpretability (does attribution match the profiler?)  ✅
- **Ada.** Analytic occupancy reproduces ncu *theoretical* occupancy 22/22 (a calculator reproduction,
  not achieved occupancy — stated honestly). Dominant-penalty attribution **12/12** (8 spill via ncu
  local-mem bytes + 4 layout via bank conflicts). LOG-03.
- **C500.** Ground truth via **MCPTI-direct** — the `mcProfiler` CLI value-dump is *unimplemented*
  (grpc 12); we drive `libmcpti` via ctypes (`fusion/mcpti_profile.py`). Attribution **12/12 spill**:
  local (spill) traffic scales monotonically with `n_spills` and **dominates DRAM ~307×**, so the static
  single-compile spill count is grounded in real hardware private-memory traffic. LOG-04 §7.

## RQ3 — Cross-vendor transfer (Ada → C500 by re-parameterization)  ✅ (the core novelty)
- **Transfer.** Swapping only `DeviceConstants` + `HardwareModel` (formulas frozen), the C500 model gets
  **decision F1 0.909, recall 1.000**; re-fit `B_peak ≈ 1.05 TB/s` sanity-checks against C500 HBM.
- **Decision-flip** (the headline). `sibling_redux NOUT=32 fp32` is **beneficial on Ada** (0 spills,
  1.04×) but **toxic on C500** (100 spills; median **0.64×, 95% CI [0.638, 0.645]** on 4 independent
  GPUs × 20 rounds — significant, LOG-09), consistent across all 4 shapes. The *same* model,
  fed each device's compile report, flips its verdict. Mechanism: the C500's **64-wide wavefront**
  doubles register pressure, so the fusion spills at a *smaller* NOUT than on Ada's 32-wide warps.
- **Honest nuance.** The binary decision is spill-dominated, so applying the *Ada* constants to C500 data
  gives the *same* F1 — the transfer is carried by the **re-read hardware-specific static inputs**, not
  by re-fitting constants. Cross-vendor generalization table in `MODEL_SPEC §7`. LOG-04.

## RQ4 — Utility (vs greedy, oracle, and a real compiler)  ✅
- **Synthetic subgraphs.** The offline recommender is up to **9.85× faster than greedy** and matches the
  timed **oracle** on fp16 (a constructed worst-case for greedy — honest). LOG-03.
- **Real compiler (`torch.compile`/Inductor — runs on the C500, a first).** Fusion benefit is
  **regime-governed exactly as the roofline predicts**: memory-bound **1.9–3.65×** (mean 2.57×),
  compute-bound **0.74–1.06×** (2/4 MLP-FFNs actually *slower*). Even a production compiler **over-fuses
  net-harmfully when compute-bound** — the mistake an interpretable pruning pass catches. LOG-07 §2–3.
- **Predicting the compiler's own fusions.** Hooking `triton.compile`, the model reads Inductor's
  generated fused kernel's single-compile report and predicts its fusion benefit — sign correct **5/5**
  on elementwise chains. *Capability demo* (elementwise fusion ~always beneficial ⇒ not discriminating).
  LOG-07 §6.
- **A toxic Inductor fusion is hard to trigger (honest negative, LOG-08).** Even forced recomputation
  stays beneficial (fusion saves round-trips *and* kernel launches; toxicity needs a pathological,
  uncompilable kernel). So the model's *discriminating* power is best shown on **controlled kernels**
  (the CONTRACTION tile sweep — a spilling fp32 tile the reread model catches) and targets **weaker /
  domestic compilers** without Inductor-grade autotuning — precisely this project's cross-vendor thesis.

## Cross-cutting finding: static spill *count* → spill *traffic*
On the CONTRACTION (GEMM-epilogue) family the search-free model **failed** (recall 0/4 on the toxic
fp32 big-tile configs): the fused kernel's static spill *count* (205) is *lower* than the unfused
(234), so the model predicted "beneficial" while the fusion is measured **toxic** (0.78× mean over
shapes; median 0.82×, 95% CI [0.821, 0.827] on 4 GPUs — significant, LOG-09). MCPTI
root-cause: the fused kernel's epilogue **re-reads the spilled accumulator**, moving *more* local
traffic (950K vs 833K) than the count implies — the static count has the wrong sign. The fix (spill
traffic = count × taxonomy-derived reread) recovers **GEMM recall 0→1.0** with **reductions unchanged**
and **Ada RQ1 unchanged**, independently verified incl. leave-one-family-out (held-out GEMM recall 1.0;
0.0 with the feature ablated ⇒ it generalizes, not overfit). This is what makes Φ(v) **load-bearing**.
LOG-05/06.

## Contributions
1. A **search-free, interpretable** fusion-rejection criterion with single-compile inputs and a
   **taxonomy-derived spill-traffic** term.
2. The **first fusion characterization of the MetaX C500** and a **cross-vendor transfer with a
   documented, mechanistically-explained decision-flip**, hardware-validated via MCPTI.
3. A **real-compiler (Inductor) validation** on the C500 showing fusion benefit is roofline-governed and
   that even a production compiler over-fuses net-harmfully compute-bound.

## Honest limitations (for the writeup)
- **Interpretability is mostly one cause.** Smooth occupancy (P_occ) and layout (P_layout) never *flip*
  a decision in these sweeps; toxicity is driven by the **spill cliff** (+ now the roofline compute term
  for recomputation). The model is largely a spill-traffic + roofline detector; the "multi-cause"
  framing is honestly narrow.
- **Hardware breadth.** One consumer NVIDIA GPU + one domestic GPU; no datacenter NVIDIA, and the
  available **Ampere sm80** is not yet a third point. The **GEMM family is C500-only** (Ada GEMM run
  pending) ⇒ the cross-vendor GEMM comparison is incomplete.
- **Statistics.** The **key C500 claims now carry 95% CIs measured on 4 independent GPUs** (LOG-09):
  the decision-flip (CI [0.638, 0.645]) and the GEMM toxicity (CI [0.821, 0.827]) are toxic with
  overwhelming significance, and cross-device-reproducible to ~3 decimals. **The Ada-side claims**
  (RQ1 F1, the Ada half of the flip) still warrant the same CI pass on the Ada machine. Datasets are
  small (64–80 cases).
- **Real-compiler scope.** The Inductor prediction is a capability demo; a *toxic* Inductor fusion could
  not be triggered at tractable sizes (LOG-08), so the discriminating evidence is on controlled kernels
  (the CONTRACTION spill tile) — the model targets weaker / domestic fusers, not a tuned Inductor.

## Venue (honest)
An empirical + systems contribution — **workshop / MLSys-short scale**. The domestic-GPU cross-vendor
angle (decision-flip + first C500 characterization + MCPTI-direct recipe) is the differentiator a
Hopper-rich lab would not produce.
