# Detecting Toxic Operator Fusion: A Search-Free, Interpretable Predictor Measured Across NVIDIA Ada and the MetaX C500

**Abstract.** Operator fusion is one of the highest-impact optimizations in deep-learning compilers, but fusing "as much as possible" is a false heuristic: fusion inflates register pressure and can spill to local memory, erasing the memory savings it was meant to capture. We ask a measurement question: can a compiler decide *at compile time, without autotuning search*, when **not** to fuse, using only inputs available from a single codegen pass (per-kernel register and spill counts)? We build a search-free, interpretable roofline-style predictor and measure it on a consumer NVIDIA Ada GPU (RTX 4060, sm89) and — to our knowledge for the first time — on the domestic **MetaX C500** accelerator, grounding its attribution in hardware counters (Nsight Compute on Ada; a direct MCPTI recipe on the C500, the vendor CLI being unimplemented). We report findings honestly, including negative ones. (1) In our microbenchmarks, fusion toxicity is **dominated by register spilling**: a zero-parameter "don't fuse if the kernel spills" rule is a strong baseline that the full model ties in-sample and, on Ada, does **not** beat out-of-fold. (2) On the C500, where genuine *spill-but-beneficial* cases exist, the model beats the spill rule **modestly** out-of-fold (leave-one-dtype-out F1 0.906 vs 0.857) but still misclassifies 7 of the 8 hardest cases. (3) The **same** fused kernel is beneficial on Ada but toxic on the C500 (a "decision-flip"), which we trace to the C500 toolchain allocating **1.6–1.75× more registers per thread** than `ptxas`; we further show — and this is the paper's sharpest finding — that this toxicity is a **default-launch-configuration artifact** that re-tuning (more warps) removes, and that the static predictor correctly *tracks* the flip either way. We conclude that the useful artifact is a config-dependent, spill-focused fusion predictor for accelerators whose autotuning toolchains are immature, and we release the dataset and code.

---

## 1. Introduction

Operator fusion eliminates HBM round-trips between a memory-bound producer and its consumer and is a canonical inference optimization (FlashAttention [9] being the archetypal win). Yet fusion is not free: combining two operators increases the register footprint and tile-shape constraints of the resulting kernel. When the fused kernel exceeds the register budget it **spills** to local memory, and the extra local-memory traffic can outweigh the saved HBM round-trip — a *toxic* fusion that is slower than not fusing at all.

Production compilers resolve this in one of three unsatisfying ways: (i) greedy/heuristic fusion (blind to microarchitectural cost); (ii) autotuning search [2] (expensive, per-shape, and a black box that never says *why* a fusion is bad); or (iii) learned cost models [11], trained on datasets such as [10] (opaque, and needing large per-hardware training sets). None provides a *cheap, static, interpretable* decision computed from inputs a compiler already has after one codegen pass.

This paper is a **measurement study** of that decision. We do not claim a new algorithm; we ask how far a deliberately simple, search-free predictor gets, and we characterize exactly where and why it succeeds and fails, across two vendors. Our contributions are:

1. **A search-free, single-compile fusion-rejection predictor** and a careful measurement of its decision quality against a strong zero-parameter spill baseline — the comparison most fusion papers omit.
2. **The first fusion characterization of the MetaX C500**, a domestic accelerator with a split scalar/vector register file and a hard 4 KB/thread spill cap, including a working **MCPTI-direct profiling recipe** (the vendor `mcProfiler` CLI value-dump is unimplemented in MACA 3.7.0).
3. **A cross-vendor "decision-flip"** — the same fused kernel beneficial on Ada, toxic on the C500 — with an identified register-allocation mechanism, and the finding that **the flip is a default-launch-config artifact** that the static predictor nonetheless tracks.
4. **An honest accounting of the limits** of the static single-compile approach: the discriminating fp16 cases require a *runtime* signal, and we prove no single-compile static input separates them without a dtype-label hack.

We release all code, kernels, and datasets.

## 2. Background and Related Work

**Deep-learning compilers and fusion.** TVM [1] and its autotuner Ansor [2] generate high-performance tensor programs by search; TorchInductor (PyTorch 2) [3] performs greedy Triton-level fusion. Fusion-specialized systems model memory movement (Welder [5]) or enable memory-intensive fusion spaces (AStitch [6]); Hidet [7] exposes register-level control through a task-mapping paradigm. These systems *achieve* the outcome — avoiding bad fusions — through scheduling or search, and are not designed to emit a compile-time, interpretable *reject* signal with an attributed cause. Learned cost models such as AutoTVM's [11] — and the datasets that train them, e.g. TpuGraphs [10] — amortize search but are opaque and hardware-specific.

**Roofline and static analysis.** Our predictor is a factored roofline model [8]: each plan's time is `max(compute, memory)` scaled by an efficiency term. The novelty relative to per-kernel roofline modeling is that we use it as a *fusion go/no-go criterion* driven by single-compile static inputs and transfer it across vendors by re-parameterization.

**Kernel generation and profiling.** Kernels are written in Triton [4]. Ground truth on Ada uses NVIDIA Nsight Compute / CUPTI [12]; on the C500 we use the CUPTI-compatible MCPTI interface directly.

**Positioning.** The *outcome* (avoid bad fusions) exists in search-based systems. What we measure — a search-free, interpretable, cross-vendor spill-toxicity predictor from single-compile inputs, and a fusion characterization of the MetaX C500 — is, to our knowledge, not covered, and the C500 has no prior published fusion analysis.

## 3. Method

### 3.1 Model

Each operator node is annotated with a taxonomy class `Φ(v) ∈ {Pointwise, Reduction, Contraction, Permute}`. For a candidate fusion `u→v` we predict the time of the fused and unfused plans on a per-device roofline:

```
T_plan = max( F / (C_peak · η) ,  M / (B_peak · η) ) + L · T_launch
η_fused = min(η_u, η_v) · P_occ · P_layout
```

where `F` is flops, `M` is bytes moved, `L` is the launch count, and `η` is a multiplicative efficiency in [0,1]. `P_occ` combines a smooth occupancy term with a **register-spill discontinuity**; `P_layout` charges bank-conflict / transpose cost. The **decision rule** is: prune the fusion edge iff `T_fused > T_unfused`, and report the multiplicatively dominant penalty as the *reason*.

**Spill traffic.** The spill term is `spill_factor(s) = 1/(1 + γ · s)`, where the spill signal `s = n_spills · reread`. The `reread` multiplier is taxonomy-derived: a Contraction→Pointwise epilogue re-reads the spilled (fp32) accumulator, so `reread = 2`; otherwise `1`. This corrects a sign error we observed on the GEMM family (§5.4).

**dtype-aware compute.** fp16 arithmetic runs at ≈2× fp32 FMA throughput, so the compute term carries a fixed per-device `fp16_compute_mult` (2.0 on the C500, off on Ada; §5.1).

### 3.2 Search-free static inputs

Every input to the deployed decision comes from **one compile** of the fused candidate — no run, no autotuning:

| Quantity | NVIDIA Ada | MetaX C500 |
|---|---|---|
| registers/thread, spills | Triton `n_regs`/`n_spills` (from `ptxas -v`) | Triton `n_regs`/`n_spills` (MTregisters) |
| shared memory / tile bytes | codegen | codegen |
| flops, bytes, launches | graph shapes | graph shapes |

Profiling (§4) is used **only** to fit per-device constants and to validate — never inside the deployed pass. Transfer across vendors swaps only the per-device constants and hardware descriptor; the formulas are frozen.

## 4. Experimental Setup

**Hardware.** NVIDIA Ada (RTX 4060 Laptop, sm89; 32-thread warps, 64K registers/SM, soft spill; 8 GB, under WSL2) and the MetaX **C500** (MACA 3.7.0; 64-thread wavefronts, 128K registers/CU, split scalar/vector register file, hard 4 KB/thread spill cap; 64 GB). No Ampere sm80 was reachable; the third hardware point is blocked, not deferred.

**Benchmarks.** Triton microbenchmarks over four families: **P** pointwise chains, **R** sibling reductions (the spill-cliff knob: `NOUT` independent projections sharing an input), **T** transpose/bank-conflict (Ada, raw CUDA), and **G** GEMM-epilogue / Contraction (`C = relu(A·B + bias)`, tensor-core). Each producer→consumer pair has fused and unfused variants, swept over shapes and dtypes (fp16, fp32). After excluding degenerate rows where the "unfused" plan is a single launch identical to the fused kernel (no fusion decision to make), the Ada set has 64 genuine cases and the combined C500 set 80 (24 toxic).

**Ground truth and fitting.** Timing uses repeated rounds of min-of-N with L2 flush. Per-device constants (`C_peak`, `B_peak`, `T_launch`, occupancy knee, `γ_spill`) are fit in log-space by Nelder-Mead with random restarts. Attribution ground truth: Nsight Compute on Ada (achieved occupancy, local-memory spill bytes, bank conflicts); **MCPTI-direct** on the C500 — because `mcProfiler`'s value-dump returns `UNIMPLEMENTED` (gRPC 12), we drive `libmcpti` via `ctypes`, reading CUPTI-legacy events (`local_load/store`, `global_load/store`) as before/after deltas.

**Statistics.** Decision quality is precision/recall/F1 of *toxic-fusion* detection (positive class = "don't fuse"). Speedups carry 95% percentile CIs over repeated rounds. We report leave-one-out cross-validation on the variables that move register pressure (`NOUT`, dtype), because in-sample separability can be trivial (§5.1).

**Threats to validity.** (i) An early Ada dataset was corrupted by a **VRAM-oversubscription artifact**: under WSL2, when the caching allocator's reserved footprint oversubscribed the 8 GB card, a spilling kernel's local-memory backing store could become host-resident, inflating *spilling-kernel* fused times by 9–41×. We detected this, reproduced its mechanism, and re-collected; non-spilling kernels were unaffected in ratio, and all 24 reduction rows were re-verified end-to-end. The C500 data is excluded from the artifact by three independent lines of evidence (≈8× VRAM headroom; matrix-vs-isolated agreement within 1.5%; the vendor raises a hard error rather than silently paging). (ii) Some single-shot toxic labels sit within ~5% of 1.0; where a 20-round CI overturns a label we say so. (iii) Our CIs bound *within-process* noise; two independent 20-round runs of one Ada config gave non-overlapping intervals, so run-to-run variation exceeds the quoted width — we treat any verdict within ~5% of 1.0 as unresolved. (iv) The dataset is small (~80 cases, 3–4 families, 2 GPUs).

## 5. Results

### 5.1 RQ1 — Predictability, and the spill-detector caveat

On Ada the in-sample decision is perfect: precision = recall = F1 = 1.000 (TP 16 / FP 0 / FN 0 / TN 48). **This does not validate the model's structure.** After the degenerate filter the Ada benchmark is *perfectly separable by one raw feature* — all 48 non-spilling cases are beneficial and all 16 spilling cases toxic — so a **zero-parameter rule, "don't fuse iff the fused kernel spills," also scores F1 = 1.000**. Worse, out-of-fold the fitted model is *dominated*: holding out the highest-spill regime collapses `γ_spill` and the model misses the toxic cases (leave-one-`NOUT`-out F1 **0.667**), while the zero-parameter rule scores **1.000** under the same protocol. Ada RQ1 therefore establishes only that *spilling is a sufficient toxicity indicator on this benchmark*, not that the roofline structure adds value. Ablations agree: dropping the spill term sends F1 to **0.000**, while dropping smooth occupancy leaves it at **1.000** — the spill discontinuity is the whole signal.

The C500 is the discriminating case, because it has genuine **spill-but-beneficial** kernels (fp16 tiles that spill yet fuse profitably). Here the model is **not** strictly dominated: with the dtype-aware compute term it reaches in-sample F1 **0.873** and leave-one-dtype-out F1 **0.906**, against the spill rule's **0.857** — a *modest*, seed-stable out-of-fold gain. But it still misclassifies **7 of the 8** hardest cases: those fp16 kernels have static inputs *identical* to their toxic fp32 twins (same registers, spills, occupancy — only dtype differs), and no single-compile input separates them (§5.4).

### 5.2 RQ2 — Interpretability (attribution vs profiler)

On Ada the analytic occupancy reproduces Nsight Compute's *theoretical* occupancy 22/22 (a calculator reproduction, stated as such — not achieved occupancy), and the dominant-penalty attribution matches the profiler **12/12** (8 spill-dominated via local-memory bytes, 4 layout-dominated via bank conflicts). On the C500, MCPTI-direct measurement shows local (spill) traffic scaling monotonically with the static `n_spills` and **dominating DRAM traffic by ≈307×**; the model's dominant-penalty attribution matches **12/12** (all spill). The single-compile spill count is thus grounded in real hardware private-memory traffic on both vendors — the one place the interpretability claim is unambiguously supported.

### 5.3 RQ3 — Cross-vendor transfer and the decision-flip

Re-parameterizing the frozen model to the C500 gives decision F1 **0.909** (recall 1.000) on the reduction family. We find **two decision-flips** — the same fused kernel beneficial on Ada but toxic on the C500:

| fused kernel | Ada | C500 (default config) |
|---|---|---|
| sibling reduction, `NOUT=32`, fp32 | 0 spills, **1.04×** (beneficial) | 100 spills, **0.64×** (toxic) |
| GEMM-epilogue 128×128, fp32 | 0 spills, **1.08×** (beneficial) | 205 spills, **0.82×** (toxic) |

Both are CI-backed (C500 reduction CI [0.638, 0.645] over 4 GPUs; Ada `NOUT=32` fp32 1.040 [1.040, 1.041]). The same model, fed each device's compile report, flips its verdict via the re-read static spill count — no re-fit is even required, since the decision is spill-dominated.

**Mechanism.** For identical non-spilling kernels the C500 toolchain allocates **1.6–1.75× more registers/thread** than `ptxas` (64×64 GEMM tile: fp16 84→134, fp32 96→168). Direct compile sweeps attribute this to (a) software-pipeline multi-buffering held in registers (≈34 registers per `num_stages`; the Triton default is 3) and (b) a ≈2× general allocator-efficiency gap; the C500 *does* use its scalar register file (26 MT + 16 ST on a probe), and the accumulator layout is ruled out (the effect appears in the pipeline-free reduction too). We explicitly retract an earlier "64-wide wavefront doubles register pressure" explanation: the register file per CU is doubled in lockstep, so wave width and capacity cancel.

**The flip is a default-launch-config artifact.** This is the paper's sharpest result. The register inflation only causes a spill at the *default* launch config (`num_warps=4`, inherited from NVIDIA-tuned Triton). Re-tuning to `num_warps=8` **eliminates the spill entirely** (numerically verified correct) and makes the C500 fusion beneficial for both families — though by very different margins. The **GEMM** gain is robust: a tuning-aware comparison (best fused config vs best *re-tuned* unfused) gives **1.206** (≈20%). The **reduction** gain is real but *thin*: the same-config `num_warps=8` fused/unfused speedup is 0.641→**1.080**, but once the unfused baseline is also re-tuned the honest tuning-aware margin is only **1.041** (≈4%). So the flip is a property of the *default launch configuration*, not the hardware. Crucially, the **static predictor tracks this**: fed the config-dependent spill count it predicts TOXIC at `num_warps=4` (spills) and BENEFICIAL at `num_warps=8` (0 spills) for both cases, matching measurement (4 of 5 configs; it misses only a `num_warps=16` case where the kernel is toxic again from *over-provisioning* at zero spills — a non-spill mode the inert occupancy term cannot see). The contribution is therefore a predictor of **config-dependent** fusion toxicity — the pre-autotune signal a search-free pass provides on a toolchain whose defaults are mis-tuned — not evidence of a hardware-fundamental fusion boundary.

### 5.4 RQ4 — Utility, and an honest compiler negative

As an offline recommender over three constructed subgraphs the model matches an autotuned oracle (1.00×) and beats greedy-always-fuse by 5.81–8.65× — but that margin is a *constructed worst case for greedy* (deliberately over-fused wide layers), and this is a recommender, not a scheduler integration. Against a real compiler, we hooked `triton.compile` under TorchInductor on the C500: the model predicts the *sign* of Inductor's own fusion benefit 5/5 on elementwise chains — a capability demo, since elementwise fusion is essentially always beneficial. Attempting to make Inductor emit a *toxic* fusion is an **honest negative**: a well-tuned production compiler avoids the spilling tiles that make fusion toxic (and, per §5.3, would re-tune away from them), so the model's discriminating power is demonstrable only on controlled kernels or weaker/domestic toolchains — precisely the setting this paper targets.

**Why the GEMM sign was initially wrong, and why precision cannot be fully fixed.** On the Contraction family the *static spill count has the wrong sign*: the fused GEMM's static spill count (205) is *lower* than the unfused (234), so a naive model predicts "beneficial" while the fusion is measured toxic. MCPTI shows why — the fused epilogue re-reads the spilled accumulator, moving *more* local traffic (950K vs 833K instructions) than the static count implies. The taxonomy-derived `reread=2` term recovers GEMM **recall 0→1.0** and generalizes leave-one-family-out. It does **not** recover *precision*: the flat `reread=2` also over-rejects 4 beneficial fp16 tiles (precision 0.5). We tried to fix this with a dtype-aware spill term; a dtype-switched reread robustly lifts the metric (GEMM F1 0.667→1.0) **but misattributes the mechanism** — MCPTI shows the fp16 fused kernel does +6.5% *more* local traffic than its unfused, so setting its reread to 1 zeroes a real effect using the dtype label. The true discriminator is a *runtime* fused−unfused traffic delta that is not a single-compile static input (the static spill counts even have the wrong sign). We therefore keep `reread=2` — a *safe* precision loss (recall stays 1.0; no toxic fusion is ever taken) — and report the residual as a genuine boundary of the search-free approach.

## 6. Discussion and Limitations

Read together, our findings scope the contribution narrowly and honestly:

- **The predictor is essentially a well-calibrated spill detector.** In every sweep only the spill term ever *flips* a decision; smooth occupancy and layout never do. "Interpretable multi-cause attribution" is, in this data, spill-traffic + roofline. The one place the model demonstrably beats the zero-parameter spill rule is a modest out-of-fold gain on the C500's spill-but-beneficial fp16 cases.
- **The decision-flip is config-dependent, not hardware-fundamental.** This both weakens the naive "cross-vendor toxicity" story and clarifies the model's real use: on accelerators with immature autotuning (the C500), default launch configs are mis-tuned and a search-free static pass usefully flags "this fusion is toxic *as configured* — re-tune, don't reject."
- **The hardest cases need runtime information.** The fp16 spill-but-beneficial kernels cannot be separated from their toxic fp32 twins by any single-compile static input; we prove this for both the reduction (feature-identical twins) and the GEMM (the discriminating signal is a runtime traffic delta).
- **Scope.** Two GPUs (sm80 unreachable), 3–4 op families, ~80 genuine cases; an offline recommender rather than a scheduler integration; `γ_spill` is identifiable only from the extreme-spill regime and does not extrapolate across spill magnitudes.

None of these is hidden in the deployed pass: recall is 1.0 throughout, so the predictor never *takes* a toxic fusion; its errors are conservative rejections.

## 7. Conclusion

We measured how far a search-free, single-compile, interpretable predictor gets at deciding when *not* to fuse GPU operators, across a consumer NVIDIA GPU and the domestic MetaX C500. The honest answer: fusion toxicity here is spill-dominated, so a zero-parameter spill rule is a strong baseline the model only modestly beats, and only where spill-but-beneficial cases exist. The cross-vendor decision-flip is real but is a default-launch-config artifact that the static predictor nonetheless tracks — making the useful artifact a *config-dependent* fusion-and-tuning signal for accelerators whose toolchains are not yet autotuned. Alongside, we contribute the first fusion characterization of the MetaX C500 and a reusable MCPTI-direct profiling recipe. We release the code, kernels, and datasets for reproduction.

## References

*(Bibliographic details below were prepared for this draft and should be verified against the canonical sources before submission — see the companion author notes.)*

[1] T. Chen et al. "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning." OSDI, 2018.
[2] L. Zheng et al. "Ansor: Generating High-Performance Tensor Programs for Deep Learning." OSDI, 2020.
[3] J. Ansel et al. "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation." ASPLOS, 2024.
[4] P. Tillet, H. T. Kung, D. Cox. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." MAPL, 2019.
[5] Y. Shi et al. "Welder: Scheduling Deep Learning Memory Access via Tile-graph." OSDI, 2023.
[6] Z. Zheng et al. "AStitch: Enabling a New Multi-Dimensional Optimization Space for Memory-Intensive ML Training and Inference on Modern SIMT Architectures." ASPLOS, 2022.
[7] Y. Ding et al. "Hidet: Task-Mapping Programming Paradigm for Deep Learning Tensor Programs." ASPLOS, 2023.
[8] S. Williams, A. Waterman, D. Patterson. "Roofline: An Insightful Visual Performance Model for Multicore Architectures." Communications of the ACM, 2009.
[9] T. Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness." NeurIPS, 2022.
[10] P. M. Phothilimthana et al. "TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs." NeurIPS Datasets and Benchmarks, 2023.
[11] T. Chen et al. "Learning to Optimize Tensor Programs." NeurIPS, 2018.
[12] NVIDIA. "Nsight Compute" and "CUPTI (CUDA Profiling Tools Interface)." NVIDIA Developer Documentation.
