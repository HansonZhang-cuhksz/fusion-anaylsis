# Detecting Toxic Operator Fusion — Consolidated Results

*Reviewer-ready synthesis of LOG-01…10 (2026-07). Reproduce pointers in each subsection; frozen model
in `model/MODEL_SPEC.md`; open items in `REVIEW_FINDINGS_TODO.md`.*

## Setup
- **Hardware.** NVIDIA **Ada** (RTX 4060 Laptop, sm89; 32-thread warps, 64K reg/SM, soft spill) and the
  domestic **MetaX C500** (MACA 3.7.0; **64-thread wavefronts**, 128K reg/CU, split ST/MT register file,
  **hard 4 KB/thread spill cap**). **No Ampere sm80 is reachable** — the Ada box hosts only the RTX 4060,
  so the third-hardware-point pass is blocked, not merely deferred.
- **Model (frozen; `MODEL_SPEC.md`).** A search-free interpretable degradation model
  `η_fused = min(η_u,η_v)·P_occ·P_layout`; every input comes from **one compile** (Triton `n_regs`,
  `n_spills`, shared mem, or `ptxas -v`). Deploy-time decision: prune the fusion iff `T_fused > T_unfused`
  on a per-device roofline. The spill term uses spill **traffic** = `n_spills × reread`, where `reread`
  is a **taxonomy-derived** multiplier (CONTRACTION→POINTWISE epilogue re-reads the spilled accumulator
  ⇒ 2, else 1). Profiling is used only to fit per-device constants / validate — never at decision time.
- **Op families.** P = pointwise chains, R = sibling reductions (the spill-cliff knob), T = transpose /
  bank-conflict (Ada), **G = GEMM-epilogue / CONTRACTION** (tensor-core / compute-bound) — now run on
  **both** devices (`data/microbench_gemm_ada.csv`, `data/microbench_gemm_c500.csv`).

## ⚠ Measurement artifact in the original Ada timings (found & corrected, LOG-10)
- **What happened.** A measurement artifact inflated the fused time of **spilling kernels only**, by
  **9–41×**, in the original `data/microbench_timing.csv`. Mechanism (reproduced, not merely inferred):
  when torch's caching allocator grows until its **reserved** footprint *oversubscribes* physical VRAM
  (7.82 GB reserved on an 8 GB card with ~6.9 GB available), WSL2/WDDM permits the oversubscription
  rather than failing the allocation, and the CUDA context's local-memory backing store can be left
  host-resident. Every spill access then crosses PCIe. Non-spilling kernels are untouched in ratio
  (1.040 vs 1.043 healthy). Reproduced on demand: at 8.08 GB reserved / 0.00 GB free, redux N128 fp32
  2048×2048 measures t_f = 754 ms (0.0097×) against a healthy 78 ms (0.122×).
- **Scope and limits of the explanation (honest).** The trigger is **oversubscription (reserved >
  physical)**, *not* "free VRAM ≈ 0": replaying the entire pre-fix code path (48 pointwise cases, no
  `empty_cache`, free 0.000 GB but reserved 6.73 GB) reproduces the **corrected** values exactly
  (redux N64 fp32 2048×2048 = 0.8939, t_f = 5.362 ms). Once a context's local-memory store is
  host-resident it does **not** migrate back — `empty_cache()` does not repair an already-poisoned
  process (N64 still 9.70 ms / 0.129× after `empty_cache` restored 6.36 GB free). The `empty_cache()`
  calls added to `runner.py` / `gemm_sweep.py` / `endtoend.py` / `recommender.py` are therefore
  **prophylactic** — they keep the cache from ever reaching oversubscription — and their efficacy is a
  **design argument, not a measured result**: in a direct A/B on the same process the fix changed nothing
  (0.8939 unfixed vs 0.8937 fixed), because that replay never entered the oversubscribed regime.
  The matrix wall-clock drop (253.7 s → 52.1 s) is a *symptom* of the artifact being absent on the re-run,
  **not** evidence that `empty_cache()` caused the correction; an unfixed replay is equally fast today.
  Exact severity is history-dependent and not fully characterized: the original run inflated NOUT=64 rows
  (t_f 13.55 ms) but a fresh oversubscribed repro leaves NOUT=64 healthy (0.913×) and damages only the
  1868-spill NOUT=128 rows.
- **Confidence in the corrected dataset is HIGH and does not depend on the causal story:** all 24
  reduction rows of the CSV were independently reproduced end-to-end (n64/2048/fp32 0.8939 vs 0.8939;
  n128/2048/fp32 0.1224 vs 0.1191; n32 rows 1.00–1.08).
- **Corrected spill cliff** (`data/microbench_timing.csv`, mean speedup by NOUT): NOUT=32 (0 spills)
  fp16 **1.028** / fp32 **1.046** beneficial; NOUT=64 (300/310 spills) **0.940** / **0.897**;
  NOUT=128 (1986/1868 spills) **0.158** / **0.145** toxic. Pre-fix means were **0.077** (NOUT=64) and
  **0.008** (NOUT=128) — up to 179× slower; *that* inflation was the artifact, and the cliff is real but
  an order of magnitude shallower than first reported. At NOUT=64 only the **fp32** point is CI-backed
  (median 0.893, CI [0.893, 0.898], 20 rounds); the fp16 bucket has no CI and contains a member at
  **0.982**, so "toxic" at NOUT=64/fp16 is **not yet established** — its label comes from a single-shot
  `t_fused < t_unfused` threshold, which RQ3 below shows can flip under CIs.
- **The C500 data is unaffected.** It predates the fix and was collected under the same allocator path,
  but three independent lines of evidence exclude the artifact there. (1) *Matrix-vs-isolated agreement*:
  all 5 configurations measured both in the matrix and as isolated 20-round CIs agree to within **1.5%**
  — redux N32 fp16 1.0650 vs 1.0815 (−1.5%); N32 fp32 0.6341 vs 0.6415 (−1.2%); N64 fp32 0.2699 vs
  0.2710 (−0.4%); gemm128 fp16 1.0640 vs 1.0640 (0.0%); gemm128 fp32 0.8320 vs 0.8240 (+1.0%). The Ada
  artifact cost ~90% of speedup, so it is excluded by three orders of magnitude. (Agreement is
  *practical*, not statistical: the within-process CIs are extremely tight and all 5 matrix values fall
  just outside them, as expected cross-process.) (2) *The residual lacks the starvation signature* —
  it is bidirectional (−1.5% to +1.0%) and not monotone in spills (r(|dev|, spills) = −0.38; the
  524-spill case deviates least). (3) *The precondition is not met* — C500 has ~64 GB VRAM vs Ada's 8 GB
  (LOG-04) for an identical working set (~8× headroom), and MetaX raises `mcErrorMemoryValueTooLarge` on
  private-memory overflow rather than silently backing local memory over PCIe.
  *Limitations:* no free-VRAM measurement was recorded for the C500 run (the device is not reachable from
  the Ada box), the C500 matrix has not been regenerated post-fix, and the most spill-heavy configs
  (N128, 844/848 spills) have no isolated counterpart — so the ≤1.5% bound rests on 5 measured points and
  is extended to the rest by mechanism, not measurement.

## RQ1 — Predictability (can static inputs predict fusion toxicity?)  ✅
- **Ada** (on the *corrected* `microbench_timing.csv`; 64 genuine cases, 8 degenerate no-ops excluded):
  in-sample **precision = recall = F1 = accuracy = 1.000** (TP 16/FP 0/FN 0/TN 48). The old "single FN is
  a non-spill NOUT=32 toxicity (0.91)" blind spot **was the measurement artifact and is gone.**
- **This does *not* validate the cost model's structure — the decisive baseline is a one-liner.** After
  the degenerate filter the Ada benchmark is *perfectly separable by a single raw feature*: all 48
  non-spilling cases are beneficial, all 16 spilling cases are toxic, zero exceptions. A **zero-parameter
  rule — "don't fuse iff the fused kernel spills" — also scores F1 = 1.000** (TP 16/FP 0/FN 0/TN 48),
  matching the 5-parameter model exactly. The only baseline we report (greedy-always-fuse, **F1 0.000**)
  never predicts toxic and cannot distinguish the two. Ada RQ1 establishes only that **spilling is a
  sufficient toxicity indicator on this benchmark**; it is no evidence for the roofline model's added
  parameters. The fp16↔fp32 transfer (**leave-one-dtype-out F1 0.968**, recall 0.938) is the only
  in-family generalization signal here.
- **The degenerate filter is load-bearing and we flag it.** The 8 excluded rows (`n_launches_unfused ≤ 1`
  — pointwise K=1 / reduction NOUT≤GS, i.e. *no fusion decision to make*) carry **7 of the 23 raw toxic
  labels**, with speedups 0.969/0.997/0.995/1.005/0.999/0.988/0.996/**0.912** — all but the last in the
  noise band. Keeping them would cap recall at 16/23 = 0.696.
- **Leave-one-NOUT-out is F1 0.667 (recall 0.500) — a defect of the *fit*, not of the problem.** Holding
  out NOUT=128 collapses `gamma_spill` to ~1e-8 (the fit abandons the spill term entirely, despite the
  genuinely toxic 300-spill NOUT=64 rows remaining in training) and the model then misses all 8 toxic
  1986-spill cases. The reverse fold (hold out NOUT=64) learns `gamma_spill = 0.0067` and scores F1 1.000
  — so `gamma_spill` is **identifiable only from the extreme-spill regime**. The zero-parameter spill rule
  scores **F1 1.000 / recall 1.000** under the same protocol, i.e. it **strictly dominates the fitted
  model out-of-sample**. We therefore do *not* claim Ada RQ1 as evidence that the fit generalizes across
  spill regimes.
- **The bootstrap CI is degenerate; reported for completeness only.** Predictions are frozen at the
  in-sample fit and match labels elementwise, so **every** resample returns F1 = 1.0 (100% of B = 2000).
  "F1 1.000, 95% CI [1.000, 1.000]" is a **mathematical identity, not an estimate** — it quantifies no
  sampling error and has zero power to detect overfitting. Only a bootstrap that *refits* the constants
  inside each resample would be informative.
- **C500** combined (reduction + pointwise + GEMM, 80 cases): **F1 0.852, recall 0.958**.
- **Ablations** isolate the driver: drop the spill term → **F1 0.000**; drop smooth occupancy → F1 1.000
  (unchanged). The decisive signal is the **register-spill discontinuity**, not the smooth occupancy
  curve — but see the separability caveat above: on this benchmark that discontinuity is the *whole*
  signal, so the ablation confirms the driver rather than the model.
- *Reproduce:* `python -m model.fit data/microbench_timing.csv` (Ada);
  `python -m model.bootstrap_ci` (bootstrap); `python -m model.build_combined_c500 && …` (C500).
  LOG-02/03/06, LOG-10.

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
- **TWO decision-flips, both bilaterally CI-backed** (the headline). Both sides now carry 20-round CIs
  (Ada: `data/timing_ci_ada.csv`, LOG-10; C500: 4 independent GPUs, LOG-09):
  1. **`sibling_redux NOUT=32 fp32`** (memory-bound): **Ada beneficial** — 0 spills, median **1.040×**,
     CI [1.040, 1.041]; **C500 toxic** — 100 spills, median **0.642×**, CI [0.638, 0.645]. Consistent
     across all 4 shapes.
  2. **`gemm_128 fp32` (NEW, LOG-10)** (compute-bound, tensor-core): on the *same* fp32 128×128 tile,
     **Ada** fits the kernel in **255 regs/thread with 0 spills** and fusion is **beneficial** (sweep
     1.028–1.269; 2048×2048×512 median **1.083**, CI [1.072, 1.104]); on the **C500** the same tile takes
     **256 regs/thread, spills 205**, and fusion is **toxic** (sweep 0.777–0.832; median **0.824** across
     four GPUs, per-GPU CIs spanning [0.821, 0.827]). Ada spills nowhere in this family (`f_spills = 0` on
     all 16 configs).
  The *same* model, fed each device's compile report, flips its verdict — now demonstrated in **two
  independent families** (memory-bound reduction **and** compute-bound tensor-core GEMM).
- **Mechanism (partial — we state what we measured, not why).** The C500's Triton backend allocates
  **1.6–1.75× more registers per thread than ptxas for identical kernels** at tiles where *neither* device
  spills (64×64 fp16: 84 → 134; fp32: 96 → 168); at the 128×128 fp32 tile it saturates its 256-reg/thread
  architectural cap and spills, while Ada's allocation lands at 255, just inside its own cap. **We do NOT
  attribute this to the C500's 64-wide wavefronts** (an earlier draft did): its register file per CU is
  doubled *in lockstep* (131072 vs 65536, `fusion/hw.py`), so per-wave demand and per-CU capacity cancel
  exactly — back-solving both CSVs' `f_occ`, both devices hold **8** resident waves/warps on the reduction
  and **4** on the 128×128 GEMM. Wave width also predicts the **wrong direction**, since the C500 spreads
  the same tile over **2× the threads** (256 vs 128/block), which should *halve* per-thread demand.
  **Cause now identified (LOG-12).** Direct C500 compile sweeps show the excess is (a) **software-pipeline
  multi-buffering held in registers** — each `num_stages` adds ~34 regs (Triton default 3 → ~68 extra) —
  plus (b) a **~2× general allocator-efficiency gap** vs ptxas, visible even in the pipeline-free reduction
  (C500 needs 2× the registers for *half* the per-thread accumulator work). Ruled out: scalar-register
  underuse (MACA *does* use the split file — `26 MT + 16 ST` on a probe kernel) and accumulator layout
  (the effect is not MMA-specific).
- **⚠ The C500 toxicity is a DEFAULT-LAUNCH-CONFIG artifact, not a hardware limit (LOG-12).** The spill
  that drives *both* flip families vanishes under a C500-aware launch config: `num_warps=8` (or
  `num_stages=1`) takes the GEMM 128×128 fp32 from **205 → 0** spills and the reduction NOUT=32 fp32 from
  **100 → 0** (at 8 warps the register/thread even matches Ada). The Triton default `num_warps=4` is
  NVIDIA-tuned (32-wide warps); on the C500's 64-wide warps it under-provisions threads and overflows the
  256-reg cap. This does not invalidate the model — it correctly flags the *as-shipped default* fusion as
  toxic, the pre-autotune regime a search-free pass serves — but the flip must be framed as **"the
  default-config fusion is toxic on C500, beneficial on Ada,"** not "fundamentally toxic on C500."
  (Untested: whether the re-tuned 8-warp fusion is net-*beneficial* — spill removed ≠ fusion wins.)
- **Spilling is necessary but not sufficient for toxicity** — a counter-example lives inside our own data:
  the C500's fp16 128×128 tile spills **117** and is still **beneficial** (1.011–1.064). Likewise the
  reduction flip's dtype dependence is unexplained by register state: at R=C=2048, NOUT=32 the C500's fp16
  and fp32 rows have *identical* `f_regs = 256`, `f_spills = 100`, `f_occ = 0.25`, yet fp16 is 1.065
  (beneficial) and fp32 is 0.634 (toxic).
- **No Ada GEMM configuration is significantly toxic.** Two of 16 sweep rows carry single-shot toxic
  labels (0.973, 0.957); on repeat measurement the fp16 128×128 row is **beneficial** (1.0250,
  CI [1.025, 1.029]) and the fp32 64×64 row is **inconclusive** (0.9904, CI [0.964, 1.055] — spans 1.0).
  We do not claim it is benign, only that the single-shot label is unsupported.
  *CI caveat:* `gemm_sweep`'s one-shot labels are noise-prone within ~5% of 1.0, and our CIs are
  within-process percentiles over min-of-N timings — they bound **short-term noise only**. Two independent
  20-round runs of the identical 2048×2048×512 fp32 BM128 config produced **non-overlapping** intervals
  ([1.111, 1.118] vs [1.072, 1.104]), so run-to-run variation on this thermally-throttling laptop exceeds
  the quoted CI width. **Treat any verdict within ~5% of 1.0 as unresolved.** (The C500's [0.821, 0.827]
  is likewise a *cross-GPU envelope* of four per-GPU CIs, not a single interval.)
- **Honest nuance.** The binary decision is spill-dominated, so applying the *Ada* constants to C500 data
  gives the *same* F1 — the transfer is carried by the **re-read hardware-specific static inputs**, not
  by re-fitting constants. Cross-vendor generalization table in `MODEL_SPEC §7`. LOG-04.

## RQ4 — Utility (vs greedy, oracle, and a real compiler)  ✅
- **Synthetic subgraphs** (post-fix re-run, `logs/run_endtoend.log`). The offline recommender is
  **8.65×** faster than greedy on `wide_multiproj`, **5.81×** on `mixed_widths`, **7.42×** on `fp32_block`,
  and matches the timed **oracle exactly** on all three (**1.000×** of oracle). Still a **constructed
  worst-case for greedy** — honest. TOTALS ms: wide none 25.248 / greedy 192.140 / model = oracle 22.215;
  mixed 17.481 / 85.484 / 14.703; fp32 5.977 / 38.946 / 5.252. LOG-03, LOG-10.
- **RQ4 was *not* affected by the VRAM artifact — a robustness result, not a correction.** The end-to-end
  harness allocates few enough buffers that it never entered the starved regime (pre-fix greedy(w128) =
  151.4 ms sits with the *unstarved* matrix value ~173 ms, not the starved ~3142 ms). Post-fix times rose
  by a uniform 1.22–1.37× across spilling and non-spilling kernels alike, with **no ~10× deflation of the
  spilling greedy kernel**; the shift from the pre-fix 9.72/6.17/7.43 is **run-to-run drift that largely
  cancels in the ratio**. The conclusion holds in **both** regimes, and is additionally robust to the 14×
  refit of `gamma_spill` (0.0807 → 0.00567), which left every fusion decision identical.
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

**On Ada the fix is vacuous by construction, not validated (LOG-10).** The reread multiplier attaches only
to CONTRACTION→POINTWISE pairs — i.e. the 16 GEMM rows — and every one of them compiles to `f_spills = 0`
on Ada's register file (255 regs, no spill; the same fp32 128×128 tile spills 205 on the C500). Since the
spill term is `spill_factor(spills × reread)`, the reread column **multiplies zero**: the no-reread and
reread-aware training sets yield a **bit-identical objective** (loss delta 0.0 at every parameter setting),
hence identical fits (`gamma_spill = 0.04881` both ways; overall F1 0.875 / recall 0.778 both ways). This
is an **arithmetic identity, not a measurement** — it is **no evidence** that the fix is correct, and
equally none that it is harmless, because on Ada the feature has no path to affect any decision. **We
therefore do not claim Ada as a control** (a control must be able to fail). The fix's evidence is the
**C500 result alone** (held-out GEMM recall 1.0 with reread=2 vs 0.0 with reread=1, LOG-06) — validated
only where the failure mode exists. Ada's real contribution here is the cross-vendor GEMM decision-flip
itself; the fix's inertness is a *corollary* of that flip, not a separate finding.
*Caveat on the per-family number:* the "GEMM F1/recall = 0.000 (n_toxic = 2)" printed by
`model/refit_combined_ada.py` is **not a meaningful score** — both "toxic" labels are the timing noise that
20-round CIs overturn (see RQ3), so the two FNs are the model disagreeing with a *wrong label*. Ada has
zero genuinely toxic GEMM fusions, so **GEMM recall on Ada is undefined and should read n/a**, not 0.000.

## Contributions
1. A **search-free, interpretable** fusion-rejection criterion with single-compile inputs and a
   **taxonomy-derived spill-traffic** term.
2. The **first fusion characterization of the MetaX C500** and a **cross-vendor transfer with TWO
   documented, CI-backed decision-flips** — one memory-bound (reduction), one compute-bound
   (tensor-core GEMM) — hardware-validated via MCPTI. The flips are **empirically solid and
   bilaterally significant**; their *mechanism* is only partially resolved (register-allocation
   divergence between the two toolchains, cause unidentified — see RQ3).
3. A **real-compiler (Inductor) validation** on the C500 showing fusion benefit is roofline-governed and
   that even a production compiler over-fuses net-harmfully compute-bound.

## Honest limitations (for the writeup)
- **Interpretability is mostly one cause.** Smooth occupancy (P_occ) and layout (P_layout) never *flip*
  a decision in these sweeps; toxicity is driven by the **spill cliff** (+ now the roofline compute term
  for recomputation). The model is largely a spill-traffic + roofline detector; the "multi-cause"
  framing is honestly narrow.
- **Ada RQ1's perfect score is the dataset's, not the model's.** The 64 genuine Ada cases are perfectly
  separable by `f_spills > 0`, so a **zero-parameter rule ties the 5-parameter model at F1 1.000**
  in-sample and **beats it out-of-sample** (leave-one-NOUT-out 1.000 vs 0.667). Ada RQ1 supports "spilling
  ⇒ toxic on this benchmark", not the roofline model's structure. The accompanying bootstrap CI
  [1.000, 1.000] is a degenerate identity with no power.
- **On the C500 the model beats the spill heuristic — modestly, and not on the hard cases (LOG-11).**
  Where discriminating (**spill-but-beneficial**) cases exist (C500), the fitted model beats the trivial
  `f_spills>0 ⇒ toxic` rule (0.857) **both in-sample (F1 0.873) and out-of-fold (leave-one-dtype-out
  0.906)**, seed-stable — so it is not *merely* a spill detector. This uses a **now-integrated,
  principled dtype-aware compute term** (`fp16_compute_mult`: fp16's ~2× FMA throughput scales the
  compute roofline; a FIXED per-device constant, enabled on C500, off by default). Integrating it lifted
  C500 in-sample 0.852→0.873 and out-of-fold 0.873→0.906, and **strengthened the decision-flip** (NOUT=32
  fp32 predicted-toxic 3/4→4/4) with no family regressing. **But** both models still get **7 of the 8**
  discriminating cases **wrong**: those fp16 cases have **static inputs identical to their toxic fp32
  twins** (f_regs=256, f_spills=100, f_occ=0.25; only dtype differs), so the fp16-vs-fp32 distinction is
  **not cracked** — the model beats trivial by higher precision (rescues 1 case) + recall, not by solving
  the hard cases, and the margin is modest on 80 cases. The term is **regime-specific**: on Ada (0
  discriminating cases, spill-separable) it gives no benefit and mildly regresses out-of-fold, so it is
  correctly left off there. *Independently verified; two of my own analysis errors (a scripting bug and
  under-converged fits) were caught and corrected in the process.* ⇒ Honest claim: the model adds
  **modest, converged value over the spill heuristic and strengthens the flip**, not a decisive win on
  the hardest cases; its clearest contribution remains the **cross-vendor transfer of the spill signal**.
- **The spill penalty does not extrapolate across spill magnitudes.** Leave-one-NOUT-out **F1 0.667
  (recall 0.500)**: `gamma_spill` is identifiable *only* from the extreme-spill (1986) regime — hold that
  out and the fit collapses it to ~1e-8, abandoning the spill term despite toxic 300-spill rows in
  training. A calibrated model at one spill magnitude does **not** transfer to another.
- **Hardware breadth.** One consumer NVIDIA GPU + one domestic GPU; no datacenter NVIDIA. **sm80 is not
  reachable at all** (the Ada box hosts only the RTX 4060) — the third hardware point is **blocked**, not
  pending. The **GEMM family now runs on both devices** (LOG-10), so the cross-vendor GEMM comparison is
  complete; and the flip's **mechanism is now explained (LOG-12)** — the C500 compiler demands 1.6–1.75×
  more regs/thread because of pipeline multi-buffering held in registers (~34/`num_stages`) plus a ~2×
  allocator-efficiency gap vs ptxas (we retract the earlier 64-wide-wavefront explanation — the register
  file is doubled in lockstep). **The residual honesty cost is that this makes the C500 toxicity
  default-config-dependent** (`num_warps=8`/`num_stages=1` removes the spill for both families), so the
  flip is a property of the as-shipped default launch config, not a hardware necessity — a caveat we now
  state explicitly rather than a mechanism we cannot explain.
- **The reread fix is validated on the C500 only, and Ada cannot help.** On Ada the feature multiplies
  zero spills, so its inertness is an arithmetic identity — **not** a passing control. Ada carries no
  evidence for or against the fix.
- **Statistics.** Both sides now carry 95% CIs. **C500** (LOG-09, 4 independent GPUs): the reduction flip
  (CI [0.638, 0.645]) and GEMM toxicity (envelope [0.821, 0.827]) are toxic with overwhelming
  significance, cross-device-reproducible to ~3 decimals. **Ada** (LOG-10, `data/timing_ci_ada.csv`, 20
  rounds): redux_N32_fp16 1.011 [1.010, 1.013]; redux_N32_fp32 1.040 [1.040, 1.041]; redux_N64_fp32 0.893
  [0.893, 0.898]; gemm_128_fp16 1.164 [1.124, 1.190]; gemm_128_fp32 1.083 [1.072, 1.104] — all excluding
  1.0. **But these CIs bound within-process noise only:** two 20-round runs of the *same* Ada config gave
  non-overlapping intervals ([1.111, 1.118] vs [1.072, 1.104]), so run-to-run/thermal variation exceeds
  them and verdicts within ~5% of 1.0 are unresolved. Coverage is also partial — redux_N64_**fp16** and the
  spill-heavy C500 N128 configs have no CI at all. Datasets are small (64–80 cases).
- **Provenance.** The Ada Bucket-1 artifacts (`data/microbench_gemm_ada.csv`, `data/timing_ci_ada.csv`,
  the regenerated `data/microbench_timing.csv`, `logs/LOG-10-ada-bucket1.md`, the `empty_cache` fixes) are
  **working-tree state, not yet committed**, and the 0.9904 / 1.0250 dismissal CIs are prose in LOG-10 with
  no committed script that reproduces them (`fusion/timing_ci.py`'s CLAIMS list omits those configs).
  These must be committed — and a runnable probe for the artifact's two regimes added — before writeup.
- **Real-compiler scope.** The Inductor prediction is a capability demo; a *toxic* Inductor fusion could
  not be triggered at tractable sizes (LOG-08), so the discriminating evidence is on controlled kernels
  (the CONTRACTION spill tile) — the model targets weaker / domestic fusers, not a tuned Inductor.

## Venue (honest)
An empirical + systems contribution — **workshop / MLSys-short scale**. The domestic-GPU cross-vendor
angle (decision-flip + first C500 characterization + MCPTI-direct recipe) is the differentiator a
Hopper-rich lab would not produce.
