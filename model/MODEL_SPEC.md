# FROZEN MODEL SPEC — search-free interpretable fusion-decision model (Ada sm89)

This is the frozen spec that travels to the MetaX C500 session (Phase 4). It contains the exact
formulas, the fitted Ada constants, and the vendor-neutral dataset schema. **Transfer = swap the
`DeviceConstants` and the `HardwareModel`; do not change the formulas.**

> **Provenance (read before citing).** The constants in §4, the corrected `data/microbench_timing.csv`,
> and the Ada GEMM/CI results in §6 currently live in the **working tree, not in a commit** (untracked:
> `data/microbench_gemm_ada.csv`, `data/timing_ci_ada.csv`, `model/bootstrap_ci.py`,
> `logs/LOG-10-ada-bucket1.md`; modified-unstaged: `data/microbench_timing.csv`,
> `model/ada_constants.json`, the four `empty_cache` fixes). Only the C500 halves are committed. These
> must be committed — together with a runnable probe reproducing both the oversubscribed and healthy
> regimes of §4a — before this spec is cited as reproducible. The 0.9904/1.0250 GEMM CIs in §6 appear
> only in LOG-10 prose; `fusion/timing_ci.py`'s CLAIMS list does not contain those configs, so they are
> not reproducible from the repo as it stands.

## 1. Static inputs (single compile — the only things the deployed pass may read)
Per candidate kernel, from ONE codegen/compile (Triton `k.n_regs / k.n_spills / metadata.shared /
num_warps`, or `nvcc -Xptxas -v`):
- `regs`  — registers/thread
- `spills` — spill count (0 ⇒ no local-memory spill)   ← the P_occ discontinuity trigger
- `smem`  — shared bytes/block
- `threads` — threads/block (= num_warps·32)
Plus graph-level analytic quantities the compiler already has: `flops`, `bytes_fused`,
`bytes_unfused`, `n_launches_unfused`, and (if a transpose/permute edge) a bank-conflict/layout
descriptor.

## 2. Analytical occupancy (hw.py) — reproduces the ncu THEORETICAL occupancy calculator (MAE 0.000, 22/22)
<!-- This validates the calculator re-implementation, NOT achieved occupancy (which differs 21.7 pts
     mean / 88 max and is deliberately not predicted — the spill term captures the real degradation). -->
`occupancy(regs, smem, threads)` on the sm89 `HardwareModel`:
- warps/block = ceil(threads/32); registers/warp = ceil(regs·32, 256); warps rounded to gran 4.
- blocks/SM = min over limiters {regs: regs_per_sm//regs_per_block, smem: smem_per_sm//ceil(smem,128),
  warps: 48//warps_per_block, threads: 1536//threads, blocks: 24}.
- occupancy = blocks/SM · warps/block / 48. Reports the **binding limiter** (interpretability).
- sm89 constants: SMs 24, 48 warps/SM, 1536 threads/SM, 65536 regs/SM, regs>255 ⇒ hard fail,
  reg_alloc_unit 256, warp_gran 4, smem/SM 102400, smem_unit 128.
- **Portability knobs** (parameters, not hardcoded): `split_regfile` (C500 ST+MT files),
  `spill_cap_bytes` (C500 hard 4096 B/thread ⇒ launch failure — a hard P_occ cliff, not a soft cost).

## 3. Degradation model (costmodel.py)
Factored, interpretable efficiency (faithful to PROPOSAL §5.2, extended so the memory/latency-bound
regime is representable):

    eff(plan) = lam(occ) · spill_factor(spills) · layout_factor(bank_conf_per_elem)
              = [ P_occ_occ ]   · [ P_occ_spill ]  · [ P_layout ]
    P_occ = P_occ_occ · P_occ_spill

    lam(occ)          = clip(occ / occ_knee, occ_floor, 1)
    spill_factor(s)   = 1 / (1 + gamma_spill · s)          # spills override register-occupancy
    layout_factor(bc) = 1 / (1 + beta_layout · bc)

    T_plan = max( flops/(C_peak·eff),  bytes/(B_peak·eff) ) + n_launch·T_launch

**Decision:** prune (don't fuse) iff `T_fused > T_unfused`.
**Attribution:** harm_x = −log(P_x); dominant = argmax(harm_occ, harm_layout); the occupancy branch is
split into `spill` vs `occupancy` by which factor is smaller. (spills ⇒ "spill"; bank conflicts with
no spill ⇒ "layout"; neither ⇒ "none".)

## 4. Fitted Ada constants (model/ada_constants.json)
Refit on the corrected `data/microbench_timing.csv` (see §4a — the pre-fix dataset inflated spilling
kernels, so the pre-fix constants in brackets are superseded).

| const | value | note |
|---|---|---|
| C_peak | 4.26e11 flop/s | under-constrained (workloads memory-bound); not decision-critical [was 4.18e11] |
| **B_peak** | **1.81e11 B/s** | ~181 GB/s — an *effective* bandwidth ~20% **above** the RTX 4060 laptop's ~151 GB/s spec (128-bit GDDR6); it absorbs L2 hits and is not a physical peak [was 1.51e11] |
| T_launch | 1.34e-4 s | per-launch overhead (WSL2 Python/Triton) — a fitted catch-all for per-launch cost, not a measured latency; ~3× the pre-fix 4.43e-5 [was 4.43e-5] |
| occ_knee | 0.08 | hit floor ⇒ occupancy above ~8% does not drive toxicity on Ada; **spills do** |
| gamma_spill | **0.00567** | spill sensitivity (the dominant toxic term). 14× smaller than the pre-fix 0.0807 — the old value was fit to artifact-inflated spilling kernels. Every RQ4 fusion decision is **identical** under both values (§6). [was 0.0807] |
| beta_layout | 0.327 | bank-conflict sensitivity (fit from the cuda_layout PAD0/PAD1 slowdown; raw-CUDA data, unaffected by the artifact) |

Fit objective: combined log(speedup) (weight 1.0) + log(absolute time) (0.3), occ_knee bounded
[0.08, 0.35] (Ada saturates HBM by ~⅓ occupancy). Nelder–Mead, 6 restarts.

### 4a. Harness requirement (LOG-10) — a correctness condition of the measurement protocol
Any benchmark process that builds many cases **MUST call `torch.cuda.empty_cache()` before building
each case** (`fusion/runner.py::run_case`, `fusion/gemm_sweep.py::run_one`,
`fusion/endtoend.py::time_layer_width`, `model/recommender.py::score_grouping`).

Without it, torch's caching allocator hoards VRAM until its **reserved** footprint *oversubscribes*
physical VRAM (observed: 7.82 GB reserved on an 8 GB card with ~6.9 GB available). WSL2/WDDM permits
the oversubscription rather than failing the allocation, and the CUDA context's local-memory backing
store can be left host-resident — so every spill access crosses PCIe. This inflates the fused time of
**spilling kernels only** by 9–41×, leaves non-spilling kernels untouched, and therefore silently
fabricates a **too-severe spill cliff** (the pre-fix NOUT=128 mean speedup read 0.008, i.e. up to 179×
slower, vs the corrected 0.15).

Stated honestly, and load-bearing for any future small-VRAM device:
- The trigger is **oversubscription (reserved > physical), NOT "free VRAM ≈ 0"**. Free VRAM at 0.00 GB
  with reserved *below* physical is harmless: replaying the whole pre-fix path (48 pointwise cases, no
  `empty_cache`, free 0.000 GB, reserved 6.73 GB) reproduces the *corrected* values exactly.
- Severity is spill-magnitude- and history-dependent, not a uniform "10×": a fresh oversubscribed repro
  damages only the 1868-spill NOUT=128 rows and leaves the 300-spill NOUT=64 rows healthy (sp=0.913),
  whereas the original run inflated NOUT=64 too.
- Once a context's local-memory store is host-resident it does **not** migrate back: `empty_cache()`
  does not repair an already-poisoned process. The calls are therefore **prophylactic** — they keep the
  cache from ever reaching oversubscription — and their efficacy is a **design argument, not a measured
  result**. In a direct A/B on one process the fix changed nothing (0.8939 unfixed vs 0.8937 fixed),
  because that replay never entered the oversubscribed regime.
- The matrix wall-clock drop (253.7 s → 52.1 s) is a *symptom* of the artifact being absent on the
  re-run, **not** evidence that `empty_cache()` caused the correction.
- Confidence in the corrected dataset is **high and does not depend on the causal story**: all 24
  reduction rows were independently reproduced end-to-end (n64/2048/fp32 0.8939 vs 0.8939;
  n128/2048/fp32 0.1224 vs 0.1191; n32 rows 1.00–1.08).

## 5. Vendor-neutral dataset schema (concept columns, NOT raw metric names)
`data/microbench_timing.csv` (label + static features), `data/microbench_ncu.csv` (ground truth):
- keys: family, op_pair, producer_class, consumer_class, dtype, R, C, param_*
- static: f_regs, f_spills, f_smem, f_threads, f_occ(analytic), f_occ_binder, u_regs, u_occ,
  n_launches_unfused, bytes_fused, bytes_unfused, flops, arith_intensity
- measured (fit/validate only): t_fused_ms, t_unfused_ms, speedup, beneficial(label)
- ncu ground truth (concepts): occ_achieved/theoretical, spill(local)_bytes, bank_conf,
  dram_bytes, tensor_pct, dur_ns (ncu kernel duration, ns)  → dominant_penalty ∈ {spill, layout, none}

## 6. Headline Ada results (to reproduce on C500 and compare)
- RQ1 decision (64 genuine cases; 8 degenerate no-op rows — `n_launches_unfused<=1`, pointwise K=1,
  no fusion decision to make — excluded from scoring): **in-sample precision=recall=F1=acc=1.000**
  (TP=16, FP=0, FN=0, TN=48). Greedy-always-fuse F1=0.000. Ablations: drop the spill term ⇒ F1=0.000;
  drop smooth occupancy (occ=1) ⇒ F1=1.000 (the occupancy term is inert on Ada).
  - **This does NOT validate the model's structure.** After the degenerate filter the Ada benchmark is
    perfectly separable by one raw feature: all 48 non-spilling cases are beneficial, all 16 spilling
    cases are toxic, zero exceptions. A **zero-parameter rule — "don't fuse iff the fused kernel
    spills" — also scores F1=1.000 in-sample**, matching the 5-parameter model exactly. F1=1.000 is the
    score of the *dataset*, not of the model. The only baseline reported (greedy, F1=0.000) never
    predicts toxic and cannot discriminate the two. Ada RQ1 establishes only that **spilling is a
    sufficient toxicity indicator on this benchmark**.
  - The filter is **load-bearing**: 7 of the 8 dropped rows carry toxic labels (speedups 0.91–1.01, the
    noise band), so it removes **7 of the 23 raw toxic labels**; keeping them would cap recall at
    16/23=0.696. Defensible (no fusion exists to make) and disclosed, but not free.
  - F1 by fold scheme: shape-CV **0.941** (P=0.889/R=1.000; spills are constant across (R,C) shapes ⇒
    ~in-sample w.r.t. the decision), leave-one-dtype-out **0.968** (P=1.000/R=0.938; fp16-held 0.933,
    fp32-held 1.000) — **the only genuine in-family generalization signal**; leave-one-NOUT-out
    **0.667** (P=1.000/R=0.500) — the honest number.
  - **Why leave-one-NOUT-out fails is a defect of the fit, not of the problem.** Holding out NOUT=128
    collapses `gamma_spill` to its floor (~1e-8) — the fit abandons the spill term entirely despite the
    genuinely toxic 300-spill NOUT=64 rows remaining in training — and then misses all 8 toxic
    1986-spill cases. The reverse fold (hold out NOUT=64) learns gamma_spill=0.0067 and scores F1=1.000.
    So `gamma_spill` is **identifiable only from the extreme-spill regime** — a parameter-identifiability
    failure, *not* a calibration-range/extrapolation error. The zero-parameter spill rule scores
    F1=1.000 (recall 1.000) under the same protocol, i.e. it **strictly dominates the fitted model
    out-of-sample**. We do not claim Ada RQ1 as evidence that the fitted model generalizes across spill
    regimes. (Reporting nit: `fit.py` prints held-out NOUT=32 as "F1=0.000 recall=0.000" — that fold
    contains **zero** toxic cases, so recall is undefined; it actually scores acc=1.000, 8/8 TN. The
    pooled 0.667 is correct.)
  - Bootstrap (`model/bootstrap_ci.py`, B=1000) reports F1=1.000, 95% CI [1.000, 1.000] — reported for
    completeness only. It is a **mathematical identity, not an estimate**: predictions are frozen at the
    in-sample fit and match the labels elementwise, so 100% of resamples return exactly 1.0. It
    quantifies no sampling error and has zero power to detect overfitting; only a bootstrap that refits
    the constants inside each resample would be informative.
- RQ2a occupancy: analytic model **reproduces ncu *theoretical* occupancy exactly (MAE=0.000, 22/22)**
  = the CUDA occupancy calculator by construction; it does **not** predict *achieved* occupancy (off
  21.7 pts mean / 88 max) — the spill term handles the real degradation.
- RQ2b attribution: **100%** on cases with a profiled dominant penalty (spill branch);
  layout branch validated by the raw-CUDA bank-conflict study (spills=0, conflicts drive harm).
- RQ4 utility: recommender **5.8–8.7× faster than greedy-always-fuse** (wide_multiproj **8.65×**,
  mixed_widths **5.81×**, fp32_block **7.42×**); **matches the timed oracle exactly (1.000×) on all
  three subgraphs**, at zero timing cost (compiles only). See `logs/run_endtoend.log`.
  *Caveat:* the subgraphs are deliberately built with wide layers greedy over-fuses, so the 8.65×
  headline is a constructed upper bound on greedy's badness, not a typical-workload speedup.
  *Not a correction:* RQ4 was **not** affected by the §4a artifact — the end-to-end harness allocates
  few enough buffers that it never entered the oversubscribed regime (pre-fix greedy(w128)=151.4 ms
  sits with the unstarved matrix value ~173 ms, not the starved ~3142 ms). Post-fix times rose by a
  uniform 1.22–1.37× across spilling and non-spilling kernels alike, with **no ~10× deflation of the
  spilling greedy kernel**; the shift from the pre-fix 9.72/6.17/7.43 is run-to-run drift that largely
  cancels in the ratio. RQ4's conclusion holds in **both** regimes — a robustness result — and is
  additionally robust to the 14× refit of gamma_spill (0.0807 → 0.00567), which left every fusion
  decision identical.
- **GEMM cross-vendor decision-flip** (`data/microbench_gemm_ada.csv` vs `data/microbench_gemm_c500.csv`):
  a *second* flip, in the GEMM-epilogue family. On the same fp32 128×128 tile, Ada's compiler fits the
  kernel in **255 regs/thread with 0 spills and fusion is beneficial** (sweep 1.028–1.269;
  2048×2048×512 median **1.083**, 95% CI [1.072, 1.104]), while the C500 allocates **256 regs/thread,
  spills 205, and fusion is toxic** (sweep 0.777–0.832; median **0.824** across four GPUs, per-GPU 95%
  CIs spanning [0.821, 0.827] — an envelope, not one interval). Ada spills nowhere in this family
  (`f_spills=0` on all 16 configs).
  - *Mechanism (partial — we state what we measured, not why).* The C500's Triton backend allocates
    **1.6–1.75× more registers per thread than ptxas** for identical kernels at tiles where neither
    device spills (fp16 64×64: 84 → 134; fp32 64×64: 96 → 168); at the 128×128 fp32 tile it saturates
    its 256-reg/thread architectural cap and spills, while Ada lands at 255, just inside its own cap.
    We do **NOT** attribute this to the C500's 64-wide wavefronts: its register file per CU is doubled
    in lockstep (131072 vs 65536), so per-wave demand and per-CU capacity cancel exactly — both devices
    hold 8 resident waves/warps on the reduction and 4 on the 128×128 GEMM (back-solved from the
    reported `f_occ`). Wave width also predicts the **wrong direction**: the C500 spreads the same tile
    over 2× the threads (256 vs 128/block), which should *halve* per-thread demand. Separating compiler
    register-allocation maturity from an ISA/accumulator-layout cause needs a per-wave register-demand
    measurement we do not have.
  - *Spilling is necessary but not sufficient for toxicity:* the C500's fp16 128×128 tile spills 117 and
    is still **beneficial** (1.011–1.064).
  - *No Ada GEMM config is significantly toxic.* Two of 16 sweep rows carry single-shot toxic labels
    (0.973, 0.957); on repeat measurement the fp16 128×128 row is beneficial (1.0250, CI [1.025, 1.029])
    and the fp32 64×64 row is **inconclusive** (0.9904, CI [0.964, 1.055] — spans 1.0). We do not claim
    it is benign, only that the single-shot label is unsupported.
  - *CI caveat:* `gemm_sweep`'s one-shot labels are noise-prone within ~5% of 1.0, and our CIs are
    within-process percentiles over min-of-N timings — they bound short-term noise only. Two independent
    20-round runs of the identical 2048×2048×512 fp32 BM128 config produced **non-overlapping**
    intervals ([1.111, 1.118] vs [1.072, 1.104]), so run-to-run variation on this thermally-throttling
    laptop exceeds the quoted CI width. **Treat verdicts within ~5% of 1.0 as unresolved.**
- Spill cliff (`data/microbench_timing.csv`, mean speedup by NOUT): NOUT=32 (0 spills) fp16 1.028 /
  fp32 1.046 beneficial; NOUT=64 (300/310 spills) fp16 0.940 / fp32 0.897; NOUT=128 (1986/1868 spills)
  0.158 / 0.145 toxic. At NOUT=64 only the **fp32** point is CI-backed (median 0.893, CI [0.893, 0.898],
  20 rounds); the fp16 bucket has no CI and contains a member at 0.982, so "toxic" at NOUT=64/fp16 is
  **not yet established** — labels there come from a single-shot `t_fused < t_unfused` threshold, which
  the GEMM CIs above showed can flip.
- Ada finding: the **only decision-flipping** toxic mechanism *within Ada* is the register-spill cliff
  (P_occ). Layout penalties degrade but do not overturn round-trip savings on Ada.

## 7. C500 cross-vendor transfer — DONE (Phase 4)
Full log: `logs/LOG-04-c500-transfer.md`; reproduce: `python -m model.transfer_c500`. The model
transferred by **swapping `DeviceConstants` + `HardwareModel` only — formulas unchanged.**
`METAX_C500` in `fusion/hw.py` (env `FUSION_HW=c500`); constants `model/c500_constants.json`; ground
truth via `fusion/mcpti_profile.py` (MCPTI-direct — the `mcProfiler` CLI value-dump is unimplemented).

**Cross-vendor generalization table (reproducible):**
| property | Ada sm89 | MetaX C500 |
|---|---|---|
| wavefront (warp) size | 32 | **64** |
| register file / CU · spill cap | 64K · soft | 128K · **hard 4 KB/thread** |
| genuine cases / toxic | 64 / 16 | 64 / 20 |
| decision F1 | 1.000 (in-sample; **trivially separable** — see §6) | 0.909 |
| precision / recall | 1.000 / 1.000 | 0.833 / 1.000 |
| errors | none in-sample (0.667 leave-one-NOUT-out) | 4 FP (NOUT=32 fp16) |
| attribution (model==profiled) | 12/12 (8 spill ncu + 4 layout) | **12/12 (spill, MCPTI)** |
| gamma_spill · B_peak (fit) | **0.00567 · 181 GB/s** | 0.0068 · **1.05 TB/s** |
| spill-cliff onset (NOUT) | 64 | **32 (earlier; cause not established)** |
| GEMM fp32 128×128 tile | 255 regs, **0 spills**, beneficial 1.083 | 256 regs, **205 spills**, toxic 0.824 |

- **Decision-flip 1 (reduction):** `sibling_redux NOUT=32 fp32` = beneficial on Ada (0 spills, 1.04×) →
  **toxic on C500** (100 spills, 0.67×), all 4 shapes. The model flips its verdict via the re-read
  static spill count.
- **Decision-flip 2 (GEMM) — the flip is now bilateral across families**, i.e. it is no longer confined
  to the reduction family: the fp32 128×128 GEMM-epilogue tile is beneficial on Ada (0 spills, 1.083)
  and toxic on the C500 (205 spills, 0.824). Both benchmark families now exercise the same
  static-spill-driven verdict flip, in the same direction (Ada-beneficial → C500-toxic). See §6 for the
  measured contrast and its (partial) mechanism.
- **Mechanism caveat — do not repeat the "64-wide waves double register pressure" story.** It is
  contradicted by this repo's own numbers: the C500's register file per CU is doubled in lockstep
  (131072 vs 65536), absolute residency is identical on both devices, and the C500 spreads the same tile
  over 2× the threads (which predicts *lower* per-thread demand). The binding constraint is the
  **per-thread architectural cap** (255 NVIDIA / 256 C500) — both devices sit at it — which is
  independent of wave width. The unrefuted (simpler) explanation is that the C500's Triton/MetaX
  register allocator simply demands ~1.7× more regs/thread than ptxas for the same kernel; nothing in
  the data isolates wave width from compiler maturity. The reduction flip's dtype dependence is
  *also* unexplained by register state (C500 NOUT=32 fp16 and fp32 have identical regs=256 /
  spills=100 / occ=0.25, yet 1.083 beneficial vs 0.672 toxic), so it cannot corroborate the GEMM flip.
- **The re-read (reread-aware spill-traffic) feature is inert wherever nothing spills — vacuous by
  construction on Ada, not validated there.** The reread multiplier attaches only to
  CONTRACTION→POINTWISE pairs (the 16 GEMM rows), and every one compiles to `f_spills=0` on Ada. Since
  the spill term is `spill_factor(spills × reread)`, the reread column multiplies zero: the no-reread and
  reread-aware training sets give a **bit-identical objective** (loss delta 0.0 at every θ) and identical
  fits (gamma_spill=0.04881 both ways). This is an **arithmetic identity, not a measurement** — it is
  **no evidence** that the fix is correct, and equally none that it is harmless, because the feature has
  no path to affect any Ada decision. **Ada is therefore NOT a control.** The fix's evidence is the C500
  result alone (held-out GEMM recall 1.0 with reread=2 vs 0.0 with reread=1, LOG-06) — validated only
  where the failure mode exists. Ada's real contribution is the GEMM flip itself; the fix's inertness is
  a *corollary* of that flip, not a separate finding. (The "GEMM F1/recall=0.000 (n_toxic=2)" printed by
  `model/refit_combined_ada.py` is **not** a meaningful score: both "toxic" labels are timing noise that
  20-round CIs overturn, so the 2 FNs are the model disagreeing with a wrong label. Ada has zero
  genuinely toxic GEMM fusions ⇒ GEMM recall on Ada is **undefined; report n/a, not 0.000**.)
- **Honest caveats** (feed TODO G2/G3): the binary decision is spill-dominated, so Ada constants on
  C500 give the same F1 — the flip is carried by re-read static inputs, not re-fit constants; and the
  4 C500 FPs (`NOUT=32 fp16`, same 100 spills as the toxic fp32) show spill *count* alone can't
  separate them. NOUT=128 does **not** cross the 4 KB cap into a launch failure (spills-but-runs).
- Deferred: achieved-occupancy on C500 needs the MCPTI Metric API (no raw `waves` event); low value
  (occupancy term inert).
