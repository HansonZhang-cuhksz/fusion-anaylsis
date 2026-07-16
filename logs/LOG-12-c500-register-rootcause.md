# LOG-12 — #2: why does the C500 toolchain allocate 1.6–1.75× more registers/thread?

Date: 2026-07-15 · Machine: MetaX C500 (MACA 3.7.0), env `fusion`, idle GPU 1/2 · From the Ada
adversarial review (LOG-10 §2), which refuted the "64-wide wavefront doubles register pressure"
explanation and left the *cause* of the C500's higher register allocation unidentified. This log
identifies it.

## The phenomenon (grounded in committed data)
Identical Triton kernels, default launch config, non-spilling so the count is true allocator demand:
| kernel (64×64 GEMM tile) | Ada `f_regs` | C500 `f_regs` | ratio |
|---|---|---|---|
| fp16 | 84 | 134 | 1.60× |
| fp32 | 96 | 168 | 1.75× |

At the 128×128 tile this inflation slams the C500 into its **256-MTregister cap** and spills
(fp32 205, fp16 117) where Ada fits (232/255, 0 spills) — the GEMM half of the decision-flip. The
reduction NOUT=32 does the same (C500 256 regs + 100 spills vs Ada 255 regs + 0 spills).

## Experiments (all on C500; `scratchpad/{stage_sweep,spill_sens,redux_sens2,probe_regs}.py`)

**1. Register cost is dominated by software-pipeline multi-buffering (~34 regs/stage).**
GEMM 64×64 fp32, 4 warps, vary `num_stages`: 1→116, 2→134, 3→168, 4→204, 5→236 regs
(≈ +34/stage; fp16 ≈ +26/stage). The Triton default is `num_stages=3`, so pipelining alone adds
~68 regs over a single-stage kernel. Each stage keeps its own copy of the loaded A/B operand tile
fragments **in registers**.

**2. MACA *does* use the split ST/MT register file — so this is NOT scalar-register underuse.**
`cucc -resource-usage` on a test kernel: `Used 26 MTregisters, 16 STregisters`. The scalar file is
active; the high MT count is genuine vector-register demand, not a failure to offload uniform values.

**3. The accumulator layout is NOT the driver.** At the default 4 warps the C500 block is
4×64 = **256 threads** (vs Ada's 4×32 = 128). A 64×64 fp32 tile spread over 256 threads is only 16
acc-regs/thread (Ada: 32/thread). So the C500 carries *half* the per-thread accumulator yet uses
*more* total registers — the excess is per-thread pipeline/addressing state, not the accumulator.
(Corollary: the effect is not MMA-specific — the reduction, which has no `tl.dot`/pipelining, inflates
too.)

**4. ⇒ The residual is a general allocator-efficiency gap.** Ada fits 2× the per-thread accumulator
work into 255 regs; C500 cannot fit half that into 256 → MACA needs ≈2× the registers per unit of
per-thread work (less live-range splitting / rematerialization / reuse than ptxas). Compiler maturity,
not ISA.

## The load-bearing consequence: the C500 toxicity is a DEFAULT-LAUNCH-CONFIG artifact
The spill that drives the decision-flip **disappears** when the NVIDIA-tuned default launch config is
adjusted for the C500's 64-wide wavefronts:

| case (fp32, default → tuned) | default (4 warps, 3 stages) | tuned | spill |
|---|---|---|---|
| GEMM 128×128, `num_stages=1` | 256 regs / **205 spills** | 252 regs / **0** | eliminated |
| GEMM 128×128, `num_warps=8` | 256 / **205** | 204 / **0** | eliminated |
| reduction NOUT=32, `num_warps=8` | 256 / **100** | 220 / **0** | eliminated |
| reduction NOUT=64, `num_warps=8` | 256 / 524 | 256 / **104** | 5× reduced |

The default `num_warps=4` is inherited from Triton's NVIDIA (32-wide-warp) tuning; on the C500's
64-wide warps it under-provisions threads so each thread's share of the tile + MACA's less-efficient
allocation overflows the 256-reg cap. Splitting the work across more warps (or shortening the pipeline)
fits it with zero spills — at 8 warps the register/thread even matches Ada (GEMM 64×64 fp32: 102 vs 96).

## Honest implications (must be disclosed)
- **The register inflation (1.6–1.75×) is real and now explained:** pipeline multi-buffering in
  registers + a ~2× general allocator-efficiency gap vs ptxas. NOT wavefront width (refuted), NOT
  scalar-register underuse (ruled out), NOT accumulator layout (ruled out).
- **The decision-flip's C500 toxicity is default-config-contingent, not a hardware limit.** For BOTH
  families the spill vanishes under a C500-aware config (`num_warps=8` / `num_stages=1`). This does not
  invalidate the model — it correctly reads the single-compile static spill count and flags the
  *as-shipped default* fusion as toxic, which is exactly the pre-autotune regime a search-free pass
  serves — but the flip must be framed as **"the default-config fusion is toxic on C500, beneficial on
  Ada,"** not "the fusion is fundamentally toxic on C500."
- **Follow-up NOW MEASURED (LOG-13):** the *re-tuned* (8-warp) fused kernel **is net-beneficial** on
  C500 for both families (reduction 1.080, GEMM 1.206), so the flip is genuinely tunable-away; the cost
  model tracks the TOXIC→beneficial flip via the config-dependent spill count. Caveat: toxicity is
  non-monotonic (nw16 reduction toxic again via over-provisioning, which the model misses). See LOG-13.

## Status
- [x] #2 root cause identified with direct C500 evidence (pipeline multi-buffering + allocator gap;
      candidates ruled out) and the config-artifact consequence documented.
- [x] Follow-up: timed the 8-warp re-tuned fused-vs-unfused (LOG-13) — re-tuning flips the decision back
      to beneficial for both families; model tracks it. Non-monotonic over-provisioning caveat added.
