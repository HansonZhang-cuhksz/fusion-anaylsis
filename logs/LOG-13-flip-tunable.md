# LOG-13 — #2 follow-up: is the C500 decision-flip tunable-away? (YES — but the model still tracks it)

Date: 2026-07-15 · Machine: MetaX C500 (MACA 3.7.0), env `fusion`, idle GPU 2 · Follow-up to LOG-12
(the C500 spill is a default-launch-config artifact). Question: if the compiler re-tunes past the
default `num_warps=4` (removing the spill), does the fusion become *beneficial* on C500 — i.e. is the
decision-flip a property of the default config rather than the hardware? **Answer: yes.** Data:
`data/flip_tunable_c500.csv`. Independently reproduced + adversarially verified (workflow `wf_42172126`).

## Measurement (fp32, R=C=2048 reduction; 2048×2048×512 GEMM; 15 rounds × min-of-50; `c.check()` gated)
| case | num_warps | spills | fused ms | unfused ms | speedup (u/f) | verdict |
|---|---|---|---|---|---|---|
| reduction NOUT=32 | **4 (default)** | 100 | 2.240 | 1.436 | **0.641** | TOXIC |
| reduction NOUT=32 | **8** | 0 | 1.225 | 1.323 | **1.080** | benef |
| reduction NOUT=32 | 16 | 0 | 1.562 | 1.276 | **0.817** | TOXIC |
| GEMM 128×128 | **4 (default)** | 205 | 0.804 | 0.663 | **0.825** | TOXIC |
| GEMM 128×128 | **8** | 0 | 0.127 | 0.153 | **1.206** | benef |

**Tuning-aware decision** (best fused config vs best unfused config, each re-tuned over the warp grid):
- reduction: best_fused=nw8 (1.225 ms, 0 spills) vs best_unfused=nw16 (1.276 ms) ⇒ **1.041, beneficial** — but the margin is **thin (~4%)**.
- GEMM: best_fused=nw8 vs best_unfused=nw8 ⇒ **1.206, beneficial** — **robust (~20%)**.

⇒ **Both decision-flips are tunable-away.** The default (NVIDIA-tuned) `num_warps=4` fusion is toxic
on C500 because it spills; a C500-aware `num_warps=8` fusion removes the spill and is *beneficial*. The
flip is a property of the **default launch configuration, not the hardware.** (Verified fair: the
unfused side is re-tuned too; even a fully-tuned unfused loses to the fused; the GEMM ~20% gap equals
the saved M×N epilogue round-trip, confirmed by a GEMM-only probe.)

## The cost model TRACKS the spill-driven flip (re-derived on hardware)
Feeding the config-dependent single-compile static inputs to `decide()` (`c500_combined_constants.json`):
| case | nw4 (spills) → pred | nw8 (0 spills) → pred |
|---|---|---|
| reduction NOUT=32 fp32 | f_spills=100 → **TOXIC** (u/f 0.687) | f_spills=0 → **beneficial** (u/f 1.135) |
| GEMM 128×128 fp32 | f_spills=205 → **TOXIC** (u/f 0.626) | f_spills=0 → **beneficial** (u/f 1.010) |

The model flips its verdict with the launch config, driven by the static spill count — matching the
measured direction on both devices and both configs. This is the pre-autotune signal a search-free pass
provides: a user compiling with defaults on C500 gets a toxic fusion and the model flags it; re-tune and
the model tracks that too.

## ⚠ Honest limitation: toxicity is NON-MONOTONIC in num_warps, and the model only tracks the spill branch
The reduction is toxic@4 (spill) → beneficial@8 → **toxic again@16** (0.817, *zero spills*). The nw16
toxicity is **over-provisioning** (512→1024 threads for a small tile collapses occupancy: the model's
own `f_occ`→0.000 there), *not* spilling. The model, fed 0 spills at nw16, predicts **beneficial** (u/f
1.009) — **wrong**. So the model tracks the **spill-driven** branch of the toxicity curve but misses the
**occupancy/over-provisioning** branch (the same inert-P_occ limitation disclosed for RQ1). This is why
nw8 (not "as many warps as possible") is the actual sweet spot, and the model cannot by itself find it —
it can only rule out the spilling configs.

## Net effect on the headline
- **Supersedes** the "untested: whether the re-tuned 8-warp fusion is net-beneficial" caveat (LOG-12,
  RESULTS.md) — it is (reduction 1.04, GEMM 1.21).
- **Reframes** the decision-flip: *"the same **default-config** fused kernel is beneficial on Ada but
  toxic on C500 (register spill under the C500 toolchain's allocation); the cost model predicts this from
  the static spill count. The toxicity is removable by C500-specific tuning (num_warps=8), which the
  model also tracks."* Not a hardware-fundamental flip; a correct, config-dependent decision.
- **Adds** the non-monotonic / over-provisioning caveat above.

## Method note
An initial run measured the reduction nw4 fused at 7.0 ms (speedup 0.205) — a GPU-contention anomaly on a
non-idle GPU. Re-run on confirmed-idle GPU 2 (and independently by the verifier) gives 2.24 ms (0.641),
consistent with the committed `microbench_timing_c500.csv` (~0.634). The clean CSV reflects 0.641.
