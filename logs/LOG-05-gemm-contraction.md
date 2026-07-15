# LOG-05 — CONTRACTION (GEMM-epilogue) family: breadth (G4) + the spill-collapse failure (G2/G3)

Date: 2026-07-15 · Machine: MetaX C500 ×4, env `fusion` · TODO gaps G4 (breadth) + G2/G3 (model collapse)

## 1. What was built
`fusion/kernels/gemm_epilogue.py` — Family **G / CONTRACTION**, the declared-but-missing taxonomy
class and the canonical *compute-bound* fusion (vs the memory/spill-bound Family R):
`C = relu(A@B + bias)`; **fused** = one Triton GEMM (`tl.dot`) applying bias+relu in-register;
**unfused** = GEMM→HBM `T`, then a pointwise `relu(T+bias)` kernel. Fusion saves the M×N round-trip.
- `tl.dot` (tensor-core/MMA path) **works on C500** (fp16 & fp32, correctness exact, max|Δ|=0).
- Swept 16 configs (shape × tile × dtype) via a **4-GPU parallel workflow** (`fusion/gemm_sweep.py`,
  one agent per C500 GPU) → `data/microbench_gemm_c500.csv`. Tile 128×128 pushes occupancy to 0.125.

## 2. G4 — breadth result (CONTRACTION characterized on C500)
Fusion **beneficial 12/16, toxic 4/16**. Pattern (consistent across all 4 shapes):
| tile | dtype | regs | spills | occ | speedup | verdict |
|---|---|---|---|---|---|---|
| 64×64 | fp16 | 134 | 0 | 0.375 | 1.05–1.21× | ✓ beneficial |
| 64×64 | fp32 | 168 | 0 | 0.25 | 1.03–1.14× | ✓ beneficial |
| 128×128 | fp16 | 256 | 117 | 0.125 | 1.01–1.06× | ✓ beneficial |
| **128×128** | **fp32** | **256** | **205** | **0.125** | **0.78–0.83×** | **✗ toxic** |

The epilogue-fusion win (saved round-trip) holds broadly; only the fp32 128×128 tile flips toxic.

## 3. G2 — does the model collapse to "did it spill?" → **YES, and here it FAILS**
Two findings, one confirming the design and one exposing its limit:

**(a) Occupancy alone does not flip the decision** (confirms the spill-dominant model). At the *lowest*
occupancy (0.125, 128×128 tile), fp16 (spill 117) is beneficial while fp32 (spill 205) is toxic — the
3× occupancy loss vs the 64×64 tile does not by itself cause toxicity. Across two very different op
families (R reductions, G GEMMs) the decisive signal is the **register-spill cliff**, not occupancy.

**(b) But the search-free spill-count model FAILS to generalize to GEMM's toxic cases.** Running the
model (C500 constants, fit on reduction+pointwise) on the 16 GEMM configs:
> **P=0.000 R=0.000 F1=0.000 acc=0.750 — misses ALL 4 toxic cases (FN=4).**

Root cause — the static spill count has the **wrong sign** here: the toxic fp32 128×128 configs have
fused `f_spills=205` but unfused-GEMM `u_spills=234`, so the model reasons *"fused spills less →
beneficial"* — yet the fusion is measured **toxic (0.78×)**. The single-compile spill *instruction
count* undercounts the fused kernel's real cost (its epilogue re-reads the spilled accumulator, adding
runtime spill traffic the static count misses). **This is the concrete G2/G3 failure regime:** a
fusion toxic for a reason the search-free spill signal cannot see.

## 4. Implications (actionable)
- **G4:** CONTRACTION now implemented + characterized; the dataset is broadened beyond memory-bound
  reductions to a compute-bound tensor-core family.
- **G2/G3:** we now have a *reproducible* case where the spill-count model is wrong (recall 0/4 on
  GEMM toxic). The fix is a **non-spill / runtime-spill-aware cost signal** — e.g., use the fused
  kernel's *shared/local-mem bytes* or an occupancy×compute term instead of (or alongside) the static
  spill count. This is the sharpened next step for the "interpretable multi-cause" claim.
- **Caveat (G5):** the fp32-big-tile toxicity is consistent across 4 shapes measured on 4 independent
  GPUs, but timing CIs would confirm it is not a measurement artifact (cf. the earlier Ada fp32
  artifact). Worth a repeat-with-CIs pass.

## 5. Verified real + root-caused (not a measurement artifact)
Re-timed with 150 iters (min-time, robust to noise): fp32 128×128 speedup **0.826 (mean 0.825, min
0.826)** — the toxicity is **real and consistent**, not an artifact (cf. the earlier Ada fp32 artifact).
MCPTI (fused-plan vs unfused-plan local/global instruction counts, `scratchpad/gemm_rootcause.py`):
| config | f_spill / u_spill | **fused local** | **unfused local** | fused global | unfused global | speedup(min) |
|---|---|---|---|---|---|---|
| fp32 64×64 | 0 / 0 | 0 | 0 | 328K | 348K | 1.145 ✓ |
| fp16 128×128 | 117 / 118 | 461K | 433K | 98K | 124K | 1.063 ✓ |
| **fp32 128×128** | **205 / 234** | **950K** | **833K** | 172K | 198K | **0.826 ✗** |

**Mechanism:** the fused kernel does **more local (spill) traffic (950K > 833K)** than the unfused
GEMM, even though its static spill *instruction count* is LOWER (205 < 234) — its epilogue **re-reads
the spilled accumulator** (bias+relu on spilled data). That extra local traffic (+117K) outweighs the
saved global round-trip (−26K) → net toxic. **The single-compile spill COUNT ≠ runtime spill
TRAFFIC**, so the search-free static signal has the wrong sign for epilogue-into-spilling-GEMM.

## 6. Fix proof-of-concept (still search-free: a taxonomy-aware spill-TRAFFIC estimate)
Model spill signal = `f_spills × reread_mult`, where `reread_mult = 2` when the fusion applies a
full-tile epilogue over a spilling producer (≈ store + reload) — derivable from Φ(v)
(CONTRACTION producer + POINTWISE consumer), else 1. Result on the 16 GEMM configs
(`scratchpad/gemm_fix_poc.py`):
- **baseline** (`reread_mult=1`): P=0 **R=0** F1=0 (FN=4 — misses all toxic).
- **fix** (`reread_mult=2`): **R=1.0** (catches all 4 toxic), P=0.5 F1=0.667.
Recovers the safety-critical recall (0→1). A *flat* ×2 over-rejects the beneficial fp16 big-tile
(precision 0.5), so the proper fix is to add this reread-aware spill-traffic feature and **RE-FIT on
the combined dataset** (reductions + pointwise + GEMM) to calibrate the threshold across families —
which also finally gives the Φ(v) taxonomy a real job (closes G7).

## 7. Artifacts
- `fusion/kernels/gemm_epilogue.py`, `fusion/gemm_sweep.py`; `data/microbench_gemm_c500.csv` (16 rows).
- Investigation: `scratchpad/gemm_modelcheck.py` (generalization fail), `gemm_rootcause.py` (MCPTI +
  robust timing), `gemm_fix_poc.py` (the fix). Promote to `model/` when the re-fit lands.
