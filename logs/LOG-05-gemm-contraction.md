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

## 5. Artifacts
- `fusion/kernels/gemm_epilogue.py`, `fusion/gemm_sweep.py`; `data/microbench_gemm_c500.csv` (16 rows).
- Model-generalization check: `scratchpad/gemm_modelcheck.py` (promote to `model/` if kept).
