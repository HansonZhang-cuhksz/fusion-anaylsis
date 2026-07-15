# Ada-session handoff — Bucket 1 (hardware-gated items)

*You are on the **Ada machine** (RTX 4060 Laptop, sm89, WSL2). You have the repo via git but **not** the
MetaX session's chat history — everything you need is in the repo. The project's consolidated results
are in [`RESULTS.md`](RESULTS.md); the frozen model in `model/MODEL_SPEC.md`; per-topic logs in
`logs/LOG-01…09`. This file lists the three items that could only be done on your hardware.*

## 0. Setup (Ada)
```
git pull                      # get the C500 work: GEMM family, timing_ci, reread fix, RESULTS.md
source tooling/env.sh         # puts nvcc/ncu on PATH, pins host gcc-11, exports sm89 constants
# conda env: profiling  (torch 2.7.0+cu128, triton 3.3.0, nvcc 12.8, ncu 2025.1.1)
```
Static-input extraction defaults to the Ada `HardwareModel` (`FUSION_HW` unset ⇒ `ADA_SM89`). All the
code below is already committed; you are running it on Ada and comparing to the committed C500 data.

---

## Task A1 — Run the CONTRACTION (GEMM-epilogue) family on Ada
**Why:** the GEMM family + the reread-aware spill-traffic fix (LOG-05/06) were only run on the C500, so
the cross-vendor GEMM story is one-sided. This closes it.
```
python -m fusion.gemm_sweep --configs all --out data/microbench_gemm_ada.csv
```
This sweeps 16 configs (shape × tile × dtype); `tl.dot` should work under CUDA. **Report / check:**
- Do the same fp32 128×128 tiles that are **toxic on C500** (`data/microbench_gemm_c500.csv`; f_spills=205,
  speedup ~0.82) also **spill and go toxic on Ada**? Ada has half the register file (64K vs 128K/CU) but
  32-wide warps — the spill point may differ. Three possible outcomes, all interesting:
  (a) Ada also spills+toxic ⇒ the reread fix generalizes cross-vendor (strongest);
  (b) Ada spills less / stays beneficial ⇒ a **second decision-flip** (GEMM safe on Ada, toxic on C500);
  (c) different tile is the cliff ⇒ document it.
- Then score the model on the Ada GEMM data with the reread feature (mirror `model/build_combined_c500.py`
  → make a `build_combined_ada.py` combining `data/microbench_timing.csv` + `data/microbench_gemm_ada.csv`,
  `f_reread=spill_reread(...)`), re-fit, and confirm **GEMM recall on Ada** the way LOG-06 did for C500.
- **DoD:** `data/microbench_gemm_ada.csv` committed; an Ada-vs-C500 GEMM comparison paragraph; whether
  the reread fix holds on Ada.

## Task A2 — Ada-side timing CIs (finish the statistical-rigor pass)
**Why:** LOG-09 put 95% CIs on the key *C500* claims across 4 GPUs. The **Ada half** is not yet CI'd —
in particular the **decision-flip's Ada side** and RQ1.
```
python -m fusion.timing_ci --rounds 20 --out data/timing_ci_ada.csv
```
Note the `CLAIMS` `expected` labels in `fusion/timing_ci.py` are written for C500. **On Ada the verdicts
flip for the reductions** — most importantly **`redux_N32_fp32` should be significantly BENEFICIAL on Ada**
(median ~1.0–1.1×, CI excluding 1.0). Combined with the C500 result (median 0.64, CI [0.638, 0.645],
toxic), that makes the cross-vendor **decision-flip statistically bulletproof on *both* sides** — the
headline result. Report each claim's Ada median + 95% CI and whether it excludes 1.0.
- **RQ1 F1 CI:** add a bootstrap to the Ada decision quality — resample the 64 genuine cases B≈1000×,
  recompute F1 via `model.fit.decisions`/`prf`, report the F1 point + 95% CI (or, cheaper, report the
  leave-one-shape/NOUT/dtype fold spread you already compute in `model.fit`).
- **DoD:** `data/timing_ci_ada.csv`; the flip's Ada side significant; RQ1 F1 with a CI.

## Task A3 — Add the Ampere sm80 as a third hardware point *(only if the sm80 GPU is reachable)*
**Why:** two devices (Ada + C500) can look like a coincidence; a third (Ampere sm80) de-risks
single-device overfit and strengthens "transfer by re-parameterization." **Template:** copy exactly what
was done for the C500 — see `METAX_C500` in `fusion/hw.py` and the C500 transfer in `logs/LOG-04`.
1. `HardwareModel("ampere_sm80", …)` in `fusion/hw.py` from `torch.cuda.get_device_properties` +
   `nvcc -Xptxas -v` (SMs, warps/SM, regs/SM, warp_size=32, max_regs=255, smem/SM, spill=soft); wire an
   `FUSION_HW=sm80` branch into `default_hw()`.
2. `FUSION_HW=sm80 python -m fusion.runner data/microbench_timing_sm80.csv` (matrix) and
   `python -m fusion.gemm_sweep …` + `python -m fusion.timing_ci …` for sm80.
3. Re-fit `DeviceConstants` for sm80 (`model/sm80_constants.json`); run `model.transfer_c500`-style
   comparison (generalize it to a 3-way Ada / C500 / sm80 table); check the model transfers to sm80 too.
- **DoD:** a **3-way generalization table** (Ada / C500 / sm80); note any new decision-flips.

---

## Hand-back
Commit each task on `main` (this repo is the transport between machines; the MetaX session pulls it):
`git add … && git commit -m "Ada Bucket-1: GEMM on Ada / Ada CIs / sm80"` (end messages with the
`Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer). Then **fold the new Ada results into
`RESULTS.md`** (the cross-vendor GEMM row, the flip's Ada-side CI, the 3-way table) so it stays the single
source of truth. The MetaX session owns anything else needing the C500 (env there is `fusion`, not
`profiling`; MCPTI recipe in memory `metax-c500-profiling`).
