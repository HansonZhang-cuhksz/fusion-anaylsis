# FROZEN MODEL SPEC — search-free interpretable fusion-decision model (Ada sm89)

This is the frozen spec that travels to the MetaX C500 session (Phase 4). It contains the exact
formulas, the fitted Ada constants, and the vendor-neutral dataset schema. **Transfer = swap the
`DeviceConstants` and the `HardwareModel`; do not change the formulas.**

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
| const | value | note |
|---|---|---|
| C_peak | 4.01e11 flop/s | under-constrained (workloads memory-bound); not decision-critical |
| **B_peak** | **1.52e11 B/s** | physically correct for RTX 4060 laptop (128-bit GDDR6) |
| T_launch | 8.16e-5 s | per-launch overhead (WSL2 Python/Triton) |
| occ_knee | 0.08 | hit floor ⇒ occupancy above ~8% does not drive toxicity on Ada; **spills do** |
| gamma_spill | 0.0887 | spill sensitivity (the dominant toxic term) |
| beta_layout | 0.406 | bank-conflict sensitivity (fit from the cuda_layout PAD0/PAD1 slowdown) |

Fit objective: combined log(speedup) (weight 1.0) + log(absolute time) (0.3), occ_knee bounded
[0.08, 0.35] (Ada saturates HBM by ~⅓ occupancy). Nelder–Mead, 6 restarts.

## 5. Vendor-neutral dataset schema (concept columns, NOT raw metric names)
`data/microbench_timing.csv` (label + static features), `data/microbench_ncu.csv` (ground truth):
- keys: family, op_pair, producer_class, consumer_class, dtype, R, C, param_*
- static: f_regs, f_spills, f_smem, f_threads, f_occ(analytic), f_occ_binder, u_regs, u_occ,
  n_launches_unfused, bytes_fused, bytes_unfused, flops, arith_intensity
- measured (fit/validate only): t_fused_ms, t_unfused_ms, speedup, beneficial(label)
- ncu ground truth (concepts): occ_achieved/theoretical, spill(local)_bytes, bank_conf,
  dram_bytes, tensor_pct, dur_ns (ncu kernel duration, ns)  → dominant_penalty ∈ {spill, layout, none}

## 6. Headline Ada results (to reproduce on C500 and compare)
- RQ1 decision (72 genuine cases; 16 degenerate no-op rows — NOUT∈{8,16}, unfused==fused — excluded
  from scoring): **recall=1.00 across every CV** (never keeps a toxic fusion). F1 by fold scheme:
  shape-CV **0.91** (spills are constant across (R,C) shapes ⇒ ~in-sample), leave-one-NOUT-out
  **1.00** (spill signal perfectly separates the reduction cases on this clean microbench),
  leave-one-dtype-out **0.91** (fp16-held 0.96, fp32-held 0.86 — a mild transfer cost). Greedy F1=0.00.
- RQ2a occupancy: analytic model **reproduces ncu *theoretical* occupancy exactly (MAE=0.000, 22/22)**
  = the CUDA occupancy calculator by construction; it does **not** predict *achieved* occupancy (off
  21.7 pts mean / 88 max) — the spill term handles the real degradation.
- RQ2b attribution: **100%** on cases with a profiled dominant penalty (spill branch);
  layout branch validated by the raw-CUDA bank-conflict study (spills=0, conflicts drive harm).
- RQ4 utility: recommender **up to 9.85× faster than greedy-always-fuse**; **matches the timed
  oracle exactly (1.00×) on the fp16 subgraphs** and is **3.03× off oracle on the fp32 subgraph**
  (where discounting smooth occupancy under-penalises non-spilling fp32 width tuning — still 2.80×
  faster than greedy), at zero timing cost (compiles only). See `logs/run_endtoend.log`.
  *Caveat:* the subgraphs are deliberately built with wide layers greedy over-fuses, so 9.85× is a
  constructed upper bound on greedy's badness, not a typical-workload speedup.
- Ada finding: the **only decision-flipping** toxic mechanism is the register-spill cliff (P_occ).
  Layout penalties degrade but do not overturn round-trip savings on Ada.

## 7. C500 transfer checklist (Phase 4)
1. Build `HardwareModel("metax_c500", ...)`: SMs, warps/SM, regs/SM, `split_regfile=True`,
   `spill_cap_bytes=4096`. Occupancy formula unchanged.
2. Re-fit `DeviceConstants` on C500 (B_peak, C_peak, T_launch, gamma_spill, beta_layout) from the
   C500 microbench timing (same `runner.py`, Triton ports).
3. Ground truth via MCPTI (WAVES, conflict cycles, MMA duty, Dnoc/L2C). Map to the SAME concept
   columns (§5).
4. **Decision-flip hunt:** the 4 KB/thread private cap turns the soft spill cost into a HARD launch
   failure — a fusion safe on Ada (spills but runs) can be *illegal* on C500. Expect ≥1 flip in the
   sibling-reduction family around the NOUT where spill bytes/thread cross 4 KB.
