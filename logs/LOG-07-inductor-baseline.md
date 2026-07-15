# LOG-07 — Real-compiler (torch.compile / Inductor) fusion baseline on C500 (G4)

Date: 2026-07-15 · Machine: MetaX C500 ×4, env `fusion` · TODO gap G4 (real compiler baseline).

## 1. Inductor works on the C500
`torch.compile(backend="inductor")` **runs correctly on the MetaX C500** (max|err|=0 vs eager) — to our
knowledge the first demonstration. This unlocks the real-compiler baseline the proposal names (vs the
prior synthetic microbenchmarks + hand-built greedy/oracle).

## 2. Fusion benefit vs regime (`fusion/inductor_baseline.py`, 4-GPU sweep → `data/microbench_inductor_c500.csv`)
Two real subgraphs, eager (multi-kernel) vs `torch.compile` (fused), across sizes:
| subgraph | arith. intensity | regime | fusion speedup (eager/compiled) |
|---|---|---|---|
| pointwise+residual chain | 1.3 | **memory** | **1.91–3.65× (mean 2.57×)** |
| MLP-FFN (Linear→GELU→Linear) | 683–1170 | **compute** | **0.74–1.06× (mean 0.92×)** |

## 3. Findings (validate the model's thesis on a real compiler + real patterns)
- **Memory-bound fusion is a big win** (up to 3.65×): fusing the pointwise/residual chain into one
  kernel saves the HBM round-trips — exactly what the search-free roofline model flags beneficial.
- **Compute-bound fusion is ~free, and often TOXIC** (2/4 MLP configs are *slower* under
  `torch.compile`: 0.74×, 0.86×): the GEMMs dominate, so epilogue fusion buys nothing, and Inductor's
  codegen sometimes loses to the vendor GEMM path. **Even a production compiler over-fuses net-harmfully
  in the compute-bound regime** — precisely the mistake an interpretable roofline/spill pruning pass
  would catch.
- The regime split (fusion benefit governed by **arithmetic intensity / memory-boundedness**) is exactly
  the search-free model's core prediction — now confirmed against a **real production compiler on real
  transformer patterns**, not just our microbenchmarks vs a hand-built oracle. This substantially
  strengthens RQ4 / the "utility" claim.

## 4. Honest scope
The connection here is at the **roofline-regime** level (arithmetic intensity predicts where fusion
pays), not a full per-kernel static-input decision: Inductor's generated kernels are not our Triton
kernels, so scoring them with the model's single-compile inputs would need `TORCH_COMPILE_DEBUG` to
extract Inductor's Triton + its register/spill report (future work — then the model could *predict
Inductor's own fusion outcomes* per kernel). The regime-level agreement + the toxic-over-fusion cases
already make the point.

## 5. Artifacts
`fusion/inductor_baseline.py`; `data/microbench_inductor_c500.csv` (8 configs).
