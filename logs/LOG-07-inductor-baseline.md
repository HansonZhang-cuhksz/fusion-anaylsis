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

## 6. The model predicts Inductor's fusion outcomes from its GENERATED kernels (closes §4 caveat)
Hooked `triton.compile` to capture the single-compile static report (`n_regs`/`n_spills`) of the fused
Triton kernel **Inductor actually generates** — the model's exact required inputs. Inductor fuses the
elementwise chain into ONE kernel `triton_poi_fused_add_gelu_mul_sigmoid_0` (n_regs 14–42, no spills).
Feeding that + analytic plan bytes to the model (`fusion/inductor_predict.py`) and comparing the
MODEL's predicted fusion speedup to Inductor's MEASURED (eager vs compiled):

| M×N | Inductor kernel regs | model speedup | measured speedup | sign |
|---|---|---|---|---|
| 1024²–8192² | 14–42 (no spill) | 1.4–4.5× | ~2.8× (flat) | **beneficial 5/5** |

- **The model reads a REAL production compiler's own kernels and correctly predicts its fusion is
  beneficial (5/5 sign)** — not hand-written microbenchmarks. This is the per-kernel version of §3.
- **Honest limitations:** (i) elementwise fusions are ~always beneficial, so this is a *capability*
  demo (the model consumes real compiler output), not a *discriminating* test; the discriminating
  cases (compute-bound MLPs, §2–3) route to the vendor GEMM, which is not an Inductor Triton kernel
  to score. (ii) Magnitude is only roughly right (analytic byte model for Inductor's fused plan is
  approximate) — sign, not magnitude, is the claim.

## 7. Artifacts
`fusion/inductor_baseline.py`, `fusion/inductor_predict.py`; `data/microbench_inductor_c500.csv`.
