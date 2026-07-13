# HANDOFF → Ada machine (NVIDIA sm89) Claude Code session

You are picking up the **primary research work** for the project described in `PROPOSAL.md`.
Read `PROPOSAL.md` first — this file is the operational quickstart, the frozen decisions, and
the context you must design for even though you can't see the MetaX hardware.

---

## 0. TL;DR of what you're building

A **search‑free, interpretable cost model that decides when NOT to fuse two GPU operators**, and
attributes the harm to occupancy/spill vs layout/bank‑conflict. Your machine (Ada sm89) is the
**primary** target. A second machine has 4× MetaX C500 GPUs for a later **cross‑vendor transfer**
phase — design for portability now, but you do NOT touch MetaX. The user runs that phase later on
the MetaX session.

Your scope = **Phases 1–3** in `PROPOSAL.md §8`. Phase 0 (env validation) is already done on MetaX.

---

## 1. Frozen decisions (do not re‑litigate; changing these needs the user)

1. **Target = Ada sm89 primary + MetaX C500 transfer.** No Hopper/Blackwell (no hardware).
2. **Drop the warp‑specialization penalty `P_div`.** Ada and C500 have no async warpgroup/TMA;
   the term is ~0. The model is `η_fused = min(η_u,η_v) · P_occ · P_layout`. Do not add P_div back.
3. **Search‑free.** The deployed pass may **compile once** and read the resource report, but must
   **not** autotune or profile at decision time. Profiling is for *fitting/validation only*.
4. **Interpretability is a first‑class deliverable**, not a nicety — the pass must output the
   dominant penalty per rejected edge, and we validate that attribution against the profiler.
5. **Novelty rests on: interpretable attribution + cross‑vendor transfer (incl. a domestic GPU).**
   Not on discovering a new microarch phenomenon. Frame honestly (workshop/short‑paper scale).
6. **Fallback if Inductor integration is too deep:** an offline recommender that scores a fusion
   plan and emits edits + attributions. This still answers RQ1–RQ4.

---

## 2. Ada environment bring‑up checklist (Phase 1, do first)

- [ ] `nvidia-smi` — confirm sm89 (e.g. RTX 4090 / L40 / L4), driver, CUDA version
- [ ] `nvcc --version`; confirm Triton + `torch.compile` work (a trivial `@torch.compile` fn)
- [ ] `ncu --version` and **verify you can actually collect counters** (profiling often needs
      `NVreg_RestrictProfilingToAdminUsers=0`, or run `ncu` with sudo / `--target-processes all`).
      If counters are blocked, resolve this before building the dataset — it's the usual Ada
      gotcha and it will silently return empty metrics otherwise.
- [ ] Confirm `nvcc -Xptxas -v` (a.k.a. `--resource-usage`) prints **registers/thread + spill
      stores/loads + shared mem** per kernel. This is your static `P_occ` input — the NVIDIA
      counterpart of MetaX's `cucc -resource-usage`.
- [ ] Port `tooling/probe_kernel.cu` and write an NVIDIA analogue of
      `tooling/check_profiling_stack.sh` (call it `tooling/check_profiling_stack_nvidia.sh`).

## 3. The counter map you'll need (Ada `ncu` → concept → later C500 equivalent)

| Concept (model term) | Ada / `ncu` metric | C500 equivalent (for later) |
|---|---|---|
| Registers/thread | `launch__registers_per_thread` (or `-Xptxas -v`) | `cucc -resource-usage` MT+ST regs |
| Achieved occupancy | `sm__warps_active.avg.pct_of_peak_sustained_active` | `Achieved waves`/`WAVES` |
| Static occupancy | CUDA Occupancy API / `--resource-usage` | `staticMaxWarps/PEU` |
| Register spills (P_occ cliff) | `-Xptxas -v` spill stores/loads; local‑mem traffic | private‑mem cap **4 KB/thread** (hard fail) |
| Bank conflicts (P_layout) | `l1tex__data_bank_conflicts_pipe_lsu_*` | `average conflict cycles per instruction` |
| DRAM traffic | `dram__bytes.sum` | `Global Read/Write Instructions`, `Dnoc`, `L2C` |
| Tensor‑Core util | `sm__pipe_tensor_op_*` | `AP MMA Duty ratio` |

Keep your dataset schema **vendor‑neutral** (concept columns, not raw metric names) so Phase 4
transfer is a re‑parameterization, not a rewrite.

## 4. MetaX facts to design around now (so transfer is smooth later)

- C500 has a **split scalar/vector register file** (`STregisters` + `MTregisters`) — unlike Ada's
  unified file. Your `P_occ` should treat the register‑file model as a **parameter**, not hardcode
  a single-pool assumption.
- C500 enforces a **hard 4 KB/thread private (local/spill) memory cap**; exceeding it fails the
  launch (`mcErrorMemoryValueTooLarge`). This is the sharpest spill cliff and the most likely
  source of a **decision‑flip** (a fusion safe on Ada, toxic on C500). Make sure your model can
  represent a discontinuous spill penalty, not just a smooth occupancy curve.
- C500 profiling uses **MCPTI** (CUPTI‑compatible field names) or `mcProfiler`; NVIDIA binaries do
  NOT run there. `torch 2.8.0+metax` + `triton 3.0.0` are installed, so your Triton kernels port.

## 5. First concrete tasks (in order)

1. Ada env checklist (§2). Commit `tooling/check_profiling_stack_nvidia.sh`.
2. Build the microbench matrix (fused/unfused op‑pairs across the taxonomy; shape sweep). Start
   with the highest‑signal pair: **LayerNorm/Softmax (reduction) → GEMM epilogue** — the classic
   tile‑shape clash — plus a **pointwise→GEMM** easy‑win control.
3. Automate `ncu` → tidy vendor‑neutral dataset with ground‑truth `beneficial?` + `dominant_penalty`.
4. Implement `η_fused` from single‑compile inputs; report RQ1/RQ2 numbers.
5. Only then attempt the Inductor pass (or the offline‑recommender fallback).

## 6. When to hand back to the MetaX session

After **Phase 3 DoD**: end‑to‑end speedup on ≥2 real subgraphs AND a **frozen model spec** (exact
formulas + fitted constants + the vendor‑neutral dataset schema). At that point the user returns to
the MetaX (this) session for Phase 4 transfer. Leave the frozen spec in `PROPOSAL.md`/a `model/`
dir so it travels via git.

---

*Repo artifacts already present:* `PROPOSAL.md`, `tooling/check_profiling_stack.sh` (MetaX),
`tooling/probe_kernel.cu`. This machine's git repo is the transport between sessions — keep
everything reproducible and committed.
