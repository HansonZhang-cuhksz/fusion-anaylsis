# Detecting Toxic Operator Fusion
### An Interpretable, Search‑Free Cost Model for GPU Operator Fusion, Validated Across NVIDIA Ada and a Domestic GPU (MetaX C500)

*Working proposal v2 — supersedes `research-proposal.pdf`. Last updated 2026‑07‑13.*

---

## 0. What changed from v1 (and why)

v1 ("Beyond the Fusion Silver Bullet") proposed a *grand unified framework* — a taxonomy + analytical cost model + GNN surrogate — targeting Hopper/Blackwell, claiming "no unified framework exists." A literature scan (Welder, SpaceFusion, Hidet, Neptune, TpuGraphs, Mirage, plus 2026 analytical GPU models) showed each pillar is substantially covered by recent work, and the "first/only" framing is not defensible. We also have **no Hopper/Blackwell hardware**. This version is deliberately narrower and honest:

| v1 | v2 (this doc) | reason |
|---|---|---|
| Hopper/Blackwell target | **NVIDIA Ada (sm89)** primary + **MetaX C500** cross‑vendor | that is the hardware we actually have |
| `η_fused = min·P_occ·P_layout·P_div` | drop **P_div** (warp‑specialization) | Ada & C500 have no async warpgroup/TMA — the term is ~0; keeping it would be dishonest |
| "analytical, no profiling" | **search‑free, single‑compile static inputs** | register/occupancy come from *one compile*, not a run; sidesteps the intractable static‑register‑prediction problem |
| GNN surrogate pillar | **dropped** (or optional stretch goal) | redundant with the analytical model and with TpuGraphs; a hedge, not a contribution |
| "first comprehensive taxonomy" | **interpretable + cross‑vendor transfer** as the novelty | this is the slice the frontier has *not* nailed |

**One‑line thesis (v2):** *A compiler can decide — at compile time, without autotuning search — whether fusing two operators is net‑harmful, attribute the harm to a specific microarchitectural cause (occupancy loss vs layout/bank‑conflict cost), and this single interpretable criterion transfers from a consumer NVIDIA Ada GPU (RTX 4060) to a domestic accelerator by re‑parameterization rather than retraining.*

---

## 1. Motivation

Operator fusion eliminates HBM round‑trips for memory‑bound producers/consumers and is the single highest‑impact inference optimization (FlashAttention being the canonical win). But "fuse as much as possible" is a false heuristic: combining operators inflates per‑thread register pressure and forces tile/layout compromises, which reduce occupancy, spill to local memory, or inject shared‑memory transposes and bank conflicts — often erasing the memory savings. This is well documented (systematic studies report diminishing/negative returns beyond ~3 fused kernels; register pressure and occupancy collapse are the named culprits).

Today's compilers resolve this in one of three unsatisfying ways:

1. **Greedy/heuristic** (early TVM, naive Inductor rules): fast but blind to microarchitectural cost.
2. **Search/autotuning** (Ansor, Welder, Neptune): finds good plans but is expensive, must be re‑run per shape/arch, and is a **black box** — it never tells you *why* a fusion was rejected.
3. **Learned cost models** (TpuGraphs, learned TPU model): amortize search but need large per‑hardware training sets and are equally opaque.

**The gap we target:** a *cheap, static, interpretable* decision that (a) needs no search, (b) is computed from inputs a compiler already has after a single codegen pass, (c) explains its verdict, and (d) **ports across vendors** — including to domestic GPUs whose compiler ecosystems lack mature fusion tooling and whose microarchitecture (e.g., MetaX C500's split scalar/vector register file and hard 4 KB/thread spill cap) makes the "toxic fusion" cliff sharper and closer than on NVIDIA.

---

## 2. Related work & honest positioning

| System (venue) | What it does | What it does **not** do (our room) |
|---|---|---|
| **Welder** (OSDI'23) | Tile‑graph + tile‑traffic cost model; auto fusion+tiling; trades intra/inter‑op reuse | Search/schedule‑based; not interpretable; NVIDIA‑only; no compile‑time attribution |
| **SpaceFusion / ++** (EuroSys'25/'26) | Space‑Mapping Graph models inter+intra‑op spatial deps to schedule fusion | Scheduler, not a *reject* criterion; not cross‑vendor; not an interpretable penalty budget |
| **Hidet** (ASPLOS'23) | Task‑mapping, register‑level control, post‑scheduling fusion | Programming paradigm, not a predictive fusion‑profitability model |
| **Neptune** (2025) | Advanced attention fusion w/ algebraic repair; empirical autotuning | Search‑based; no static microarch penalty model |
| **TpuGraphs / learned TPU model** (NeurIPS'23) | GNN predicts tile/fusion perf to prune autotuning | Opaque; heavy per‑hardware training; TPU |
| **Mirage** (OSDI'25) | Multi‑level superoptimizer over μGraphs incl. fusion | Search + equivalence verification; not a lightweight static pass |
| **Microbench‑driven analytical models** (2026) | Analytical latency/occupancy incl. register pressure for Blackwell/CDNA3 | Per‑kernel perf modeling, **not** a fusion go/no‑go criterion; not cross‑vendor‑as‑transfer |

**Net:** the *outcome* (avoid bad fusions) is achieved by search‑based systems. What is **not** in the literature is a **search‑free, interpretable, cross‑vendor fusion‑rejection criterion** driven by single‑compile static inputs, and there is **no published fusion characterization of the MetaX C500 at all**. That is our defensible contribution surface — modest but real ("a bit innovative and useful").

---

## 3. Problem statement & research questions

**RQ1 (Predictability).** Can a fusion's net profitability be predicted at compile time from *statically obtainable* inputs (per‑kernel register/occupancy from one compile + graph‑level tile/layout descriptors), without autotuning or profiling?

**RQ2 (Interpretability).** Can we attribute a "don't fuse" verdict to a dominant microarchitectural cause (occupancy/spill vs layout/bank‑conflict), and does that attribution match what a profiler measures?

**RQ3 (Transfer).** Does *one* model structure, re‑parameterized per device, predict fusion profitability on both NVIDIA Ada and MetaX C500 — and does it correctly identify at least one **decision‑flip** case (a fusion safe on one vendor and toxic on the other)?

**RQ4 (Utility).** As a static pruning pass over real subgraphs (attention, MLP, LayerNorm→GEMM, MoE expert), does it beat greedy‑always‑fuse and the compiler default, approaching an autotuned oracle — at ~zero search cost?

---

## 4. Key idea & contributions

1. **A search‑free, interpretable fusion‑degradation model** `η_fused = min(η_u, η_v) · P_occ · P_layout`, where every input is obtained from a *single* codegen/compile pass plus static graph metadata — no autotuning, no profiling in the deployed pass.
2. **A single‑compile static‑input recipe** that resolves the classic "you can't predict registers statically" objection: we do not *predict* registers, we *read them* from one compiler resource report (`nvcc --resource-usage` / `cucc -resource-usage`, which also emits static max‑warps/occupancy).
3. **A cross‑vendor instantiation and the first fusion characterization of the MetaX C500**, showing the model transfers by re‑parameterization and exposing vendor‑dependent fusion decisions (split scalar/vector RF + 4 KB/thread spill cap ⇒ a sharper occupancy cliff than Ada).
4. **A prototype static pruning pass** (Triton/PyTorch‑Inductor) plus an interpretability report (per‑edge dominant‑penalty attribution).

---

## 5. Methodology

### 5.1 Pattern taxonomy Φ(v)
Each op node annotated with `Φ(v) = ⟨C_v, I_v, T_v, L_v⟩`: topological class (Pointwise / Broadcast / Reduction / Contraction / Permute), operational intensity, preferred tile geometry, layout/alignment. A fusion edge `u→v` is a *candidate* only if tile/layout compatibility is satisfiable; otherwise a transpose/repack cost is charged to `P_layout`.

### 5.2 Cost model
- Unfused: `T_unfused = Σ max(W_i/(C·η_i), M_i/B) + 2·T_launch`
- Fused:   `T_fused   = max((W_u+W_v)/(C·η_fused), (M_in+M_out)/B) + T_sync`
- **Degradation coefficient:** `η_fused = min(η_u, η_v) · P_occ · P_layout`
  - **P_occ** = `ActiveWarps(RF_fused, SMEM_fused) / MaxWarpsPerSM` — occupancy after fusion, *plus* a spill discontinuity when `RF_fused` forces local‑memory spilling (on C500, a hard cliff at the 4 KB/thread private‑memory cap).
  - **P_layout** = `1 − (T_transpose + T_bank_conflict)/T_compute` — cost of reconciling incompatible thread‑data mappings (e.g., reduction row‑major block vs GEMM 128×128 tile).
- **Decision rule:** prune the fusion edge iff `T_fused > T_unfused`. Report the multiplicatively dominant penalty as the *reason*.

### 5.3 Static inputs (the crux — how we stay search‑free)
| Quantity | NVIDIA Ada | MetaX C500 |
|---|---|---|
| Registers/thread (fused & unfused) | `nvcc -Xptxas -v` / `--resource-usage` | `cucc -resource-usage` → `MTregisters`+`STregisters` |
| Static occupancy | CUDA Occupancy API / `--resource-usage` | `staticMaxWarps/PEU` (printed by `-resource-usage`) |
| Shared‑mem / tile bytes | codegen | codegen |
| Spill cliff | RF > 255 or local‑mem > 0 | private mem > **4 KB/thread** ⇒ `mcErrorMemoryValueTooLarge` |
The model consumes these + Φ(G) — all available after **one compile of the fused candidate**. Ground truth for *fitting/validation only* comes from profiling (§5.5), never from the deployed pass.

### 5.4 Compiler integration
Prototype as a static pruning hook. Simplest viable path: a pass over the Inductor/`torch.compile` scheduler (or a standalone Triton‑level pass) that, for each candidate fusion, compiles the fused kernel once, evaluates `η_fused`, and vetoes net‑harmful edges. Fallback deliverable if deep Inductor integration is too costly: an *offline recommender* that scores a fusion plan and reports edits + attributions.

### 5.5 Cross‑vendor instantiation & validation
Same code/kernels compiled on both vendors. Fit per‑device constants (`C_peak`, `B_peak`, `MaxWarps`, spill cap, register‑file model). Validate predicted `η_fused` and the fuse/don't‑fuse decision against **profiled** ground truth:
- **Ada:** Nsight Compute (`ncu`) — achieved occupancy, registers/thread, local‑mem spills, shared bank conflicts, DRAM throughput.
- **C500:** MCPTI (preferred — CUPTI‑compatible field names) or `mcProfiler`; metrics `WAVES`/`Achieved waves`, `average conflict cycles per instruction`, `AP MMA/busy Duty`, `Global/Dnoc/L2C` traffic. (See `tooling/` and memory `metax-c500-profiling`.)

---

## 6. Evaluation plan

**Datasets.** (a) *Microbench matrix*: producer→consumer op pairs spanning the taxonomy (pointwise↔reduction, broadcast↔GEMM‑epilogue, LayerNorm/Softmax↔GEMM, permute↔contraction), each in fused/unfused variants, swept over shapes/dtypes. (b) *Real subgraphs*: attention block, MLP/FFN, LayerNorm+Linear, one MoE expert.

**Baselines.** greedy‑always‑fuse; Inductor default fusion; autotuned oracle (upper bound); ablations (drop P_occ, drop P_layout).

**Metrics.**
- Decision quality: precision/recall/F1 of *toxic‑fusion* detection vs profiled ground truth.
- End‑to‑end: latency vs baselines (target: beat greedy & default; approach oracle at ~0 search cost).
- Transfer: fit‑on‑Ada → evaluate‑on‑C500 accuracy; ≥1 documented decision‑flip.
- Interpretability: fraction where predicted dominant penalty == profiled dominant penalty.
- Cost: compile‑time overhead of the pass (should be ≪ autotuning).

---

## 7. Hardware & tooling status

- **Ada (sm89)** — *primary; on a separate machine.* Main research runs there. Setup checklist in `HANDOFF.md`.
- **MetaX C500 ×4 (this machine)** — *validated & ready for the transfer phase.* MACA 3.7.0; `cucc`/`mxcc` compile CUDA; `-resource-usage` gives MT/ST registers + static max‑warps; `mcProfiler`/MCPTI expose occupancy, bank‑conflict cycles, MMA util, DRAM traffic (NVIDIA‑CUPTI‑identical names at the MCPTI/API layer); `torch 2.8.0+metax` + `triton 3.0.0` across all 4 GPUs. Known gotcha: `mcProfiler perf_exec` value‑dump is metric‑group‑sensitive — prefer MCPTI directly. Reproduce with `bash tooling/check_profiling_stack.sh`.

---

## 8. Workplan / TODO (problem → paper)

Legend: **[ADA]** = do on the Ada machine's Claude Code session · **[MX]** = do here (MetaX) · **[EITHER]**.

### Phase 0 — Environment validation ✅ DONE (this machine)
- [x] Confirm NVIDIA tools absent; MetaX MACA stack present & functional
- [x] Verify compile (`cucc`), static registers (`-resource-usage`), profiler (`mcProfiler`/MCPTI)
- [x] Confirm torch+triton on C500; deliver `tooling/check_profiling_stack.sh`

### Phase 1 — Ada bring‑up & ground‑truth harness  **[ADA]** ✅ DONE (see logs/LOG-01, LOG-02)
- [x] Set up Ada env: torch/Triton/`nvcc`/`ncu` in conda env `profiling`; **ncu counters work under WSL2** (`--target-processes all`); nvcc host-compiler pinned to conda gcc-11 (`tooling/env.sh`)
- [x] `tooling/check_profiling_stack_nvidia.sh` (NVIDIA analogue, PASS=11/0/0); `-Xptxas -v` register/spill/occupancy captured; Triton `n_regs`/`n_spills` as the single-compile static input
- [x] Microbench matrix in Triton (families P pointwise, R sibling-reduction) + raw CUDA (family T transpose/bank-conflict)
- [x] Automated `ncu` → vendor-neutral dataset (`data/microbench_timing.csv` 88 rows; 72 genuine + 16 degenerate no-ops excluded from scoring, plus `data/microbench_ncu.csv`), with `beneficial?` + `dominant_penalty`
- **DoD:** ✅ reproducible dataset with per-fusion ground-truth label + dominant penalty

### Phase 2 — Model formalization & fit  **[ADA]** ✅ DONE (LOG-02, LOG-03)
- [x] Φ(v) taxonomy + tile/layout descriptors (`fusion/kernels/base.py`); analytical sm89 occupancy (`fusion/hw.py`) reproduces ncu *theoretical* occupancy exactly (the CUDA occupancy calculator; *achieved* occupancy is handled by the spill term, not predicted)
- [x] `η_fused = min·P_occ·P_layout` from single-compile inputs (`model/costmodel.py`); `P_occ` (occupancy+spill), `P_layout` (bank-conflict) calibrated on Ada
- [x] **RQ1 (72 genuine cases): recall=1.00 in every CV; F1 0.91 shape-CV / 1.00 leave-one-NOUT-out / 0.91 leave-one-dtype-out**; **RQ2a occupancy MAE=0.000 vs ncu theoretical**; **RQ2b attribution 12/12**
- [x] Ablations: drop-spill F1→0.55 (spills dominant), drop-occupancy F1=0.91; vs greedy F1=0.00
- **DoD:** ✅ recall=1.00 in every CV; F1 robust to held-out NOUT (1.00) and held-out dtype (0.91), on the 72 genuine cases; attribution matches profiler

### Phase 3 — Compiler pass & end‑to‑end  **[ADA]** ✅ DONE (LOG-03)
- [x] Offline recommender (`model/recommender.py`) — the PROPOSAL §5.4 sanctioned fallback; decides from single-compile static resource reports
- [x] End-to-end on 3 subgraphs vs greedy/oracle/none (`fusion/endtoend.py`): **model up to 9.85× vs greedy** (a constructed worst-case for greedy — subgraphs built with wide over-fused layers), **= oracle on fp16**, **3.03× off oracle on fp32** (honest limit); decide wall ≤26 ms (compiles only, no timing)
- [x] Frozen model spec (`model/MODEL_SPEC.md`) + fitted constants (`model/ada_constants.json`)
- **DoD:** ✅ end-to-end speedup on ≥2 subgraphs; frozen spec ready for MX phase

### Phase 4 — Cross‑vendor transfer to C500  **[MX]** *(user returns to this session)*
- [ ] Re‑parameterize constants for C500 (RF model, spill cap, `MaxWarps`, `C_peak`, `B_peak`)
- [ ] Re‑run microbench matrix on C500 (Triton + `cucc`); collect ground truth via **MCPTI**
- [ ] Validate transfer (RQ3); find & analyze ≥1 **decision‑flip** case (safe on Ada, toxic on C500 or vice‑versa) — likely driven by the 4 KB spill cliff or the MT/ST split
- [ ] Cross‑vendor generalization table
- **DoD:** transfer accuracy reported; ≥1 documented, explained decision‑flip

### Phase 5 — Write‑up & artifact  **[EITHER]**
- [ ] Draft paper (intro/related/method/eval); build all figures/tables
- [ ] Reproducibility artifact (scripts, datasets, model code)
- [ ] Internal adversarial review of every claim; target venue submission
- **DoD:** submittable paper + artifact

---

## 9. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Novelty too thin vs Welder/SpaceFusion | Med‑High | Lean on the two things they lack: **interpretable attribution** + **cross‑vendor transfer incl. a domestic GPU**; frame as workshop/short‑paper‑scale honestly |
| Static register report misestimates real fused RF | Med | We *read* registers from the actual fused compile (not predict); validate P_occ vs profiled occupancy |
| `P_layout` hard to model cleanly | Med | Start empirical (fit transpose/bank‑conflict cost vs tile mismatch), then simplify |
| Inductor integration too deep for timeline | Med | Offline‑recommender fallback still yields RQ1‑RQ4 results |
| MetaX MCPTI value extraction friction | Low‑Med | Already scoped; use MCPTI API not the CLI; `tooling/` documents the path |
| Ada & C500 "too similar/too different" to tell a transfer story | Low | The MT/ST split + 4 KB spill cap guarantee a structural contrast → the decision‑flip narrative |

---

## 10. Expected contributions & target venue

**Contributions:** (1) a search‑free, interpretable fusion‑rejection criterion with single‑compile inputs; (2) the first fusion characterization + cost‑model transfer to the MetaX C500; (3) a working static pruning prototype with penalty attribution; (4) an open dataset + artifact.

**Honest framing:** an *empirical + systems* contribution, not a theoretical breakthrough. Realistic homes: **MLSys / a compiler or systems‑for‑ML workshop**, or an arXiv report + thesis chapter. The domestic‑GPU cross‑vendor angle is the differentiator that a Hopper‑rich lab would not produce.

*Pointers:* `HANDOFF.md` (Ada onboarding) · `tooling/` (env checks) · memory `project-fusion-cost-model`, `metax-c500-profiling`, `research-scoping-decisions`.
