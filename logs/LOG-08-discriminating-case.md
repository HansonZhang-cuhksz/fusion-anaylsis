# LOG-08 — Hunting a *discriminating* real-compiler case (honest negative + reframing)

Date: 2026-07-15 · Machine: MetaX C500, env `fusion` · Follows LOG-07 §6.

## Goal
LOG-07 §6 showed the model can read Inductor's generated kernels and predict its fusion outcomes — but
non-discriminatingly (elementwise fusion ~always beneficial ⇒ "predicts beneficial 5/5" is a weak test).
The goal here: find a case where a REAL compiler (Inductor) emits a **toxic** Triton fusion, so the
model's prediction is non-trivial.

## Attempt: forced recomputation (`fusion/inductor_toxic_probe.py`)
A fusing compiler can be net-harmful by **recomputing** an expensive shared producer for each consumer.
We forced this (`torch._inductor.config.realize_reads_threshold/opcount_threshold` set very high so
Inductor never materializes the shared intermediate) with an expensive producer (2·D sin/cos ops)
feeding `fanout` consumers, and measured eager vs `torch.compile`, capturing the fused kernel's static
and scoring it with the model.

## Result — Inductor's Triton fusion is robustly beneficial (no tractable toxic case)
| tensor | D, fanout | fused regs/spill | eager→compiled | verdict |
|---|---|---|---|---|
| 2048² (memory-bound) | 8, 8 | 76 / 0 | 0.98→0.27 ms | **3.6× beneficial** |
| 256² (compute/launch-bound) | 8, 8 | 48 / 0 | 0.47→0.04 ms | **11× beneficial** |

Fusion **wins in every regime**: memory-bound → it saves the HBM round-trips; small/launch-bound → it
collapses ~(D+fanout) eager kernel launches into ONE (launch-overhead saving). To make the recomputed
compute dominate *both* of those savings needs a huge D×fanout, which makes Inductor generate an
enormous fused kernel that **does not compile in tractable time** (D≥32 hangs the codegen). So on a
well-engineered, autotuning production compiler, a toxic Triton fusion is **very hard to trigger**.

## Interpretation (reframes the model's utility — honestly)
- The model's **discriminating** power (catching a net-harmful fusion) is real but is demonstrated on
  **controlled kernels**: the CONTRACTION tile sweep (LOG-05) where a specific fp32 128×128 tile spills
  and fusion goes toxic (0.78×) — which the reread-aware model catches — and the cross-vendor
  **decision-flip** (LOG-04). A naive/greedy fuser or an autotuner's *candidate* kernels would hit these.
- On a real, well-tuned compiler (Inductor) the model **agrees** but has little to correct, because
  Inductor's fusion heuristics + autotuning already avoid the toxic tiles. So the model's clearest value
  is for **(a) the register-spill cliff on specific kernels/tiles, and (b) weaker / immature compilers —
  notably domestic-GPU stacks (the C500's own compiler) that lack Inductor-grade tuning.** That is
  exactly the paper's cross-vendor / domestic-accelerator thesis.
- This is an **honest negative** that sharpens the contribution rather than weakening it: the search-free
  check is a cheap safety net most useful precisely where a mature autotuning compiler is absent.

## Status
`fusion/inductor_toxic_probe.py` kept as the harness. Not pursuing giant-kernel recompute further
(uncompilable). A cleaner future discriminating case: intercept an *autotuner's candidate* tiles (before
it discards the spilling one) and show the model would prune it — or run against the C500's native
(non-Inductor) fuser.
