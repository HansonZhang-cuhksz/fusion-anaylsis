# Author notes for `toxic-fusion-measurement.md` (read before submitting)

This draft was assembled from the repository's verified artifacts (RESULTS.md, LOG-01…14, the data
CSVs, and the code). It is written to be **factually faithful to what the repo actually shows** and to
foreground the honest/negative findings rather than hide them. Two classes of things need your action
before this is submission-ready.

## 1. Citations — verified, but do a final citation-manager pass
The reference list was initially generated from the model's background knowledge, then **independently
web-verified in the fact-check pass**: all 12 references [1]–[12] came back REAL with title/venue/year
correct (e.g. [1] TVM OSDI'18 dblp ChenMJZYSCWHCGK18; [2] Ansor OSDI'20 pp.863–879; [3] PyTorch 2
ASPLOS'24 pp.929–947; [5] Welder OSDI'23 pp.701–718; [6] AStitch 10.1145/3503222.3507723; [7] Hidet
10.1145/3575693.3575702; [8] Roofline CACM 52(4) 2009). No hallucinated or misattributed entries were
found. Still: run one citation-manager pass to attach DOIs and full author lists before submission
(standard practice, not a correctness worry now). Two citation-*usage* imprecisions were flagged and
already fixed in the draft: [5] Welder is a scheduling/cost-model compiler, not an autotuner, so §1 now
cites only [2] for autotuning search; [10] TpuGraphs is a dataset/benchmark, not itself a learned model,
so §1/§2 now describe it as such.
The proposal (`PROPOSAL.md` §2) lists further related work the author intended to cite (SpaceFusion,
Neptune, Mirage); I omitted those because I could not verify their metadata — add them back only with
verified citations.

## 2. Every quantitative claim and its source (spot-check against these)
All numbers below are traceable to repo artifacts; a fact-check pass (workflow `verify`) cross-checked
them. If you change any underlying result, update the paper.
- Ada RQ1 F1=1.000, TP16/FP0/FN0/TN48; zero-param spill rule also 1.000; leave-one-NOUT-out model 0.667
  vs rule 1.000; leave-one-dtype-out 0.968 — RESULTS.md RQ1; LOG-02/03/10.
- Ablations drop-spill→0.000, drop-occupancy→1.000 — RESULTS.md RQ1.
- C500 combined in-sample F1 0.873 (0.852 without the dtype-compute term), leave-one-dtype-out 0.906,
  trivial rule 0.857, 7/8 discriminating wrong — LOG-11; `model/c500_combined_constants.json`.
- RQ2: Ada occupancy 22/22 (theoretical), attribution 12/12 (8 spill + 4 layout); C500 MCPTI 12/12
  spill, local≫DRAM ≈307× — RESULTS.md RQ2; LOG-03/04.
- RQ3: C500 reduction transfer F1 0.909; flips reduction N32 fp32 (Ada 1.04 / C500 0.64) and GEMM128
  fp32 (Ada 1.08 / C500 0.82); reg inflation 84→134, 96→168; num_warps=8 → 0 spills, reduction 1.080 /
  GEMM 1.206; model tracks 4/5 configs — LOG-04/12/13; `data/flip_tunable_c500.csv`.
- RQ4: end-to-end 5.81–8.65× vs greedy (constructed), =oracle on 3 subgraphs; Inductor sign 5/5;
  toxic-Inductor honest negative — RESULTS.md RQ4; LOG-07/08.
- GEMM sign/precision (reread): recall 0→1.0, precision residual 0.5, MCPTI 950K vs 833K, fp16 +6.5% —
  RESULTS.md cross-cutting; LOG-05/14.
- CIs: reduction flip [0.638,0.645] (4 GPUs); Ada N32 fp32 [1.040,1.041]; GEMM128 fp32 [1.072,1.104];
  within-process-noise caveat + run-to-run non-overlap — RESULTS.md statistics; LOG-09/10.
- VRAM artifact: 9–41× inflation of spilling-kernel fused times, corrected; C500 excluded — RESULTS.md
  artifact section; LOG-10.

## 3. Framing choices (deliberate, defensible, but yours to accept)
- The paper is scoped as a **measurement study**, not a "novel cost model," per the honest assessment
  that every headline was de-risked to "modest." If you target a full conference, you need a new
  load-bearing result (a real scheduler integration, or a genuine non-spill interpretability case);
  neither exists yet.
- The **config-artifact** finding (§5.3) is presented as the sharpest result and as an *asset*
  (config-dependent fusion signal for immature toolchains), not buried as a caveat. This is the honest
  reframe that makes the cross-vendor story defensible.
- Numbers within ~5% of 1.0 are treated as unresolved (the paper says so); do not over-claim any such
  verdict as significant.

## 4. Still missing for a complete artifact
- Figures/plots (spill cliff, decision-flip, config sweep) — the data exist (`figures/` has some PNGs;
  `data/flip_tunable_c500.csv`, `data/microbench_*`); regenerate camera-ready versions.
- A reproducibility appendix mapping each claim → command (the pointers are in RESULTS.md/logs).
- sm80 third hardware point remains blocked (no such GPU reachable).
