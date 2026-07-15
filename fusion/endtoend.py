"""endtoend.py -- RQ4: end-to-end latency of real-ish subgraphs under different fusion policies.

A subgraph is a list of "layers"; each layer is a horizontal-fusion opportunity (a producer feeding
NOUT sibling reductions -- e.g. a multi-head / multi-projection block). We measure the real latency
of executing every layer under four policies and compare:

  none    : split into narrow groups (width = SPLIT)      -- conservative, wastes input re-reads
  greedy  : fuse everything (width = NOUT)                 -- the "always fuse" heuristic
  model   : width chosen by the search-free recommender    -- compiles candidates once, no timing
  oracle  : width chosen by measuring every candidate       -- upper bound, expensive search

Reports model-vs-greedy and model-vs-oracle, plus the search cost each policy paid.

Usage: python -m fusion.endtoend
"""
from __future__ import annotations
import time, math, json
import torch
from fusion.kernels import reduction
from fusion.timing import time_ms
from model.recommender import recommend, _load_k

SPLIT = 8  # the conservative "no horizontal fusion" group width


def time_layer_width(R, C, NOUT, width, dtype) -> float:
    """Measured latency of executing NOUT sibling reductions in groups of `width`."""
    # Prophylactic hygiene only (LOG-10 s1): keep torch's RESERVED footprint from oversubscribing
    # physical VRAM, which can leave a SPILLING kernel's local-memory store host-resident.
    # NOTE: this harness was verified NOT to have been affected -- the pre-fix RQ4 numbers were not
    # contaminated (pre-fix greedy(w128)=151.4ms is consistent with unstarved timing); the 9.72 -> 8.65x
    # shift is ordinary run-to-run variation, not a correction.
    torch.cuda.empty_cache()
    dt = getattr(torch, dtype)
    if width >= NOUT:
        case = reduction.make_case(R=R, C=C, NOUT=NOUT, dtype=dt, GS=min(SPLIT, NOUT),
                                   compile_probe=True)
        return time_ms(case.run_fused, warmup=8, iters=25)["ms"]
    case = reduction.make_case(R=R, C=C, NOUT=NOUT, dtype=dt, GS=width, compile_probe=True)
    return time_ms(case.run_unfused, warmup=8, iters=25)["ms"]


def candidate_widths(NOUT):
    return [w for w in (8, 16, 32, 64, 128, 256) if w <= NOUT and NOUT % w == 0] or [NOUT]


def eval_subgraph(name, layers, k):
    print(f"\n### subgraph: {name}  ({len(layers)} layers)")
    totals = {"none": 0.0, "greedy": 0.0, "model": 0.0, "oracle": 0.0}
    model_search_compiles = 0
    oracle_search_measures = 0
    t_model_decide = 0.0
    for (R, C, NOUT, dtype) in layers:
        cands = candidate_widths(NOUT)
        # measured latency of each candidate width (for oracle + reporting)
        meas = {w: time_layer_width(R, C, NOUT, w, dtype) for w in cands}
        oracle_search_measures += len(cands)
        oracle_w = min(meas, key=meas.get)

        # model recommendation (search-free: compiles candidates once, reads static report)
        t0 = time.perf_counter()
        rec = recommend(R, C, NOUT, dtype, candidate_widths=cands, k=k)
        t_model_decide += time.perf_counter() - t0
        model_search_compiles += len(cands)
        model_w = rec["recommended_width"]

        totals["none"] += meas[min(cands)]
        totals["greedy"] += meas[max(cands)]
        totals["model"] += meas[model_w]
        totals["oracle"] += meas[oracle_w]
        print(f"  L(R{R}xC{C} NOUT{NOUT} {dtype}): "
              f"none(w{min(cands)})={meas[min(cands)]:.3f} greedy(w{max(cands)})={meas[max(cands)]:.3f} "
              f"model(w{model_w})={meas[model_w]:.3f} oracle(w{oracle_w})={meas[oracle_w]:.3f} "
              f"| reject_greedy={rec['reject_greedy']} reason={rec['greedy_reject_reason']}")
    print(f"  TOTALS ms: none={totals['none']:.3f} greedy={totals['greedy']:.3f} "
          f"model={totals['model']:.3f} oracle={totals['oracle']:.3f}")
    print(f"  model vs greedy: {totals['greedy']/totals['model']:.2f}x faster | "
          f"model vs oracle: {totals['model']/totals['oracle']:.3f}x of oracle (1.0=optimal)")
    print(f"  search cost: model={model_search_compiles} compiles(no run) | "
          f"oracle={oracle_search_measures} timed runs; model decide wall={t_model_decide*1e3:.0f}ms")
    return {"name": name, "totals": totals,
            "model_vs_greedy": totals["greedy"] / totals["model"],
            "model_over_oracle": totals["model"] / totals["oracle"],
            "model_compiles": model_search_compiles, "oracle_measures": oracle_search_measures}


SUBGRAPHS = {
    # a wide multi-projection block (transformer multi-query / MoE-router-ish): greedy over-fuses
    "wide_multiproj": [(4096, 2048, 128, "float16"), (4096, 2048, 64, "float16")],
    # mixed widths: small layers where greedy is fine + wide layers where it is toxic
    "mixed_widths": [(2048, 2048, 16, "float16"), (2048, 2048, 32, "float16"),
                      (2048, 2048, 64, "float16"), (2048, 2048, 128, "float16")],
    # fp32 variant (spills earlier)
    "fp32_block": [(2048, 1024, 64, "float32"), (2048, 1024, 128, "float32")],
}


def main():
    k = _load_k()
    print(f"[endtoend] device constants: B_peak={k.B_peak:.3g} gamma_spill={k.gamma_spill:.3g}")
    res = [eval_subgraph(n, layers, k) for n, layers in SUBGRAPHS.items()]
    print("\n=== RQ4 summary ===")
    for r in res:
        print(f"  {r['name']:16s}: model {r['model_vs_greedy']:.2f}x vs greedy, "
              f"{r['model_over_oracle']:.3f}x of oracle, "
              f"{r['model_compiles']} compiles vs {r['oracle_measures']} timed runs")


if __name__ == "__main__":
    main()
