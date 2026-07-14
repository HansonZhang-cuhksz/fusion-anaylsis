"""recommender.py -- the search-free offline fusion recommender (PROPOSAL §5.4 fallback deliverable).

Given a horizontal-fusion opportunity (a producer feeding NOUT sibling consumers), decide how many
to fuse per kernel. It compiles each candidate grouping ONCE, reads the static resource report
(registers / spills / shared → analytical occupancy), scores it with the cost model, and picks the
best grouping — plus a per-decision attribution. No autotuning, no timing at decision time.

Greedy-always-fuse == the single widest group (all NOUT in one kernel); on Ada this spills and is
catastrophic at large NOUT. The recommender vetoes that and emits the edit + reason.
"""
from __future__ import annotations
import json, math
import torch
from fusion.kernels import reduction
from fusion.static import from_triton
from model.costmodel import DeviceConstants, predict_time, decide


def _load_k(path="model/ada_constants.json") -> DeviceConstants:
    try:
        with open(path) as f:
            return DeviceConstants(**json.load(f))
    except FileNotFoundError:
        return DeviceConstants()


def score_grouping(R, C, NOUT, width, dtype, k: DeviceConstants) -> dict:
    """Compile ONE width-`width` sibling kernel, read static inputs, predict the full-plan time."""
    dt = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    # build a width-sized case to obtain the compiled kernel's static inputs (single compile)
    case = reduction.make_case(R=R, C=C, NOUT=width, dtype=dt, GS=width,
                               BLOCK_R=32, BLOCK_C=32)
    sf = from_triton(case.fused_kernels[0])
    n_groups = math.ceil(NOUT / width)
    isz = torch.tensor([], dtype=dt).element_size()
    # one width-kernel: reads X once, its slice of W, writes its outputs
    per_flops = R * C * width * 2
    per_bytes = R * C * isz + width * C * isz + R * width * 4
    per = predict_time(per_flops, per_bytes, sf.occupancy, sf.n_spills, n_launch=1, k=k)
    total_t = n_groups * per["t"]           # groups run sequentially
    return {"width": width, "n_groups": n_groups,
            "regs": sf.n_regs, "spills": sf.n_spills, "occ": round(sf.occupancy, 3),
            "pred_t": total_t, "pred_per_kernel": per["t"],
            "p_occ": per["p_occ"], "p_occ_spill": per["p_occ_spill"]}


def recommend(R, C, NOUT, dtype="float16", candidate_widths=None,
              k: DeviceConstants = None) -> dict:
    k = k or _load_k()
    if candidate_widths is None:
        candidate_widths = [w for w in (8, 16, 32, 64, 128, 256) if w <= NOUT and NOUT % w == 0]
        if NOUT not in candidate_widths:
            candidate_widths.append(NOUT)
    scored = [score_grouping(R, C, NOUT, w, dtype, k) for w in sorted(set(candidate_widths))]
    best = min(scored, key=lambda s: s["pred_t"])
    greedy = max(scored, key=lambda s: s["width"])   # all-fused
    # attribution for the greedy (widest) group if the recommender rejects it
    reject_greedy = best["width"] != greedy["width"]
    reason = None
    if reject_greedy:
        reason = ("spill" if greedy["spills"] > 0 else
                  "occupancy" if greedy["occ"] < best["occ"] else "none")
    return {"R": R, "C": C, "NOUT": NOUT, "dtype": dtype,
            "recommended_width": best["width"], "greedy_width": greedy["width"],
            "reject_greedy": reject_greedy, "greedy_reject_reason": reason,
            "pred_speedup_vs_greedy": round(greedy["pred_t"] / best["pred_t"], 3),
            "candidates": scored, "best": best, "greedy": greedy}


if __name__ == "__main__":
    for NOUT in (16, 64, 128, 256):
        r = recommend(2048, 2048, NOUT, "float16")
        print(f"NOUT={NOUT:3d}: recommend width={r['recommended_width']:3d} "
              f"(greedy={r['greedy_width']}) reject_greedy={r['reject_greedy']} "
              f"reason={r['greedy_reject_reason']} pred_speedup_vs_greedy={r['pred_speedup_vs_greedy']}")
