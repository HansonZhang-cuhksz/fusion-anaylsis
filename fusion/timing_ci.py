"""timing_ci.py -- statistical rigor for the key fuse/don't-fuse claims (G5).

The results elsewhere use point-estimate (min-of-N) timings; one fp32 measurement artifact was caught
earlier (the A1 case). This re-measures the decisive claims with REPEATED independent rounds and reports
the speedup's median + 95% CI, flagging whether the CI excludes 1.0 (i.e. beneficial/toxic is
statistically significant, not noise). Run on 4 independent C500 GPUs for cross-device reproducibility.

Usage: MACA_VISIBLE_DEVICES=<g> FUSION_HW=c500 python -m fusion.timing_ci [--rounds 20] [--out f.csv]
"""
from __future__ import annotations
import argparse, csv, os, warnings
import numpy as np, torch
warnings.filterwarnings("ignore")
from fusion.kernels import reduction, gemm_epilogue
from fusion.timing import time_ms

# (label, builder) -- the claims RESULTS.md rests on.
CLAIMS = [
    ("redux_N32_fp16", lambda: reduction.make_case(2048, 2048, 32, torch.float16, GS=16)),
    ("redux_N32_fp32", lambda: reduction.make_case(2048, 2048, 32, torch.float32, GS=16)),
    ("redux_N64_fp32", lambda: reduction.make_case(2048, 2048, 64, torch.float32, GS=16)),
    ("gemm_128_fp16", lambda: gemm_epilogue.make_case(2048, 2048, 512, torch.float16, BM=128, BN=128)),
    ("gemm_128_fp32", lambda: gemm_epilogue.make_case(2048, 2048, 512, torch.float32, BM=128, BN=128)),
]

# Expected verdicts are DEVICE-SPECIFIC: the two fp32 claims are exactly the cross-vendor
# decision-flips (C500's 64-wide wavefronts double register pressure -> it spills where Ada does
# not). Selected by --expect / $FUSION_HW; c500 is the default so the committed C500 run reproduces.
EXPECTED = {
    "c500": {"redux_N32_fp16": "beneficial", "redux_N32_fp32": "TOXIC(flip)",
             "redux_N64_fp32": "TOXIC", "gemm_128_fp16": "beneficial", "gemm_128_fp32": "TOXIC"},
    # Ada: 0 spills on both fp32 claims -> both flip to beneficial (see data/microbench_gemm_ada.csv,
    # data/microbench_timing.csv). redux_N64 still spills (310) on Ada -> stays toxic.
    "ada":  {"redux_N32_fp16": "beneficial", "redux_N32_fp32": "beneficial(flip)",
             "redux_N64_fp32": "TOXIC", "gemm_128_fp16": "beneficial",
             "gemm_128_fp32": "beneficial(flip)"},
}


def speedup_rounds(run_fused, run_unfused, rounds, iters):
    sp = []
    for _ in range(rounds):
        tf = time_ms(run_fused, warmup=8, iters=iters)["ms_min"]
        tu = time_ms(run_unfused, warmup=8, iters=iters)["ms_min"]
        sp.append(tu / tf)
    return np.array(sp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20); ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--expect", default=None, choices=["ada", "c500"],
                    help="which device's expected verdicts to check against; default follows "
                         "$FUSION_HW (unset => ada, matching hw.default_hw())")
    a = ap.parse_args()
    dev = a.expect or ("c500" if os.environ.get("FUSION_HW", "").lower() in ("c500", "metax", "mx")
                       else "ada")
    exp_map = EXPECTED[dev]
    print(f"[timing_ci] expectations for device={dev}", flush=True)
    rows = []
    for label, build in CLAIMS:
        exp = exp_map[label]
        case = build()
        sp = speedup_rounds(case.run_fused, case.run_unfused, a.rounds, a.iters)
        med = float(np.median(sp)); lo, hi = np.percentile(sp, [2.5, 97.5])
        sig = "sig" if (hi < 1.0 or lo > 1.0) else "NS"          # CI excludes 1.0?
        verdict = "TOXIC" if hi < 1.0 else ("beneficial" if lo > 1.0 else "ambiguous")
        rows.append({"claim": label, "device": dev, "expected": exp, "median_speedup": round(med, 3),
                     "ci_lo": round(float(lo), 3), "ci_hi": round(float(hi), 3),
                     "verdict": verdict, "significant": sig, "rounds": a.rounds})
        print(f"{label:16s} exp={exp:12s}: median={med:.3f} 95%CI=[{lo:.3f},{hi:.3f}] "
              f"-> {verdict} ({sig})", flush=True)
    if a.out:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
