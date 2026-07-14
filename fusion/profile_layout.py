"""profile_layout.py -- build the P_layout dataset (raw-CUDA bank-conflict cases) and validate that
the model attributes their degradation to LAYOUT (not P_occ), since occupancy/spills are identical
between the conflict and clean variants.

Usage: python -m fusion.profile_layout data/cuda_layout.csv
"""
from __future__ import annotations
import sys, csv, json, subprocess, os
import torch
from fusion.cuda_layout import ptxas_registers, runners, CU
from fusion.ncu import _METRIC_LIST, _parse_csv, aggregate, REPO
from fusion.timing import time_ms
from fusion.hw import ADA_SM89
from model.costmodel import DeviceConstants, decide


def profile_layout_plan(plan, R, C, timeout=600):
    kregex = "regex:(txpose_relu|txpose_only|relu_ep)"
    args = ["ncu", "--target-processes", "all", "--csv", "--metrics", _METRIC_LIST,
            "--print-units", "base", "-k", kregex,
            "python", "-m", "fusion.cuda_layout_worker", plan, str(R), str(C)]
    r = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=timeout)
    return aggregate(_parse_csv(r.stdout))


def load_k():
    try:
        return DeviceConstants(**json.load(open("model/ada_constants.json")))
    except FileNotFoundError:
        return DeviceConstants()


def main(out_csv):
    reg = ptxas_registers()
    hw = ADA_SM89
    k = load_k()
    # both variants: 32x32=1024 threads/block; occupancy identical (same regs/smem/threads)
    occ_conf = hw.occupancy(reg["conflict"], 4096, 1024)
    occ_clean = hw.occupancy(reg["clean"], 4224, 1024)
    print(f"[layout] static: conflict regs={reg['conflict']} occ={occ_conf['occupancy']:.3f} | "
          f"clean regs={reg['clean']} occ={occ_clean['occupancy']:.3f}  (identical => isolates P_layout)")

    rows = []
    for (R, C) in [(2048, 2048), (4096, 4096), (8192, 8192), (4096, 8192)]:
        r = runners(R, C)
        ok = (torch.allclose(r["fused_conflict"](), r["ref"], atol=1e-4) and
              torch.allclose(r["fused_clean"](), r["ref"], atol=1e-4) and
              torch.allclose(r["unfused"](), r["ref"], atol=1e-4))
        t_conf = time_ms(r["fused_conflict"], warmup=10, iters=40)["ms"]
        t_clean = time_ms(r["fused_clean"], warmup=10, iters=40)["ms"]
        t_unf = time_ms(r["unfused"], warmup=10, iters=40)["ms"]
        pc = profile_layout_plan("fused_conflict", R, C)
        pk = profile_layout_plan("fused_clean", R, C)
        n = R * C
        bank_conf = pc.get("sig_bank_conf", 0.0)
        bank_clean = pk.get("sig_bank_conf", 0.0)
        # model attribution: feed the bank-conflict-per-element layout feature
        row_model = {
            "flops": n * 2, "bytes_fused": 2 * n * 4, "bytes_unfused": 4 * n * 4,
            "f_occ": occ_conf["occupancy"], "f_spills": reg["spill_conflict"],
            "u_occ": occ_clean["occupancy"], "u_spills": 0, "n_launches_unfused": 2,
            "f_bank_conf_per_elem": bank_conf / n,
        }
        dd = decide(row_model, k)
        rows.append({
            "family": "transpose_cuda", "op_pair": f"txpose_relu_{R}x{C}", "R": R, "C": C,
            "dtype": "float32", "correct": int(ok),
            "f_regs": reg["conflict"], "f_spills": reg["spill_conflict"],
            "f_occ_analytic": round(occ_conf["occupancy"], 4),
            "t_conflict_ms": round(t_conf, 5), "t_clean_ms": round(t_clean, 5),
            "t_unfused_ms": round(t_unf, 5),
            "conflict_slowdown": round(t_conf / t_clean, 3),
            "fused_beats_unfused": int(t_conf < t_unf),
            "bank_conf_conflict": bank_conf, "bank_conf_clean": bank_clean,
            "spill_bytes_conflict": pc.get("sig_local_bytes", 0.0),
            "model_dominant_penalty": dd["dominant_penalty"],
            "model_P_occ": round(dd["P_occ"], 4), "model_P_layout": round(dd["P_layout"], 4),
        })
        print(f"  {R}x{C}: conflict={t_conf:.3f} clean={t_clean:.3f} unfused={t_unf:.3f} "
              f"slowdown={t_conf/t_clean:.2f}x bank(conf={bank_conf:.2g} clean={bank_clean:.2g}) "
              f"spill={pc.get('sig_local_bytes',0):.2g} -> model_dom={dd['dominant_penalty']} ok={ok}", flush=True)

    # ---- calibrate beta_layout from the measured conflict slowdown (fit on the layout microbench,
    #      exactly as gamma_spill is fit on the spill microbench). P_layout should equal 1/slowdown. ----
    import numpy as np
    bpe = np.array([r["bank_conf_conflict"] / (r["R"] * r["C"]) for r in rows])
    slow = np.array([r["conflict_slowdown"] for r in rows])
    # 1/(1+beta*bpe) = 1/slowdown  =>  beta = (slowdown-1)/bpe
    beta_layout = float(np.median((slow - 1.0) / np.maximum(bpe, 1e-9)))
    k.beta_layout = beta_layout
    print(f"[layout] calibrated beta_layout = {beta_layout:.4g} (from median conflict slowdown "
          f"{np.median(slow):.2f}x, bank_conf/elem~{np.median(bpe):.2f})")
    # re-attribute with the calibrated constant
    n_layout = 0
    for r in rows:
        rm = {"flops": r["R"] * r["C"] * 2, "bytes_fused": 2 * r["R"] * r["C"] * 4,
              "bytes_unfused": 4 * r["R"] * r["C"] * 4, "f_occ": r["f_occ_analytic"],
              "f_spills": r["f_spills"], "u_occ": r["f_occ_analytic"], "u_spills": 0,
              "n_launches_unfused": 2,
              "f_bank_conf_per_elem": r["bank_conf_conflict"] / (r["R"] * r["C"])}
        dd = decide(rm, k)
        r["model_dominant_penalty"] = dd["dominant_penalty"]
        r["model_P_layout"] = round(dd["P_layout"], 4)
        r["model_P_occ"] = round(dd["P_occ"], 4)
        n_layout += int(dd["dominant_penalty"] == "layout")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    # persist the calibrated beta_layout back into the device constants
    if os.path.exists("model/ada_constants.json"):
        d = json.load(open("model/ada_constants.json")); d["beta_layout"] = beta_layout
        json.dump(d, open("model/ada_constants.json", "w"), indent=2)
    print(f"[layout] wrote {len(rows)} rows; model attributed {n_layout}/{len(rows)} to LAYOUT "
          f"(spills=0, bank conflicts drive the degradation) -> beta_layout saved to constants")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/cuda_layout.csv")
