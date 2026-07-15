"""fit.py -- fit the cost model's per-device constants on Ada and report RQ1/RQ4 metrics.

- Fits DeviceConstants to measured fused/unfused times (log-space, search-free features only).
- Reports toxic-fusion DECISION quality: precision / recall / F1 vs profiled(=timed) ground truth.
- Held-out-shape cross-validation (RQ1 "held-out shapes").
- Baselines: greedy-always-fuse, and ablations (no-P_occ, no-spill).

Usage: python -m model.fit data/microbench_timing.csv
"""
from __future__ import annotations
import sys, json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from model.costmodel import DeviceConstants, predict_time, decide

PARAM_NAMES = ["C_peak", "B_peak", "T_launch", "occ_knee", "gamma_spill"]
# fit in log-space for the positive scale params; occ_knee is bounded (0,1]
LOG_PARAMS = {"C_peak", "B_peak", "T_launch", "gamma_spill"}


def unpack(theta) -> DeviceConstants:
    d = {}
    for name, val in zip(PARAM_NAMES, theta):
        d[name] = float(np.exp(val)) if name in LOG_PARAMS else float(val)
    # occ_knee physically bounded: memory-bound Ada kernels saturate HBM by ~1/3 occupancy,
    # so occupancy above ~0.35 buys no more latency-hiding. Prevents over-penalising the
    # low-occupancy-but-bandwidth-saturated reductions.
    d["occ_knee"] = min(0.35, max(0.08, d["occ_knee"]))
    return DeviceConstants(**d)


def pack(k: DeviceConstants):
    return [np.log(getattr(k, n)) if n in LOG_PARAMS else getattr(k, n) for n in PARAM_NAMES]


def predict_times_row(row, k: DeviceConstants):
    f = predict_time(row["flops"], row["bytes_fused"], row["f_occ"], row["f_spills"], 1, k,
                     reread=row.get("f_reread", 1.0))
    u = predict_time(row["flops"], row["bytes_unfused"], row["u_occ"], row["u_spills"],
                     int(row["n_launches_unfused"]), k, reread=row.get("u_reread", 1.0))
    return f["t"], u["t"]


def loss(theta, df, w_ratio=1.0, w_abs=0.3):
    """Combined objective: the fuse/don't-fuse decision depends only on the fused/unfused time
    RATIO, so we fit log(speedup) primarily (w_ratio); a down-weighted absolute-time term (w_abs)
    keeps B_peak / T_launch physically meaningful for the roofline interpretation."""
    k = unpack(theta)
    err = 0.0
    for _, row in df.iterrows():
        tf, tu = predict_times_row(row, k)
        pred_ratio = np.log(tu) - np.log(tf)
        true_ratio = np.log(row["t_unfused_ms"]) - np.log(row["t_fused_ms"])
        err += w_ratio * (pred_ratio - true_ratio) ** 2
        err += w_abs * ((np.log(tf) - np.log(row["t_fused_ms"] * 1e-3)) ** 2 +
                        (np.log(tu) - np.log(row["t_unfused_ms"] * 1e-3)) ** 2)
    return err / len(df)


def fit(df, restarts=6, seed=0) -> DeviceConstants:
    rng = np.random.default_rng(seed)
    best, best_loss = None, np.inf
    x0_base = pack(DeviceConstants())
    for r in range(restarts):
        x0 = np.array(x0_base) + (rng.standard_normal(len(x0_base)) * 0.7 if r else 0)
        res = minimize(loss, x0, args=(df,), method="Nelder-Mead",
                       options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-7})
        if res.fun < best_loss:
            best_loss, best = res.fun, res.x
    k = unpack(best)
    return k, best_loss


def decisions(df, k: DeviceConstants):
    pred, true = [], []
    for _, row in df.iterrows():
        d = decide(row, k)
        pred.append(1 - d["pred_beneficial"])   # positive class = TOXIC (don't fuse)
        true.append(1 - int(row["beneficial"]))
    return np.array(pred), np.array(true)


def prf(pred, true):
    tp = int(((pred == 1) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    tn = int(((pred == 0) & (true == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / len(pred)
    return dict(precision=prec, recall=rec, f1=f1, accuracy=acc, tp=tp, fp=fp, fn=fn, tn=tn)


def nondegenerate(df):
    """Drop rows whose 'unfused' plan is a single launch identical to the fused kernel
    (n_launches_unfused <= 1). There is nothing to fuse, so the fused-vs-unfused comparison is a
    no-op and the beneficial label is pure timing noise -- it must not be scored as a fusion
    decision. This is the real defect condition (review item 7), independent of family: it catches
    reduction NOUT<=GS AND pointwise K=1 (a single elementwise op is not a fusion). Returns
    (filtered_df, n_dropped)."""
    if "n_launches_unfused" not in df.columns:
        return df, 0
    deg = df["n_launches_unfused"] <= 1
    return df[~deg], int(deg.sum())


def main(csv):
    df = pd.read_csv(csv)
    print(f"[fit] {len(df)} rows | toxic(don't-fuse)={int((df.beneficial==0).sum())} "
          f"beneficial={int((df.beneficial==1).sum())}")

    # ---- full-data fit ----
    k, l = fit(df)
    print("\n=== fitted DeviceConstants (Ada sm89) ===")
    print(json.dumps(k.as_dict(), indent=2))
    print(f"fit log-MSE(time) = {l:.4f}")

    # ---- degenerate-row filter for DECISION scoring --------------------------------
    # The roofline FIT above uses all rows (they are near-neutral; full-data fit reproduces the
    # deployed constants exactly). But rows where the unfused plan is a no-op copy of the fused
    # kernel (n_launches_unfused<=1, NOUT<=GS) carry a noise label and must not be scored as a
    # fuse/don't-fuse decision -- exclude them from all decision metrics below.
    dfe, n_deg = nondegenerate(df)
    if n_deg:
        print(f"\n[fit] excluding {n_deg} degenerate no-op rows (n_launches_unfused<=1: pointwise "
              f"K=1 and/or reduction NOUT<=GS) from decision scoring; {len(dfe)} genuine cases remain.")

    pred, true = decisions(dfe, k)
    m = prf(pred, true)
    print("\n=== RQ1: toxic-fusion DECISION quality (in-sample, genuine cases) ===")
    print(f"precision={m['precision']:.3f} recall={m['recall']:.3f} F1={m['f1']:.3f} "
          f"acc={m['accuracy']:.3f}  (TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})")

    # ---- baselines ----
    greedy = np.zeros(len(dfe))  # always fuse -> never predicts toxic
    mg = prf(greedy, true)
    print("\n=== baselines ===")
    print(f"greedy-always-fuse: F1={mg['f1']:.3f} acc={mg['accuracy']:.3f} "
          f"(misses all {int(true.sum())} toxic fusions)")

    # ---- held-out-shape cross validation ----
    print("\n=== RQ1: held-out-shape cross-validation ===")
    shapes = sorted(dfe.apply(lambda r: (r["R"], r["C"]), axis=1).unique())
    allp, allt = [], []
    for held in shapes:
        te = dfe[dfe.apply(lambda r: (r["R"], r["C"]) == held, axis=1)]
        tr = dfe[dfe.apply(lambda r: (r["R"], r["C"]) != held, axis=1)]
        kk, _ = fit(tr, restarts=3)
        p, t = decisions(te, kk)
        mm = prf(p, t)
        allp.extend(p); allt.extend(t)
        print(f"  held-out R{held[0]}xC{held[1]} (n={len(te)}): acc={mm['accuracy']:.3f} F1={mm['f1']:.3f}")
    mcv = prf(np.array(allp), np.array(allt))
    print(f"  pooled CV: precision={mcv['precision']:.3f} recall={mcv['recall']:.3f} "
          f"F1={mcv['f1']:.3f} acc={mcv['accuracy']:.3f}")
    print("  NOTE: spills (the toxicity driver) are identical across (R,C) shapes, so this CV is")
    print("        ~ in-sample w.r.t. the decision. The stronger CVs below hold out the variables")
    print("        that actually change spills (NOUT, dtype).")

    # ---- held-out-DECISION-VARIABLE cross validation (the honest generalization test) ----
    # Folds on the variables that move register pressure / spills, not on (R,C) which does not.
    print("\n=== RQ1: held-out-DECISION-VARIABLE cross-validation ===")
    # leave-one-NOUT-out over the reduction family: each NOUT is a distinct spill regime.
    gp, gt = [], []
    for no in sorted(int(x) for x in dfe["param_NOUT"].dropna().unique()):
        te = dfe[dfe["param_NOUT"] == no]
        tr = dfe[(dfe["param_NOUT"] != no) | (dfe["param_NOUT"].isna())]
        kk, _ = fit(tr, restarts=3)
        p, t = decisions(te, kk); gp.extend(p); gt.extend(t)
        mm = prf(p, t)
        print(f"  held-out NOUT={no} (n={len(te)}): F1={mm['f1']:.3f} recall={mm['recall']:.3f}")
    mno = prf(np.array(gp), np.array(gt))
    print(f"  leave-one-NOUT-out POOLED: precision={mno['precision']:.3f} recall={mno['recall']:.3f} "
          f"F1={mno['f1']:.3f} acc={mno['accuracy']:.3f}")
    # leave-one-dtype-out: fp16 <-> fp32 calibration transfer (the honest generalization cost).
    gp, gt = [], []
    for dt in sorted(dfe["dtype"].unique()):
        te = dfe[dfe["dtype"] == dt]; tr = dfe[dfe["dtype"] != dt]
        kk, _ = fit(tr, restarts=3)
        p, t = decisions(te, kk); gp.extend(p); gt.extend(t)
        mm = prf(p, t)
        print(f"  held-out dtype={dt} (n={len(te)}): F1={mm['f1']:.3f} recall={mm['recall']:.3f}")
    mdt = prf(np.array(gp), np.array(gt))
    print(f"  leave-one-dtype-out POOLED: precision={mdt['precision']:.3f} recall={mdt['recall']:.3f} "
          f"F1={mdt['f1']:.3f} acc={mdt['accuracy']:.3f}")

    # ---- ablations ----
    print("\n=== ablations (in-sample, genuine cases) ===")
    # no-spill: zero out spill feature
    df_ns = dfe.copy(); df_ns["f_spills"] = 0; df_ns["u_spills"] = 0
    kns, _ = fit(df_ns, restarts=3)
    p, t = decisions(df_ns, kns)
    print(f"  drop spill term: F1={prf(p,t)['f1']:.3f} (spills are the dominant toxic signal)")
    # no-P_occ: force occupancy to 1 everywhere
    df_no = dfe.copy(); df_no["f_occ"] = 1.0; df_no["u_occ"] = 1.0
    kno, _ = fit(df_no, restarts=3)
    p, t = decisions(df_no, kno)
    print(f"  drop P_occ (occ=1): F1={prf(p,t)['f1']:.3f}")

    # persist fitted constants -- preserve fields this fit does NOT calibrate (e.g. beta_layout,
    # which is fit separately on the layout microbench by fusion/profile_layout.py) so that
    # re-running fit after profile_layout does not clobber the layout calibration back to default.
    out = k.as_dict()
    try:
        with open("model/ada_constants.json") as f:
            prev = json.load(f)
        for key in ("beta_layout",):
            if key in prev:
                out[key] = prev[key]
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    with open("model/ada_constants.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n[fit] wrote model/ada_constants.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/microbench_timing.csv")
