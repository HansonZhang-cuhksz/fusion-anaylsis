"""costmodel.py -- the search-free, interpretable fusion-degradation cost model.

Faithful to PROPOSAL 5.2, adapted so the memory/latency-bound regime our microbenches live in is
representable (the toxic cases are spill/occupancy bound, so the degradation must reach the memory
term, not only the compute term):

    eta_fused = min(eta_u, eta_v) * P_occ * P_layout          (factored, interpretable)

    T_plan = max( F / (C_peak * eff),  M_eff / (B_peak * eff) ) + L * T_launch

where every input is STATIC (single compile): occupancy from the analytical sm89 model, spill
count from the compiler, analytic bytes/flops from graph shapes. Ground-truth timing is used only
to FIT the per-device constants (C_peak, B_peak, T_launch, occ_knee, gamma_spill, beta_layout) --
never inside the deployed decision.

Decision: prune the fusion edge iff  T_fused > T_unfused.
Attribution: report the multiplicatively dominant penalty (P_occ vs P_layout) as the reason.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math


@dataclass
class DeviceConstants:
    name: str = "ada_sm89"
    C_peak: float = 2.0e13      # effective flop/s (fitted)
    B_peak: float = 2.0e11      # effective HBM bytes/s (fitted)
    T_launch: float = 5.0e-6    # per-launch overhead, seconds (fitted)
    occ_knee: float = 0.5       # occupancy at which bandwidth/latency hiding saturates (fitted)
    gamma_spill: float = 2.0e-3  # spill sensitivity: eff *= 1/(1+gamma*spills) (fitted)
    beta_layout: float = 1.0e-3  # bank-conflict sensitivity (fitted; per-conflict-cycle)
    occ_floor: float = 0.06     # minimum efficiency floor to avoid div-by-0

    def as_dict(self):
        return asdict(self)


def lam_occ(occ: float, k: DeviceConstants) -> float:
    """Latency-hiding / bandwidth efficiency vs occupancy: rises linearly then saturates at 1."""
    return max(k.occ_floor, min(1.0, occ / max(1e-6, k.occ_knee)))


def spill_factor(spills: int, k: DeviceConstants) -> float:
    """Multiplicative efficiency hit from register spills (local-memory traffic + serialization)."""
    return 1.0 / (1.0 + k.gamma_spill * max(0, spills))


def layout_factor(bank_conf_per_elem: float, k: DeviceConstants) -> float:
    """Multiplicative efficiency hit from bank conflicts / transpose (P_layout). 0 when compatible."""
    return 1.0 / (1.0 + k.beta_layout * max(0.0, bank_conf_per_elem))


def plan_efficiency(occ: float, spills: int, k: DeviceConstants,
                    bank_conf_per_elem: float = 0.0) -> dict:
    """Effective throughput fraction 'eff' for one plan, with its interpretable factors."""
    p_occ_occ = lam_occ(occ, k)          # occupancy component
    p_occ_spill = spill_factor(spills, k)  # spill component (part of P_occ)
    p_layout = layout_factor(bank_conf_per_elem, k)
    eff = p_occ_occ * p_occ_spill * p_layout
    return {"eff": max(k.occ_floor * 0.1, eff),
            "p_occ_occ": p_occ_occ, "p_occ_spill": p_occ_spill,
            "p_occ": p_occ_occ * p_occ_spill, "p_layout": p_layout}


def predict_time(flops: float, bytes_moved: float, occ: float, spills: int, n_launch: int,
                 k: DeviceConstants, bank_conf_per_elem: float = 0.0) -> dict:
    e = plan_efficiency(occ, spills, k, bank_conf_per_elem)
    eff = e["eff"]
    t_compute = flops / (k.C_peak * eff)
    t_mem = bytes_moved / (k.B_peak * eff)
    t = max(t_compute, t_mem) + n_launch * k.T_launch
    return {"t": t, "t_compute": t_compute, "t_mem": t_mem,
            "bound": "compute" if t_compute > t_mem else "memory", **e}


def decide(row: dict, k: DeviceConstants) -> dict:
    """Search-free fuse/don't-fuse decision + attribution from a dataset row's STATIC features.

    Expected static keys: flops, bytes_fused, bytes_unfused, f_occ, f_spills, u_occ, u_spills,
    n_launches_unfused, and optional f_bank_conf_per_elem / u_bank_conf_per_elem.
    """
    fused = predict_time(row["flops"], row["bytes_fused"], row["f_occ"], row["f_spills"],
                         n_launch=1, k=k,
                         bank_conf_per_elem=row.get("f_bank_conf_per_elem", 0.0))
    unfused = predict_time(row["flops"], row["bytes_unfused"], row["u_occ"], row["u_spills"],
                          n_launch=row["n_launches_unfused"], k=k,
                          bank_conf_per_elem=row.get("u_bank_conf_per_elem", 0.0))
    pred_beneficial = fused["t"] < unfused["t"]

    # ---- attribution: why is the fused plan degraded? compare penalty factors (lower = worse) ----
    # convert each penalty to a positive "harm" = -log(factor); dominant = largest harm.
    harm_occ = -math.log(max(1e-6, fused["p_occ"]))
    harm_layout = -math.log(max(1e-6, fused["p_layout"]))
    if max(harm_occ, harm_layout) < 1e-3:
        dominant = "none"
    elif harm_occ >= harm_layout:
        # split occupancy vs spill for a finer reason
        dominant = "spill" if fused["p_occ_spill"] < fused["p_occ_occ"] else "occupancy"
    else:
        dominant = "layout"
    return {
        "pred_t_fused": fused["t"], "pred_t_unfused": unfused["t"],
        "pred_speedup": unfused["t"] / fused["t"],
        "pred_beneficial": int(pred_beneficial),
        "fused_bound": fused["bound"],
        "P_occ": fused["p_occ"], "P_layout": fused["p_layout"],
        "P_occ_occ": fused["p_occ_occ"], "P_occ_spill": fused["p_occ_spill"],
        "dominant_penalty": dominant,
        "harm_occ": harm_occ, "harm_layout": harm_layout,
    }
