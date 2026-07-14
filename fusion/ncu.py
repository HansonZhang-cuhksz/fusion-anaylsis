"""ncu.py -- automate Nsight Compute collection into vendor-neutral concept metrics.

Profiles a fusion plan (fused / unfused) by wrapping fusion.ncu_worker in `ncu --csv`, then maps
the raw NVIDIA metric names onto the vendor-neutral CONCEPTS the cost model reasons about. Keeping
the dataset schema in concept space (not raw metric names) is what makes the Phase-4 MetaX transfer
a re-parameterisation rather than a rewrite (HANDOFF section 3).
"""
from __future__ import annotations
import json, subprocess, csv, io, os
from statistics import median

# concept -> ncu metric name
METRICS = {
    "occ_achieved_pct":  "sm__warps_active.avg.pct_of_peak_sustained_active",
    "occ_theoretical_pct": "sm__maximum_warps_per_active_cycle_pct",
    "regs_per_thread":   "launch__registers_per_thread",
    "smem_per_block":    "launch__shared_mem_per_block",
    "waves_per_sm":      "launch__waves_per_multiprocessor",
    "dram_bytes":        "dram__bytes.sum",
    "local_ld_bytes":    "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum",   # spill loads
    "local_st_bytes":    "l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum",   # spill stores
    "bank_conf_ld":      "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "bank_conf_st":      "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    "l2_sectors":        "lts__t_sectors.sum",
    "tensor_pct":        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "duration_ns":       "gpu__time_duration.sum",   # ncu base unit for time is NANOSECONDS
    "sm_busy_pct":       "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_busy_pct":     "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
}
_METRIC_LIST = ",".join(sorted(set(METRICS.values())))
_INV = {v: k for k, v in METRICS.items()}

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _to_float(s: str) -> float:
    s = s.strip().strip('"').replace(",", "")
    if s in ("", "N/A", "n/a"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def profile_plan(family: str, plan: str, params: dict, timeout: int = 600) -> list[dict]:
    """Return a list of per-kernel metric dicts (concept-keyed) for one plan execution."""
    p = dict(params)
    # restrict profiling to our Triton microbench kernels (exclude curand/copy setup kernels).
    kregex = "regex:(sibling_redux|fused_chain|one_step|producer|consumer|trans_|epilogue|xpose)"
    args = ["ncu", "--target-processes", "all", "--csv", "--metrics", _METRIC_LIST,
            "--print-units", "base", "-k", kregex,
            "python", "-m", "fusion.ncu_worker", family, plan, json.dumps(p)]
    env = dict(os.environ)
    r = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=timeout, env=env)
    if r.returncode != 0 and "==ERROR==" in r.stderr:
        raise RuntimeError(f"ncu failed for {family}/{plan}/{params}:\n{r.stderr[-800:]}")
    return _parse_csv(r.stdout)


def _parse_csv(text: str) -> list[dict]:
    # find the header line (ncu prints a preamble); the CSV block starts at the line with "ID",
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith('"ID"')), None)
    if start is None:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(lines[start:])))
    # ncu long format: one row per (kernel launch ID, metric). Group by ID.
    kernels: dict[str, dict] = {}
    for row in reader:
        kid = row.get("ID", "")
        mname = row.get("Metric Name", "")
        mval = row.get("Metric Value", "")
        if kid not in kernels:
            kernels[kid] = {"kernel": row.get("Kernel Name", ""), "id": kid}
        if mname in _INV:
            kernels[kid][_INV[mname]] = _to_float(mval)
    return list(kernels.values())


def aggregate(kernels: list[dict]) -> dict:
    """Collapse a plan's per-kernel rows into plan-level features the model consumes."""
    if not kernels:
        return {}
    total_dur = sum(k.get("duration_ns", 0.0) for k in kernels)
    total_dram = sum(k.get("dram_bytes", 0.0) for k in kernels)
    total_local = sum(k.get("local_ld_bytes", 0.0) + k.get("local_st_bytes", 0.0) for k in kernels)
    total_bank = sum(k.get("bank_conf_ld", 0.0) + k.get("bank_conf_st", 0.0) for k in kernels)
    # the "signature" kernel = the longest-running one (dominates the plan)
    sig = max(kernels, key=lambda k: k.get("duration_ns", 0.0))
    return {
        "n_kernels": len(kernels),
        "dur_ns_total": total_dur,            # nanoseconds (ncu gpu__time_duration, base units)
        "dram_bytes_total": total_dram,
        "local_bytes_total": total_local,     # spill traffic (P_occ ground truth)
        "bank_conf_total": total_bank,        # bank conflicts (P_layout ground truth)
        "sig_occ_achieved": sig.get("occ_achieved_pct", 0.0),
        "sig_occ_theoretical": sig.get("occ_theoretical_pct", 0.0),
        "sig_regs": sig.get("regs_per_thread", 0.0),
        "sig_smem": sig.get("smem_per_block", 0.0),
        "sig_local_bytes": sig.get("local_ld_bytes", 0.0) + sig.get("local_st_bytes", 0.0),
        "sig_bank_conf": sig.get("bank_conf_ld", 0.0) + sig.get("bank_conf_st", 0.0),
        "sig_tensor_pct": sig.get("tensor_pct", 0.0),
        "sig_dram_busy": sig.get("dram_busy_pct", 0.0),
        "sig_sm_busy": sig.get("sm_busy_pct", 0.0),
        "sig_kernel": sig.get("kernel", ""),
    }
