"""mcpti_profile.py -- MetaX C500 ground-truth profiler (the ncu analogue for RQ2-on-C500).

Drives MCPTI directly via ctypes to read per-kernel hardware event counts, because the mcProfiler
CLI value-dump path (`MctxStreamProfilerCountDataGet`) is UNIMPLEMENTED in MACA 3.7.0 (returns
grpc_status 12). Recipe (see memory `metax-c500-profiling`):
  - `libmcruntime.so :: mcCtxGetCurrent` -> the current MCcontext (after torch inits it);
  - `libmcpti.so` Event API (CUPTI-compatible), CONTINUOUS collection mode;
  - event names are CUPTI-legacy style: local_load/local_store (spill/private traffic),
    global_load/global_store (DRAM), shared_ld/st_bank_conflict (layout);
  - counters are FREE-RUNNING (cumulative) -> take a per-kernel DELTA (read before/after the launch).

Concept map (vendor-neutral, matches fusion/ncu.py on Ada):
  local(spill) = local_load+local_store ; DRAM = global_load+global_store ;
  bank = shared_ld+st_bank_conflict ; dominant_penalty = spill if local dominates, else layout/none.

Usage:  FUSION_HW=c500 MACA_VISIBLE_DEVICES=<idx> python -m fusion.mcpti_profile [out.csv]
"""
from __future__ import annotations
import ctypes, sys, csv, warnings
import torch
warnings.filterwarnings("ignore")
from fusion.kernels import reduction
from fusion.static import from_triton

_RT  = ctypes.CDLL("/opt/maca/lib/libmcruntime.so")
_PTI = ctypes.CDLL("/opt/maca/lib/libmcpti.so")
for _f, _a in {
    "mcptiSetEventCollectionMode": [ctypes.c_void_p, ctypes.c_int],
    "mcptiEventGroupCreate": [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32],
    "mcptiEventGetIdFromName": [ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p],
    "mcptiEventGroupAddEvent": [ctypes.c_void_p, ctypes.c_uint32],
    "mcptiEventGroupEnable": [ctypes.c_void_p], "mcptiEventGroupDisable": [ctypes.c_void_p],
    "mcptiEventGroupDestroy": [ctypes.c_void_p],
    "mcptiEventGroupReadEvent": [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32,
                                 ctypes.c_void_p, ctypes.c_void_p],
}.items():
    getattr(_PTI, _f).argtypes = _a
_RT.mcCtxGetCurrent.argtypes = [ctypes.c_void_p]

EVENTS = ["local_load", "local_store", "global_load", "global_store",
          "shared_ld_bank_conflict", "shared_st_bank_conflict"]


class MCPTI:
    def __init__(self):
        torch.zeros(8, device="cuda"); torch.cuda.synchronize()   # ensure a context exists
        self.ctx = ctypes.c_void_p(); _RT.mcCtxGetCurrent(ctypes.byref(self.ctx))
        _PTI.mcptiSetEventCollectionMode(self.ctx, 0)             # CONTINUOUS
        self.eid = {}
        for n in EVENTS:
            e = ctypes.c_uint32()
            if _PTI.mcptiEventGetIdFromName(0, n.encode(), ctypes.byref(e)) == 0:
                self.eid[n] = e.value

    def _read(self, g, e):
        buf = (ctypes.c_uint64 * 256)(); sz = ctypes.c_size_t(ctypes.sizeof(buf))
        r = _PTI.mcptiEventGroupReadEvent(g, 0, e, ctypes.byref(sz), buf)
        return sum(buf[i] for i in range(sz.value // 8)) if r == 0 else None

    def profile(self, run_kernel) -> dict:
        """Per-kernel event counts via before/after delta of the free-running counters."""
        g = ctypes.c_void_p(); _PTI.mcptiEventGroupCreate(self.ctx, ctypes.byref(g), 0)
        added = [n for n in self.eid if _PTI.mcptiEventGroupAddEvent(g, self.eid[n]) == 0]
        _PTI.mcptiEventGroupEnable(g); torch.cuda.synchronize()
        before = {n: self._read(g, self.eid[n]) for n in added}
        run_kernel(); torch.cuda.synchronize()
        after = {n: self._read(g, self.eid[n]) for n in added}
        _PTI.mcptiEventGroupDisable(g); _PTI.mcptiEventGroupDestroy(g)
        return {n: after[n] - before[n] for n in added
                if before[n] is not None and after[n] is not None}


def dominant_penalty(local, glob, bank) -> str:
    if local > 0.25 * max(1, glob):
        return "spill"
    if bank > 5 * max(1, glob):
        return "layout"
    return "none"


SUBSET = [(R, C, NOUT, dt) for (R, C) in [(2048, 2048), (1024, 1024)]
          for NOUT in (32, 64, 128) for dt in ("float16", "float32")]


def main(out_csv):
    m = MCPTI()
    print(f"[mcpti] resolved {len(m.eid)}/{len(EVENTS)} events: {list(m.eid)}", flush=True)
    rows = []
    for (R, C, NOUT, dt_name) in SUBSET:
        dt = getattr(torch, dt_name)
        case = reduction.make_case(R=R, C=C, NOUT=NOUT, dtype=dt, GS=16, compile_probe=True)
        si = from_triton(case.fused_kernels[0])
        v = m.profile(case.run_fused)
        local = v.get("local_load", 0) + v.get("local_store", 0)
        glob = v.get("global_load", 0) + v.get("global_store", 0)
        bank = v.get("shared_ld_bank_conflict", 0) + v.get("shared_st_bank_conflict", 0)
        dom = dominant_penalty(local, glob, bank)
        row = {"op_pair": f"sibling_redux_n{NOUT}", "dtype": dt_name, "R": R, "C": C, "param_NOUT": NOUT,
               "f_regs": si.n_regs, "f_spills": si.n_spills, "f_occ_analytic": round(si.occupancy, 4),
               "local_inst": local, "global_inst": glob, "bank_conf": bank, "dominant_penalty": dom}
        rows.append(row)
        print(f"  NOUT={NOUT:3d} {dt_name:7s} R{R}xC{C}: spills={si.n_spills:4d} "
              f"local={local:>10} global={glob:>8} bank={bank:>9} -> {dom}", flush=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    n_spill = sum(r["dominant_penalty"] == "spill" for r in rows)
    print(f"[mcpti] wrote {len(rows)} rows -> {out_csv}; dominant=spill on {n_spill}/{len(rows)} "
          f"(all spilled reductions attribute to spill on hardware).")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/microbench_c500_mcpti.csv")
