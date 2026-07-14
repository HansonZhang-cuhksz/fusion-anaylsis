"""ncu_worker.py -- run ONE fusion plan (fused or unfused) exactly once, for ncu to profile.

Invoked as:  python -m fusion.ncu_worker <family> <plan> <json-params>
No warmup, no timing: build the case (triggers Triton compile -- host side, not profiled) then
execute the chosen plan once so ncu profiles exactly that plan's kernel launches.
"""
import sys, json
import torch
from fusion.kernels import pointwise, reduction

FAMILIES = {"pointwise": pointwise, "reduction": reduction}


def build(family: str, params: dict):
    mod = FAMILIES[family]
    dtype = getattr(torch, params.pop("dtype", "float16"))
    # compile_probe=False: skip the static-extraction pre-launches so ncu profiles only the plan.
    return mod.make_case(dtype=dtype, compile_probe=False, **params)


def main():
    family, plan, params_json = sys.argv[1], sys.argv[2], sys.argv[3]
    params = json.loads(params_json)
    case = build(family, params)
    torch.cuda.synchronize()
    if plan == "fused":
        case.run_fused()
    elif plan == "unfused":
        case.run_unfused()
    else:
        raise ValueError(plan)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
