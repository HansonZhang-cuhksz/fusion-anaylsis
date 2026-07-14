"""cuda_layout.py -- P_layout study in raw CUDA (bank conflicts control the layout penalty).

Provides, for Y[C,R]=relu(s*X^T+b):
  fused_conflict : one kernel, transpose+relu, shared tile PAD=0  -> 32-way bank conflicts (P_layout)
  fused_clean    : same fusion, PAD=1                              -> conflict-free (control)
  unfused        : clean padded transpose -> HBM, then a relu epilogue (2 kernels)

Static registers come from a single nvcc -Xptxas -v compile (search-free input); the runnable
kernels come from torch load_inline (cached). ncu supplies the bank-conflict ground truth. This is
the layout-dominant branch that makes RQ2 attribution non-trivial (spills=0, bank conflicts high).
"""
from __future__ import annotations
import os, re, subprocess, functools
import torch

CONDA = os.environ.get("PROFILING_ENV", "/home/shuhan/miniconda3/envs/profiling")
CCBIN = f"{CONDA}/bin/x86_64-conda-linux-gnu-gcc"
HERE = os.path.dirname(os.path.abspath(__file__))
CU = os.path.join(HERE, "cuda_kernels", "layout.cu")


def ptxas_registers() -> dict:
    """Return {'conflict': regs, 'clean': regs, 'spill_conflict':bytes, 'spill_clean':bytes}
    from a single nvcc -Xptxas -v compile (the single-compile static P_occ input)."""
    r = subprocess.run(["nvcc", "-ccbin", CCBIN, "-arch=sm_89", "-Xptxas", "-v",
                        "-c", CU, "-o", "/dev/null"], capture_output=True, text=True)
    log = r.stderr + r.stdout
    out = {}
    # ptxas prints one block per entry function; match the templated txpose_relu instantiations.
    blocks = re.split(r"ptxas info\s+:\s+Compiling entry function", log)
    for blk in blocks:
        m_regs = re.search(r"Used (\d+) registers", blk)
        m_spill = re.search(r"(\d+) bytes spill stores", blk)
        if "txpose_relu" not in blk or not m_regs:
            continue
        regs = int(m_regs.group(1)); spill = int(m_spill.group(1)) if m_spill else 0
        # PAD=0 (conflict) uses [32][32]=4096B smem; PAD=1 (clean) uses [32][33]=4224B smem
        if "ILi0E" in blk or "0E" in blk[:200]:
            out["conflict"] = regs; out["spill_conflict"] = spill
        elif "ILi1E" in blk or "1E" in blk[:200]:
            out["clean"] = regs; out["spill_clean"] = spill
    # fall back: if we could not disambiguate by mangling, both variants have ~equal regs
    if "conflict" not in out or "clean" not in out:
        allr = [int(x) for x in re.findall(r"Used (\d+) registers", log)]
        if allr:
            out.setdefault("conflict", max(allr)); out.setdefault("clean", max(allr))
            out.setdefault("spill_conflict", 0); out.setdefault("spill_clean", 0)
    return out


@functools.lru_cache(maxsize=1)
def build_module():
    from torch.utils.cpp_extension import load_inline
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
    os.environ.setdefault("CC", CCBIN)
    os.environ.setdefault("CXX", f"{CONDA}/bin/x86_64-conda-linux-gnu-g++")
    cuda_src = open(CU).read() + r'''
#include <torch/extension.h>
torch::Tensor l_fused(torch::Tensor x, int pad, float s, float b){
  int R=x.size(0), C=x.size(1); auto y=torch::empty({C,R}, x.options());
  dim3 blk(32,32), grd((C+31)/32,(R+31)/32);
  if(pad==0) txpose_relu<0><<<grd,blk>>>(x.data_ptr<float>(),y.data_ptr<float>(),R,C,s,b);
  else       txpose_relu<1><<<grd,blk>>>(x.data_ptr<float>(),y.data_ptr<float>(),R,C,s,b);
  return y;
}
torch::Tensor l_unfused(torch::Tensor x, float s, float b){
  int R=x.size(0), C=x.size(1);
  auto xt=torch::empty({C,R}, x.options()); auto y=torch::empty({C,R}, x.options());
  dim3 blk(32,32), grd((C+31)/32,(R+31)/32);
  txpose_only<<<grd,blk>>>(x.data_ptr<float>(),xt.data_ptr<float>(),R,C);
  long n=(long)R*C; relu_ep<<<(n+255)/256,256>>>(xt.data_ptr<float>(),y.data_ptr<float>(),n,s,b);
  return y;
}
'''
    cpp = ("torch::Tensor l_fused(torch::Tensor x, int pad, float s, float b);\n"
           "torch::Tensor l_unfused(torch::Tensor x, float s, float b);")
    return load_inline(name="fusion_layout", cpp_sources=cpp, cuda_sources=cuda_src,
                       functions=["l_fused", "l_unfused"],
                       extra_cuda_cflags=["-arch=sm_89", f"-ccbin={CCBIN}"], verbose=False)


def runners(R: int, C: int):
    m = build_module()
    x = torch.randn(R, C, device="cuda", dtype=torch.float32)
    return {
        "x": x,
        "fused_conflict": lambda: m.l_fused(x, 0, 1.0, 0.1),
        "fused_clean":    lambda: m.l_fused(x, 1, 1.0, 0.1),
        "unfused":        lambda: m.l_unfused(x, 1.0, 0.1),
        "ref": torch.relu(x.t().contiguous() * 1.0 + 0.1),
    }
