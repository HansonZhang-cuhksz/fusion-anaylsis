"""cuda_layout_worker.py -- run ONE CUDA layout plan once, for ncu to profile.
Usage: python -m fusion.cuda_layout_worker <plan> <R> <C>   plan in {fused_conflict,fused_clean,unfused}
"""
import sys
import torch
from fusion.cuda_layout import runners


def main():
    plan, R, C = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    r = runners(R, C)
    torch.cuda.synchronize()
    r[plan]()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
