# tooling/env.sh  --  source this to get the NVIDIA/Ada research toolchain on PATH.
#   usage:  source tooling/env.sh
#
# The whole toolchain lives inside the conda env "profiling" (torch 2.7.0+cu128,
# triton 3.3.0, nvcc 12.8, Nsight Compute 2025.1.1). nvcc + ncu are NOT on the
# base PATH, so this script exports them and pins the correct host compiler.
#
# Machine: RTX 4060 Laptop GPU (Ada, sm89), 24 SMs, 8 GB, WSL2, driver 610.74.
export PROFILING_ENV="${PROFILING_ENV:-/home/shuhan/miniconda3/envs/profiling}"
export NSIGHT_DIR="$PROFILING_ENV/nsight-compute-2025.1.1"

# nvcc / ptxas / cuobjdump / nvdisasm live in $PROFILING_ENV/bin; ncu lives in NSIGHT_DIR.
export PATH="$PROFILING_ENV/bin:$NSIGHT_DIR:$PATH"

# CUDA 12.8 nvcc rejects the system GCC 14 headers (noexcept clash on cospi/sinpi).
# Pin the conda-provided GCC 11.2 as the host compiler for all raw-CUDA compiles.
export NVCC_CCBIN="$PROFILING_ENV/bin/x86_64-conda-linux-gnu-gcc"
# Convenience alias mirroring what the build scripts pass to nvcc.
export NVCC="nvcc -ccbin $NVCC_CCBIN -arch=sm_89"

# sm89 hardware constants (source of truth for the analytical occupancy model).
export SM_ARCH=sm_89
export SM_COUNT=24

echo "[env] profiling toolchain ready:"
echo "      nvcc : $(command -v nvcc)  ($(nvcc --version 2>/dev/null | grep -o 'release [0-9.]*'))"
echo "      ncu  : $(command -v ncu)"
echo "      ccbin: $NVCC_CCBIN"
