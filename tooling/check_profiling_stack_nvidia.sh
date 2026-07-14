#!/usr/bin/env bash
# ============================================================================
# check_profiling_stack_nvidia.sh   --  NVIDIA/Ada analogue of
# check_profiling_stack.sh (which targets MetaX MACA).
#
# Validates the NVIDIA profiling toolchain on the Ada (sm89) primary machine and
# maps the cost-model's needs (P_occ / P_layout / mem-traffic / MMA-util) onto
# the metrics ncu actually exposes here. Read-only + a couple of tiny kernels.
#
# Concept  ->  NVIDIA metric (this script probes each of these end to end):
#   registers/thread    -> ptxas -v  AND  ncu launch__registers_per_thread
#   register spills      -> ptxas -v "spill stores/loads"; local-mem traffic
#   achieved occupancy   -> sm__warps_active.avg.pct_of_peak_sustained_active
#   bank conflicts       -> l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum
#   DRAM traffic         -> dram__bytes.sum
#   tensor-core util     -> sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_...
#
# Usage:  source tooling/env.sh && bash tooling/check_profiling_stack_nvidia.sh
# ============================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(mktemp -d /tmp/probe_prof_nv.XXXXXX)"
CCBIN="${NVCC_CCBIN:-/home/shuhan/miniconda3/envs/profiling/bin/x86_64-conda-linux-gnu-gcc}"
ARCH="${SM_ARCH:-sm_89}"
PASS=0; FAIL=0; WARN=0
ok(){   printf '  \033[32m[ OK ]\033[0m %s\n' "$*"; PASS=$((PASS+1)); }
no(){   printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
warn(){ printf '  \033[33m[WARN]\033[0m %s\n' "$*"; WARN=$((WARN+1)); }
hdr(){  printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
have(){ command -v "$1" >/dev/null 2>&1; }

hdr "1. NVIDIA toolchain present"
for t in nvidia-smi nvcc ptxas ncu cuobjdump nvdisasm; do
  if have "$t"; then ok "$t  $(command -v $t)"; else no "$t missing (did you 'source tooling/env.sh'?)"; fi
done

hdr "2. Device (nvidia-smi)"
if have nvidia-smi; then
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader 2>/dev/null | sed 's/^/  /'
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  [ "$cc" = "8.9" ] && ok "compute capability 8.9 (Ada sm89) confirmed" || warn "compute_cap=$cc (expected 8.9)"
else no "cannot query device"; fi

hdr "3. Compile probe kernel (nvcc, pinned host compiler)"
if have nvcc && nvcc -ccbin "$CCBIN" -arch="$ARCH" -o "$OUT/probe" "$HERE/probe_kernel.cu" 2>"$OUT/cc.log"; then
  ok "nvcc compiled probe_kernel.cu -> $OUT/probe"
else no "nvcc compile failed (see $OUT/cc.log)"; fi

hdr "4. Static register/spill report (ptxas -v == MetaX cucc -resource-usage)"
if have nvcc && nvcc -ccbin "$CCBIN" -arch="$ARCH" -Xptxas -v -c "$HERE/probe_kernel.cu" -o /dev/null >"$OUT/res.log" 2>&1; then
  if grep -q 'Used .* registers' "$OUT/res.log"; then
    grep -E 'Compiling entry|Used .* registers|spill' "$OUT/res.log" | sed 's/^/  /'
    ok "static registers + spill stores/loads reported (this is the P_occ static input)"
  else warn "compiled but no 'Used N registers' lines (see $OUT/res.log)"; fi
else warn "resource-usage probe failed (see $OUT/res.log)"; fi

hdr "5. ncu counter collection (the WSL2 make-or-break check)"
cat > "$OUT/mm.py" <<'PY'
import torch
a = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
b = torch.randn(2048, 2048, device='cuda', dtype=torch.float16)
for _ in range(20):
    c = a @ b
torch.cuda.synchronize()
PY
NEED="sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,dram__bytes.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active"
if have ncu && timeout 300 ncu --target-processes all --launch-count 1 --launch-skip 5 \
      --metrics "$NEED" python "$OUT/mm.py" >"$OUT/ncu.log" 2>&1; then
  if grep -q 'sm__warps_active' "$OUT/ncu.log" && ! grep -q 'No kernels were profiled' "$OUT/ncu.log"; then
    ok "ncu collected real counters under WSL2"
    grep -E 'sm__warps_active|launch__registers_per_thread|dram__bytes|bank_conflicts|tensor_op_hmma' "$OUT/ncu.log" | sed 's/^/    /'
  else no "ncu ran but profiled no kernels / no counters (see $OUT/ncu.log)"; fi
else no "ncu invocation failed (see $OUT/ncu.log)"; fi

hdr "6. Triton static register/spill introspection (single-compile input, no ncu)"
cat > "$OUT/tt.py" <<'PY'
import torch, triton, triton.language as tl
@triton.jit
def add_k(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0); off = pid*BLOCK + tl.arange(0, BLOCK)
    m = off < n
    tl.store(o_ptr+off, tl.load(x_ptr+off, mask=m) + tl.load(y_ptr+off, mask=m), mask=m)
n=1<<20; x=torch.randn(n,device='cuda'); y=torch.randn(n,device='cuda'); o=torch.empty_like(x)
k = add_k[(triton.cdiv(n,1024),)](x,y,o,n,BLOCK=1024)
print("TRITON_STATIC n_regs=%d n_spills=%d shared=%d" % (k.n_regs, k.n_spills, k.metadata.shared))
PY
if timeout 180 python "$OUT/tt.py" >"$OUT/tt.log" 2>&1 && grep -q TRITON_STATIC "$OUT/tt.log"; then
  ok "Triton exposes per-kernel n_regs/n_spills/shared from one compile"
  grep TRITON_STATIC "$OUT/tt.log" | sed 's/^/    /'
else warn "Triton static introspection failed (see $OUT/tt.log)"; fi

hdr "VERDICT"
printf '  PASS=%d  WARN=%d  FAIL=%d\n' "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -eq 0 ]; then
  echo "  => NVIDIA/Ada profiling stack is present and functional."
  echo "     Static P_occ input: ptxas -v  OR  Triton k.n_regs/n_spills."
  echo "     Ground truth:        ncu (--target-processes all) collects counters under WSL2."
else
  echo "  => Toolchain gap detected (see FAIL lines above)."
fi
echo "  artifacts: $OUT"
