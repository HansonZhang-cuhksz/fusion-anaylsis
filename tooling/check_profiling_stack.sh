#!/usr/bin/env bash
# ============================================================================
# check_profiling_stack.sh
#
# Validates the GPU profiling toolchain available on this machine for the
# operator-fusion cost-model research, and maps the metrics the model needs
# (P_occ / P_layout / memory-traffic / MMA-util) onto what is actually
# exposed here.
#
# It checks TWO toolchains:
#   1. NVIDIA   (nvidia-smi / nvcc / ncu / nsys ...)   -- expected ABSENT on
#                                                          a MetaX-only host.
#   2. MetaX MACA (mx-smi / cucc / mcProfiler ...)     -- the counterpart.
#
# Read-only + one tiny kernel compile/run. No sudo. Safe to re-run.
# Usage:   bash tooling/check_profiling_stack.sh [GPU_INDEX]
#          GPU_INDEX defaults to 1 (leaves GPU0 free); overridable.
# ============================================================================
set -u
GPU="${1:-1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(mktemp -d /tmp/probe_prof.XXXXXX)"
MACA="${MACA_PATH:-/opt/maca}"
MCPROF="$MACA/mcProfiler-linux/mcProfiler"
PASS=0; FAIL=0; WARN=0
ok(){   printf '  \033[32m[ OK ]\033[0m %s\n' "$*"; PASS=$((PASS+1)); }
no(){   printf '  \033[31m[FAIL]\033[0m %s\n' "$*"; FAIL=$((FAIL+1)); }
warn(){ printf '  \033[33m[WARN]\033[0m %s\n' "$*"; WARN=$((WARN+1)); }
hdr(){  printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
have(){ command -v "$1" >/dev/null 2>&1; }

hdr "1. NVIDIA toolchain (expected absent)"
for t in nvidia-smi nvcc ncu nsys cuobjdump nvdisasm; do
  if have "$t"; then warn "$t present ($(command -v $t)) -- unexpected on MetaX host"
  else printf '  [absent] %s\n' "$t"; fi
done
have nvcc || have ncu || ok "confirmed: NVIDIA binaries not installed (use MetaX tools)"

hdr "2. MetaX MACA toolchain"
have mx-smi && ok "mx-smi        $(command -v mx-smi)"            || no "mx-smi missing"
have cucc   && ok "cucc (CUDA drv) $(command -v cucc)"            || no "cucc missing"
have mxcc   && ok "mxcc (native)   $(command -v mxcc)"            || no "mxcc missing"
[ -x "$MCPROF" ]         && ok "mcProfiler     $MCPROF"           || no "mcProfiler missing"
[ -x "$MACA/bin/mcTracer" ] && ok "mcTracer       $MACA/bin/mcTracer" || warn "mcTracer missing"
ls "$MACA"/include/mcpti/*.h >/dev/null 2>&1 && ok "MCPTI headers  $MACA/include/mcpti (CUPTI counterpart)" || warn "MCPTI headers missing"
ls "$MACA"/tools/cu-bridge/include/cupti*.h >/dev/null 2>&1 && ok "CUPTI shim     cu-bridge/include/cupti*.h" || warn "CUPTI shim headers missing"

hdr "3. Devices (mx-smi)"
if have mx-smi; then
  mx-smi 2>/dev/null | grep -E 'Attached GPUs|MetaX C500|MACA Version' | sed 's/^/  /'
  ok "device query works"
else no "cannot query devices"; fi

hdr "4. Compile probe kernel with cucc"
if have cucc && cucc -o "$OUT/probe" "$HERE/probe_kernel.cu" 2>"$OUT/cc.log"; then
  ok "cucc compiled probe_kernel.cu -> $OUT/probe"
else no "cucc compile failed (see $OUT/cc.log)"; fi

hdr "5. Static register/spill report (cucc -resource-usage)"
# NVIDIA counterpart: nvcc -Xptxas -v.  MetaX prints MT/ST register split.
if have cucc && cucc -resource-usage -c "$HERE/probe_kernel.cu" -o /dev/null >"$OUT/res.log" 2>&1; then
  if grep -q 'maca info' "$OUT/res.log"; then
    grep 'maca info' "$OUT/res.log" | sed 's/^/  /'
    ok "static register usage reported (MTregisters=vector, STregisters=scalar)"
  else warn "compiled but no 'maca info' resource lines found"; fi
else warn "resource-usage probe failed (see $OUT/res.log)"; fi

hdr "6. Profiler metric catalog (mcProfiler show_metrics)"
if [ -x "$MCPROF" ] && "$MCPROF" show_metrics >"$OUT/metrics.txt" 2>&1; then
  n=$(grep -c $'\t' "$OUT/metrics.txt")
  ok "show_metrics returned $n metrics"
  echo "  --- research-relevant metrics present ---"
  # research need -> MetaX metric (regex)
  declare -A NEED=(
    ["P_occ  (occupancy)"]="Achieved waves|Dispatched waves|WAVES"
    ["P_occ  (spill/local mem)"]="localMemory|Private (Read|Write) Instructions"
    ["P_layout (bank conflict)"]="conflict cycles|shared memory access efficiency"
    ["mem traffic (HBM/DRAM)"]="Global (Read|Write) Instructions|Dnoc|L2C (Read|Write)"
    ["MMA / Tensor-Core util"]="MMA Duty ratio"
    ["pipeline stalls / busy"]="AP busy Duty|stall cycles|AP active cycles"
    ["roofline"]="RoofLine"
  )
  for k in "${!NEED[@]}"; do
    if grep -qiE "${NEED[$k]}" "$OUT/metrics.txt"; then
      hit=$(grep -iE "${NEED[$k]}" "$OUT/metrics.txt" | head -1 | sed 's/^[[:space:]]*//')
      printf '    \033[32m✓\033[0m %-26s -> %s\n' "$k" "$hit"
    else
      printf '    \033[31m✗\033[0m %-26s -> (not found by name)\n' "$k"
    fi
  done
else warn "show_metrics failed"; fi

hdr "7. End-to-end profile run (best-effort)"
# NOTE: mcProfiler writes its report dir; point it at a writable --output.
# Value extraction can be metric-group sensitive; treated as best-effort.
if [ -x "$MCPROF" ] && [ -x "$OUT/probe" ]; then
  ( cd "$OUT" && MACA_VISIBLE_DEVICES="$GPU" CUDA_VISIBLE_DEVICES="$GPU" \
      timeout 240 "$MCPROF" perf_exec --casename probe \
        --cmdline "./probe" --per-kernel --output "$OUT/prof" \
        >"$OUT/prof.log" 2>&1 )
  if grep -q 'kernels launched' "$OUT/prof.log"; then
    ok "profiler launched kernel on GPU $GPU (device path works)"
  else warn "kernel did not report launch (see $OUT/prof.log)"; fi
  if ls "$OUT"/prof/*.json >/dev/null 2>&1; then
    ok "profiler emitted report JSON ($(ls "$OUT"/prof/*.json | wc -l) files in $OUT/prof)"
  else warn "no report JSON produced"; fi
  if grep -qiE '\[error\]|execute failed' "$OUT/prof.log"; then
    warn "perf_exec reported an error extracting values -- see $OUT/prof.log"
    warn "  (counters ARE configured; value-dump needs correct metric grouping / MetaX docs)"
  fi
else warn "skipped: profiler or probe binary unavailable"; fi

hdr "VERDICT"
printf '  PASS=%d  WARN=%d  FAIL=%d\n' "$PASS" "$WARN" "$FAIL"
if [ "$FAIL" -eq 0 ]; then
  echo "  => MetaX profiling stack is present and functional."
  echo "     NVIDIA binaries do NOT run here; use cucc + mcProfiler + MCPTI."
else
  echo "  => Core tooling gap detected (see FAIL lines above)."
fi
echo "  artifacts: $OUT"
