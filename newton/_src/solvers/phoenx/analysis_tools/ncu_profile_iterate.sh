#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Nsight Compute profile of the PhoenX multi-world hot iterate kernel.
#
# Answers the "is this kernel bandwidth-, compute-, or latency-bound?" question
# that drives PhoenX perf work, by capturing SpeedOfLight + memory-workload +
# occupancy + warp-stall sections for ONE steady-state launch of the fused
# ``prepare_plus_iterate`` kernel.
#
# GPU performance counters require elevated privileges, so this must run with
# sudo:
#
#     sudo bash ncu_profile_iterate.sh [SCENE] [NUM_WORLDS] [OUT_FILE]
#
# Defaults: SCENE=dr_legs, NUM_WORLDS=4096, OUT_FILE=/tmp/phoenx_ncu_report.txt
# SCENE is any key in ``benchmarks/scenarios`` (dr_legs, h1_flat, g1_flat, ...).
#
# Gotchas this script handles so ncu actually works on PhoenX:
#   * capture_while greedy colouring builds conditional CUDA graphs that ncu
#     cannot replay in eager mode -- it is forced to the fixed-loop colourer.
#   * the profiled window is gated with the CUDA profiler API
#     (cudaProfilerStart/Stop + ncu --profile-from-start off), so warmup /
#     JIT launches are excluded without needing the optional ``nvtx`` package.
#   * HOME is pinned so root reuses the user's Warp kernel cache (no recompile).
set -u

SCENE="${1:-dr_legs}"
NUM_WORLDS="${2:-4096}"
OUT="${3:-/tmp/phoenx_ncu_report.txt}"

# Repo root = five levels up from this file (.../newton/_src/solvers/phoenx/analysis_tools).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../../.." && pwd)"
PY="${PHOENX_PY:-$REPO/.venv/bin/python3}"
NCU="$(command -v ncu || echo /usr/local/cuda/bin/ncu)"
RUNNER="$(mktemp /tmp/phoenx_ncu_runner.XXXXXX.py)"

# Reuse the invoking user's Warp cache (sudo runs as root otherwise).
export HOME="${SUDO_USER:+/home/$SUDO_USER}"
[ -n "$HOME" ] && [ -d "$HOME" ] || export HOME="$(cd "$REPO/.." && pwd)"
export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO" PYTHONUTF8=1

echo "scene=$SCENE worlds=$NUM_WORLDS  ncu=$NCU  py=$PY  out=$OUT"
[ -x "$NCU" ] || { echo "ERROR: ncu not found (install Nsight Compute)"; exit 1; }
[ -x "$PY" ]  || { echo "ERROR: python not found at $PY (set PHOENX_PY)"; exit 1; }

cat > "$RUNNER" <<PYEOF
import ctypes, warp as wp
wp.init()
# CUDA profiler-API gate (paired with ncu --profile-from-start off).
_cudart = None
for _n in ("libcudart.so", "libcudart.so.12", "libcudart.so.13", "libcudart.so.11"):
    try:
        _cudart = ctypes.CDLL(_n); break
    except OSError:
        pass
_start = (lambda: _cudart.cudaProfilerStart()) if _cudart else (lambda: None)
_stop = (lambda: _cudart.cudaProfilerStop()) if _cudart else (lambda: None)
# Force the fixed-loop colourer (capture_while conditional graphs break ncu).
import newton._src.solvers.phoenx.graph_coloring.graph_coloring_incremental as gci
gci.IncrementalContactPartitioner.set_capture_while_greedy = (
    lambda self, enabled=False: setattr(self, "_use_capture_while_greedy", False)
)
from newton._src.solvers.phoenx.benchmarks.scenarios import SCENARIOS
h = SCENARIOS["$SCENE"].build(
    num_worlds=$NUM_WORLDS, solver_name="phoenx", substeps=4, solver_iterations=8
)
for _ in range(12):
    h.simulate_one_frame()
wp.synchronize_device()
_start()
for _ in range(3):
    h.simulate_one_frame()
wp.synchronize_device()
_stop()
PYEOF

echo "=== running ncu (one kernel, a few metric passes; a couple of minutes) ==="
"$NCU" \
  --target-processes all \
  --replay-mode kernel \
  --profile-from-start off \
  --kernel-name "regex:prepare_plus_iterate" \
  --launch-count 1 \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --section WarpStateStats \
  --section SchedulerStats \
  "$PY" "$RUNNER" > "$OUT" 2>&1
NCU_RC=$?
rm -f "$RUNNER"
chmod 644 "$OUT" 2>/dev/null

echo ""
echo "==================== KEY METRICS (full report: $OUT) ===================="
grep -iE "Compute \(SM\) Throughput|Memory Throughput|DRAM Throughput|Achieved Occupancy|Theoretical Occupancy|Block Limit|Stall|Issued Warp|Active Threads Per Warp|Duration|L2 Cache Throughput|L1/TEX" "$OUT" | head -40
echo "========================================================================"
if grep -qiE "Compute \(SM\) Throughput" "$OUT"; then
  echo "OK: profile captured."
else
  echo "WARNING: no SpeedOfLight rows (ncu rc=$NCU_RC). Tail of $OUT:"; tail -25 "$OUT"
fi
