#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Nsight Compute profile of the PhoenX *single-world* hot contact kernel
# (the Kapla tower's persistent PGS iterate -- ~53% of GPU time).
#
# Answers "is the contact iterate bandwidth-, compute-, or latency-bound, and
# is there headroom for software prefetch / more memory-level parallelism?" by
# capturing SpeedOfLight + memory-workload + occupancy + warp-stall +
# scheduler + launch sections for a handful of steady-state launches.
#
# GPU performance counters require elevated privileges, so run with sudo:
#
#     sudo bash ncu_profile_kapla.sh [LAUNCH_COUNT] [OUT_FILE] [KERNEL_REGEX] [LAUNCH_SKIP]
#
# Defaults: LAUNCH_COUNT=8, OUT=/tmp/phoenx_kapla_ncu.txt,
#           KERNEL_REGEX=singleworld_persistent, LAUNCH_SKIP=9.
# The skipped matches are the prepare sweep's nine colors, so profiling
# starts at the first iterate sweep.
#
# Notes:
#   * The runner forces a NON-conditional CUDA graph -- ``mass_splitting_unrolled``
#     (host-side fixed unroll instead of the PGS ``wp.capture_while``) +
#     ``capture_while_greedy_coloring=False`` (fixed-loop colourer). ncu cannot
#     replay conditional graph nodes, and the Kapla pipeline is not safe to run
#     eagerly (CUDA 700), so the captured graph is the only option and it must
#     be conditional-free. The unroll changes only the dispatch, not the
#     persistent iterate kernel being measured.
#   * The window is gated with the CUDA profiler API (cudaProfilerStart/Stop +
#     --profile-from-start off) so build / warmup launches are excluded.
#   * HOME is pinned so root reuses the user's Warp kernel cache (no recompile).
#   * The three persistent kernels (prepare / iterate / relax) share a closure
#     name; the iterate is the most frequent and ~6-7 us. The summary prints
#     each profiled launch's duration so the iterate is easy to pick out.
set -u

LAUNCH_COUNT="${1:-8}"
OUT="${2:-/tmp/phoenx_kapla_ncu.txt}"
KERNEL_REGEX="${3:-singleworld_persistent}"
LAUNCH_SKIP="${4:-9}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../../.." && pwd)"
PY="${PHOENX_PY:-$REPO/.venv/bin/python3}"
NCU="$(command -v ncu || echo /usr/local/cuda/bin/ncu)"
RUNNER="$(mktemp /tmp/phoenx_kapla_runner.XXXXXX.py)"

export HOME="${SUDO_USER:+/home/$SUDO_USER}"
[ -n "$HOME" ] && [ -d "$HOME" ] || export HOME="$(cd "$REPO/.." && pwd)"
export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO" PYTHONUTF8=1

echo "launch_count=$LAUNCH_COUNT  launch_skip=$LAUNCH_SKIP  kernel=/$KERNEL_REGEX/  ncu=$NCU  out=$OUT"
[ -x "$NCU" ] || { echo "ERROR: ncu not found (install Nsight Compute)"; exit 1; }
[ -x "$PY" ]  || { echo "ERROR: python not found at $PY (set PHOENX_PY)"; exit 1; }

cat > "$RUNNER" <<'PYEOF'
import ctypes
import functools
import sys

import warp as wp

sys.argv = ["kapla", "--viewer", "null", "--num-frames", "100"]

import newton.examples  # noqa: E402
from newton._src.solvers.phoenx import solver_phoenx as _sp  # noqa: E402
from newton._src.solvers.phoenx.examples import example_kapla_tower as ex  # noqa: E402

# Force a conditional-free graph so ncu can replay the persistent kernels:
# host-unrolled mass-splitting PGS + fixed-loop greedy colouring.
_orig_init = _sp.PhoenXWorld.__init__


@functools.wraps(_orig_init)
def _patched_init(self, *a, **k):
    k.setdefault("mass_splitting_unrolled", True)
    k.setdefault("capture_while_greedy_coloring", False)
    return _orig_init(self, *a, **k)


_sp.PhoenXWorld.__init__ = _patched_init

# CUDA profiler-API gate (paired with ncu --profile-from-start off).
_cudart = None
for _n in ("libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11"):
    try:
        _cudart = ctypes.CDLL(_n)
        break
    except OSError:
        pass
_start = (lambda: _cudart.cudaProfilerStart()) if _cudart else (lambda: None)
_stop = (lambda: _cudart.cudaProfilerStop()) if _cudart else (lambda: None)

viewer, args = newton.examples.init()
example = ex.Example(viewer, args)

# Warm up via the captured graph so the steady contact set is established.
for _ in range(15):
    example.step()
wp.synchronize_device()

_start()
example.step()  # one frame: replays the conditional-free graph
wp.synchronize_device()
_stop()
PYEOF

echo "=== running ncu (eager kapla, $LAUNCH_COUNT launches x a few passes; a couple minutes) ==="
"$NCU" \
  --target-processes all \
  --replay-mode kernel \
  --profile-from-start off \
  --kernel-name "regex:$KERNEL_REGEX" \
  --launch-skip "$LAUNCH_SKIP" \
  --launch-count "$LAUNCH_COUNT" \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --section WarpStateStats \
  --section SchedulerStats \
  --section LaunchStats \
  "$PY" "$RUNNER" > "$OUT" 2>&1
NCU_RC=$?
rm -f "$RUNNER"
chmod 644 "$OUT" 2>/dev/null

echo ""
echo "==================== KEY METRICS (full report: $OUT) ===================="
grep -iE "Compute \(SM\) Throughput|Memory Throughput|DRAM Throughput|L2 Cache Throughput|L1/TEX Cache Throughput|Achieved Occupancy|Theoretical Occupancy|Block Limit Registers|Registers Per Thread|Stall Long Scoreboard|Stall MIO|Stall Wait|Stall Short|Eligible Warps Per Scheduler|Issued Warp Per Scheduler|Active Threads Per Warp|Duration|Waves Per SM" "$OUT" | head -60
echo "========================================================================"
if grep -qiE "Compute \(SM\) Throughput" "$OUT"; then
  echo "OK: iterate profile captured after skipping the nine-color prepare sweep."
else
  echo "WARNING: no SpeedOfLight rows (ncu rc=$NCU_RC). Likely ERR_NVGPUCTRPERM -- rerun with sudo. Tail of $OUT:"
  tail -25 "$OUT"
fi
