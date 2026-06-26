#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Nsight Compute profile of the multi-world block_world prepare+iterate kernel
# on the Anymal RL walk-training env -- the ~59%-of-sim hot kernel for the
# real RL application.
#
# Answers the warp-fill / occupancy / register-pressure question for the
# block_world iterate (one CUDA block per world): SpeedOfLight + memory +
# occupancy + warp-stall + scheduler + launch sections for a few steady-state
# launches.
#
# GPU performance counters require elevated privileges, so run with sudo:
#
#     sudo bash ncu_profile_anymal.sh [NUM_WORLDS] [LAUNCH_COUNT] [OUT_FILE]
#
# Defaults: NUM_WORLDS=4096, LAUNCH_COUNT=6, OUT=/tmp/phoenx_anymal_ncu.txt
#
# Notes:
#   * Forces the fixed-loop greedy colourer (capture_while conditional graphs
#     are not ncu-replayable) and drives the env eagerly (no graph capture).
#   * Profiler-API gated + HOME-pinned so sudo reuses the user's warp cache.
#   * The iterate is ``..._prepare_plus_iterate_kernel...``; relax is a separate
#     kernel. The summary prints Registers Per Thread, Achieved Occupancy,
#     Active Threads Per Warp, Eligible Warps Per Scheduler and the Long
#     Scoreboard stall -- the warp-fill / latency-hiding signals.
set -u

NUM_WORLDS="${1:-4096}"
LAUNCH_COUNT="${2:-6}"
OUT="${3:-/tmp/phoenx_anymal_ncu.txt}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../../.." && pwd)"
PY="${PHOENX_PY:-$REPO/.venv/bin/python3}"
NCU="$(command -v ncu || echo /usr/local/cuda/bin/ncu)"
RUNNER="$(mktemp /tmp/phoenx_anymal_runner.XXXXXX.py)"

export HOME="${SUDO_USER:+/home/$SUDO_USER}"
[ -n "$HOME" ] && [ -d "$HOME" ] || export HOME="$(cd "$REPO/.." && pwd)"
export PYTHONNOUSERSITE=1 PYTHONPATH="$REPO" PYTHONUTF8=1

echo "worlds=$NUM_WORLDS launch_count=$LAUNCH_COUNT  ncu=$NCU  out=$OUT"
[ -x "$NCU" ] || { echo "ERROR: ncu not found (install Nsight Compute)"; exit 1; }
[ -x "$PY" ]  || { echo "ERROR: python not found at $PY (set PHOENX_PY)"; exit 1; }

cat > "$RUNNER" <<PYEOF
import ctypes
import warp as wp

# Fixed-loop colourer (capture_while conditional graphs break ncu replay).
import newton._src.solvers.phoenx.graph_coloring.graph_coloring_incremental as gci
gci.IncrementalContactPartitioner.set_capture_while_greedy = (
    lambda self, enabled=False: setattr(self, "_use_capture_while_greedy", False)
)
from newton._src.solvers.phoenx.rl_training.anymal import (
    EnvAnymalPhoenX, ConfigEnvAnymalPhoenX, ACTION_DIM_ANYMAL,
)
env = EnvAnymalPhoenX(
    ConfigEnvAnymalPhoenX(
        world_count=$NUM_WORLDS, sim_substeps=4, solver_iterations=8,
        velocity_iterations=1, reward_mode="dense_command",
        command=(1.0, 0.0, 0.0, 0.0), auto_reset=False,
    ),
    device=wp.get_device(),
)
actions = wp.zeros(($NUM_WORLDS, ACTION_DIM_ANYMAL), dtype=wp.float32, device=env.device)
for _ in range(15):
    env.step(actions)
wp.synchronize_device()

_cudart = None
for _n in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
    try:
        _cudart = ctypes.CDLL(_n); break
    except OSError:
        pass
if _cudart: _cudart.cudaProfilerStart()
env.step(actions)
wp.synchronize_device()
if _cudart: _cudart.cudaProfilerStop()
PYEOF

echo "=== running ncu (eager anymal, $LAUNCH_COUNT launches x a few passes; a couple minutes) ==="
"$NCU" \
  --target-processes all \
  --replay-mode kernel \
  --profile-from-start off \
  --kernel-name "regex:block_world_prepare_plus_iterate" \
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
grep -iE "Compute \(SM\) Throughput|Memory Throughput|DRAM Throughput|L2 Cache Throughput|Registers Per Thread|Block Limit Registers|Achieved Occupancy|Theoretical Occupancy|Active Threads Per Warp|Eligible Warps Per Scheduler|Issued Warp Per Scheduler|Stall Long Scoreboard|Stall MIO|Stall Wait|Warp Cycles Per Issued|Duration|Waves Per SM|Block Size|Grid Size" "$OUT" | head -60
echo "========================================================================"
if grep -qiE "Compute \(SM\) Throughput" "$OUT"; then
  echo "OK: profile captured."
else
  echo "WARNING: no SpeedOfLight rows (ncu rc=$NCU_RC). Likely ERR_NVGPUCTRPERM -- rerun with sudo. Tail of $OUT:"
  tail -25 "$OUT"
fi
