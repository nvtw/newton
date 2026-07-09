#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Profile exactly one steady-state projected-maximal tree recurrence launch.
# Warmup and graph capture occur before the CUDA profiler-API window.
#
# Run from anywhere:
#   sudo bash newton/_src/solvers/phoenx/analysis_tools/ncu_profile_maximal_projector.sh
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../../.." && pwd)"
PY="${PHOENX_PY:-$REPO/.venv/bin/python3}"
OUT_BASE="${1:-/tmp/phoenx_g1_maximal_projector_latest}"
REPORT="${OUT_BASE}.ncu-rep"

if [ -x /usr/local/cuda-13.2/bin/ncu ]; then
  NCU=/usr/local/cuda-13.2/bin/ncu
elif [ -x /usr/local/cuda-13.1/bin/ncu ]; then
  NCU=/usr/local/cuda-13.1/bin/ncu
else
  NCU="$(command -v ncu 2>/dev/null || true)"
fi

if [ -z "$NCU" ] || [ ! -x "$NCU" ]; then
  echo "ERROR: Nsight Compute CLI was not found." >&2
  exit 1
fi
if [ ! -x "$PY" ]; then
  echo "ERROR: Newton virtual-environment Python was not found at $PY." >&2
  exit 1
fi

if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != root ]; then
  export HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
fi
export HOME="${HOME:-$REPO/..}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO"
export PYTHONUTF8=1

printf "Profiling one projected-maximal recurrence launch\n  ncu: %s\n  python: %s\n  report: %s\n" "$NCU" "$PY" "$REPORT"

"$NCU" \
  --target-processes all \
  --replay-mode kernel \
  --profile-from-start off \
  --kernel-name "regex:_project_maximal_tree_kernel.*" \
  --launch-count 1 \
  --kill 1 \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section Occupancy \
  --section WarpStateStats \
  --section SchedulerStats \
  --force-overwrite \
  --export "$OUT_BASE" \
  "$PY" -m newton._src.solvers.phoenx.benchmarks.profile_g1_reduced_kernels \
    --articulation-mode maximal_projected \
    --projector-block-dim 128 \
    --world-count 8192 \
    --replays 1 \
    --warmup-replays 2 \
    --sim-substeps 5 \
    --solver-iterations 2 \
    --velocity-iterations 1
NCU_RC=$?

if [ -f "$REPORT" ]; then
  chmod 644 "$REPORT" 2>/dev/null || true
  echo "OK: $REPORT"
  exit 0
fi

echo "ERROR: ncu exited with code $NCU_RC and did not create $REPORT" >&2
exit "$NCU_RC"
