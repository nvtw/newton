#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Capture targeted counter reports for the five current G1 physics hotspots.
# Run from anywhere with one command:
#   sudo bash /home/twidmer/Documents/git/newton/newton/_src/solvers/phoenx/analysis_tools/ncu_profile_g1_hotspots.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../../../.." && pwd)"
OUT=/tmp/phoenx_g1_ncu_production_latest
REAL_USER="${SUDO_USER:-${USER:-root}}"
mkdir -p "$OUT"

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

{
  echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo=$REPO"
  echo "commit=$(git -C "$REPO" rev-parse HEAD)"
  echo "ncu=$NCU"
  "$REPO/.venv/bin/python3" -c "import warp as wp; wp.init(); print(wp.get_device(\"cuda:0\"))"
  git -C "$REPO" status --short
} > "$OUT/metadata.txt"

names=(reduced_advance reduced_advance_publish reduced_factor reduced_contact_rows reduced_contact_solve)
for name in "${names[@]}"; do
  script="$HERE/ncu_profile_${name}.sh"
  base="$OUT/${name}"
  echo "=== $name ==="
  if [ -s "$base.ncu-rep" ] && [ -s "$base.csv" ]; then
    echo "Reusing completed $base.ncu-rep"
    continue
  fi
  bash "$script" "$base" 2>&1 | tee "$base.log"
  "$NCU" --import "$base.ncu-rep" --page details --csv > "$base.csv" 2> "$base.import.log"
done

if id "$REAL_USER" >/dev/null 2>&1; then
  chown -R "$REAL_USER:$REAL_USER" "$OUT" 2>/dev/null || true
fi
chmod -R a+rX "$OUT"
echo "WROTE $OUT (five .ncu-rep files, CSV exports, logs, and metadata)"
