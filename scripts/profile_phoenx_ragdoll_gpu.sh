#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $EUID -ne 0 || -z ${SUDO_USER:-} || ${SUDO_USER:-} == root ]]; then
    echo "Run: sudo ./scripts/profile_phoenx_ragdoll_gpu.sh" >&2
    exit 1
fi

owner=$SUDO_USER
repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
owner_home=$(getent passwd "$owner" | cut -d: -f6)
uv_bin=""
for candidate in "$owner_home/.local/bin/uv" "$owner_home/.cargo/bin/uv" /usr/local/bin/uv /usr/bin/uv; do
    if [[ -x $candidate ]]; then
        uv_bin=$candidate
        break
    fi
done
[[ -n $uv_bin ]] || { echo "Cannot find uv for $owner." >&2; exit 1; }
[[ -x $repo_dir/.venv/bin/python ]] || { echo "Project .venv is missing." >&2; exit 1; }

ncu_bin=""
for candidate in /usr/local/cuda-13.2/bin/ncu /usr/local/cuda-13.1/bin/ncu /usr/local/cuda/bin/ncu; do
    if [[ -x $candidate ]]; then
        ncu_bin=$candidate
        break
    fi
done
[[ -n $ncu_bin ]] || { echo "Nsight Compute is missing." >&2; exit 1; }

# Counters require the application to retain root privileges. Use private
# writable caches and the existing project environment, without syncing it.
report_dir=$(mktemp -d /tmp/phoenx-ragdoll-ncu.XXXXXXXX)
trap 'chown -R "$owner":"$(id -gn "$owner")" "$report_dir"' EXIT
export UV_CACHE_DIR="$report_dir/uv-cache"
export WARP_CACHE_PATH="$report_dir/warp-cache"
mkdir -p "$UV_CACHE_DIR" "$WARP_CACHE_PATH"
warp_version=$("$uv_bin" run --directory "$repo_dir" --no-sync python -c 'import warp; print(warp.config.version)')
if [[ -d $owner_home/.cache/warp/$warp_version ]]; then
    cp -a --reflink=auto -- "$owner_home/.cache/warp/$warp_version" "$WARP_CACHE_PATH/"
fi
echo "Reports: $report_dir"
echo "Building the full scene and warming 300 frames; progress prints every 60 frames."
"$ncu_bin" --target-processes all --graph-profiling node --profile-from-start off \
    --kernel-name 'regex:phoenx_singleworld_iterate_.*' --launch-count 2 \
    --section SpeedOfLight --section Occupancy --section WarpStateStats \
    --section MemoryWorkloadAnalysis --export "$report_dir/iterate" \
    "$uv_bin" run --directory "$repo_dir" --no-sync \
    python "$repo_dir/scripts/profile_phoenx_ragdoll_iter.py"

[[ -s $report_dir/iterate.ncu-rep ]] || { echo "No kernel report was produced." >&2; exit 1; }
"$ncu_bin" --import "$report_dir/iterate.ncu-rep" --page raw --csv > "$report_dir/metrics.csv"
"$ncu_bin" --import "$report_dir/iterate.ncu-rep" --page details > "$report_dir/summary.txt"
echo "Completed: $report_dir/iterate.ncu-rep"
echo "Metrics:   $report_dir/metrics.csv"
echo "Summary:   $report_dir/summary.txt"
