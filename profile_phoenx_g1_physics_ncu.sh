#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

set -u

RUN_PREFIX="${1:-/tmp/newton_phoenx_g1_physics_ncu}"
RUN_DIR="$(mktemp -d "${RUN_PREFIX}.XXXXXX")"
OUT_BASE="$RUN_DIR/newton_phoenx_g1_physics_ncu"
SUMMARY="${OUT_BASE}_summary.txt"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${NEWTON_NCU_PY:-$HERE/.venv/bin/python3}"
NCU="${NEWTON_NCU_BIN:-/usr/local/cuda/bin/ncu}"

if [ -n "${SUDO_USER:-}" ] && [ -d "/home/$SUDO_USER" ]; then
    export HOME="/home/$SUDO_USER"
fi
export PYTHONNOUSERSITE=1 PYTHONPATH="$HERE" PYTHONUTF8=1

[ -x "$NCU" ] || { echo "ERROR: Nsight Compute not found at $NCU"; exit 1; }
[ -x "$PY" ] || { echo "ERROR: Python not found at $PY"; exit 1; }

FAMILIES=(advance factor contact_solve contact_build)
FILTERS=(
    'regex:.*(_advance_reduced_articulations_warp_kernel|_advance_and_publish_reduced_articulations_warp_kernel).*'
    'regex:.*_factor_reduced_multidof_kernel.*'
    'regex:.*_solve_patch_contact_tile_kernel.*'
    'regex:.*_build_packed_patch_rows_warp_kernel.*'
)
COUNTS=(12 4 4 4)

printf 'PhoenX G1 physics Nsight Compute summary\n' > "$SUMMARY"
printf 'Worlds: 2048; warmed captured environment; two profiled graph replays.\n\n' >> "$SUMMARY"
FAILED=0
for index in "${!FAMILIES[@]}"; do
    family="${FAMILIES[$index]}"
    filter="${FILTERS[$index]}"
    count="${COUNTS[$index]}"
    base="${OUT_BASE}_${family}"
    report="${base}.ncu-rep"
    text="${base}.txt"
    details="${base}_details.csv"
    raw="${base}_raw.csv"

    echo "Profiling $family (up to $count matched launches)."
    "$NCU" \
        --target-processes all \
        --replay-mode kernel \
        --kernel-name "$filter" \
        --launch-count "$count" \
        --section SpeedOfLight \
        --section MemoryWorkloadAnalysis \
        --section Occupancy \
        --section WarpStateStats \
        --section SourceCounters \
        --section SchedulerStats \
        --section LaunchStats \
        --import-source yes \
        --export "$base" \
        --force-overwrite \
        "$PY" -m newton._src.solvers.phoenx.benchmarks.profile_g1_reduced_kernels \
            --world-count 2048 --warmup-replays 5 --replays 2 \
            > "$text" 2>&1
    ncu_rc=$?

    if [ -f "$report" ]; then
        "$NCU" --import "$report" --page details --csv > "$details" 2>&1
        "$NCU" --import "$report" --page raw --csv > "$raw" 2>&1
    fi
    {
        printf '== %s ==\n' "$family"
        grep -iE 'Kernel Name|Duration|Compute \(SM\) Throughput|Memory Throughput|DRAM Throughput|L2 Cache Throughput|Achieved Occupancy|Theoretical Occupancy|Registers Per Thread|No Eligible|Eligible Warps Per Scheduler|Waves Per SM' "$details" | head -120
        printf '\nReport: %s\nDetails: %s\nRaw: %s\nDriver log: %s\n\n' "$report" "$details" "$raw" "$text"
    } >> "$SUMMARY"
    if ! grep -qi 'Compute (SM) Throughput' "$details"; then
        echo "ERROR: $family captured no performance-counter metrics (ncu exit $ncu_rc)." | tee -a "$SUMMARY"
        tail -30 "$text" >> "$SUMMARY"
        FAILED=1
    elif [ "$ncu_rc" -ne 0 ]; then
        echo "WARNING: $family counters are valid; target exited $ncu_rc after CUDA-graph kernel replay." | tee -a "$SUMMARY"
    fi
done

chmod 644 "$SUMMARY" "${OUT_BASE}"* 2>/dev/null
if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
    chown -R "$SUDO_USER" "$RUN_DIR"
fi

echo
echo "Run directory: $RUN_DIR"
echo "Summary: $SUMMARY"
echo "Per-family reports: ${OUT_BASE}_{advance,factor,contact_solve,contact_build}.ncu-rep"
echo "Per-family CSVs: ${OUT_BASE}_<family>_{details,raw}.csv"
exit "$FAILED"
