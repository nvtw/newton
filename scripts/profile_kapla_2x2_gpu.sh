#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

# Capture GPU activity for four Kapla towers in one PhoenX world.
# From anywhere: sudo /path/to/newton/scripts/profile_kapla_2x2_gpu.sh
set -euo pipefail

if [[ ${EUID} -ne 0 || -z ${SUDO_USER:-} ]]; then
    echo "Run this script with sudo so Nsight Systems can read GPU counters." >&2
    exit 1
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
user_home="$(getent passwd "${SUDO_USER}" | cut -d: -f6)"
nsys_bin="$(command -v nsys || true)"
if [[ -z ${nsys_bin} && -x /usr/local/bin/nsys ]]; then
    nsys_bin=/usr/local/bin/nsys
fi
uv_bin="${user_home}/.local/bin/uv"
report_base=/tmp/kapla_2x2_gpu_metrics

if [[ -z ${nsys_bin} || ! -x ${nsys_bin} ]]; then
    echo "Nsight Systems (nsys) was not found." >&2
    exit 1
fi
if [[ -z ${user_home} || ! -x ${uv_bin} ]]; then
    echo "uv was not found at ${uv_bin}." >&2
    exit 1
fi

cd "${repo_dir}"
"${nsys_bin}" profile \
    --trace=cuda,nvtx \
    --cuda-graph-trace=node \
    --gpu-metrics-devices=0 \
    --gpu-metrics-frequency=10000 \
    --sample=none \
    --run-as="${SUDO_USER}" \
    --output="${report_base}" \
    --force-overwrite=true \
    /usr/bin/env HOME="${user_home}" XDG_CACHE_HOME="${user_home}/.cache" \
    UV_CACHE_DIR="${user_home}/.cache/uv" "${uv_bin}" run --no-sync python \
    scripts/benchmark_kapla_optix.py --tower-grid 2x2 --num-frames 80

"${nsys_bin}" export --type=sqlite --output="${report_base}.sqlite" \
    --force-overwrite=true "${report_base}.nsys-rep"
if ! kernel_count="$(sqlite3 "${report_base}.sqlite" 'SELECT count(*) FROM CUPTI_ACTIVITY_KIND_KERNEL;')"; then
    echo "The report contains no CUDA kernel table; the benchmark did not run." >&2
    exit 1
fi
if [[ ${kernel_count} -eq 0 ]]; then
    echo "The report contains no CUDA kernels; the benchmark did not run." >&2
    exit 1
fi
chown "${SUDO_USER}:" "${report_base}.nsys-rep" "${report_base}.sqlite"
echo "Captured ${kernel_count} CUDA kernel launches."
echo "GPU metric report: ${report_base}.nsys-rep"
echo "SQLite data:      ${report_base}.sqlite"
