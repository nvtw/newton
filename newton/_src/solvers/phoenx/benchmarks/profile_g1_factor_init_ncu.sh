#!/usr/bin/env bash
# Profile one steady G1 reduced-factor initialization launch and export metrics.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
OUTPUT_DIR=/tmp/phoenx_g1_factor_init_latest
REPORT="${OUTPUT_DIR}/factor_init"
NCU=/usr/local/cuda-13.1/bin/ncu
PYTHON="${ROOT_DIR}/.venv/bin/python"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

"${NCU}" \
  --set full \
  --target-processes all \
  --replay-mode kernel \
  --kernel-name "regex:_initialize_reduced_factor_kernel.*" \
  --launch-count 1 \
  --force-overwrite \
  --export "${REPORT}" \
  "${PYTHON}" -m newton._src.solvers.phoenx.benchmarks.bench_g1_shared_physics \
    --world-count 8192 \
    --warmup-replays 1 \
    --measure-replays 1 \
    --articulation-mode reduced \
    --solver-iterations 2 \
    --velocity-iterations 1

"${NCU}" --import "${REPORT}.ncu-rep" --csv --page raw >"${OUTPUT_DIR}/factor_init_metrics.csv"
chmod -R a+rX "${OUTPUT_DIR}"
printf "Report: %s.ncu-rep\nMetrics: %s/factor_init_metrics.csv\n" "${REPORT}" "${OUTPUT_DIR}"
