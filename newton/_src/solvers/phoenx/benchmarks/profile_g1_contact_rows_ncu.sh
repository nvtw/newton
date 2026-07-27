#!/usr/bin/env bash
# Profile one steady G1 point-contact row build and export metrics and source counters.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
OUTPUT_DIR=/tmp/phoenx_g1_contact_rows_latest
REPORT="${OUTPUT_DIR}/contact_rows"
NCU=/usr/local/cuda-13.1/bin/ncu
PYTHON="${ROOT_DIR}/.venv/bin/python"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

set +e
"${NCU}" \
  --set full \
  --target-processes all \
  --replay-mode kernel \
  --kernel-name "regex:.*build_packed_generalized_contact_rows_kernel.*" \
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
capture_status=$?
set -e

if [[ ! -r "${REPORT}.ncu-rep" ]]; then
  printf "ERROR: ncu exited with code %d and did not create %s.ncu-rep\n" "${capture_status}" "${REPORT}" >&2
  exit 1
fi

"${NCU}" --import "${REPORT}.ncu-rep" --csv --page raw >"${OUTPUT_DIR}/contact_rows_metrics.csv" || true
"${NCU}" --import "${REPORT}.ncu-rep" --csv --page source >"${OUTPUT_DIR}/contact_rows_source.csv" || true
chmod -R a+rX "${OUTPUT_DIR}"
printf "Report: %s.ncu-rep\n" "${REPORT}"
printf "Metrics: %s/contact_rows_metrics.csv\n" "${OUTPUT_DIR}"
printf "Source: %s/contact_rows_source.csv\n" "${OUTPUT_DIR}"
if (( capture_status != 0 )); then
  printf "The application exited with code %d after capture; the saved report is still usable.\n" "${capture_status}"
fi
