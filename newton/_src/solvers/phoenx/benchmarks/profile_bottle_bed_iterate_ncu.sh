#!/usr/bin/env bash
# Profile the PhoenX island iterate sweep on a packed bottle bed (the
# bottling plant's pasteuriser bed, bench_bottle_bed.py) and export metrics
# and SASS source counters with stall reasons. Run with sudo; writes to
# /tmp/phoenx_bottle_bed_ncu. Set PYTHON to the interpreter of a Newton
# environment (default: the checkout's .venv).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
OUTPUT_DIR=/tmp/phoenx_bottle_bed_ncu
REPORT="${OUTPUT_DIR}/bed_iterate"
NCU=/usr/local/cuda-13.2/bin/ncu
PYTHON="${PYTHON:-${ROOT_DIR}/.venv/bin/python}"

mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

# Step 240 frames (two steps each) uncaptured, then profile the first two
# biased sweeps of frame 241: each frame launches 24 iterate sweeps. Kernel
# replay of captured graph nodes is unreliable, hence the eager frames.
set +e
PYTHONPATH="${ROOT_DIR}" "${NCU}" \
  --set full \
  --import-source yes \
  --target-processes all \
  --replay-mode kernel \
  --kernel-name "regex:phoenx_iterate_islands.*" \
  --launch-skip 5760 \
  --launch-count 2 \
  --force-overwrite \
  --export "${REPORT}" \
  "${PYTHON}" -m newton._src.solvers.phoenx.benchmarks.bench_bottle_bed --bed 6 --warmup 240 --eager 1
capture_status=$?
set -e

if [[ ! -r "${REPORT}.ncu-rep" ]]; then
  printf "ERROR: ncu exited with code %d and did not create %s.ncu-rep\n" "${capture_status}" "${REPORT}" >&2
  exit 1
fi

"${NCU}" --import "${REPORT}.ncu-rep" --csv --page raw >"${OUTPUT_DIR}/bed_iterate_metrics.csv" || true
"${NCU}" --import "${REPORT}.ncu-rep" --csv --page details >"${OUTPUT_DIR}/bed_iterate_details.csv" || true
"${NCU}" --import "${REPORT}.ncu-rep" --csv --page source --print-source sass >"${OUTPUT_DIR}/bed_iterate_source.csv" || true
chmod -R a+rX "${OUTPUT_DIR}"
printf "Report: %s.ncu-rep\n" "${REPORT}"
printf "Metrics: %s/bed_iterate_metrics.csv\n" "${OUTPUT_DIR}"
if (( capture_status != 0 )); then
  printf "The application exited with code %d after capture; the saved report is still usable.\n" "${capture_status}"
fi
