#!/usr/bin/env bash
# Profile implicit-BVH vs Warp BVH query kernels with Nsight Compute (run with sudo).
# Writes the report, raw metrics, source counters and the launch manifest to OUTPUT_DIR.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUTPUT_DIR=/tmp/implicit_bvh_ncu
REPORT="${OUTPUT_DIR}/queries"
NCU=/usr/local/cuda/bin/ncu
PYTHON="${ROOT_DIR}/.venv/bin/python"

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
cd "${ROOT_DIR}"

set +e
"${NCU}" \
  --set full \
  --import-source yes \
  --target-processes all \
  --replay-mode kernel \
  --profile-from-start off \
  --force-overwrite \
  --export "${REPORT}" \
  "${PYTHON}" local_studies/implicit_tree_bvh/ncu_driver.py "${OUTPUT_DIR}/manifest.txt"
capture_status=$?
set -e

if [[ ! -r "${REPORT}.ncu-rep" ]]; then
  printf "ERROR: ncu exited with code %d and did not create %s.ncu-rep\n" "${capture_status}" "${REPORT}" >&2
  exit 1
fi

"${NCU}" --import "${REPORT}.ncu-rep" --csv --page raw >"${OUTPUT_DIR}/metrics.csv" || true
"${NCU}" --import "${REPORT}.ncu-rep" --csv --page details >"${OUTPUT_DIR}/details.csv" || true
"${NCU}" --import "${REPORT}.ncu-rep" --csv --page source --print-source sass >"${OUTPUT_DIR}/source_sass.csv" || true
chmod -R a+rwX "${OUTPUT_DIR}"
printf "Done. Results in %s (report, metrics.csv, details.csv, source_sass.csv, manifest.txt)\n" "${OUTPUT_DIR}"
if (( capture_status != 0 )); then
  printf "The application exited with code %d after capture; the saved report is still usable.\n" "${capture_status}"
fi
