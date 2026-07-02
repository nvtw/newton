#!/usr/bin/env bash
# Full-set Nsight Compute capture of the top reduced-pipeline kernels
# (ABA advance, factorization, contact gather, contact tile solve) on the
# steady G1 physics state. One launch per kernel keeps runtime bounded.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
REPORT="${1:-/tmp/phoenx_g1_reduced_pipeline}"
REPORT="${REPORT%.ncu-rep}"
NCU=/usr/local/cuda-13.1/bin/ncu
PYTHON=${ROOT_DIR}/.venv/bin/python

cd "${ROOT_DIR}"
set +e
"${NCU}" \
  --set full \
  --target-processes all \
  --replay-mode kernel \
  --kernel-name "regex:_advance_reduced_articulations_warp_kernel.*|_factor_reduced_warp_kernel.*|_gather_reduced_contact_blocks_kernel.*|_solve_generalized_contact_tile_kernel.*|_finish_and_publish_reduced_warp_kernel.*" \
  --launch-count 5 \
  --force-overwrite \
  --export "${REPORT}" \
  "${PYTHON}" -m newton._src.solvers.phoenx.benchmarks.bench_g1_shared_physics \
    --world-count 8192 \
    --warmup-replays 1 \
    --measure-replays 1 \
    --articulation-mode reduced \
    --solver-iterations 2 \
    --velocity-iterations 1
ncu_status=$?
set -e

if [[ ! -f "${REPORT}.ncu-rep" ]]; then
  exit "${ncu_status}"
fi

printf "Nsight Compute report: %s.ncu-rep\n" "${REPORT}"
if [[ "${ncu_status}" -ne 0 ]]; then
  printf "The profiled process exited with status %s after the report was captured.\n" "${ncu_status}" >&2
fi
