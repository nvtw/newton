#!/usr/bin/env bash
# Full-counter capture of one contact-inclusive biased solve/ABA/publish
# megakernel launch. The single launch keeps this privileged run bounded.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
REPORT="${1:-/tmp/phoenx_g1_biased_megakernel}"
REPORT="${REPORT%.ncu-rep}"
NCU=/usr/local/cuda-13.1/bin/ncu
PYTHON=${ROOT_DIR}/.venv/bin/python

cd "${ROOT_DIR}"
set +e
"${NCU}" \
  --set full \
  --target-processes all \
  --replay-mode kernel \
  --kernel-name "regex:_biased_contact_advance_publish_kernel.*" \
  --launch-count 1 \
  --force-overwrite \
  --export "${REPORT}" \
  "${PYTHON}" -m newton._src.solvers.phoenx.benchmarks.bench_g1_shared_physics \
    --world-count 8192 \
    --warmup-replays 1 \
    --measure-replays 1 \
    --articulation-mode reduced \
    --reduced-articulation-path persistent \
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
