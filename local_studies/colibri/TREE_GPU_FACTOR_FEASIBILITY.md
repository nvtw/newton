# Tree factor GPU feasibility

The live example remains at 30 substeps per 120 Hz collision update. These
are frozen-matrix studies, with no production solver changes.

## Actual vendor factor costs

`benchmark_tree_joint_gpu.py` reads the exported physical 46x46 reduced
mass, 14x46 loop velocity rows, and two drive slack columns. It captures
Cholesky, triangular solve, assembly/scaling and complete 48x14 QR using
Torch's CUDA backend. On the RTX PRO 6000 Blackwell:

| Operation | Median time |
| --- | ---: |
| Mass Cholesky | 67.58 us |
| Triangular solve | 43.00 us |
| Complete QR | 145.42 us |
| All operations together | 257.98 us |

The combined result agrees with the CPU QR input to 7.63e-15 relative.
The independently measured pieces need not sum exactly to the combined
capture. Results: `/tmp/colibri_tree_gpu_breakdown.json`.

## Dedicated complete QR

`benchmark_small_tree_qr.py` implements FP64 Householder QR with one block
of exactly 256 threads, fixed at 48x14. Cooperative warp reductions compute
norms and reflector products. Padded shared storage avoids row-access bank
conflicts. The input has unit-scaled columns; this is not an arbitrary-size
or arbitrary-magnitude QR API. It retains all 14 selected independent rows.

The initial scalar-norm prototype was slower than the vendor implementation.
The final cooperative version measures **74.63 us**, approximately 49% less
than the vendor QR component. No full-frame speedup is established.

Validation on frames 330, 1, 30, and 120:

- Physical mobility relative error versus the CPU tree reference is at most
  1.01e-12. This is not a new comparison against a high-precision oracle.
- Remaining original loop/drive row residual is at most 1.32e-11.
- Frame-330 QR reconstruction error is 4.21e-16 and orthogonality error
  8.89e-16.
- Repeated frame-330 factorization is byte-identical on this GPU. This does
  not establish cross-device or whole-solver determinism.

Results: `/tmp/colibri_small_tree_qr_gpu.json`.

## Dedicated mass preparation and QR

`benchmark_tree_prepare_gpu.py` adds a fixed-size FP64 Cholesky and
forward solve before the same QR, in one 256-thread block. The mass factor
has no diagonal floor or regularization: invalid pivots set a failure
status. The study uses the same actual physical matrices.

The combined captured setup measures **174.01 us**, versus 257.98 us for the
vendor sequence (approximately 33% less). This includes export of the
Cholesky factor for independent validation. At two setups per frame the
measured component would cost approximately 0.348 ms, excluding all remaining
operations listed below. It is not a live FPS result.

Across all four saved poses, using the actual GPU Cholesky in the physical
response reconstruction:

- Reduced mass reconstruction error is at most 1.94e-16 relative.
- Physical mobility error versus the CPU tree reference is at most 4.55e-12.
- QR reconstruction and orthogonality errors remain below 1.2e-15.
- Repeated frame330 QR output remains byte-identical on the tested GPU.

Results: `/tmp/colibri_tree_prepare_gpu.json`.

## Invalid-input controls

`check_tree_prepare_rejection.py` exposed two defects in the local prototype:
zero/nonfinite joint rows could retain a success status, and invalid mass
factors still overwrote the QR outputs. The fix validates normalization,
synchronizes the shared failure flag, and skips QR on failure. It adds no
mass floor, regularization, or physical-mode cutoff.

All six controls (negative, zero, NaN and infinite mass pivots; zero and NaN
joint rows) failed the required rejection/output-preservation checks before
the fix and pass afterward. The four valid pose checks still pass, with
unchanged numerical results and repeated output bytes. The post-fix setup
measurement is168.80us; its small difference from174.01us should not be
interpreted as a separate optimization gain.

Before/after evidence: `/tmp/colibri_tree_prepare_rejections_before.json`
and `/tmp/colibri_tree_prepare_rejections.json`. This remains a local
fixed-size prototype; the controls are not a generic rank-certification API.

## Integration constraints

These timings exclude tree construction, mass assembly, reaction recovery,
contact iterations, and state scatter. Rebuilding the vendor factors every
substep would cost approximately 15.48 ms per displayed frame before those
operations. A final-unbiased-relaxation-only experiment would require two
setups per frame, approximately 0.516 ms of vendor factor cost.

The actual biased closed-loop recovery targets are not all compatible with
the exact joint nullspace. Do not silently omit inconsistent equations to
enable a global biased solve. Retaining the native biased path and testing
an exact final unbiased solve is the bounded next integration route.

An accurate joint operator alone is insufficient: the 64-sweep CPU contact
reference still has normal/tangent residuals 0.01161/0.00642 m/s. A stronger
coupled contact solve must demonstrate convergence before this factor is
promoted. See `TREE_CONDENSED_JOINT_REFERENCE.md` and
`/tmp/colibri_joint_tree_sweeps64.json`.

Reproduce from this worktree:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.benchmark_tree_joint_gpu
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.benchmark_small_tree_qr
```

Both scripts require the exported reference matrices and use Torch only
inside the local benchmark. They introduce no production dependency.
