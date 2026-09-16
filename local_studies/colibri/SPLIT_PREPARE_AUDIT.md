# Copy-aware parallel contact preparation

The local prototype separates independent per-point geometry preparation from
ordered per-column warm-start accumulation. Each column retains its existing
copy counts, slot lookup, point sum order and body/copy writeback. The existing
broadcast and average schedule remains unchanged. Shared header writes are
forbidden in the point kernel. Production guards and defaults remain unchanged.

## Validation

- Tiny ragged 3+1+1-point fixture with changing copy counts over eight updates:
  baseline and candidate poses, velocities, derived rows, impulses, material
  anchors and copy state are array-exact. All six momentum components and
  non-increasing energy pass without changing their thresholds.
- Actual full-source first preparation enters with bit-identical state and
  headers. Only 222 effective-mass entries differ, at most 3 FP32 ULP (absolute
  5.82e-11). All other derived rows, anchors, impulses, body velocities and copy
  velocities remain exact.
- Independent FP64 calculation over all 5289 source mass rows gives both routes
  maximum relative error 5.015e-7 and maximum absolute error 1.490e-9.
  Mean relative errors are 4.876e-8 baseline and 4.806e-8 candidate.
  The source mass expression is unchanged; compiler arithmetic differences are
  consistent with this evidence, though individual generated instructions have
  not been attributed.
- Full 1200-frame / 20-second candidate passes fresh contact, joint, coloring
  and contact-point ownership checks: maximum penetration 184.77 um, final
  29.48 um. No physical tolerance, mass, damping or SOR change.

The frozen trace contains mixed colored and overflow constraints, ragged contact
columns, kinematic/ground endpoints and the actual mechanism. A tiny fixture
alone had missed the arithmetic differences. Full trajectories therefore are
not bit-identical and must not be described as such.

## Timing

Controlled 300-frame runs use 30 temporal steps per 120 Hz update, exact measured
PhysX shape offsets, 5 mm predictive search cap, source SDF/cylinders/gravity,
chunk size six, mass-split batch two, head launch chunk one and unit-aware tail.

| Route | Mean physics time | FPS |
| --- | ---: | ---: |
| Serial geometry baseline | 25.613 ms | 39.04 |
| Local parallel geometry | 22.642 ms | 44.17 |

This remains below the requested 60 FPS target.

## Files and artifacts

- parallel_prepare_split.py: process-local installer.
- Canonical private geometry factory and _contact_cached_warmstart_split now
  live in constraint_contact_cloth.py. The full source clone was removed.
- Production test_rigid_split_prepare.py exercises mixed colored/overflow
  columns, changing copies, FP64 mass bounds and all six momentum components.
  Substituting the existing unsplit warm-start helper reproduces divergence
  starting at frame one; the copy-aware helper passes exact comparisons.
- test_parallel_prepare_split.py: tiny invariant and exactness regression.
- trace_split_prepare.py: first-five-preparation full-source snapshots.
- audit_split_prepare_precision.py: bitwise snapshot and FP64 checks.
- check_parallel_prepare_split.py: full validation and saved final state.
- /tmp/colibri_splitprep_trace_reference_inertia.npz
- /tmp/colibri_splitprep_trace_candidate.npz
- /tmp/colibri_splitprep_{reference,candidate}_head1_300.json
- /tmp/colibri_splitprep_candidate_head1_1200.json
- Corresponding full-run NPZ files include final contact state and graph CSR.
