# Exact-zero cached warm-start experiments

Current public Colibri, group four/grid64, 30 temporal steps per 120 Hz query,
endpoint-corrected meshes, unchanged default predictive admission.

At warmed frame330, 319/347 contact columns (91.9%) contain only zero normal
and tangent impulses. Only47/1999 points have any nonzero impulse. Snapshot:
`/tmp/colibri_warm_column_snapshot.npz`; population counts:
`/tmp/colibri_warm_column_population.json`.

Two local candidates preserve the live solve and all contact metadata:

- `zero_warmstart_split.py` skips geometry/wrench accumulation for each exactly
  zero point, retaining ordered accumulation and column application.
- `zero_column_warmstart.py` scans impulses and returns before geometry/body
  response work only when the entire column is exactly zero.

Both pass96 frozen uint32-exact comparisons across all normal/tangent zero
patterns, ragged1/3/7-point columns, and direct/mixed/copy endpoints with unequal
copy counts. Both also pass the production three-body/two-manifold moving-COM
momentum and energy tests with the local callback installed.

Isolated public330-frame comparisons exclude30 warmup frames and rendering/
checks from step timing. All eight saved numeric arrays match byte-for-byte,
including all330 pose and velocity samples.

| Candidate | Reference ms/frame | Candidate ms/frame | Result |
| --- | ---: | ---: | --- |
| Per-point skip | 13.0741 | 13.2578 | 1.4% slower |
| Whole-column skip | 13.2118 | 13.2329 | No measurable improvement |

Reports: `/tmp/colibri_zero_split_{reference,candidate}330.json` and
`/tmp/colibri_zero_column_{reference,candidate}330.json`, with corresponding
state NPZs. GPU processes were checked absent before each isolated pair and
other agents held launches. Neither optimization merits canonical integration
on this evidence. Sparse impulses alone do not establish a GPU speed benefit.
