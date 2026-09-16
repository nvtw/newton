# Public contact chunk integration audit

The canonical point-column chunk implementation matches all four local
prototype kernel ASTs exactly. Scratch allocation moves into world
construction; buffers do not resize afterward. Each ingest fully refreshes
source headers, source-pair mapping, counts and offsets. Splitting occurs
before warm-start gather and coloring; ownership is restored after gather.

Canonical regression coverage passes:
- Point ranges, source pairs, endpoints and owner metadata survive splitting
  and shrinking input ranges.
- Disabled/empty/None contacts and unsupported options behave explicitly.
- CUDA compound face contacts preserve physical momentum and do not add
  kinetic energy, including reversed body order and recycled point matches.
- Two-world CUDA graph replay retains all points and distinct endpoint pairs.

Removing only post-gather owner restamping makes both compound tests fail:
three of four contact IDs incorrectly remain the original unsplit owner.
With restamping both pass. No global monkeypatch is used in production.

## Public versus legacy trajectory hashes

Local wrapper: `hash_solver_trajectory.py`. It captures generated contacts
before ingestion and complete headers/maps/history after ingestion and
restamping at both120Hz updates, then hashes post-frame q/qd. Hashes use raw
bits, including packed-header NaNs. Captured copies and CPU hashing make
these accuracy diagnostics, not performance measurements.

Exact runner configuration: full36-prefix/37-body source model,30substeps
per120Hz update, one iteration, source SDF/cylinders, exact per-shape contact
offsets, predictive search cap.005m, chunk6, mass-split batch2, head launch1
and batch-aware unit tail. Legacy uses local fused joints/chunk installation
and split parallel preparation; public uses constructor-owned block joints,
chunks and parallel preparation. No damping/SOR/model changes.

The first120-frame pair has bit-identical q/qd, impulses, headers and maps
throughout. First difference atframe18/outer1 is a generated contact normal
BEFORE ingestion (max5.96e-8, identical witness points). That contact has
zero normal/tangent impulse. Shapes1/116 are:
`Hypocycloid_Gear__3x_02/Hypocycloid_Gear__3x_0_mesh` and
`Frame/Cylinders/Cylinder_10`.
Raw snapshots: `/tmp/colibri_hash_legacy/frame0018.npz`,
`/tmp/colibri_hash_public/first_difference.npz`.

The extended1200-frame pair first differs in qd atframe316 (maximum
1.788e-6rad/s); q first differs at317. Generated-normal differences precede
this, while headers/source-pair/owner maps still agree at316. Both runs
PASS physical checks. Public peak fresh depth is.122596mm, peak joint
anchor.920991mm, axis.00730915rad. End trajectories differ materially
(max q component.10394, qd component2.3796), so no trajectory identity is
claimed despite both passing.

Artifacts:
- `/tmp/colibri_hash_{legacy,public}_1200/hashes.json`
- compact SI q/qd history in each directory's `states.npz`
- public `first_state_difference.npz` includes both captured query body
  poses, shape transforms, generated contacts, headers and histories
- `first_state_reference.npz` contains counterpart legacy q/qd
- corresponding `/tmp/colibri_hash_*_1200_run.json` reports

The legacy1200 process imported just before the public velocity_relaxation
API was added; the public process imported afterward. The default policy
and existing local temporal wrapper were intentionally preserved, but this
source-provenance difference remains explicit.

These results do not demonstrate a public chunk integration regression.
They establish small contact-generation differences and sensitivity of the
long mechanism trajectory. They do not explain the separate transient
frame1133 failure; that robustness issue remains open.
