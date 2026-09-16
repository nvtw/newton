# Actual PhysX GPU friction comparison

Scope: source comparison against `/home/twidmer/Documents/git/PhysX`, plus
native PhoenX CPU preparation and captured two-body frame330. No production
changes, no stationarity fix claimed.

## Concrete persistence discrepancy

PhoenX `constraints/constraint_contact_cloth.py:730` resets material anchors and
zeros both tangent multipliers when their previous norm reaches98% of the static
cone radius. This includes strictly interior sticking states. The separate
geometric separation and2mm material-drift break conditions remain relevant.

Actual PhysX GPU `physx/source/gpusolver/src/CUDA/solverBlockTGS.cuh:337–449`
first sums actual normal impulses over the friction patch, forms the current
unconstrained two-dimensional tangent trial, and marks `broken` only when the
trial norm exceeds the static radius. `contactConstraintBlockPrep.cuh:256–301`
retains unbroken matching patch anchors, with normal/geometric correlation checks.
This is not just a different previous-impulse threshold: it uses actual current
solve information and persistent patch identity.

`check_interior_anchor_break.py` executes actual native preparation on CPU with
lambda_t=0.99*actual_mixed_mu*lambda_n. It reproduces anchor and tangent-history
erasure while keeping collision witnesses unchanged, then verifies recovery from
new displacement. This is a diagnostic asserting observed behavior, not a
passing regression asserting correct sticking. Original log:
`/tmp/colibri_interior_anchor_break.log` (session68755 exit0).

`audit_anchor_break_ring.py` reads actual before/after preparation boundaries in
the trajectory-byte-verified two-body frame330. Among35 strictly interior
previous near-cap impulses, all35 have their material anchors changed and tangent
multipliers cleared;23 are overlapping contacts. Artifact:
`/tmp/colibri_anchor_break_ring.json`. This does not establish that every current
unconstrained trial would remain inside its cone: the original solver does not
save that trial or a broken bit.

## What is already equivalent, and what is not

- PhoenX already transfers material anchors through matched contact identity;
  `contact_ingest.py:963–988`. An unconditional history-loss claim is false.
- PhysX GPU uses patch-total actual normal impulse. Canonical PhoenX still uses
  the previously diagnosed diagonal recovery subtraction; local total-normal
  controls correct that convention but do not eliminate drift.
- PhysX `concludeContactBlockTGS` zeroes friction bias before velocity passes
  (`solverBlockTGS.cuh:631–652`). Its velocity-pass loop follows all position
  passes (`PxgTGSCudaSolverCore.cpp:1560–1623`). One final velocity pass per30
  temporal passes is therefore not itself a discovered mismatch.
- PhysX patch anchors couple patch load and friction at at most two anchors.
  PhoenX per-point/chunk friction is not equivalent; porting patch friction changes
  torque capacity/distribution and requires explicit yaw and sliding validation.
- Neither ordered sequential solve guarantees both joint and contacts remain
  converged after one sweep. Mill's native replay proves the immediate joint
  solution is correct; subsequent contacts and averaging reopen it. Merely moving
  the joint last risks moving the residual into ground friction instead.

## Bounded actionable correction

Replace the previous98%-cap break inference with explicit actual-projection
break state, updated by the tangent solve when its unconstrained trial exceeds
its static cone. Carry that bit with matched material anchors and packed/scattered
contact rows. Preserve independent geometric-break conditions. Do not assume a
boundary impulse necessarily implies slip: an exactly static limiting load can
also lie on the boundary. Do not replace98% with another arbitrary percentage.

Before a live comparison, test interior99% sticking, exact limiting static load,
true sliding with mu_static>mu_dynamic, rotating normals/contact reorder, and
warm-start-disabled persistence. Then compare two-body drift and all actual joint
and contact residuals under the same30x1 budget. This addresses a real premature
history-erasure path; it is not evidence that coupled convergence is solved.

## Executed explicit-break candidate

`friction_break_state.py` installs a source overlay before importing Newton.
A thirteenth persistent material-history row records the latest actual tangent
projection's static-cone exceedance. It is overwritten, allowing resticking,
and consumed during contact ingest at120Hz, not each temporal prepare. Matching,
packed gather/scatter and new-contact reset carry/consume it explicitly.
Ordinary scalar/frame/metric point projections preserve their numerical impulse
operations. This local control leaves all positive-gap geometric resets intact.

Native fail-before/pass-after tests cover interior99% sticking, true sliding,
exact cone boundary, changing/zero normal load, resticking, no premature
microstep consumption, packed matching, warm-start-disabled transfer, and
recontact. CPU expected impulse/witness snapshots use `.copy()` because Warp
CPU NumPy views alias storage. Clean-cache session15370 passed3tests; added
restick/unload test43079 and alias-safe preparation1026 passed separately.
An initial combined process exited139 after unittestOK; subsequent isolated,
combined, and clean-cache runs exited0. No unexplained exit is counted as a pass.

Matched60s controls, same authored two-body model/30x1/120Hz:

| Law/history | Base10–60s net translation | Base10–60s rotation | Final hinge angle |
|---|---:|---:|---:|
| Original |41.5420mm|12.9356deg|25.08306deg|
| Original + explicit break |41.6771mm|12.9723deg|25.08294deg|
| Actual total normal |1.99029mm|0.113413deg|25.08173deg|
| Total normal + explicit break |0.198381mm|0.033109deg|25.08189deg|
| Above + prepare positive-gap retention (root control) |0.192299mm|0.006293deg|25.08197deg|

Both explicit-break60s runs passed geometry checks and source guards. They do
NOT establish stationarity: late50–60s speed remains5.23µm/s with break+totalnormal
and4.07µm/s with the additional gap control. The hinge remains near25.082deg;
base improvement did not correct its coupled constitutive equilibrium. PhysX's
15.729deg is a comparison, not an analytic correctness target.

Artifacts `/tmp/colibri_base_frame_break_{original,totalnormal}3600.*` and
`/tmp/colibri_friction_break_paired_motion.json`. No production defaults changed.

## Analytic native load-law regression

`test_stabilized_coulomb_support.py` constructs an independent full coupled
sagittal two-support normal/friction equilibrium, with actual native soft normal
diagonal and stabilization in the same physical velocity. A native projected
sweep must leave that equilibrium fixed under either point ordering and copy
counts1/3/7. It audits physical linear/angular impulse after1/N averaging and
negative tangential work. The old diagonal-subtraction rule fails by99.25µm/s;
actual-total-normal loading passes all six cases. This is a constitutive fixed
point test, not a requirement to match another simulator's pose.

Logs `/tmp/colibri_stabilized_support_failbefore.log` and
`/tmp/colibri_stabilized_support_passafter.log`. Review-only production proposals:
`/tmp/colibri_friction_break_production.patch` and
`/tmp/colibri_total_normal_production.patch`, with staged file trees beside them.
The proposals also cover articulation-owned point friction; that broader path
still needs its native regression run before promotion. Patch-friction history
retains its old separate policy pending a dedicated migration. Production source
remains unchanged.
