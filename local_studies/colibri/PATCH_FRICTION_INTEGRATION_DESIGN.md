# General persistent patch-friction integration: bounded source audit

> **Updated architecture decision:** production integration will share coherent
> patch **history**, while retaining every physical contact's local Coulomb
> capacity and common-point force/torque application. Any collective sticking
> solve requires a feasibility certificate and fallback. The earlier two-anchor
> force-replacement proposal below is background, not the accepted force design.

This is an internal integration proposal, not completed implementation or
physical acceptance. No core edits or GPU runs were made. Canonical coding and
public API guidelines were read. The user requests arbitrary rigid-body pairs,
contact counts, physical forces and torques, and FP32 production arithmetic.

## What currently exists

- `constraints/contact_patch_friction.py:17` defines `ContactPatchFriction`,
  indexed by **contact column**, with one world tangent impulse, one pair of
  levers, one 2x2 effective mass and one bias. It has neither persistent two
  friction anchors nor patch broken/history state.
- `constraint_contact_cloth.py:789` accumulates point normals/levers/errors.
  The patch path prepares their centroid and later (`1878+`) solves one2D
  tangent block after point normals. This cannot represent a general patch's
  pure twisting couple without a resultant tangent force. It is not the new
  two-anchor PhysX reference, despite the similar name.
- `contact_patch_friction.py:110-189` eligibility accepts convex shape pairs;
  raw mesh/heightfield and compound body-pair grouping retain point friction.
  Warm start uses the first matched point's prior column impulse. This is not
  a safe persistent identity for split/merged patches or contact chunks.
- `solver_phoenx.py:3206-3224,3325+` provides useful graph-safe history-copy and
  post-ingestion gather hooks. It copies previous live ranges with stable
  pointers before updating current counts. Keep that lifecycle property.
- Existing option `contact_friction_model='patch'` explicitly documents the
  old central-row algorithm in `solver.py:298-305`. World guards exclude mass
  splitting, deformables and sleeping; cached prepare also excludes patches.
  Contact chunks and optimized point-only paths have additional guards.
- Standard single/multi-world kernel factories pass `patch_friction` through
  preparation/iteration. Reduced articulation patch rows have a separate
  gather/finalize/tile-solve implementation in `reduced_contact_block.py`
  (`731`, `1100`, `2326+`), not the same world-container solve. Public reduced
  setup passes world point friction and routes the articulation separately.
  Maximal/direct owned callbacks likewise bypass ordinary contact iteration.

**Conclusion:** a flag or replacement projection helper alone cannot integrate
persistent two-anchor patches across all current owners. The container,
identity, schedule and owner-specific physical response all need explicit wiring.

## Minimal reusable internal data contract

Keep the public entry point on `SolverPhoenX`; do not expose new top-level types.
A private patch state can replace/extend the existing internal storage after
its old semantics and tests are intentionally migrated:

1. **Membership:** device patch count, per-world ranges, point-to-patch map,
   CSR point offsets/indices, body0/body1, owning solver and owning schedule
   item. Contact count is arbitrary; no32point assumption. Point normal rows
   retain their original normals, offsets, soft coefficients and activation.
2. **Identity:** canonical ordered shape pair, world ID, material/flags and
   compatible target motion, normal-cluster identity, current-to-previous
   patch map. Initial implementation can keep different shape pairs separate;
   it need not pool an entire compound body pair to support compounds.
3. **Persistent anchors:** up to two local material positions per side,
   anchor count/validity, normal/tangent frame, broken-state lifecycle, and
   world/vector tangent impulse per anchor. Preserve the reviewed FP32
   differential-reference mechanism as needed; no FP64 production state.
4. **Prepared rows:** current common forcepoint/levers, bias/error, current
   normal-load sum and anchor allocation, target velocity, and response terms.
   Generic rigid, direct/maximal and reduced solvers consume the same physical
   rows through their appropriate mobility operator.
5. **Previous state:** separate fixed-capacity storage/counts and stable device
   copy at generation boundary. Never swap Python pointers inside captured
   execution or allocate scratch based on a host-read active count.

Capacity: at most one nonempty patch per accepted contact; allocate patch
capacity from rigid-contact capacity and2anchors/patch. Membership has exactly
one slot/contact. Preallocate segmentation/sort/scan/matching scratch. Support
empty input. On insufficient configured capacity, surface a capacity failure;
never silently drop points, duplicate normal load, or change the friction law.

## Lifecycle API (private host launch functions plus Warp functions)

- `update_patches(state, contacts, ingest_maps, materials, generation)`:
  copy previous live state; build deterministic membership; correlate patches
  and anchors; initialize genuinely new/broken anchors. Operates before
  coloring/ownership finalization, after accepted contacts and materials exist.
- `prepare_patch_rows(state, bodies, motion, dt, phase, response_backend)`:
  derive geometry/error/response for all active anchors. Phase explicitly
  distinguishes biased position solve, conclude, and unbiased velocity solve.
- `solve_patch(state, patch_id, point_rows, backend, phase)`:
  solve its complete normal member set, sum the **actual** normal impulses,
  then solve each anchor under the chosen reviewed friction law. Apply each
  impulse exactly once through the backend. No diagonal bias subtraction from
  normal capacity. No accidental point-friction solve in addition to patches.
- `commit_patch_history(state)` at the established generation boundary:
  preserve references and the reviewed broken latch/update semantics. Do not
  substitute a last-clipped-bit or98%capacity heuristic for that contract.

These are conceptual signatures; use concrete Warp structs and compile-time
backend specializations rather than Python polymorphism in device loops.
Preserve actual GPU TGS friction projection semantics if that is the accepted
model; the older full2x2 disk projection is not automatically interchangeable
with two-anchor sequential PhysX rows.

## Matching and grouping rules

Use world + ordered shape pair + material/static/dynamic coefficients + flags
and target-motion compatibility, then normal clustering using the reviewed
source rule. Never merge opposite face normals just because bodies match.
Keep original point normal directions for the normal solve. The patch frame
is separate data, not a rewrite of collision witnesses or force activation.

Deterministic ties must use a stable contact identity/geometric ordering, not
atomic allocation order. Two separated anchors are selected geometrically;
no fixed contact indices or ground-plane special case. An unbroken anchor may
remain valid even if its original contact ID disappears, provided its material
reference meets the reviewed correlation criteria.

Split/merge is not a first-match copy: duplicating a predecessor impulse into
two child patches doubles applied load. Start with deterministic one-to-one
lineage, reinitializing ambiguous additional children/new merged anchors under
the specified birth semantics. A more elaborate conservative impulse transfer
requires a wrench audit; do not quietly claim it preserves strong friction.
Expose split/merge/rebirth diagnostics in tests.

## Scheduling and physical response

A **patch is a solve unit**, independent of preparation chunks. If preparation
splits129 contacts into smaller chunks, retain one patch membership and one
normal-load sum/friction solve. Splitting a patch into separately frictional
chunks changes the model and cannot be called a launch-only optimization.

Ordinary coloring should own all bodies a solve unit writes. For direct/reduced
contacts, ownership is the articulation response domain, not just the two link
IDs. Two links of the same articulation require cross mobility; two distinct
articulations require both responses, and a mixed free-body/articulation pair
must update both. Reuse their actual J M^-1 J^T operators and J^T scatter,
including physical drives; never substitute independent rigid inverse masses.
All points in a patch read the same current velocity phase.

Keep current splitting guards until physical copy ownership is implemented and
tested. Overflow splitting needs one logical normal load, correctly partitioned
anchor impulse storage and a single physical writeback; copying a patch impulse
to every mass copy is invalid. User preference remains ordinary coloring with
splitting only as overflow, not unconditional splitting for speed.

For every dynamic pair, impulse application must be `(-p,-r0 cross p)` and
`(+p,+r1 cross p)` with the accepted common physical forcepoint convention.
Check `x0+r0 == x1+r1` to FP32 geometric accuracy and audit world torque. The
source's two independently transported material anchors can separate under
error; blindly using those separated points as equal/opposite forcepoints
creates a couple. Persistent error references and forcepoint/scatter semantics
must be reviewed explicitly. Do not advertise exact PhysX row equivalence and
exact angular-momentum conservation if they require different conventions.
Kinematic/static partners supply prescribed velocity and reaction accounting;
they must not be treated as arbitrary dynamic infinite-mass bodies in tests.

## Missing regression matrix

Existing `test_contact_patch_friction.py` covers convex eligibility, compound
fallback, history reorder, rest/static/kinetic friction, coupled2x2 response,
rigid-pair P/L, graph capture and single/block-world paths. These are valuable
but validate the old central-row algorithm. Add independent tests for:

| Axis | Required cases / acceptance |
|---|---|
| Membership |0,1,2,3,31,32,33,64,129points; unchanged normals; each point consumed once; multiple normal clusters; mixed materials/target velocities|
| Shape/body scope |mesh, convex, compound; dynamic/dynamic, dynamic/static, dynamic/kinematic; no plane-specific assumptions; reversed endpoint order|
| Persistence |point/column reorder, source contact removal, anchor reuse, genuine slip break, normal rotation, separation/recontact, split/merge; no duplicated impulse|
| Physical wrench |opposing anchor forces give zero resultant and nonzero yaw torque; free pair P/L; off-center forces; unequal and rotated inertias; no extra torsion row without specification|
| Law |static versus kinetic coefficient, mu0, zero normal load, positive gap, actual normal budget, anisotropic response, support load/time-step units, conclude/velocity phases|
| Owner |ordinary PGS; maximal/direct tree; reduced; same-articulation self-contact; distinct articulation pair; mixed articulation/free body; exact ownership/write count|
| Scheduling |single/multi-world isolated equivalence, world-ID collision, reverse order, all-small/head-tail transitions, preparation chunking invariance, overflow splitting or explicit unsupported guard|
| Capture |preallocated scratch, changing counts0→many→0, poisoned previous storage, externally overwritten/reset state, repeated CUDA replay, no stale references|
| Physics quality |loaded resting and driven tests over time, force+torque equilibrium and drift, sliding work/restitution conventions, actual arbitrary-pair momentum ledger|

Use scalar/high-precision offline physical references only as test oracles;
production stays FP32. Fail-before assertions should target missing two-anchor
wrench/persistence behavior, not mirror implementation details. No performance
claim until all required physical gates pass. Integrate dispatch paths in
reviewable stages with explicit unsupported errors, never silently route one
owner to a different contact law.

## Implemented additive history container

`constraints/friction_patch_history.py` now provides fixed-capacity Warp state
and lifecycle only. `allocate_friction_patch_history`,
`update_friction_patch_history`, and `check_friction_patch_history` do not apply
impulses, select a friction law, move contact sites or alter solver dispatch.
Storage is FP32:14birth fields preserve p0(3),p1(3),q0_xyzw(4),q1_xyzw(4).
Caller-supplied CSR membership and material-point matches determine identity;
world/body/material/normal compatibility alone never joins disjoint patches.

Retention requires complete injective one-to-one predecessor membership.
New/rolling/reentering points, split/merge ambiguity, duplicated matches,
incompatible groups or caller-declared broken history cause rejection/rebirth.
A caller must not pass a recycled collision feature ID as proof of the same
material point. Broken state is explicit; no per-point clip OR is interpreted
as collective infeasibility. Age counts retained contact generations only.

Device validation rejects capacity overflow, malformed/duplicate/missing CSR
membership and incompatible group keys/normals. Errors leave live/previous
history unchanged. Consumers must gate the device error before interpreting
history under new input counts; host check is explicitly outside capture.

`tests/test_friction_patch_history.py`:8CPU and9CUDA cases PASS, including129
contacts, disjoint same-key groups, birth/reentry, split/merge/duplicate lineage,
material/normal/world separation, explicit break, empty reuse and captured
count changes/overflow. RuffPASS. Logs `/tmp/colibri_patch_history_cpu.log` and
`/tmp/colibri_patch_history_cuda.log`. No existing solver callsite/default changed.
This is storage validation, not evidence that a shared-history force adapter
solves drift or preserves rolling. Those remain separate physical gates.
