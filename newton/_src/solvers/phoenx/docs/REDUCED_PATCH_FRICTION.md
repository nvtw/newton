# Reduced-coordinate convex patch friction

## Experiment outcome

A complete implementation of this design was tested and rejected on 3 July
2026. It passed focused CUDA-graph behavior gates, including multi-point
sliding-to-rest, arbitrary three-page column counts with contact disappearance
and reappearance, compound-column point fallback, and the unchanged default
ABA row comparison.

The separate normal and patch phases did not amortize. At 8,192 G1 worlds the
prototype removed about 26% of generalized rows but measured 1.73-1.74M
physics steps/s versus 1.88-1.89M for point friction, a 7.4-8.0% regression.
At 1,024 worlds it was about 4% slower. The implementation and its feature
tests were fully reverted; only the small generic row-response extraction was
retained. A future retry needs a design that avoids the additional gather,
build, and solve page rather than reimplementing this two-stage schedule.

## Objective

Reduce generalized contact-row construction without changing point-normal
resolution, hard-Hertz contact behavior, deterministic ordering, or momentum
conservation. The opt-in maximal solver establishes the physical model; the
reduced solver needs a different storage and scheduling strategy because its
generalized response rows are paged by contact point.

For P contact points in C eligible convex columns, the current reduced solver
builds 3P rows. The target layout builds P normal rows followed by 2C tangent
rows. A live 1,024-world G1 snapshot contains 34,816 eligible points in 18,432
columns, reducing the generalized row count by 31.37%.

## Required invariants

- Every contact point retains its independent unilateral normal row.
- One coupled 2D tangent block is used only when both shapes are provably
  convex and the contact column represents exactly one shape pair.
- Raw meshes, heightfields, compound body-pair columns, deformable contacts,
  and otherwise unproven columns retain the existing three-row point solve.
- Both sides of a patch impulse act at one shared world point. This preserves
  linear and angular momentum.
- Patch warm start is stored as a world-space impulse and recovered through
  persistent contact matching, never through a transient column index.
- Ordering is deterministic. No atomics choose patch membership or row order.
- Contact and column counts remain arbitrary. Pagination is internal and no
  user-visible maximum patch size or tuning parameter is introduced.
- CUDA graph topology and launch dimensions are fixed after construction.

## Two-stage solve

### Stage 1: point normals

Gather the existing point pages, but build only one generalized row per point
for eligible columns. Ineligible columns retain their three point rows. Solve
all normal pages before any central patch row so a patch sees the complete
physical normal load, even when one column spans multiple point pages.

The normal friction load excludes Baumgarte-only impulse exactly as the current
PhoenX contact projection does. Speculative points outside the friction shell
contribute no patch load.

### Stage 2: whole-column tangent blocks

Page the already deterministic reduced contact-column schedule, not contact
points. A column is never split across patch pages. For each eligible column:

1. Accumulate its shared midpoint, average normal, tangent drift, and total
   physical normal load over every contact in the column.
2. Build a deterministic tangent frame and two generalized response rows at
   the shared midpoint.
3. Form the full 2-by-2 tangent response, including the off-diagonal term.
4. Solve and project the accumulated tangent impulse with PhoenX static and
   kinetic Coulomb disks.
5. Apply the equal-and-opposite impulse through the articulation response and
   publish the world-space warm-start impulse.

Ineligible columns are solved entirely by the existing point path. The first
prototype may select the compact path only when every deferred column in an
articulation is eligible; mixed articulations must fall back as a unit until a
variable row map is implemented. This is a conservative specialization, not a
change in physics.

## Storage and graph schedule

- Keep point-normal response pages separate from patch response pages so the
  bias and relaxation passes can reuse both without rebuilding rows.
- Derive patch counts from the existing per-articulation column schedule.
- Patch pages contain whole columns and use the same fixed maximum generalized
  row tile as point pages. Page count is device-side.
- Reuse the existing articulated response builder through an explicit row
  descriptor: source body, spatial wrench, row velocity, and row count.
- Publish effective masses outside the generic response traversal. Point
  normals write one scalar; patch blocks form their coupled 2-by-2 matrix from
  the two row responses.

## Acceptance gates

The reduced path is retained only if all of the following hold under CUDA graph
capture:

- velocity-level linear and angular momentum match the point solver tolerance;
- static/kinetic friction, sliding-to-rest, stacking, hard-Hertz contact force,
  and speculative-contact tests pass;
- raw-mesh, heightfield, compound, mixed maximal/reduced, single-world, and
  multi-world fallbacks are correct;
- repeated runs are bit deterministic;
- frozen-policy G1 walking gates remain passing at both established seeds;
- a scheduler-matched G1 bracket shows a repeatable end-to-end gain, followed
  by a short training-throughput bracket;
- diverse contact-rich robots and stacks do not regress.

The measured row reduction gives an optimistic ceiling of about 11.9% of
reduced physics-solver work, or 8.7% of total GPU work in the current training
profile. A complicated implementation or a gain confined to one scene should
be rejected.
