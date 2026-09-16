# Block-joint inactive inequality preparation

The block joint solver separately builds physical bilateral and drive rows.
Its color callback nevertheless calls legacy inequality preparation for every
joint. In particular, an unlimited frictionless revolute still builds and
inverts an anchor effective-mass matrix used only for axial joint friction.

Disabling the whole joint column would also disable its bilateral callback.
Returning before all preparation would lose axis/levers, revolution tracking,
and release of stale limit/friction multipliers. The local
`inactive_joint_prepare.py` instead preserves those updates.

The shortcut is limited to block-owned rigid revolute/prismatic joints with
no position-level writers, zero/nonpositive friction, zero/nonpositive velocity
limit, and a currently inactive position limit. Its position predicate exactly
matches the existing axial prepare: disabled lower>upper, or current coordinate
inside the inclusive interval. There is no predictive position-limit band in
that helper. Every active limit or friction/velocity-limit case delegates to
the unchanged canonical preparation.

Only unused axial/friction effective masses and the inactive soft-limit cache
are left unrefreshed. Clamp state and both accumulated axial multipliers are
cleared. The next active preparation recomputes all masses at the current pose.
The live inequality iteration and the bilateral callback are unchanged.

## Validation

`test_inactive_joint_prepare.py` compares revolute/prismatic, split/unsplit
trajectories through ten steps including a finite limit crossed within the first
timestep, stale multipliers when disabling limits, friction activation,
velocity-limit activation, and a finite-effort drive enabled via property
notification. All used fields, poses, velocities, and multipliers agree bitwise;
all six momentum components are independently checked at each step.

The test additionally checks that the shortcut leaves the unused initial axial
mass untouched, while the first active-limit step recomputes the same mass as
the reference. This distinguishes real skipped preparation from a bypassed hook.

The actual public330-frame default trajectory also matches all eight saved
numeric arrays byte-for-byte, including pose/velocity histories:
`/tmp/colibri_inactive_joint_prepare330.npz` versus
`/tmp/colibri_zero_column_reference330.npz`. This run overlapped correctness work
and is not a performance comparison. No canonical change has been made.

## Isolated timing and integration

Two 330-frame public runs in each order, 30 warmup frames, 30 temporal steps per 120 Hz update, one iteration:

- Reference then candidate: 13.09347 -> 12.82888 ms/frame (2.02% faster).
- Candidate then reference: candidate 12.49839 vs reference 13.70029 ms/frame (8.77% faster).

All eight physical/history arrays were byte-identical in both pairs. Timing magnitude varies; the direction repeated. The canonical block-only, rigid-only wrapper retains the original full preparation for active rows. Production regression failed before (unused axial inverse mass 2.5 rather than zero), then passed all revolute/prismatic and split/unsplit transition comparisons, including independent six-component momentum checks. Dormant metric fields intentionally remain untouched; the first active crossing recomputes them exactly.
