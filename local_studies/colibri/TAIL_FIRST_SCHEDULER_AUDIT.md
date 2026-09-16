# Tail-first scheduler regression audit

Production test: `newton/_src/solvers/phoenx/tests/test_tail_first_dispatch.py`.

The three synthetic CUDA graph cases execute the real head/tail kernels with a
small deterministic per-constraint operation. They check small/large transitions,
forward/reverse ordering, serial mass-splitting overflow batches, exactly-once
updates, and zero head launches when all colors fit the tail. The all-small test
failed before integration (head count 1 versus expected 0). All three pass after.

The real compact hanging-cloth fixture now also runs with actual cube/triangle
contacts. It asserts finite state, drained color cursor, pinned displacement under
5 mm, and edge length ratios between 0.3 and 3, retaining the existing fixture's
physical thresholds. Four final tests passed in 1.073 seconds after cached builds
(`/tmp/tail_first_all_final.log`).

This test exposed a compile error in the current cloth specialization: a mixed
static/runtime condition did not prune rigid-only lever and inertia expressions,
leaving `r1` undefined. The fix places those expressions inside a standalone
`wp.static(not cloth_support)` branch. Existing cloth projection arguments and
physical equations are unchanged. The failure is recorded in
`/tmp/tail_first_cloth_test.log`; the rigid normal-first equivalence regression
also passes after the fix (`/tmp/cloth_branch_rigid_normal.log`).

## Limit of the real-cloth comparison

Independent current-worktree head-first runs were not bitwise identical either:
three-frame combined pose/velocity differences reached 0.00286462 in the control
(`/tmp/cloth_head_repeat.log`). This establishes nondeterminism before the
scheduler change **in the current worktree**, not in the original branch HEAD.
The real-cloth test therefore does not claim bitwise physical equivalence; exact
operator/order coverage is provided by the synthetic cases. One warmed head/tail
sample differed by 0.214 mm in particle position and 0.0162 m/s in particle
velocity, with rigid body pose essentially identical. These are diagnostic
observations, not new acceptance tolerances.

Bounded follow-up: capture the first divergence of identical head-first runs and
attribute it to generated contacts, graph ordering, or force accumulation before
attempting any further cloth determinism fix. No such broad follow-up is included
in this scheduler change.
