# Position-level state-update infrastructure

Self-contained subtree that adds the pose-snapshot + XPBD
finite-difference sync needed for any position-based constraint
solve (XPBD cloth, PBD ragdolls, joint anchor projection, …) to
coexist with PhoenX's existing velocity-based PGS pipeline.

## Why this exists

Newton PhoenX is a sequential-impulse solver: every constraint
mutates `bodies.velocity[b]` and `bodies.angular_velocity[b]`, and
the substep loop integrates pose *once* per substep. That works well
for everything PhoenX ships today (Box2D-v3 TGS-soft contacts,
Jitter2-style PD springs for cables and drives), but it doesn't fit
position-based formulations like XPBD, where the constraint solve
mutates `position` / `orientation` directly and *recovers* velocity
afterward via finite differencing
(``v = (p - p_pre) / dt``, Macklin et al. 2020).

This subtree adds the substep-loop machinery position-based
constraint solves need without touching PhoenX's velocity hot path:

* A **pre-pass pose buffer** so position deltas can be measured
  against a known start-of-pass pose.
* A **snapshot kernel** that captures the pre-pass pose into that
  buffer.
* An **XPBD-style sync kernel** that re-derives velocity /
  angular-velocity from the position delta after the position-level
  iterate sweeps complete.
* A **`PositionPass` orchestrator** that owns the buffer + the
  iterate-callable hook + the kernel launches, so a cloth (or
  ragdoll, or whatever) solver only has to provide its own iterate
  function.

The constraints themselves live elsewhere — XPBD distance for cloth,
position-level joint anchor for stiff articulations, etc. This
subtree is *only* the integration plumbing. Same shape as the
``mass_splitting`` sibling subtree (commit ``eda8d614``): land the
infrastructure first, plug specific constraints into it later.

## File layout

| File          | Purpose |
|---------------|---------|
| `snapshot.py` | The two ``@wp.kernel`` primitives: ``snapshot_pose_kernel`` and ``sync_position_to_velocity_kernel``. |
| `position_pass.py` | ``PositionPass`` orchestrator. Owns buffers, exposes ``snapshot()`` / ``sync_to_velocity()`` / ``run(callable, n)`` so callers (cloth solver, PhoenXWorld) can string the pieces together. (File spelled out because ``pass.py`` collides with the Python keyword.) |
| `__init__.py` | Public API re-exports. |
| `tests/`      | Round-trip + zero-regression tests. |

## How callers use it

The intended pattern is **standalone** -- cloth (or any other
position-based solver) drives its own substep loop and calls into
:class:`PhoenXWorld.step` *between* the snapshot and sync, so the
velocity-level PhoenX solve sees the XPBD-recovered velocities and
does its TGS-soft thing on top:

```python
from newton._src.solvers.phoenx.position_level import PositionPass

# One-time setup, sized for the body store. Buffers are allocated
# lazily on first ``snapshot()`` call so unused passes pay zero.
pp = PositionPass(num_bodies=num_bodies, device=device)

# Per substep:
pp.snapshot(body_position, body_orientation)
for _ in range(position_iterations):
    cloth.iterate(body_position, body_orientation)   # XPBD projection
pp.sync_to_velocity(
    body_position,
    body_orientation,
    body_velocity,
    body_angular_velocity,
    inv_dt,
)

# Then PhoenX (or any other velocity-level solver) runs normally
# with the velocity field already containing the XPBD-recovered
# values.
```

### Why no ``PhoenXWorld`` hook (yet)

The plan in ``PLAN_ACCESS_MODE.md`` mentioned a ``position_iterations``
parameter on :class:`PhoenXWorld` that would run the snapshot / sync
inside the substep loop. We do not ship that in this commit because
of a footgun: an iterate callback that touches only some bodies
(typical in mixed cloth + rigid scenes) would leave non-touched
bodies' positions unchanged, and the unconditional sync would then
overwrite their velocities with ``(p - p_pre) / dt = 0`` -- silently
zeroing rigid bodies the user expected to keep their velocities.

Working that out properly needs either (a) a per-body
"participates" mask, (b) snapshotting the velocity too and skipping
non-touched bodies on sync, or (c) restricting the pass to a body
*range*. All three are reasonable; none is forced by an active use
case yet. So this commit ships only the standalone primitives, and
the concrete cloth integration that lands next can pick the variant
that matches its body-layout decisions.

PhoenX's substep loop in ``solver_phoenx.py`` is **bit-for-bit
unchanged** by this commit. Scenes that don't import this subtree
pay nothing.

## XPBD finite-difference sync (the one piece worth re-deriving)

Source: Macklin et al., *Detailed Rigid Body Simulation with
Extended Position Based Dynamics* (2020), Algorithm 2 — and
``BodyTypes.cs:268-304`` in the C# PhoenX reference.

After ``position_iterations`` of XPBD position projection, the body
pose has moved from ``(p_pre, q_pre)`` to ``(p, q)``. The recovered
velocity / angular velocity that produces this same motion under the
substep's velocity-level integrator is:

```
velocity = (p - p_pre) * inv_dt
delta_q = q * conj(q_pre)
angular_velocity = 2 * inv_dt * delta_q.xyz   # sign-flipped if delta_q.w < 0
```

That's all the sync kernel does: one read of each pre-pass pose, one
quaternion conjugate-multiply, one ``2 * inv_dt * Im(...)`` extract,
one write to ``body_velocity`` / ``body_angular_velocity``. Costs
nothing when nothing changed (``v_recovered == v_original`` modulo
float32 noise on a no-op pass — see the round-trip test).

## Zero-overhead invariants

* **No allocation when off.** ``PositionPass`` allocates the
  pre-pass buffers lazily on first ``snapshot()`` call; if
  ``position_iterations == 0`` the orchestrator is never
  constructed.
* **No kernel launches when off.** PhoenXWorld guards the snapshot
  / sync launches behind a ``position_iterations > 0`` host check
  *outside* the captured graph. The captured per-step graph in
  velocity-only scenes contains *zero* references to this subtree.
* **No new fields on ``BodyContainer``.** The pre-pass buffers
  are world-local scratch, mirroring how ``set_global_linear_damping``
  allocates its damping array. ``BodyContainer`` stays unchanged.

## What's NOT here (intentionally)

* **Constraint kernels.** No XPBD distance, no XPBD joint anchor,
  no soft body. Cloth will write its own constraint kernels and
  hand the iterate function to the orchestrator. The pattern is
  documented in ``pass.py``'s ``run_iterate`` docstring.
* **Per-particle state.** ``PositionPass`` operates on the same
  rigid-body SoA the rest of PhoenX uses. If cloth eventually wants
  particle-only state (no orientation, just position), the same
  pattern works on a particle SoA — the kernels in this subtree are
  trivially specialisable to that shape.

## References

* Macklin et al., *Detailed Rigid Body Simulation with Extended
  Position Based Dynamics*, 2020 — the XPBD finite-difference
  reintegration formula.
* Müller et al., *Position Based Dynamics*, 2007 — the original PBD
  paper.
* PhoenX C# reference: ``BodyTypes.cs:66-413`` (``ConstraintAccessMode``
  enum + ``SynchronizeVelAndPosStateUpdates`` algorithm).
* Newton PhoenX integration loop:
  ``solver_phoenx.py:1133-1958`` (substep ordering),
  ``solver_phoenx.py:1899-1912`` (velocity integration kernel
  semantics).
* Already-shipped sibling: ``newton/_src/solvers/phoenx/mass_splitting/``
  (commit ``eda8d614``) — same factoring, different infrastructure.
