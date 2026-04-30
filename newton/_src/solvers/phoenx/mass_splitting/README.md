# Mass splitting (Tonge 2012) — port from PhoenX C#

Isolated port of PhoenX's mass-splitting machinery from
`/home/twidmer/Documents/git/experimentalsim/PhoenX/src/PhoenX/MassSplitting/`
into Newton/Warp idioms. Not yet wired into :class:`SolverPhoenX`; the
goal at this stage is to land each architectural piece as a
self-contained, user-friendly component that can be unit-tested in
isolation.

## What mass splitting is

> Tonge, Benevolenski, Voroshilov, *Mass Splitting for Jitter-Free
> Parallel Rigid Body Simulation* (2012).

Per-body **velocity-state copies** indexed by partition, plus an
``invFactor``-scaled impulse rule, let independent sets of constraints
solve concurrently without racing on shared body velocity. Between
PGS iterations, an *AverageAndBroadcast* consensus step keeps the
copies coherent and conserves momentum. The whole machinery comes
down to four moving parts:

1. **`TinyRigidState`** — the per-(body, partition) velocity-state
   copy: ``orientation, position, velocity, angular_velocity,
   access_mode``.
2. **`InteractionGraph`** — a host-built, device-resident lookup that
   maps a ``(constraint_id, body_id)`` pair to an index into the
   ``TinyStates`` array, plus the inverse-factor count.
3. **`ContactPartitions`** — independent-set graph coloring of the
   constraint graph: ``partition_data_concat`` + ``partition_ends`` +
   ``num_partitions``. One partition's constraints can solve
   concurrently because no two of them share a body.
4. **Broadcast / Average / WriteBack kernels** — the substep-loop
   plumbing that copies between the body store and the per-partition
   states.

Plus the constraint-side **ReadState / WriteState** wrapper API that
turns "load body velocity" into "load this body's slice of the right
partition".

## File layout (mirrors the C# split)

| Python module             | C# source                                                    | Purpose |
|---------------------------|--------------------------------------------------------------|---------|
| `state.py`                | `CudaKernels/Common/BodyTypes.cs` (`TinyRigidState`)         | Per-(body, partition) velocity-state copy + access-mode helpers. |
| `interaction_graph.py`    | `MassSplitting/MassSplittingRigidBodyInteractionGraph.cs`    | Host builder + device-side lookup struct. |
| `partitions.py`           | `MassSplitting/ContactPartitions.cs`                          | Independent-set partitioning of the constraint graph. |
| `kernels.py`              | `CudaKernels/Solver/SolverKernels.cu` (mass-splitting parts) | Broadcast / average / writeback Warp kernels. |
| `read_state.py`           | `CudaKernels/Constraints/ConstraintHelper.cuh`               | Constraint-side ``read_state`` / ``write_state`` wrappers. |
| `mass_splitting.py`       | `MassSplitting/MassSplitting.cs`                              | Top-level orchestrator that wires the pieces. |

## Per-substep dance

```
substep start:
  broadcast_rigid_to_copy_states          # body.vel  -> TinyStates[*, p] for all p
  iterate_kernel(prepare, partition=p)    # for p in partitions, parallel-safe
  average_and_broadcast                   # TinyStates -> averaged -> TinyStates

  for iter in range(solver_iterations):
    iterate_kernel(solve, partition=p)    # for p in partitions
    average_and_broadcast

substep end:
  copy_state_into_rigids                  # TinyStates[*, 0] -> body.vel
```

The crucial point is that *partitions run concurrently*. Without
copy states they would race on body velocity. With copy states, each
partition has its own state slice; averaging is the periodic
consensus.

## Velocity-level vs position-level passes (the access-mode pattern)

Same C# pattern, no parallel infrastructure needed. The
:class:`TinyRigidState` carries an ``access_mode`` field; the
:func:`tiny_rigid_state_synchronize` helper bridges the two
regimes lazily; ``read_state`` takes a ``new_access_mode`` argument
that flips into the regime the constraint wants. **Both regimes
ride on the same broadcast / write-back kernels** -- there's
nothing extra to wire.

A constraint kernel that wants to project positions XPBD-style
just passes ``ACCESS_MODE_POSITION_LEVEL`` to ``read_state``,
mutates ``state.position`` / ``state.orientation``, and calls
``write_state``. At end-of-substep the
``copy_state_into_rigids_kernel`` runs the XPBD finite-difference
``v = (state.position - body_position) * inv_dt`` recovery and the
angular-velocity log-map -- same kernel as for velocity-level
solves, because the synchronize helper inside
:func:`tiny_rigid_state_write_back` does the regime conversion
based on whatever ``access_mode`` the state ended up in.

Recipe (cloth-style XPBD distance constraints between particles):

```python
from newton._src.solvers.phoenx.mass_splitting import (
    ACCESS_MODE_POSITION_LEVEL,
    MassSplitting,
)

ms = MassSplitting(
    num_bodies=num_particles,
    num_constraints=num_distance_constraints,
    max_partitions=12,
)
ms.setup_from_graph(constraint_bodies)   # cloth's distance constraints

# Per substep:
ms.broadcast(body_position, body_orientation, body_velocity, body_angular_velocity, dt)
for _ in range(position_iterations):
    for partition_index in range(ms.partitions.num_partitions):
        # Cloth's distance-iterate kernel reads state with
        # ACCESS_MODE_POSITION_LEVEL, mutates state.position,
        # writes back. No average() needed if mass splitting is off
        # (every body in 1 partition); ms.average(...) does it
        # automatically when partitions overlap.
        cloth_distance_iterate_kernel(
            ms.graph.data, partition_index, ACCESS_MODE_POSITION_LEVEL, ...,
        )
    ms.average(body_position, body_orientation, inv_dt)  # consensus pass
ms.write_back(body_position, body_orientation, body_velocity, body_angular_velocity, inv_dt)
```

Mixing velocity-level and position-level constraints in the same
substep works: each constraint just passes its preferred
``new_access_mode`` to ``read_state``. The synchronize helper
fires on regime changes (typically zero per call inside one
sweep, since one sweep usually targets one mode).

## What's NOT in this port (yet)

* **GPU-side MIS partitioning kernels.** The C# code uses Luby's
  algorithm on device (`PartitioningColoringKernel`); this port ships
  a host-side greedy MIS partitioner that produces an equivalent
  result. Newton's existing `graph_coloring/` already has greedy +
  JP MIS implementations on device; wiring those in is a follow-up.

* **Wiring into :class:`SolverPhoenX`.** Every constraint kernel
  (contacts, ADBS joints, cables) would need to switch from direct
  body-velocity reads to ``read_state`` / ``write_state``. That's
  ~15 functions per the prior project memory; it's a follow-up task
  once this isolated port is validated.

* **`MaximumRecoveryVelocity`-style adaptive batching**
  (`ConstraintBatchSizeInfo`, `MassSplittingInfoGpu.batchSize`). C#'s
  parallel-id offsetting for the additional Jacobi partition is
  noted in `mass_splitting.py` but not exercised by the kernels yet.

## Numerical / ordering invariants (carried over from C#)

* **`broadcast_rigid_to_copy_states` MUST run BEFORE the iterate
  loop.** Without it, the per-partition copies hold stale state.
* **`average_and_broadcast` MUST run AFTER each iterate sweep** (and
  after the prepare sweep in C#). Skipping it lets partitions drift
  apart and breaks momentum conservation.
* **`copy_state_into_rigids` MUST run ONCE at the end of the
  substep.** Reading any other partition's slot also works because
  averaging made them equal, but only run it once.
* **`alternative_averaging = True` kills momentum conservation**
  (sets `inv_factor = 1` regardless of partition count). Same flag
  as the `MassSplittingTypes.cuh:198` warning. Don't flip it.

## Why isolated for now

Wiring mass splitting into PhoenX's hot path means changing every
constraint kernel's body-state read/write to go through the
``read_state`` / ``write_state`` indirection. That's a big surface
change to the solver and easy to break in hard-to-debug ways. So
this port lands the standalone components first; the integration
PR can be reviewed and benched separately on top of a known-good
isolated foundation.
