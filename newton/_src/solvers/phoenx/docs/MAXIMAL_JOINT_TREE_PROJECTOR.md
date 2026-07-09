# Maximal joint-tree projector

## Purpose

The experimental `articulation_mode="maximal_projected"` path keeps
PhoenX body state and transient contact handling in maximal coordinates while
removing slow constraint propagation through a known acyclic joint graph.
Eligible free-root revolute forests use the specialized projector; unsupported
topologies retain the established hybrid implementation while the general
motion-subspace kernel is developed.

Joint/body incidence is persistent and can be preprocessed. Contact incidence
is not: contact attachments must be rebuilt after collision detection and must
retain ordinary deterministic Gauss–Seidel fallback.

## Mass-metric projection

Let `v_i*` be the unconstrained maximal twist of body `i`, with spatial mass
matrix `M_i`. A tree joint relates child and parent twists by

```text
v_i = X_i v_parent + S_i qdot_i + c_i.
```

`S_i` spans allowed joint motion. `c_i` is zero for bias-free relaxation; in
the positional pass it is a particular locked-direction twist reconstructed
from PhoenX's prepared joint biases. The desired state minimizes

```text
sum_i 1/2 (v_i - v_i*)^T M_i (v_i - v_i*)
```

subject to every tree relation. This is the exact kinetic-energy projection,
not fake mass, damping, or a dominance rule.

For a subtree whose minimized cost is

```text
E_i(v_i) = 1/2 v_i^T A_i v_i - h_i^T v_i + constant,
```

eliminate the joint velocity with

```text
D_i = S_i^T A_i S_i
P_i = A_i - A_i S_i D_i^-1 S_i^T A_i
g_i = h_i - A_i S_i D_i^-1 S_i^T h_i.
```

The contribution passed to the parent is

```text
A_parent += X_i^T P_i X_i
h_parent += X_i^T (g_i - P_i c_i).
```

After the root solve, a top-down pass recovers every `qdot_i` and body twist.
The work is linear in the tree size. A free root uses a six-DoF motion
subspace; an anchored root has no allowed motion.

## Reaction recovery

The projector correction must be visible to warm starting, force reporting,
and later sweeps. Its physical joint reaction is recovered bottom-up:

```text
r_i = M_i (v_i - v_i*) + sum_child X_child^T r_child.
```

Optimality gives `S_i^T r_i = 0`: the projector applies no impulse along an
allowed coordinate. A floating root has zero reaction, which is the momentum
conservation condition. For a revolute joint, the reaction is decomposed into
the same anchor-1 and anchor-2 impulses used by the existing ADBS row and added
to its accumulated multipliers.

## Current evidence

- Dense KKT comparisons cover 1–32 body anchored and floating trees, mixed
  fixed/revolute/prismatic/ball subspaces, and six decades of mass and inertia.
  The recursion matches the dense projection, applies no allowed-coordinate
  impulse, and never increases kinetic energy.
- In the real graph-captured G1 experiment, maximal 5x2 changes from all 32
  worlds terminating by control step 2 to all worlds surviving the 50-step
  gate. Locked velocity residual is approximately `1e-7`; anchor RMS error is
  approximately `1.5e-6 m` after reaction publication.
- Two paired 200-step G1 exploration brackets measure about `23.5k` diagnostic
  control-steps/s versus `11.9k` for reduced coordinates. Reduced survives
  modestly longer under the deliberately aggressive random actions, so policy
  training quality remains an open gate.
- The same free-root/revolute path reduces joint error on ANYmal and H1 without
  robot-specific tuning.
- The production adapter passes CUDA-graph tests for two independent
  articulations in one world, exact reaction recovery, allowed-coordinate
  orthogonality, floating momentum, and safe mixed-tree fallback.
- Correcting stale world inertia on non-continuation state imports reduces the
  repeated floating-tree angular-momentum error from about `9.3e-3` to
  `8e-6` in both maximal and projected-maximal execution.
- The retained ideal GPU benchmark processes 8,192 branched 29-body trees and
  recovers all reactions in about `325 us` per captured replay on RTX PRO 6000
  Blackwell. This is not an end-to-end training number.

Run the retained benchmark with:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.experimental.bench_maximal_joint_tree_projector --worlds 8192 --bodies 29 --replays 100
```

## Promotion gates

Before this mode is considered a general replacement for the established
hybrid backend, it must cover:

- free and anchored roots; revolute, prismatic, ball, fixed, and supported D6
  trees; multiple articulations per world; and an arbitrary number of worlds;
- clean loop-joint exclusion and ordinary GS fallback;
- motors, limits, friction, live armature/gear updates, and multiplier/force
  reporting;
- mixed ordinary constraints and transient contacts, including topology
  changes and contact-rich scenes;
- floating momentum, energy, joint closure, determinism, and high mass ratios;
- graph capture in both PhoenX world layouts;
- G1, ANYmal, H1, Go2, and structurally different task coverage; and
- successful paired-seed PhoenX-RL training with lower expected time to the
  frozen held-out policy gate.

Unsupported configurations must retain the established solver path. No
contact topology may be cached across collision frames.
