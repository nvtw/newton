# Support-tree contact preconditioner research

Status: isolated research result; not integrated into `PhoenXWorld`.

## Motivation

The graph-captured high-mass-ratio benchmark exposes two distinct modes in a
1 kg box supporting a 400 kg box:

- ordinary low-iteration PGS does not propagate the vertical supported load;
- normal-only shock propagation transfers the mean load exactly, but leaves a
  horizontal internal slip mode of roughly 0.18 / 0.16 m/s RMS.

The latter mode conserves horizontal momentum to about 2e-6 kg m/s. It is a
convergence error, not an unbalanced-impulse bug. Main PGS feeds the mode
slightly and one bias-free relax sweep only reduces it slightly.

Several local corrections were tested and rejected because they diverged over
time at 400:1:

- a coupled top-only tangent wrench;
- row-wise top-only Coulomb friction;
- existing convex patch friction after a normal-only shock solve;
- tangent-only true-mass PGS, including one sweep through the production
  contact update;
- existing PhoenX mass splitting, which was bit-identical to ordinary PGS in
  this control even at batch size one.

The evidence is that changing the normal fixed point independently invalidates
the coupled friction fixed point. A useful preconditioner must propagate normal
and frictional response together.

## Full-patch support-tree solve

Assume a grounded support forest whose selected edges are full-rank sticking
contact patches. A full patch constrains the relative spatial velocity of its
two bodies, so each grounded component is a temporary rigid cluster during the
preconditioner.

For body `i`, let its desired velocity change be `(delta_v_i, delta_omega_i)`.
Its momentum change about the world origin is

```text
delta_p_i = m_i delta_v_i
delta_l_i = I_i delta_omega_i + x_i cross delta_p_i.
```

For a subtree rooted at body `r`, sum these quantities bottom-up. If the parent
patch is centered at `c`, the exact impulse wrench that the parent must apply
to that subtree is

```text
P_r = sum(delta_p_i)
L_r(c) = sum(delta_l_i) - c cross P_r.
```

Internal patch impulses cancel from the subtree momentum balance. Therefore
this recursion is the exact KKT solution for a grounded full-patch tree, not a
mass scaling or dominance approximation. Its cost is linear in bodies and
edges, and it avoids the conditioning problems of a dense Delassus solve.

## Conservative physical acceptance

An exact support wrench is accepted only if it can be reconstructed from the
actual contact points with every point force inside its Coulomb cone.

For contact-point force vector `f`, build the patch map

```text
A_i f_i = [f_i, r_i cross f_i]
A = [A_0 ... A_n].
```

The fast path computes the minimum-norm exact distribution

```text
f = A^T (A A^T)^-1 wrench.
```

It accepts only when reconstruction residual is small and every point obeys
`normal >= 0` and `length(tangent) <= mu * normal`. A different feasible
distribution may exist when this test rejects, so rejection is conservative
and ordinary true-mass GS remains the fallback. Sliding, separating, tipping,
single-point, rank-deficient, nonconvex, cyclic, or otherwise unsupported
components must not be welded by the fast path.

## Current evidence

The isolated planar and spatial prototypes show:

- named 100:1, 400:1, and 4096:1 stacks reach machine-precision residual;
- the bottom-up recursion matches dense KKT solutions to about 1e-14 in the
  named 3-D cases;
- a branched 3-D support tree matches dense KKT to 2.1e-14;
- 249/250 randomized 1--8 body 3-D chains spanning masses from 0.01 to 10,000
  admit an exact Coulomb distribution;
- the conservative minimum-norm gate retains 239 of those 249 feasible cases;
- accepted cases reconstruct patch wrenches to at most 2.7e-12 and never
  increase kinetic energy;
- explicit large-slide, separation, and tipping cases fail the wrench gate;
- the graph-captured Warp prototype processes 8,192 five-body trees in a
  10.24 microsecond median replay on an RTX PRO 6000 Blackwell; 8,012/8,192
  randomized trees pass the conservative gate.

Run the GPU solve/acceptance benchmark with:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.experimental.bench_support_tree_patch
```

The benchmark excludes dynamic forest construction and PhoenX state updates,
so it is a feasibility/cost measurement rather than an end-to-end speed claim.

## Integration constraints

- Joint topology is persistent and may precompute parent/order/articulated
  response data. Contact topology is transient and must be rebuilt or validated
  after collision detection.
- The selected contact forest must be rooted at static or kinematic support.
  One deterministic parent patch per body prevents cycles; unsupported edges
  remain ordinary PGS constraints.
- Eligibility must be component-local. A sliding patch in one independent
  stack must not disable all other stacks in a large single world.
- The correction should run in the bias-free relax phase, after temporal
  position integration. This avoids erasing the Hertz/bias velocity used for
  penetration recovery.
- Existing accumulated lambdas and the proposed delta distribution must be
  checked together against each contact cone before any state is published.
- Results and lambdas must be published atomically per accepted component; a
  rejected component leaves the original PGS state untouched.
- Single-world and multi-world modes, arbitrary numbers of trees per world,
  end-to-end CUDA graph capture, deterministic classic PhoenX, and changing
  contact graphs are required.

Before retention, tests must cover momentum, kinetic-energy nonincrease for
accepted plastic corrections, Hertz/TGS position recovery, sliding, tipping,
separation, impacts, moving kinematics, branches, cycles, nonconvex/mesh
contacts, arbitrary manifold counts, mixed joints and contacts, and broad
non-stack scenes. End-to-end performance must include forest construction,
acceptance, publication, and fallback cost.
