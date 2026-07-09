# Articulation-aware colored Gauss--Seidel

## Goal

This experimental path keeps PhoenX body state in maximal coordinates, but
uses the exact constrained mobility of each acyclic joint tree while solving
contacts. It targets the convergence of a reduced-coordinate contact solve
without publishing one generalized response row per contact or serializing
the entire GPU behind one articulation.

The established `maximal_projected` path remains the fallback. No contact
topology is cached across collision frames.

## Constrained mobility

Let body twists be `v = K qdot`, body spatial mass be block diagonal `M`,
and generalized mass be

```text
H = K^T M K.
```

The velocity response to a maximal-coordinate impulse is

```text
G = K H^-1 K^T.
```

`G` is never formed densely. The existing tree factorization supplies the
subtree articulated mass `A_i`, joint motion subspace `S_i`, and
`D_i = S_i^T A_i S_i`. Define

```text
Q_i = S_i D_i^-1 S_i^T
F_i = (I - Q_i A_i) X_i.
```

For a floating root,

```text
C_root = A_root^-1
C_i = F_i C_parent F_i^T + Q_i.
```

`C_i` is exactly the diagonal block `G_ii`. For two bodies `a` and
`b` with lowest common ancestor `l`,

```text
G_ab = T(a <- l) C_l T(b <- l)^T,
```

where `T` is the product of the `F` maps along the path. This covers
self-contact without assuming one contact patch or a bounded contact count.
An anchored root uses `C_root = 0`.

The recurrence was checked against dense KKT mobility for every diagonal and
cross block in 100 randomized trees. The standalone Warp implementation
matches dense float32 results and computes all diagonal blocks for 8,192
29-body trees in about 114 microseconds on RTX PRO 6000 Blackwell.

## Contact ordering

Ordinary body coloring is insufficient once a contact impulse is propagated
through a joint tree: contacts on different links can update the same
generalized velocity. Treat each articulation as one graph node for the
contact-only coloring. Then at most one contact column per articulation is
active in a color, preserving true Gauss--Seidel ordering.

For each contact color:

1. solve each contact column using `J G J^T` from the diagonal/cross
   mobility recurrence;
2. scatter its spatial impulse to the touched body or bodies;
3. apply all independent articulation impulse batches with one factored tree
   response; and
4. continue to the next color from a joint-consistent maximal body state.

The tree response reuses `A_i` and `D_i^-1`; it only performs vector
up/down passes. The standalone GPU implementation takes about 86 microseconds
for 8,192 29-body trees, around 3.7 times less than rerunning the complete
mass-metric projector.

Contact columns remain arbitrary-sized and retain PhoenX's hard Hertz damping,
friction cone, warm start, and reporting semantics. Convex-pair patch
assumptions are not required. Non-convex mesh contacts remain ordinary
independent columns after contact reduction.

## Why articulation coloring is required

The reproducible randomized benchmark compares two sweeps over four
three-row contact blocks on 16-body trees:

| Mass ratio | Exact articulation GS median residual | Free-body PGS then project | Body-colored articulated Jacobi |
| ---: | ---: | ---: | ---: |
| 100:1 | 0.115 | 0.635 | 2.356 |
| 10,000:1 | 0.122 | 0.706 | 2.053 |
| 1,000,000:1 | 0.137 | 0.739 | 1.684 |

The body-colored variant can diverge because joint response couples links
that the colorer considered independent. Under-relaxation would mask rather
than remove that error. Articulation-aware coloring restores the GS
independence invariant and remains effective across six mass decades.

Run the probe with:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.experimental.bench_articulated_color_gs
```

## Integration constraints

- Build a contact-only articulation graph after collision detection. Structural
  tree joints must not inflate its color count.
- Prepare structural joint geometry once, then reuse the factorization and
  mobility during every contact color.
- Motors, limits, friction, loop joints, and unsupported joint types retain
  ordinary deterministic PhoenX rows unless an exact block path owns them.
- Multiple articulations per world and both world layouts are mandatory.
- Use fixed capacities and launch shapes so the complete step remains CUDA
  graph capturable.
- Preserve equal-and-opposite maximal impulses and exact tree response, hence
  floating-articulation momentum.
- Promotion requires complementarity, friction, impacts, warm starting,
  force reporting, determinism, contact topology changes, high mass ratios,
  G1/ANYmal/H1/Go2, closed loops, and end-to-end training gates.

## Parallel tree contraction

The first implementation uses one warp per articulation with parallel work
inside each tree depth. This already exposes thousands of independent warps
in multi-world RL and one warp per articulation in single-world scenes.

Rake/compress contraction remains a second execution path for deep or very
large trees. It can reduce the dependency depth from `O(height)` to
`O(log n)`, but it adds matrix work and storage. It should be promoted only
when measured topology regimes beat the simpler depth-parallel factor/response
without changing the equations.
