# Improving an Iterative Physics Solver Using a Direct Method

Notes from `docu.pdf`, a Roblox presentation by Maciej Mizerski.

## Core Problem

The solver computes constrained rigid-body motion. Constraints are differentiable functions of body poses. At the velocity level, the constraint Jacobian maps body-space velocities into constraint-space velocities:

```text
dC/dt = J_C V_B
```

An Euler step with impulses gives:

```text
V_1 = V_0 + W external_forces dt + W J^T Lambda
```

The solver finds impulses `Lambda` such that the post-step constraint-space velocity satisfies the constraint conditions.

For equality constraints, linearization gives:

```text
K Lambda = R
K = J W J^T
```

`K` is symmetric positive semidefinite. In Kamino terms this is the Delassus matrix already built by `DualProblem`.

## PGS and LDL-PGS

Traditional engines use projected Gauss-Seidel (PGS). A scalar row update is:

```text
delta_R_i = R_i - K_i Lambda
Lambda_i <- Lambda_i + D_i^-1 delta_R_i
project Lambda_i when the row is constrained by an inequality cone
```

The DVI presentation rewrites the expensive `K_i Lambda` product as `J_i delta_V`, where `delta_V = W J^T Lambda`. This is the Kaczmarz-style impulse solver used by DVI engines.

The LDL-PGS variant groups equality constraints and solves their block with sparse LDL, while inequality/contact/friction constraints continue to use projected updates. The numeric phase is:

1. Run `N - 1` PGS iterations.
2. Compute an LDL decomposition of the equality block.
3. Run one block Gauss-Seidel update for equality constraints.
4. Continue with projected updates for inequalities.

## Regularization

The equality block can be singular or ill-conditioned. The presentation regularizes it with scaled diagonal compliance:

```text
H_tilde = H + diag(epsilon_i)
epsilon_i = epsilon * H_ii
```

This is equivalent in spirit to CFM in Bullet/ODE and adds controlled compliance.

## Sparse Block LDL

The sparse method treats the constraint matrix as a graph:

- Constraint-matrix diagonal block -> graph node.
- Off-diagonal block -> graph edge between constraints that share bodies.
- Gaussian elimination -> graph elimination.

The symbolic phase runs once per mechanism and computes:

- A pivot order.
- Fill-in edges created by elimination.
- Reduction scattering indices.

The numeric phase runs every frame and performs block LDL via recursive Schur complements:

```text
N_i <- LDL(N_i)
L_*i = E_*i N_i^-1
S_i <- S_i - E_*i L_*i^T
```

The presentation notes that the reduction step dominates CPU time, so the implementation specializes small block degrees and uses packed sparse block storage.

## Kamino Integration Notes

Kamino already constructs the target equation system:

```text
v_plus = D lambda + v_f
```

where `D` is the Delassus matrix and `v_f` is the biased free constraint velocity. The DVI backend should therefore solve this existing system instead of replacing Kamino kinematics or dynamics.

The integrated solver keeps that coupled `DualProblem` system. For worlds with a usable bilateral block it now factors only the bilateral joint block, solves bilateral impulses directly, solves limits with scalar PGS, solves contact inequalities with a graph-colored projected Gauss-Seidel sweep over the existing Kamino contact rows, then re-solves the bilateral block. The outer `block_iterations` setting controls the direct-bilateral/projected-inequality loop and defaults to 32; `contact_iterations` controls colored contact sweeps per block and defaults to 4. Contacts sharing a dynamic body are assigned different colors, and each world is swept inside one block so color barriers do not require one kernel launch per color.

Contact rows use per-component diagonal scaling before projection onto the Coulomb cone. Using one scalar denominator from the maximum tangent/normal diagonal can preserve stale normal warm-start impulses when the normal diagonal is much smaller than the tangent diagonals; opening gap contacts then keep large separating forces instead of releasing.

Kamino now supports mraksha bounded recovery bias as an opt-in cap on penetration correction. Positive-gap contact rows still keep the full opening bias so stale warm-started normal impulses can release instead of pulling bodies together during separation. In the Dr Legs example the stable existing Dr Legs bias is kept (`delta=1e-3`, model/config `gamma`) with a 1 m/s safety cap; the focused DVI benchmark uses `gamma=0.05` with the same cap.

This avoids the previous dense unilateral Schur complement build. The full fallback PGS path is still used when a direct bilateral block cannot be allocated, such as heterogeneous world sets with empty joint blocks.

## mraksha Comparison

The mraksha DVI checkout uses the same high-level split: inequalities are projected updates and bilateral joints are handled by a direct/sparse block solve. The main difference is representation. mraksha updates body velocities stage-by-stage, while Kamino keeps the assembled `v_plus = D lambda + v_f` dual system and applies the split solve to that matrix.

The important performance lesson from mraksha was to avoid explicitly materializing a dense reduced contact/limit Schur complement. Kamino now follows that lesson while preserving the Kamino equation system.


## Benchmarking

The focused benchmark entrypoint used for the latest fast comparison is:

```bash
uv run --extra dev -m newton._src.solvers.kamino._src.utils.benchmark.dvi_padmm_matrix \
    --world-counts 1 \
    --contact-states no-contact contact \
    --cuda-graph-modes off on \
    --num-steps 200 \
    --mode total \
    --solver-configs padmm-fast dvi
```

Local measurements on June 16, 2026 with an RTX PRO 6000 Blackwell, 200 steps, and the colored-GS split DVI path (`block_iterations=32`, `contact_iterations=4`) show:

| Scenario | CUDA graph | PADMM fast | DVI | DVI vs PADMM fast |
| --- | --- | ---: | ---: | ---: |
| Dr Legs, 1 world, ground | off | 74.2 FPS | 227.4 FPS | 3.1x |
| Dr Legs, 1 world, ground | on | 100.5 FPS | 318.4 FPS | 3.2x |
| Dr Legs, 1 world, no ground | off | 80.6 FPS | 287.1 FPS | 3.6x |
| Dr Legs, 1 world, no ground | on | 167.8 FPS | 497.0 FPS | 3.0x |

The end-to-end `example_kamino_robot_dr_legs` loop advances two substeps per frame and therefore reports lower frame FPS than the focused benchmark. In the one-world standing contact scene, with a null viewer, CUDA graph replay, 120 warmup frames, and 300 timed frames, DVI measured 54.2 FPS while PADMM measured 25.4 FPS. Contact aggregation in the source-default DVI probe measured vertical contact force at 0.981 times the robot weight; PADMM measured 1.000 in the same probe.

A CUDA-kernel profile of DVI without graph replay over 20 example frames (40 substeps) showed:

| Kernel group | GPU time share | Calls | Average |
| --- | ---: | ---: | ---: |
| Colored contact GS | 74.4% | 1,280 | 210 us |
| Bilateral LLT solve | 17.0% | 1,320 | 46.6 us |
| Solution-vector update | 4.1% | 1,320 | 11.1 us |
| Bilateral LLT factorization | 1.3% | 40 | 114 us |

The contact GS sweep is the optimization target. A shorter `24/3` schedule reached 80.8 example FPS in a short graph replay run, but its vertical force balance dropped to 0.977 times weight and body residual velocity increased, so the default stays at `32/4`.

The mraksha hanging Dr Legs benchmark, run with cached kernels and a runtime import shim for its stale public exports, reported 74 frames/s for 250 frames with 4 DVI substeps per frame. This is not the same scene as the Kamino standing/contact benchmark, but it motivated the same important solver split: do not materialize a dense reduced contact/limit Schur complement.

## Recommendation

DVI should remain opt-in until broader benchmark coverage is stable, but the main performance concern changed. The old Schur construction was the bottleneck; the current split path is faster than PADMM fast in the single-world Dr Legs matrix while preserving the assembled Kamino dual system. Next work should focus on multi-world scaling, per-world block-iteration control, and possible sparse or velocity-space DVI kernels.

