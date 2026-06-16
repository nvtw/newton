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

The integrated solver keeps that coupled `DualProblem` system. For worlds with only unilateral rows it uses projected Gauss-Seidel on the dense Delassus matrix. For worlds with bilateral joint rows it factors the bilateral block, builds the unilateral Schur complement, solves projected Gauss-Seidel on that reduced unilateral system, and then back-substitutes the bilateral impulses. This keeps bilateral residuals near tolerance but makes contact-heavy models pay for Schur construction.

## mraksha Comparison

The mraksha DVI checkout is faster in part because it uses a different solve split:

1. Joint limits are solved and applied first as unilateral velocity constraints.
2. Bilateral joints are solved and applied second, usually with block-sparse LDL.
3. Contacts are solved and applied last with a separate sparse Jacobi or LDL contact solve.

That pipeline is close to a traditional DVI engine: every stage updates body velocities, and later stages see the impulses from earlier stages. It avoids building a dense reduced contact/limit Schur complement. The tradeoff is that it is not solving Kamino's fully coupled dual system in one pass.

The Kamino DVI implementation deliberately preserves Kamino's equation system. The current speed bottleneck is therefore not PGS iteration count. It is the Schur complement build for unilateral rows, especially when contact capacity raises `max_total_cts`. A faster next implementation should batch the bilateral solves used for Schur columns or assemble the reduced operator with a custom tiled kernel. A larger algorithm change could mimic mraksha's sequential limit, bilateral, contact split, but that would need a separate correctness decision because it changes the effective system being solved.

## Benchmarking

The focused benchmark entrypoint is:

```bash
uv run --extra dev -m newton._src.solvers.kamino._src.utils.benchmark.dvi_padmm_matrix \
    --world-counts 1 16 \
    --contact-states no-contact contact \
    --cuda-graph-modes off on \
    --num-steps 50 \
    --mode convergence \
    --solver-configs padmm-fast dvi
```

Local measurements on June 16, 2026 with an RTX PRO 6000 Blackwell show:

| Scenario | CUDA graph | PADMM fast | DVI | Notes |
| --- | --- | ---: | ---: | --- |
| Dr Legs, 1 world, no ground | off | 80 FPS | 130 FPS | DVI converges in 1 iteration. |
| Dr Legs, 1 world, ground | off | 73 FPS | 75 FPS | DVI converges reliably; Schur construction dominates. |
| Dr Legs, 1 world, no ground | on | 165 FPS | 158 FPS | Graph launch overhead reduction helps PADMM. |
| Dr Legs, 1 world, ground | on | 97 FPS | 84 FPS | Contact capacity makes current DVI slower. |
| Dr Legs, 16 worlds, no ground | on | 147 FPS | 150 FPS | Rough parity. |
| Dr Legs, 16 worlds, ground | on | 83 FPS | 79 FPS | Current DVI remains slower with contacts. |

The mraksha hanging Dr Legs benchmark, run with cached kernels and a runtime import shim for its stale public exports, reported 74 frames/s for 250 frames with 4 DVI substeps per frame. This is not the same scene as the Kamino standing/contact benchmark, but it confirms that mraksha's current DVI path is in the same single-world frame-rate range while using a different equation split.

## Recommendation

DVI should remain opt-in for now. It is clearly useful for correctness diagnostics and it often reaches tighter residuals in fewer iterations than PADMM, but the current Schur complement construction is not consistently faster than PADMM in contact-heavy Dr Legs scenes. Do not make DVI the default until the Schur build is batched or replaced and the benchmark matrix shows a stable throughput advantage without losing the coupled-system residual guarantees.
