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

The first integrated DVI solver uses projected Gauss-Seidel directly on the dense Kamino Delassus matrix. It keeps the same cone projections and De Saxce contact velocity augmentation as PADMM. Sparse block LDL is the intended next step for equality groups, but it should plug into the same `DualProblem` data rather than introducing a second equation system.
