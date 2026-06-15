# Improving an Iterative Physics Solver Using a Direct Method

Source: `docu.pdf`, Maciej Mizerski, Roblox. This Markdown file is an
implementation-oriented extraction for Phoenx articulation work.

## Solver Shape

The method augments a projected Gauss-Seidel rigid-body solver with a direct
solve for equality constraints:

- Collisions and friction remain projected inequality rows.
- Holonomic equality constraints are grouped and solved with sparse block LDL.
- The symbolic phase runs once per mechanism topology.
- The numeric phase runs every frame from the current Jacobian and mass matrix.

The high-level frame flow is:

1. Build the mechanism body graph from bodies and constraints.
2. Optionally shatter high-degree bodies to avoid dense equality blocks.
3. Build the constraint graph, where each equality constraint is a node and
   graph edges indicate shared bodies.
4. Compute an elimination sequence and reduction-scattering indices once.
5. Each frame, assemble `H = J W J^T`, regularize it, factorize with block LDL,
   and apply the inverse inside the solver.

## Constraint Equations

Each equality constraint is a differentiable function of body coordinates:

```text
c(Gamma_a, Gamma_b, ...) in R^d
c(Gamma_a, Gamma_b, ...) = 0
```

The Jacobian must have full rank on the zero locus. Its row count `d` equals the
number of degrees of freedom removed by the constraint.

For body velocities:

```text
V_beta = [v_beta, omega_beta]
dC / dt = J_C V_B
```

The global Jacobian is sparse: each constraint row touches only the two or three
bodies participating in that constraint.

## Examples

Ball socket with local pivots `x_a`, `x_b`:

```text
c(Gamma_a, Gamma_b) = p_a + o_a x_a - (p_b + o_b x_b) in R^3
dc/dt = v_a - (o_a x_a) x omega_a - v_b + (o_b x_b) x omega_b
```

Hinge:

```text
c = [
    p_a + o_a x_a - (p_b + o_b x_b),
    (o_a v_a) dot (o_b u_b),
    (o_a v_a_prime) dot (o_b u_b),
] in R^5
```

The first three rows are the ball-socket pivot match. The last two rows remove
the angular directions orthogonal to the hinge axis.

## Linearized Equality Solve

Euler integration gives:

```text
V_1 = V_0 + W f_ext dt + W J^T Lambda
```

The equality solve enforces:

```text
J V_1 = 0
K Lambda = R
K = J W J^T
R = -J(...)
```

`K` is symmetric positive semidefinite.

## PGS and Kaczmarz View

Naive Gauss-Seidel on `K Lambda = R` is expensive because `K_i Lambda` touches
many non-zero terms. The iterative impulse form keeps:

```text
delta_V = W J^T Lambda
Lambda_i <- Lambda_i + D_i^-1 (R_i - J_i delta_V)
delta_V += W J_i^T delta_Lambda_i
```

That replaces a row-times-global-matrix operation with a local Jacobian-times-
velocity operation.

## Block Partitioning

Rows can be partitioned into blocks:

```text
pi = [pi_0, pi_1, ...]
N_i = J_pi_i W J_pi_i^T
E_ij = J_pi_i W J_pi_j^T
```

Per-constraint blocks are small:

- Ball socket: `3 x 3`
- Hinge: `5 x 5`
- Cylindrical: `4 x 4`

The method groups all equality rows into one holonomic matrix:

```text
H = J_pi_0 W J_pi_0^T
```

## Regularization

`H` can be singular, large, or too dense. The direct solve regularizes by adding
scaled diagonal compliance:

```text
H_tilde = H + diag(epsilon_i)
epsilon_i = epsilon * H_ii
```

This is equivalent to constraint-force mixing in traditional PGS engines.

## Sparse Block LDL

For a symmetric positive definite matrix:

```text
A = L D L^T
```

where `L` is lower triangular with unit diagonal and `D` is block diagonal. The
inverse is applied as:

```text
A^-1 v = L^-T D^-1 L^-1 v
```

The block Gaussian elimination step for a pivot block `N_i` is:

```text
D_i = LDL(N_i)
L_*,i = E_*,i N_i^-1
S_i <- S_i - E_*,i L_*,i^T
E_*,i <- L_*,i
```

The reduction dominates runtime, so the PDF recommends:

- Template/unroll low pivot degrees `1..6`.
- Keep the dense pivot LDL simple.
- Precompute reduction-scattering indices in the symbolic phase.
- Store blocks column-major, with elements inside blocks row-major.

## Graph Interpretation

Sparse matrices map to graphs:

- Body graph nodes are rigid bodies.
- Body graph edges are constraints.
- Constraint graph nodes are constraints.
- Constraint graph edges connect constraints that share bodies.

Gaussian elimination on the matrix corresponds to graph elimination. Eliminating
a pivot connects all of its remaining neighbors, producing fill-in. The symbolic
phase should choose an elimination order that minimizes fill.

The PDF mentions these heuristics:

- Minimum degree algorithm: fast, mediocre ordering.
- Minimum edge creation algorithm: more expensive, better ordering, acceptable
  because it only runs once per mechanism.

The symbolic output is:

- Ordered pivot sequence.
- For each pivot, an ordered eliminated-edge sequence.
- Reduction-scattering indices.

## Body Shattering

Large dense submatrices happen when one body participates in many constraints.
The proposed mitigation is body shattering:

1. Find bodies with total constraint degree above a threshold, for example 20.
2. Split the body into smaller shards.
3. Join shards with rigid joints.
4. Distribute external constraints across shards.

This increases the number of constraints but makes the constraint graph much
sparser.

## Phoenx Implementation Notes

For Phoenx, the useful staged target is:

1. Factor existing joint semantics into reusable full-coordinate equation
   builders that can produce:
   - PGS row data for the existing iterative path.
   - Matrix blocks for the DVI articulation path.
2. Build articulation topology once from Newton joints and Phoenx body ids.
3. Assemble current-frame `H = J W J^T` from full-coordinate Jacobian rows.
4. Add diagonal regularization scaled by `H_ii`.
5. Prefactorize with sparse block LDL for each articulation island.
6. Use the factorized inverse for equality impulses before or inside the
   existing PGS contact/friction loop.

The first production slice should cover fixed, revolute, prismatic, ball, and
free/D6-style constraints without body shattering. Shattering can be introduced
after the direct path is validated on ordinary chain and robot mechanisms.
