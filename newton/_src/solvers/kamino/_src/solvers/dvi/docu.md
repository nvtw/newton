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

The integrated solver keeps that coupled `DualProblem` system. For worlds with a usable bilateral block it now factors only the bilateral joint block, solves bilateral impulses directly, then alternates limit/contact projected inequality sweeps with bilateral re-solves between blocks. After the last projected inequality sweep it performs one final bilateral re-solve, using the already factored block, so the final contact/limit update is not left as equality-constraint velocity. The outer `block_iterations` setting controls the direct-bilateral/projected-inequality loop and defaults to 32; `contact_iterations` controls unilateral limit/contact sweeps per block and defaults to 4. Contacts sharing a dynamic body are assigned different colors once per solve on the dense path, and each world is swept inside one block so color barriers do not require one kernel launch per color. Heterogeneous per-world DVI configs share host launches but each world stops inequality updates at its own block/contact iteration budget. Scalar contact updates use the mraksha-style trace preconditioner, `D_eff = trace(D_contact) / 3`, before projecting back to the Coulomb cone; this reduces the contact natural-map and complementarity residuals compared with per-row diagonal scaling on DR Legs. On Jacobi contact paths, `contact_jacobi_omega` and `contact_jacobi_relaxation` expose the mraksha-style update damping explicitly. Repeated direct-block bilateral re-solves use a per-world active-unilateral dimension mask on the dense path, so worlds without active limits or contacts keep the first bilateral result and skip later triangular solves. Direct-block DVI status reports `iterations = block_iterations * contact_iterations` for active unilateral worlds so benchmark iteration counts reflect projected sweep work; worlds with no active limit or contact rows report the single bilateral solve/status pass.

With Kamino sparse dynamics enabled, DVI now avoids building the full dense Delassus matrix. It assembles only the joint-joint bilateral block from the sparse constraint Jacobian, factors that dense block with the existing blocked LLT solver, and evaluates contact/limit updates through the sparse matrix-free Delassus `matvec`. Bilateral right-hand sides are also built from a sparse `D * lambda` product with joint rows zeroed before each re-solve. This is the path used by the focused `DVI` benchmark config for DR Legs.

DR Legs profiling on June 22, 2026 showed that repeated bilateral LLT solves are the main sparse-DVI hot path. Smaller LLT tiles (`block_size=16`) and the current RCM/semi-sparse LLT path were slower for this robot. The useful local optimization was to decouple projected inequality sweeps from repeated bilateral re-solves: `bilateral_solve_period=2` lets DVI run more contact sweeps per LLT solve while still finishing with a final bilateral solve.

The public Kamino DVI path builds an unpreconditioned dual problem. The diagonal dual preconditioner is useful for PADMM, but for DVI it can make small solver-space bilateral residuals correspond to large physical constraint velocities after unpreconditioning. Keeping the assembled DVI problem in physical constraint units makes contact cone projection and physics metrics refer to the same velocity scale. The direct bilateral block still uses a private diagonal row/column scaling before factorization, then scatters the solved impulses back to physical units. This local scaling stabilizes rank-deficient closed-loop robots such as DR Legs without exposing preconditioned contact rows to the projected inequality updates. Its scaled diagonal floor is deliberately small (`7.0e-7`) to reduce physical equality residuals, but not so small that the closed-loop nullspace lifts the robot out of contact; `2.5e-7` was unstable in the DR Legs first-contact regression and `5.0e-7` lost contact support in the DR Legs force-balance regression.

Contact rows use the trace-average scalar denominator before projection onto the Coulomb cone. Using one scalar denominator from the maximum tangent/normal diagonal can preserve stale normal warm-start impulses when the normal diagonal is much smaller than the tangent diagonals; opening gap contacts then keep large separating forces instead of releasing. The trace-average denominator keeps the mraksha scalar-preconditioner shape without that max-denominator release failure. Kamino also exposes opt-in full 3x3 contact block preconditioning, following the mraksha reference path, but keeps the tuned scalar update as the default until broader contact benchmarks prove the block path is consistently more reliable.

Kamino now supports mraksha bounded recovery bias as an opt-in cap on penetration correction. Positive-gap contact rows still keep the full opening bias so stale warm-started normal impulses can release instead of pulling bodies together during separation. In the Dr Legs example the stable existing Dr Legs bias is kept (`delta=1e-3`, model/config `gamma`) with a 1 m/s safety cap; the focused DVI benchmark uses `gamma=0.05` with the same cap.

The full fallback PGS/Jacobi path is still used when a direct bilateral block cannot be allocated, such as world sets with empty joint blocks; that fallback path is the one governed by `max_iterations`.

## mraksha Comparison

The mraksha DVI checkout uses the same high-level split: inequalities are projected updates and bilateral joints are handled by a direct/sparse block solve. The main difference is representation. mraksha updates body velocities stage-by-stage, while Kamino keeps the `v_plus = D lambda + v_f` dual system but can now evaluate the sparse path through matrix-free Delassus products.

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

A June 17, 2026 development sanity check on an RTX 3080 Laptop GPU used 50-step `--mode accuracy` runs on the harder one-world DR Legs contact scene. With the default scalar trace-preconditioned colored-contact update, DVI measured 64.9 ms median step time without CUDA graph replay versus 89.0 ms for PADMM fast, and 58.6 ms with graph replay versus 79.2 ms for PADMM fast. The same run reported DVI median physics residuals of `r_kinematics=4.767e-3`, `r_cts_joints=4.417e-5`, `r_cts_contacts=1.225e-4`, `r_ncp_dual=7.194e-3`, `r_ncp_compl=3.210e-6`, and `r_vi_natmap=4.184e-4`. The opt-in contact block preconditioner reduced the NCP dual/natmap residuals slightly in this run, but worsened kinematics slightly and did not improve timing enough to justify making it the default.

A June 22, 2026 30-step DR Legs contact accuracy run on the same RTX 3080 Laptop GPU compared PADMM fast, dense DVI, and sparse direct-bilateral DVI with CUDA graph replay. Median step times were 81.2 ms for PADMM fast, 60.3 ms for dense DVI, and 11.9 ms for sparse DVI (`block_iterations=16`, `contact_iterations=2`). Sparse DVI median physics residuals were `r_kinematics=4.368e-3`, `r_cts_joints=4.342e-5`, `r_cts_contacts=1.177e-4`, `r_ncp_dual=9.766e-3`, `r_ncp_compl=3.085e-5`, and `r_vi_natmap=4.706e-3`. The sparse path therefore gives the mraksha-style speedup target while keeping joint/contact constraint residuals close to dense DVI; the NCP natural-map residual remains the main tuning gap.

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

DVI should remain opt-in until broader benchmark coverage is stable, but the main performance concern changed. The sparse direct-bilateral path is now much faster than PADMM fast in the single-world Dr Legs matrix while preserving Kamino's constraint-space solution contract. Next work should focus on improving sparse unilateral NCP residuals, multi-world scaling, and replacing the dense bilateral block factor with a true sparse block LDL once the symbolic Kamino joint graph is available.

