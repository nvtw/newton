# Physical coloring and coupled overflow: momentum and convergence

CPU research/design study, 2026-09-16. No production changes or GPU experiment. The requirement is a dynamic friction-supported base with correct motion, not maximum throughput at a tolerated drift. Keep the requested 30 temporal substeps per120Hz collision update; report any extra algebraic solve work explicitly. Mencius owns the physical-head/split-overflow prototype; this document does not duplicate that implementation.

## What the literature establishes—and does not

Tonge, Benevolenski and Voroshilov motivate mass splitting by parallelism and low-iteration jitter behavior: PGS depends on ordering, whereas Jacobi sacrifices convergence speed for order independence. This does not establish accurate sticking at a fixed small iteration count. Our current momentum ledger is consistent with a conservative but underconverged method. [Author preprint, 2012](https://www.richardtonge.com/papers/Tonge-2012-MassSplittingForJitterFreeParallelRigidBodySimulation-preprint.pdf).

Erleben compares proximal contact iterations and reports more predictable convergence for Gauss–Seidel than Jacobi variants. Its convergence statements must be read with the particular friction law and iteration assumptions; they are not a blanket guarantee for our non-associated Coulomb implementation, rank-deficient contacts, soft drives, or arbitrary fixed iteration budget. [Author paper, 2017](https://erleben.github.io/pubs/2017/erleben.17/erleben.17.pdf).

Macklin et al. formulate non-smooth Newton contact solves and a GPU iterative linear solver with complementarity preconditioning. This supports investigating coupled physical solves rather than changing masses. Their smooth isotropic friction model must not be silently substituted for the current contact law. Our own original-law frozen replay still has unresolved cases, so the paper alone is not an implementation acceptance certificate. [Author manuscript, 2019](https://arxiv.org/abs/1907.04587).

The following derivations and recommendations are our analysis, not claims from those papers. The first two PDF fetches failed after indexed author-source excerpts were returned; no unread detailed theorem is relied upon.

## Momentum is independent of iteration convergence

At fixed pose, let W=M^-1, J contain actual common-point contact/joint wrenches, and update physical velocity by delta_v=W J^T delta_lambda. Every internal impulse pair changes total linear and angular momentum by zero, independent of whether delta_lambda solves the contact equations. For external supports, the same expression yields the measured support reaction. Motor torque is an equal/opposite internal couple when both endpoints are dynamic.

The exact kinetic-energy change is delta_E=(J^T delta_lambda)^T (v_before+v_after)/2. This identity is an accounting test, not a guarantee that friction dissipates energy: simultaneous stale-velocity updates can do positive net work. Preserve actual physical inertia and evaluate friction/normal residuals at the final physical velocity.

Conversely, a momentum-conserving update can leave large penetration, slip or drive residual. Our native point42 has interior Coulomb impulse and nearly zero copy slip, then0.302mm/s physical slip after averaging. That is a constitutive/convergence failure, not evidence of a missing equal/opposite impulse.

## Twelve physical colors, then overflow

Use ordinary conflict-free graph coloring. For colors0…11, solve on physical velocities and physical W, with immediate visibility of each preceding color. Constraints within a color may execute in parallel only if their dynamic response supports are disjoint. A projected articulation response can couple nominally different body endpoints; ordinary endpoint coloring is insufficient for such a response.

Twelve is a scheduling budget, not a physical parameter or proven optimum. Keep every remaining constraint in an explicit overflow set. Warm-start each owned impulse exactly once. Actual finite drive compliance/reference, predictive contacts, restitution and recovery remain unchanged. Do not classify an almost stationary body as fixed.

The existing ordinary route does not yet provide this physical head when a body has overflow copies: regular rows share slot0 but still use count-scaled inverse mass. The CPU control `check_physical_head_accounting.py` demonstrates the consequence without claiming a momentum defect. At uniform N=2/98, the head residual is-0.251/-0.497 versus roundoff for physical-head execution; both schedules pass all six momentum components. Production validation remains separate.

## Overflow alternatives

| Overflow treatment | Momentum accounting | Convergence limitation / use |
|---|---|---|
| Deterministic sequential physical rows | Paired common-point impulses, no copies | Strongest simple control; order-dependent GS can still converge slowly at400:1. Serial cost is accepted for establishing accuracy. |
| Additional conflict-free physical colors | Same as sequential rows; exact within-color independence | Avoids copying entirely. More colors/barriers; preserves chosen colored order rather than a different arbitrary serial order. |
| Body-disjoint clusters, sequential physical rows inside each | Conservative if complete owned impulses and reactions are applied | Parallel clusters must not share dynamic response; boundary coupling requires ordered outer passes. Grouping alone does not solve an internal stiff mode. |
| Coupled normal or normal+tangent blocks | Conservative with full physical operator and paired scatter | Both support interfaces must be in one block for highmass slow modes. Normal-only coupling did not resolve our dominant full-friction error. |
| Split overflow only, physical head first | Correct broadcast/average can conserve P/L | Head no longer diluted, but tail averaging can reopen head constraints. This is a useful intermediate comparison, not removal of all splitting drawbacks. |
| Additive physical Jacobi impulses | Conservative only if the same delta_lambda weights multiply both endpoint reactions | Stale velocities can overshoot, increase energy, and converge slowly. Body-specific scaling of only the scattered velocity breaks paired impulse accounting. |

## Why high mass requires coupling, not simply removing copies

For two active normal constraints with SPD mobility A=[[a,c],[c,b]], scalar GS has error factor c^2/(ab). As this approaches one, many sweeps are needed despite exact momentum conservation. Our saved two-interface400:1 operator gave0.9975887; eight sweeps retain about98.1% of that slow error. A joint block containing both rows removes this linear slow mode in one exact block solve. Separate per-interface blocks do not.

For friction, the active face changes and sliding couples normal force to tangent magnitude/direction. A cone-constrained energy QP is not automatically the same non-associated Coulomb law. A small island solver must pass original normal complementarity, friction disk and maximum-dissipation direction checks, plus actual P/L/work after storage/scatter. Pure-dual impulse gauges may be removed only with a geometric response certificate; retain every real body-response mode.

A connected component spanning support, base and attached Frame must retain the joint's hard rows and finite compliant drive. Correcting the base alone disturbs joint velocity without its reaction. Including Frame disturbs its other contacts/joints; inspect global residuals or enlarge the coupled component. A local improvement does not certify global convergence.

## Recommended bounded evaluation order

1. Establish a native twelve-color physical head followed by sequential physical overflow control. No copying or mass modification anywhere in this reference. Test off-center joint/contact momentum and warm-start ownership, static support, known sliding, motor work and highmass motion. Slow execution is acceptable.
2. Compare the same ordered rows with additional physical colors. Require identical ownership and comparable residual convergence; only then measure speed.
3. If sequential physical GS still fails highmass or base sticking, identify the dominant connected constraint response and introduce a small coupled block. Retain all rows, compliant drives and exact Coulomb law. Count extra nonlinear/linear work.
4. Compare split-overflow and additive-Jacobi variants to the physical control at equal work and equal achieved accuracy separately. Never select them solely because equal-iteration FPS is higher.

Acceptance is stationarity/sliding behavior, joint/drive constitutive residuals, physical contact gaps, six-momentum reaction accounting, and friction/drive/recovery work—not only finite state, bounded kinetic energy, or geometric survival. Timings become meaningful after those gates pass. No method in this document is promoted on research or algebra alone.
