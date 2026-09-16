# Full contact feasibility controls

These are rejected frozen CPU controls, not live solver changes. They use the
geometry-consistent joint operator, all 70 eligible unbiased contact points,
the source Coulomb coefficients and compliant drives. Speculative impulses
remain fixed. No mass, recovery, geometry, or iteration policy was changed.

## Normal block followed by metric tangents

The unsquared global normal NNLS subproblem reaches normal residuals of
1--4e-15 m/s. Subsequent sequential metric tangent updates reopen normal
constraints: after eight cycles the normal/tangent residuals are
0.0832664/0.0170075 m/s. This is worse than the pointwise control and demonstrates
that accurate normals alone do not address the strong normal/tangent coupling.

Artifacts: /tmp/colibri_joint_coupled_normal8.json and matching NPZ.
Implementation: sweep_joint_constrained_contacts.py --coupled-normals.

## Fully coupled non-associated Coulomb reference

The seed costs 64 metric point sweeps (4480 point updates and 81984 disk
bisections). The bounded nonlinear solve then stops at 100 function/Jacobian
evaluations; its 1.19 s CPU time excludes the seed and factor construction.

| Check | Result |
| --- | ---: |
| Normal residual | 0.00139322 m/s |
| Tangent residual | 0.00574627 m/s |
| Coulomb disk violation | 1.73135e-5 Ns |
| Joint/drive equation residual | 1.51182e-15 |
| Linear momentum plus static reaction | 2.069e-15 Ns |
| Angular momentum plus static reaction | 1.0243e-10 Nms |
| Kinetic energy change | -1.38589e-4 J |
| Contact midpoint work | -9.56052e-3 J |
| Actual drive midpoint work | +9.42193e-3 J |
| Work balance error | 1.14492e-16 J |

The equation and balance checks pass, but the contact KKT and cone checks fail.
The angular balance residual includes the stored contact lever precision.
Artifacts: /tmp/colibri_joint_coulomb210.json and matching NPZ.
Implementation: solve_coupled_contact210.py. Physical audit fields were appended
after the original solve; preserve this report when rerunning the solver.

## Exact pure-dual feasibility diagnosis

Exact rational arithmetic on the stored binary coefficients gives a 34-dimensional
hard-joint velocity nullspace, normal response rank 25, and full contact response
rank 29. There are therefore 181 pure contact-impulse gauge directions among
the 210 rows. No numerical singular-value cutoff establishes these ranks.

Let N span the exact hard-joint velocity nullspace and C contain the contact
wrenches. A reaction redistribution d obeying (C N)^T d = 0 preserves the
physical velocity response. It also preserves compliant-drive response:
positive drive compliance determines drive impulse change from the unchanged
velocity in its constitutive equation.

At fixed velocity, cone feasibility is convex: separating contacts require zero
impulse; sliding contacts require the Coulomb boundary direction opposite slip;
sticking contacts permit their friction disk. This is only a reaction
redistribution subproblem, not replacement of the non-associated dynamics by
a cone quadratic program.

However, the current candidate has nine normal velocities below -1e-8 m/s,
with minimum -0.0013932165644 m/s. A pure-dual redistribution cannot alter those
velocities. Thus it cannot make this candidate a feasible full Coulomb solution.
Cone repair alone would be insufficient evidence of convergence.

Artifacts: /tmp/colibri_contact_dual_gauge_diagnosis.json and
/tmp/colibri_normal_exact_rank.json. A successful next method must change the
physical velocity while managing the large exact impulse gauge space.
