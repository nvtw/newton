# Full eligible-contact joint-constrained sweeps

## Scope

Frozen final-relaxation state at frame330. Use the geometry-consistent physical joint response P=G G^T, preserving the resolved small physical modes and existing drive compliance. Start from its corrected joint-consistent state. Keep every speculative row's impulse unchanged. Update all70 eligible unbiased points in deterministic contact-column order, then stored point order. This is a sequential reference order, not a claim of equivalence to the native parallel colored schedule.

At each point, perform the non-associated Coulomb update: scalar normal nonnegative projection followed by an exact2D mass-metric tangent-disk minimization. Source friction coefficients are unchanged. Disk boundary solves use64 bisection steps. No restitution, artificial compliance, damping, rank floors or dropped eligible rows are added.

Before division, exact rational hard-joint nullspace calculations certify all70 normals have nonzero mobility and all70 tangent blocks have rank2. Singular values of the rectangular projected tangent response provide its metric without subtractive cancellation. No numerical rank cutoff is used.

## Residual comparison

All contact residuals use the same physical-W scaling, including the native baseline, so the tangent comparison is consistent.

| State | Maximum normal residual m/s | Tangent residual m/s | Joint equation residual |
|---|---:|---:|---:|
| Native saved | .0375384 | .0154903 | 1.38166 |
| Joint-only corrected | .0518981 | .0258386 | 4.44e-16 |
| 1 contact sweep | .0226693 | .0140688 | 4.22e-16 |
| 4 sweeps | .0199550 | .0116648 | 4.53e-16 |
| 8 sweeps | .0181077 | .0102199 | 4.44e-16 |

Actual drive impulse changes are accumulated through the complement drive coordinates, not held fixed. Both drive constitutive residuals remain at roundoff throughout.

This improves both contact residuals relative to the native saved state while keeping joints solved, but remains far from a converged global contact solution. Maximum contact impulse grows from0.0001201 initially to0.0044796 after one sweep and0.0314601 after eight. No full-scene physical acceptance or production promotion is claimed.

## Physical accounting

The ground supplies external momentum. Including its equal-and-opposite reactions, linear-momentum error is at most1.51e-18 N s and angular error1.81e-11 N m s. The latter is compatible with the saved FP32 contact-lever precision; joint rows remain exactly torque balanced.

At8 sweeps:
- Contact midpoint work:-1.07293566e-3 J.
- Existing compliant-drive midpoint work:+1.11638078e-3 J.
- Kinetic-energy change:+4.34451216e-5 J.
- Work-balance error:5.73e-16 J.

The kinetic increase is accounted for by drive work. The work identity alone does not establish global contact convergence or justify ignoring the growing reactions.

## Cost and artifacts

560 point updates and12608 disk bisection iterations across8 sweeps; about24ms CPU sweep time. This excludes exact-rank certification, high-precision joint QR and scene construction, and is not a production timing claim.

Script: sweep_joint_constrained_contacts.py.
Outputs: /tmp/colibri_joint_constrained_sweeps.json and .npz.
Inputs and factor construction are documented in JOINT_CONSTRAINED_MOBILITY.md.
No GPU kernels or canonical source edits.
