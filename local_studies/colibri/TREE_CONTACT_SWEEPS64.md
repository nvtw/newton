# 64-sweep unbiased tree-conditioned contact reference

The native biased path remains untouched. This CPU-only frozen diagnostic extends the same70 eligible contacts, source friction, physical joint tree mobility and actual drive compliance to16/32/64 sweeps. Speculative impulses remain fixed.

| Sweeps | Normal KKT residual m/s | Tangent KKT residual m/s | Largest impulse component N s |
|---|---:|---:|---:|
| Native saved | .0375384 | .0154903 | .0001201 |
| 16 | .0163463 | .00911243 | .0576885 |
| 32 | .0144619 | .00806407 | .104600 |
| 64 | .0116123 | .00641877 | .183980 |

Convergence remains inadequate. These results do not support GPU integration of this pointwise sweep as an accepted replacement.

## Physical checks

Joint/drive equations remain satisfied to1.53e-15. Including static-ground reactions, linear-momentum error is below1.88e-15 N s and angular error below1.28e-10 N m s. Work balance, including actual compliant drive impulse changes, stays within1.09e-16 J.

At64 sweeps, contact work=-.00558797 J and drive work=+.00552681 J, yielding deltaKE=-6.11590e-5 J. The crank speed is-2.28665 rad/s versus reference-3.49066 rad/s; its constitutive equation is solved, so this lag is now explained by reaction load rather than a residual in the drive solve.

The largest unconverged normal is saved point998, body pair[3,8], mu=.25: normal velocity-.0116123 m/s with impulse .18398 N s. Points763/773 on pair[17,2], mu=.25 remain around-.00459 m/s with impulses .078/.073 N s. Growing contact loads and persistent closing speeds warrant a stronger coupled-contact reference, not more optimistic convergence claims.

## Cost and reproduction

4480 point updates,81984 disk bisections, about.133s CPU sweep time excluding factor setup and exact-rank checks. No production timing claim.

sweep_joint_constrained_contacts.py --mobility /tmp/colibri_joint_tree_mobility.npz --baseline /tmp/colibri_joint_tree_response.npz --sweeps 64 --checkpoints 16,32,64 --output /tmp/colibri_joint_tree_sweeps64

Artifacts: /tmp/colibri_joint_tree_sweeps64.{json,npz}.
Biased-recovery incompatibility and safe factor reuse are documented in REDUCED_TREE_REUSE_DESIGN.md.
