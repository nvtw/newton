# FP64 unsquared joint mobility feasibility

Ordinary FP64 QR is sufficiently accurate for the tested geometry-consistent frozen operator. It was inadequate for the original independently rounded point levers; the explicit geometry reform is essential.

Use the same184 exact-rational independent rows as the45-digit reference, retaining the8.92836e-9 physical mode. A full Householder QR of the scaled rectangular [J L,sqrt(R)] transpose yields G and the drive slack complement. No singular-value threshold, rank floor or added compliance is used.

## Comparison on all70 eligible contacts

- Contact mobility relative Frobenius difference:8.46e-11.
- Maximum contact mobility coefficient difference:1.70e-7.
- Maximum body response difference per unit contact impulse:4.10e-6.
- Maximum drive response difference per unit contact impulse:4.28e-11.
- Maximum joint equation residual per unit contact impulse:9.24e-11.
- Joint-only baseline constitutive residual:2.40e-10.
- Baseline velocity correction differs from45-digit reference by4.12e-9.
- Virtual-work identity relative error:2.51e-15.

Unit impulses produce large responses. Their maximum numerical P/L defects are3.27e-8 N s and2.79e-9 N m s; these are reported rather than hidden by relative scaling.

## Actual-scale sweep replay

Repeating the same1/4/8 contact sweeps with FP64 baseline and mobility gives:
- Final normal residual .0181076932226 versus MP .0181076932230.
- Final tangent residual .0102199088313 versus MP .0102199088308.
- Joint/drive residual at most2.40e-10.
- Momentum plus static reactions: P error at most1.59e-11 N s, L at most1.95e-11 N m s.
- Contact plus actual-drive work balance error at most2.45e-13 J.

These meet the reference's numerical gates at the tested impulse scale. Contact convergence is still incomplete; this does not establish a live global solver or production performance.

The symbolic gauge identification is a diagnostic certificate, not proposed per-step runtime work. A production method still needs a principled geometry/rank strategy and performance validation. Multiprecision appears unnecessary in the numerical factorization once that valid geometry and independent row set are supplied.

## Reproduction

probe_fp64_joint_mobility.py writes /tmp/colibri_joint_fp64_mobility.{json,npz} and /tmp/colibri_joint_fp64_response.npz.

sweep_joint_constrained_contacts.py --mobility /tmp/colibri_joint_fp64_mobility.npz --baseline /tmp/colibri_joint_fp64_response.npz --output /tmp/colibri_joint_fp64_sweeps

All CPU-only, no canonical edits.

The parent subsequently recaptured current source and found all31 prior support snapshot arrays byte-identical. This removes the earlier source-epoch uncertainty for the saved joint operator and torque findings.
