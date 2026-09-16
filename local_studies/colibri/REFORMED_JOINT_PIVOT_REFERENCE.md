# Geometry-consistent FP64 joint diagnostic

## Important correction to earlier reference

The original saved global probe's rhs is the POSITIVE constitutive residual. A correction must solve A delta = -rhs. The first unsquared reference accidentally solved the opposite sign. Its equation-residual magnitudes described that reversed equation; its physical energy/contact results are not valid correction evidence.

Wrong-sign artifacts are preserved as /tmp/colibri_joint_unsquared_mp_wrong_sign_control.{npz,json} and /tmp/colibri_joint_unsquared_mp60_gauge20_wrong_sign_control.{npz,json}. The independent physical audit now requires rhs + J dv + R delta before reporting physical acceptance, and rejects the old control with residual2.76332.

Correct-sign original reference: /tmp/colibri_joint_original_corrected_response.{npz,json}.
Reformed reference: /tmp/colibri_joint_reformed_response.{npz,json}, using /tmp/colibri_joint_reformed_input.npz.

## Explicit operator change

For each joint's three Cartesian point rows, reconstruct each endpoint's effective lever by sum(f cross torque)/2. Their linear axes are exactly signed coordinate axes, so no rank threshold or noisy least-squares fit is necessary. Average the two inferred world application points, then recompute each lever and cross product in FP64 at that shared pivot. Positions, linear axes, angular-only rows, W, drive compliance and targets are unchanged.

This is an explicitly changed J, a geometry-consistent rounding control. It does not recover original authored anchor data; the pivot is inferred from saved rounded wrenches. It is not an arbitrary torque projection and is not yet a production implementation.

Maximum coefficient change:5.70435077e-9.
Maximum consistent residual update:1.22584529e-8.
All linear force and total world torque coefficients now sum EXACTLY to zero.

## Rank and response

Exact rational nullspace certification of [J^T; drive selector] gives:

| Quantity | Original saved J | Reformed shared-pivot J |
|---|---:|---:|
| Exact pure-dual nullity | 1 | 4 |
| Smallest retained physical factor mode | about2.73e-14 | 8.92836e-9 |
| Next small physical mode | 2.53215e-5 | 2.53215e-5 |
| Maximum impulse, chosen exact gauge | 8.92280e6 | 559.660 |
| Maximum linear velocity correction | .0684382m/s | .0707709m/s |
| Maximum angular velocity correction | 3.06836rad/s | 3.06749rad/s |
| Corrected constitutive residual | 8.47e-18 | 4.26e-16 |
| Physical response residual | 6.38e-32 | 1.50e-37 |

The three extra null directions are certified EXACT zero-response dependencies, not discarded small singular values. Thus several enormous reaction modes arose from independently rounded lever geometry. Two small physical modes remain and are solved without truncation.

Both solves use45-digit unsquared QR, original physical W and existing drive compliance. Exact independent columns select only a dual gauge for the reformed matrix. Gauge-dependent peak impulses are diagnostic; they are not unique reaction magnitudes.

## Acceptance boundary

This is a conditioning and physical-invariance diagnostic. The motion corrections are still sizable. The parent's independent correct-sign momentum/work/contact audit is required; no live solver correction or production change is accepted here.

## Reproduction

Run reform_joint_pivots.py, then:
python -m local_studies.colibri.probe_joint_unsquared_mp --input /tmp/colibri_joint_reformed_input.npz --output /tmp/colibri_joint_reformed_response

Use the repository's uv/interpreter invocation. No GPU kernels are involved.

## Independent correct-sign physical gate

The reformed response conserves all linear/angular momenta to1.63e-19 N s and1.36e-20 N m s and decreases kinetic energy by1.1920923e-4 J. Nevertheless, held-contact normal residual worsens from.0375384 to.0518981 m/s and tangent residual from.0154903 to.0258386 m/s. The joint-only correction is rejected; the geometry-consistent operator is useful only as a basis for a coupled reference.

The correct-sign original operator loses angular momentum3.6407e-4 N m s, decreases energy1.243806e-4 J, and also worsens contacts. These replace all prior wrong-sign energy/contact claims.

Evidence: /tmp/colibri_global_joint_reformed_physics.json and /tmp/colibri_global_joint_corrected_physics.json.
