# Independent drive reference

PhysX trajectories are comparison data, not analytical ground truth. In
particular, a friction-supported mechanism need not have a unique equilibrium
angle. Both implementations must satisfy the authored constitutive law and
admissible support reactions.

## Free two-body hinge

`check_analytical_free_hinge.py` removes gravity and contacts and places a
revolute joint through both bodies' centers of mass. Their axial inertias are
0.02 and 0.01 kg m²; the relative coordinate has reduced inertia
`I = 1 / (1/0.02 + 1/0.01)`. The stiffness and damping are each
0.5729577951308232 in SI units, and the target is 20 degrees. Initial angle,
angular speed, and total momentum are zero.

The independent continuous reference is the exact solution of

`I theta_ddot + kd theta_dot + ke (theta - target) = 0`.

The separate discrete reference solves the backward-Euler recurrence directly,
without calling any Newton constraint helper:

```
omega_next = (omega + h*ke/I*(target-theta)) / (1+h*kd/I+h*h*ke/I)
theta_next = theta + h*omega_next
```

GPU single-world maximal block PGS, SOR 1, no mass splitting, 120 Hz outer
steps, one biased iteration, one simulated second:

| Substeps | Final velocity passes | Max continuous angle error | Max discrete angle error |
| --- | --- | --- | --- |
| 1 | 0 | 5.46215e-4 rad | 1.73819e-7 rad |
| 30 | 0 | 1.70902e-5 rad | 1.77266e-6 rad |
| 30 | 1 | 1.70902e-5 rad | 1.77266e-6 rad |

Maximum axial angular momentum is 1.74344e-8 N m s. The two 30-substep
trajectories are identical. These results support the isolated drive law;
they do not establish convergence when contacts subsequently change the joint
velocity.

Artifacts: `/tmp/colibri_analytical_free_hinge_gpu.json`,
`/tmp/colibri_analytical_free_hinge_discrete.json`, and the corresponding GPU log.

The initial CPU attempt has irregular zero-drive intervals and does not match
the reference. It is retained separately in
`/tmp/colibri_analytical_free_hinge.json`; it is not used as drive-law evidence.
The CPU dispatch discrepancy remains to be diagnosed.

## Colibri comparison caveat

The authored USD drive is a force drive with stiffness and damping 100,
target 20 degrees, and unlimited effort. The conversion to
0.5729577951308232 N m/rad assumes centimeter and degree units. The exact
effective coefficient imported by the PhysX SDK has not yet been measured.
Its current resting pose does not balance the assumed simple spring law;
source scaling/import and actual support reactions are under investigation.
Do not use the PhysX angle as an acceptance target until that discrepancy is
resolved.

An independent scalar spring/gravity balance gives a local free-Frame root near 18.20735 degrees at the saved base orientation. Frame clears the floor by 8.187 mm. An inscribed Coulomb polygon supports the resulting assembly wrench. The current force witness projects the saved base slightly penetrating footprint (up to 4.719 micrometers) onto the plane. It is an idealized rigid support reference, not a solution of the solver penetration-dependent normal regularization. Exact-plane support is being checked separately. Reference artifacts: /tmp/colibri_two_pose_static_torque.json and /tmp/colibri_static_certificate_analytical.npz.

### Exact-plane reference completed

The exact flat-base construction has a certified equilibrium at
18.207319256 degrees, with Frame clearing the plane by 8.188824 mm. Aligning
the authored base bottom to the plane requires a 0.000189276 rad rotation.
The base stays free: this is an independent initial configuration, not a
constraint or repeated pose correction. Its inscribed Coulomb force witness
has full assembly balance error 1.39e-17 N, zero cone violation, coincident
hinge anchors, and spring/gravity torque residual 4.55e-16 N m.

Certificate:
`/tmp/colibri_static_certificate_exact_flat-base_analytical_equilibrium.npz`.
`start_analytical_equilibrium.py` validates this witness and sets only the
initial body poses for a live diagnostic. It does not seed impulses or alter
the authored drive, masses, gravity, materials or dynamic base. Small settling
under numerical normal compliance remains possible; the diagnostic does
not assume the rigid equilibrium is an exact discrete fixed point.
