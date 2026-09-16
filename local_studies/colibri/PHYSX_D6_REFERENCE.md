# Bounded FP32 GPU-source D6 row reference

physx_d6_reference.py translates the two-dynamic-body, force-spring and bilateral
subset of GPU TGS preparation/ordered solve at PhysX commit
ed6e5ca2474c9c80ad4f4826591b88476779c6ef. Full source BSD notices retained.
This CPU reference is distinct from the older D6-style experiment, which used
PhoenX equations. It does not claim GPU bitidentity or reproduce the full SDK,
importer, general joint frame construction, collision, scheduling or integration.

Source mapping:
- constraintBlockPrePrep.cu593: twist quaternion error is -2*delta.x.
- PxgConstraintHelper.h43/162: aligned hard angular Jacobians are half axes.
- jointConstraintBlockPrep.cuh171: stable solve-hint sort and group orthogonalization.
- jointConstraintBlockPrepTGS.cuh: force-spring constants and rotational basis.
- solverBlockTGS.cuh744: ordered scalar impulses and immediate body updates.
- solverBlockTGS.cuh1091: final velocity conclusion disables spring updates.

The aligned hinge emitter uses source order and geometric error. Prepared rows
support accumulated linear/angular error inputs, but their caller must supply
geometry and square-root-inertia coordinates consistently. Unsupported:
kinematics, force bounds, limits, restitution, acceleration springs, mass scaling,
slab splitting and unresolved response directions. No artificial compliance,
mode truncation or production FP64 arithmetic was introduced.

physx_d6_reference_test.py passes an independent implicit spring-step formula:
lambda=-3.32729069e-5Ns versus analytic-3.32729049e-5, constitutive residual
1.49e-12. Six-row off-center unequal-body physical impulse ledger:
P error2.35e-10Ns, L error2.92e-11Nms, midpoint-work identity6.40e-13J.
Reverse order changes velocity0.00107, confirming finite-order semantics matter.
Spring conclusion leaves velocities byte-identical. Offline physical ledgers
use higher precision only as independent reference checks.

This is a tested algorithm component, not a live creep solution.
Artifact /tmp/physx_d6_reference_test.json.
