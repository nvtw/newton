# GPU TGS motion contract and local FP32 reference

Source: PhysX commit ed6e5ca2474c9c80ad4f4826591b88476779c6ef under
`physx/source/gpusolver/src/`. This source revision is not certified to be the
installed ovphysx 0.5.11 binary revision. No production change.

## Exact coordinate and lifecycle contract

- `CUDA/preIntegration.cuh:290`: freeze symmetric world square-root inverse
  inertia S at outer interval start. Angular solver state is momocity u;
  physical angular velocity is S u.
- `CUDA/solverMultiBlockTGS.cu:710-755`: initialize contact motion accumulators
  to zero. `propagateAverageSolverBodyVelocityTGS:516-665` advances only on
  position passes, after copy averaging. Linear delta accumulates v h;
  angular delta accumulates **u h**, not world omega h or quaternion log.
- `contactConstraintBlockPrep.cuh:1066-1084` stores angular row S(r cross t).
  Therefore its dot product with angular delta equals the world row dotted
  with S times that delta. S stays frozen during these iterations.
- `solverBlockTGS.cuh:395,409`: friction error is initialError minus target
  velocity times elapsed time, plus the fixed angular rows dotted with each
  body's angular delta and t dotted with relative linear delta. This is a
  linearized predictor, not exact finite-rotation material displacement.
- `solverMultiBlockTGS.cu:562-585`: separately accumulate normalized exponential
  quaternion increments and position delta. `integration.cuh:429-431` commits
  birth position plus delta once and normalizes delta quaternion times birth
  quaternion. Velocity iterations change velocities but do not integrate pose.

## Gravity is an explicit branch, not a redistributed outer impulse

`PxSceneDesc.h:279-294` documents default false for
`eENABLE_EXTERNAL_FORCES_EVERY_ITERATION_TGS`.

Default: `preIntegration.cuh:273-283` integrates gravity/external acceleration
with outer dt. Rigid solve/motion passes do not subtract or distribute this
velocity increment. The shared solver velocities inherit it.

Enabled: `PxgTGSCudaSolverCore.cpp:826` skips initial force application;
`1346-1369` launches `applyTGSSubstepGravity` before each position pass.
`solverMultiBlockTGS.cu:1575-1683` adds acceleration times h to all remapped
copies and the average slot. Velocity iteration loop at 1623 has no such
launch. Both branches have the same free-flight final velocity; displacement
is a*T*T versus a*h*h*N*(N+1)/2 for zero initial velocity, absent damping.
This changes temporal force application and the early contact impulse budget.

The matched awake plane1mm USD has no authored
`physxScene:enableExternalForcesEveryIteration`. Schema default is false
(`omni/schema/source/physxSchema/schema.usda:318`) and importer default false
(`usdLoad/Scene.cpp:64`, read at174). A runtime SDK flag getter was not recorded;
these are source/default facts. The separate one-setting companion explicitly
requests true; do not silently change the matched baseline contract.

## CPU validation and scope

`physx_motion_reference.py` carries the full source BSD license and uses FP32
candidate arithmetic. `test_physx_motion_reference.py` verifies stationary,
constant translation/rotation, nonidentity S coordinate conversion, common
translation, and no velocity-pass pose update. Report:
`/tmp/colibri_physx_motion_reference.json`.

Translation accumulated displacement error is 1.70e-13 m against a double
closed form; quaternion error 2.40e-8. At a world coordinate of10 m the selected
small translation is lost by 30 world-coordinate updates, while the deferred
commit moves one world-coordinate ULP. This is a numerical example, not proof
of the measured creep's cause. NumPy trigonometry is not CUDA-bitwise parity.
Sleep, locks, stabilization and full momentum/energy accounting are excluded.

## Isolated PhoenX translation control

`relative_position_control.py` wraps only process-local world methods.
Allocate birth/delta arrays in construction; reset by device kernel each
world.step; accumulate preintegration v*h and publish birth+delta after the
original pose kernel. The original torque-free implicit-midpoint rotation,
inertia update and angular-momentum transport are unchanged. The original
temporary translation is overwritten before geometry consumers. No position
or velocity threshold, double precision, damping or new force is introduced.

Current test `test_relative_position_control.py` covers CPU device kernels,
constant motion, sleeping/static exclusion, overwritten reset poses and
poisoned scratch across invocations. Actual world callback integration and
CUDA graph replay remain required before live use. It is maximal-only and
must not run with an independent position-level projection modifying its
birth/delta contract. Linear velocity is untouched; orbital angular momentum
roundoff still needs a ledger, not an exact-conservation claim.

### Completed local-control gates and negative live result

CPU and CUDA `test_relative_position_integration.py` execute the actual
canonical pose kernel through a minimal world harness: q, omega, velocity,
and world inverse inertia are byte-identical after30steps with anisotropic
inertia. Only translation changes. CUDA capture replay twice reads newly
supplied outer poses and clears poisoned displacement scratch. This verifies
the local integration mechanism, not global angular momentum conservation.

The actual corrected native world then ran600frames/10seconds, all601
geometry/joint checks passing. Actual owned callback source/hash and corrected
normal-capacity/broken-state gates passed. The native wrapper preserves the
owned import interception ordering. Same initial poses, physical properties,
canonical source hashes and staged owned modules as corrected baseline.

**Negative result:** on the common5-10second interval, horizontal drift was
13.6021micrometers (2.72042micrometers/s), versus canonical12.2296micrometers
(2.44593micrometers/s). The translation-only control is not accepted as a
creep fix. This does not test the full literal PhysX contact/motion algorithm;
it isolates accumulation placement in the existing PhoenX solver.

Artifacts `/tmp/colibri_native_relative_position600.*` and
`/tmp/colibri_relative_position_comparison.json`. GPU released after completion.
No production code changed. No timing claim or long-run stationarity claim.
