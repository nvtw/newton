# PhysX authored-drive calibration — inconclusive transient fit

Local controls preserve the source hinge, force drive (100 stiffness, 100 damping,
20 degree target), masses and inertias, with gravity zero, all 35 colliders
disabled, sleep threshold zero and body damping zero. They use 30 GPU TGS position
iterations and one velocity iteration. Source assets and production code are unchanged.

Artifacts: `/tmp/colibri_physx_drive_calibration.npz` (120 Hz, 2 seconds),
`/tmp/colibri_physx_drive_calibration1200.npz` and
`/tmp/colibri_physx_drive_calibration12000.npz` (0.1 seconds each), with corresponding
JSON provenance and `.analysis.json` reports. The mass, inertia and COM getters in
the 120 Hz control are byte-identical to the matched awake two-body baseline.

The independent audit estimates axial drive torque on each body from
`axis · (dH_COM/dt - (hinge-COM) × dP/dt)`. It includes the source initial pose and
zero initial velocity before the required first SDK step. Endpoint quadrature
then fits `torque = k*(target-angle) - d*relative_angular_speed`.

| Sample rate | Samples 1:60 fitted k | Fitted d | Max torque residual | Fit condition |
| --- | ---: | ---: | ---: | ---: |
| 120 Hz | 0.06235 | 0.06496 | 0.0000702 Nm | 68.5 |
| 1200 Hz | 0.39791 | 0.48172 | 0.001934 Nm | 2.65 |
| 12000 Hz | -0.06736 | 0.62361 | 0.032487 Nm | 7.69 |

These are rejected fits, not calibrated physical coefficients. The source reset
has about 74.4 micrometers of hinge-anchor closure error. Its recovery transient
grows as the outer time step shrinks; relative angular speed after the first step
is approximately 0.024, 0.227 and 0.496 rad/s respectively. The two independently
inferred body torques disagree by up to 0.0147 Nm in the finest control. Changing
the fit window changes stiffness strongly, including negative estimates. This is
more than a late-transient collinearity problem. No monotone coefficient convergence
or source-unit mismatch has been established.

## Concrete GPU source lead

`physx/source/gpusolver/src/CUDA/jointConstraintBlockPrepTGS.cuh:85` calls
`compute1dConstraintSolverConstantsTGS` in
`physx/source/lowleveldynamics/shared/DyCpuGpu1dConstraint.h:430`.
For a force spring, its multiplier is

`x = 1 / (1 + h*(h*k+d)*unconstrained_row_response)`.

Both spring bias and damping receive this multiplier. The actual GPU solve in
`solverBlockTGS.cuh:888` adds `velMultiplier*normalVel + constant` to the stored
impulse, with no accumulated-impulse term cancelling this multiplier.
For measured inertias, the initial angular row response is 7785.9966 and nominal
SI `k=d=0.572957795`; at `h=1/3600`, `x=0.4465235`. Its effective stiffness
`x*k=0.2558391` is close to the separate static torque estimate, whose ratio to
nominal stiffness is about 0.44187. Later hard rows can change the drive velocity,
so this is a concrete finite-work coupling hypothesis, not proof of the measured
equilibrium mechanism. The refined transient controls above do not certify it.

The actual USD loader `omni/.../usdLoad/Joint.cpp:357` explicitly converts angular
stiffness/damping by 180/pi. No scale factor occurs in that function; this alone
does not certify the installed SDK's complete import path.

Composed matched PhysX gravity is 100 cm/s² with metersPerUnit=0.01, hence 1 m/s².
The Newton source builder likewise uses `(0,0,-1)`. A gravity-magnitude mismatch
does not explain the static torque discrepancy.

Next clean calibration would first remove the startup recovery transient, then
apply a measured external torque or a controlled drive-target perturbation at a
closed joint pose. It must retain an explicit diagnostic-fixture distinction and
validate momentum balance before accepting fitted coefficients.

## Exactly closed source-frame control

`prepare_physx_closed_drive.py` creates `/tmp/colibri_physx_closed_drive.usda`: both actors start at identity/origin with the original1.5worldscale. Identical source local joint frames then coincide exactly; original drive, masses/COM/inertia attributes are inherited. Contacts/gravity remain disabled. This is a deliberately changed diagnostic initializer, not the source resting-scene comparison.

Artifacts `/tmp/colibri_physx_closed_drive{120,1200,12000}.npz` and corresponding`.analysis.json` cover0.1s at each rate. SDK mass/COM/inertia differ from the rotated original only by importer floating rounding (maxmass5.96e-8kg,COM4.77nm,inertia1.14e-9kgm²).

Initial projected drive mobility from the5hard joint rows is2392.9132, effective inertia0.000417900657kgm². Nominal stiffness predicts initial instantaneous torque0.2Nm and acceleration478.58264rad/s². Measured first-interval torque increases0.017221→0.110454→0.180804Nm under timestep refinement. This is meaningful approach toward the nominal initial force, unlike the off-manifold control.

But endpoint fits are not yet a precision calibration. For samples0:12, fittedk/d are0.098688/0.099637,0.465112/0.473160,0.522658/0.672061; maximumresiduals2.01e-5,0.001109,0.021872Nm; conditionnumbers13.26,9.66,6.51. Finestfit0:60 instead gives0.485078/0.552412 with0.022046Nm residual. Residual/window sensitivity remains; do not assert an exact altereddrive coefficient.

Independent body torque disagreement is now only5.16e-8,4.35e-7,2.12e-6Nm. Totalmomentum maxima across controls are3.31e-8kgm/s and4.98e-8Nms. Nominal spring-plus-kinetic energy increases at most5.88e-7J in the finest control. These establish a much cleaner physical fixture, while finite-work and small-step floating-point effects still require separation before coefficient certification.

Exact-law correction: GPU D6 twist preparation (`constraintBlockPrePrep.cu:593`) uses `2*sin(angle_error/2)`, not the exact angle difference. Thus the nominal initial force at20degrees is0.198986Nm, versus the linear-spring approximation0.200000Nm above. This0.51% correction cannot explain the observed larger residuals. The existing regression fits intentionally remain documented as approximate, unaccepted fits.
