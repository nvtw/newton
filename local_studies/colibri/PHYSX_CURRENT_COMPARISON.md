# Current Colibri speed comparison

## Acceptance correction: neither timing is a stationary-base result

The PhoenX trajectory below drifts about 437 mm over 300 seconds. The matched
PhysX reference also moves: approximately 13.77 mm horizontally over its saved
5.5-second interval. Thus these timings do not establish a drift-free winner.
See BASE_STATIONARITY.md and /tmp/colibri_physx_matched_base_motion.json.
A new matched two-body PhysX control is required before attributing the
difference to a particular joint or friction implementation.

## Historical timing protocol

Both runs use 120 Hz collision updates and the matched temporal budget:
PhoenX 30 temporal substeps with one biased iteration and one final velocity
sweep; PhysX TGS 30 position iterations plus one velocity iteration. Source
geometry, gravity, drives and physical offsets are retained, with zero global
body damping. PhoenX uses SOR 1. Effective PhysX iteration selection is supported
by the authored scene minimum, pinned SDK defaults and the earlier runtime
trace; the timing harness does not query effective iterations at runtime.

| Run | Mean ms per 60 Hz simulation frame | FPS | Duration |
| --- | ---: | ---: | --- |
| PhoenX integrated default | 10.137728 | 98.6414 | 300 simulated seconds |
| PhysX fresh repeat | 11.404794 | 87.6824 | 300 measured frames after 30 warmup |

Rendering and validation reads are outside the timed steps. GPU jobs were
serialized and the compute-process list was empty before the PhysX repeat.
These are separate measurements, not an interleaved cross-solver comparison.
The earlier identical-state PhysX reference measured 91.5167 FPS, so timing
variation matters. The independent paired PhoenX scalar/cooperative comparison
measured an 8.83% frame-time reduction with identical 330-frame histories.

## Physical and reproducibility scope

PhoenX passed its full 300-second bounds and all ten arrays match the saved
pre-optimization trajectory byte for byte. Peak penetration is 0.762952 mm,
anchor error 0.794343 mm and axis error 0.0289467 rad. The 0.03 rad axis bound
has little remaining margin. Sixty-six contact/conservation/kinematic tests
passed in sixteen isolated modules without skips. These checks do not close
the separate difficult-mass-ratio convergence and long-term drift questions.

The fresh PhysX repeat reproduces its historical initial/final states and six
saved physical-property arrays byte for byte, with present source hashes
unchanged. Its test proves final finiteness; it does not test continuous
stability or conservation. Historical source hashes were not recorded, so
present provenance cannot retroactively prove historical source identity.
The pinned runtime is ovphysx 0.5.11; the full resolved versions and offline
command are in the provenance artifact.

Both inputs have the same authored joint misalignment: maximum anchor error
348.918 micrometers and axis error 4.437 milliradians. PhysX advances one
1/120-second initialization step before warmup and saves its initial state
after that step. The observed movement includes drive/gravity motion and
constraint correction; it cannot all be attributed to its joint-snap warning.

Six joints author PhysX articulation-friction coefficients of 1.0. PhysX
ignores these outside articulations, and the Newton literal builder also uses
zero hinge friction. This warning therefore creates no effective hinge-friction
mismatch; the coefficients must not be interpreted as 1 N m torques.

## Evidence

- `/tmp/colibri_cooperative_prepare_velocity1_long18000.{json,npz,source.json,log}`
- `/tmp/colibri_physx_cooperative_matched_repeat.{json,npz,log,provenance.json}`
- `/tmp/colibri_physx_cooperative_matched_repeat.prepared.json`
- `/tmp/colibri_physx_joint_warning_audit.json`
- `/tmp/colibri_integrated_rigid_regressions.{json,summary.json}`
- `COOPERATIVE_BILATERAL_PREPARE.md` and `INTEGRATION_REGRESSION_PLAN.md`
