# Tangential recovery versus final velocity correction

## Executed native order

In `solver_phoenx.py`, `PhoenXWorld.step` runs each temporal substep as:

1. Joint/articulation begin-substep preparation.
2. External force and gravity integration.
3. Dispatcher biased solve, including contact positional recovery.
4. Position integration using the resulting real velocities.
5. Refresh owned relaxation geometry.
6. Dispatcher unbiased velocity relaxation.
7. Inertia refresh and the remaining state bookkeeping.

The `_active_velocity_iterations` property returns zero before the last
substep when `velocity_relaxation="final_substep"`. The matched two-body
control therefore integrates thirty biased microsteps per 120 Hz collision
interval and performs its final velocity correction after their positions
have already advanced. There are two such intervals per displayed 60 Hz frame.
The direct contact path avoids the separate maximal tree pose projection.

The owned maximal contact kernel adds the two stored tangential recovery
targets to its tangential RHS only when `use_bias` is true. Normal recovery,
normal/tangent coupling and joint-bias strip/restore are separate operations.
The corrected local callback changes the normal-load and broken-history
semantics; it does not remove these tangential recovery targets.

## Consequence

An exact final zero velocity does not certify zero integrated displacement:
the interval displacement already contains the sum of h times each biased
velocity. A simple constant compatible target b gives biased velocity -b;
a final correction setting velocity to zero still leaves displacement
-interval_dt*b. This is a kinematic counterexample, not evidence that the
actual scene has constant bias or that all restorative friction is wrong.

A final correction can still improve the next interval through velocities,
contact impulses and warm history. Its effectiveness must be measured live.
Incompatible or erroneous biased targets are a distinct potential creep
channel which final-only residual checks cannot exclude. Consistent history
recovery normally restores displacement; suppressing it may worsen other
contact problems. Do not infer a general zero-recovery policy.

## Existing controls and their limited scope

- `check_high_mass_tangent_recovery.py` replaces the preparation factor
  0.08 with zero, retaining physical friction. The 400:1 8x8 benchmark worsened
  to 0.2871 m/s top RMS and about 11.9% support-load error.
  Evidence: `/tmp/high_mass_no_tangent_recovery.json` and
  `HIGH_MASS_QUALITY_AUDIT.md`. This is not the corrected Colibri owned path.
- `velocity_rigid_friction.py` passes zero tangent biases through the
  contact-cloth helper. It does not patch the current maximal owned kernel.
  The saved `/tmp/colibri_reduced_forest15_velocity_friction_20260915.json`
  uses 15 bodies, reduced joints, 10 substeps and 8 iterations; it failed a
  gear penetration check after three frames. It is not a matched two-body
  corrected-callback result.
- `tangent_recovery_trial.py` adds the stripped joint-recovery velocity to
  tangential RHS; it does not remove friction recovery.
- `shared_recovery_live.py` projects recovery targets to a compatible
  planar field; it also does not remove recovery.

No matched zero-tangential-recovery result with the corrected owned callbacks
was found in the inspected scripts, reports and filenames.

## Possible bounded diagnostic, not executed

If the final-only coupled control fails, remove only the two tangential
recovery additions from the actual corrected owned kernel for a matched
two-body control. Preserve friction coefficients, normal recovery/speculation,
joint bias, physical mass and geometry, iteration budgets and history updates.
Require executed kernel/helper identity and source fingerprints, plus the
same trajectory, contact and physical bounds. Record biased pre-integration
velocity separately from final velocity. This would isolate target influence;
it would not establish that eliminating strong-friction recovery is a valid
general solver change.

## Prepared control (CPU checked; no live run yet)

`zero_tangent_recovery_live.py` preallocates buffers during construction and
wraps only biased maximal contact calls. It saves/zeros the two tangent targets
for uniquely owned scheduled rows, executes the unchanged corrected callback,
then restores their exact prior bits. Unbiased calls bypass the wrapper.
The actual-container CPU check verifies all other rows and arrays, inactive
schedule entries, repeated updates and signed-zero restoration.

```bash
CUDA_VISIBLE_DEVICES='' uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.zero_tangent_recovery_live --cpu-test
COLIBRI_DIFFERENTIAL_REFERENCE=1 COLIBRI_DIFFERENTIAL_NORMALIZED=1 COLIBRI_STAGE_RUNNER=local_studies.colibri.zero_tangent_recovery_live uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.native_conditioned_two_body --native-path direct --body-count 2 --frames 600 --substeps 30 --save-history --output /tmp/colibri_native_zero_tangent_recovery600.json
```

The `.zero_tangent_recovery.json` companion records actual owned binding,
overlay/kernel hashes and scope. This is a standalone matched native control;
it does not silently compose with the separate final full-wrench candidate.

## Matched zero-recovery outcome

Root executed the prepared600 control: geometry PASS, 5–10 s translation
3.96653 micrometers/s versus corrected baseline2.44593;20.0807 FPS.
Artifacts: /tmp/colibri_native_zero_tangent_recovery600.*. No60-second
extension. Removing tangent recovery alone worsens this measured control;
do not promote the diagnostic or blame recovery solely from final velocity.
