# Friction correction control

The successful two-anchor hybrid changed several things simultaneously: contact
ordering, friction history, distribution of the normal-load budget, and tangent
correction scaling. Its low creep does not identify which change caused the
improvement. This control retains the native normal updates, every original
point's friction capacity, and the native impulse application. Only the tangent
velocity residual passed to the metric projection is scaled.

The residual includes recovery during the biased pass. Scaling the whole
residual retains its zero; scaling only its velocity term would change the
recovery target. Sliding/break decisions and finite-iteration trajectories can
still differ, so this is an experiment rather than an accepted solver change.
No friction anchors are pooled and global SOR remains 1.

`friction_relaxation_live.py` saves runtime source provenance. At factor 1 the
extracted kernel source is unchanged. Its two-frame live control reproduced
all pose, velocity, and timestamp bytes of the saved native trajectory:
`/tmp/colibri_friction_relaxation_unit2.*`.

The factor 0.3651483716701107 control uses the source hybrid coefficient,
30 substeps per 120 Hz collision update, and the original implicit drive law.
It must be judged against creep, analytical equilibrium, and time to accuracy;
visual stability alone is insufficient.

## Separate refresh regression finding

The existing multiworld determinism test fails with both the new parallel
refresh and a source-generated restoration of the old serial refresh. The
reported mismatch values are identical. This isolates that failure from the
refresh parallelization; it does not establish that determinism is fixed.
Control: `/tmp/check_colibri_refresh_determinism.py --serial`, output
`/tmp/colibri_serial_refresh_determinism.log`.

## Result: correction scaling alone is rejected

`/tmp/colibri_friction_relaxation0365_600.*` completed 600 frames with source
and geometry checks passing. Over seconds 5–10, horizontal net creep was
15.3515 micrometers/second, versus 2.25892 for the native unit-factor control.
The final hinge remained accurate at 18.207466 degrees (analytical equilibrium
18.207319 degrees), with final hard residual 8.53e-12 and drive residual
2.92e-9. Timing was 20.60 FPS; this experiment does not include the parallel
refresh speedup. The unchanged paired-joint impulse audit reported zero linear
defect and angular defect below 2.4e-14 Nms in the saved final state.

The successful hybrid's low creep cannot be recovered by copying its correction
factor alone. This result does not rule out every relaxation strategy, but does
reject this particular change at the tested iteration budget. Do not promote it.

## Determinism fixture correction

The failing fixture shared one stateful collision matcher between two otherwise
independent simulations. Giving each simulation its own sticky CollisionPipeline
makes the bitwise determinism test pass (0.558 seconds). The checked-in test now
uses separate pipelines; the old serial and parallel refresh both failed with
the shared history. Evidence: `/tmp/colibri_independent_pipeline_control.log`.
This fixes test isolation rather than changing simulation arithmetic.
