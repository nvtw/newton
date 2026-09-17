# Colibri readiness validation

The earlier validated configuration was ready for interactive visual inspection.
The subsequent requested scene changes have the qualification below.
The latest 120 Hz gear-contact investigation is recorded in
[PENETRATION_VALIDATION.md](PENETRATION_VALIDATION.md); penetration improved,
but the strict support-stationarity check still fails.
It uses the standard Newton renderer:

```bash
uv run -m newton.examples phoenx_colibri
```

## Subsequent counterweight and flower changes

The default is now restored to 1.0, and the base/frame axle's 20-degree position
spring and damper are disabled. The simulation results in this section refer
to the earlier 0.9 setting with the drive enabled, not the new configuration.

This revision used `--counterweight-density-scale 0.9`, applied to
`Frame/Cylinder` before mass-property assembly. Flower and slider helper shapes
are merged into the dynamic `FrameGround` body. The source USD has a separate
kinematic flower and a disabled connecting joint; the revised attachment is
physical and carries the shapes' mass/inertia on the base.

All eight scene tests, including mass/COM/inertia and body-ownership regressions,
and a 600-frame headless OpenGL run pass. However, the
revised full-minute run fails the unchanged 1 mm fresh-contact limit at
29.3167 seconds, with 1.135 mm penetration between the second hypocycloid gear
and `Frame/Cylinders/Cylinder_11`. Base-support and drive checks passed up to
that point. Artifact: `/tmp/colibri_density_flower3600.{json,npz,log}`.
Thirty substeps and eight-color grouping also failed contact checks, so neither
experiment was adopted. The default solver remains at 24 substeps/four colors.
Do not apply the earlier minute-long readiness claim below to these revised
mass/attachment defaults.

## Final measured behavior

The final contact-before-joint temporal configuration passed all 3,601 checks
in a 60-second run. Between seconds 10 and 60, net horizontal base displacement
was 1.04 µm and maximum excursion was 4.91 µm. The crank completed 33.25 turns
without stopping; mean speed was 3.4827 rad/s against a 3.4907 rad/s target
(0.23% error). Authored gains and scene masses remain unchanged.

Physics averaged 21.72 ms/frame (46 FPS) on an RTX PRO 6000 Blackwell, versus
21.13 ms/frame for the prior soft configuration. Timing excludes rendering
and diagnostic checks. Artifact: `/tmp/colibri_tracking_final3600.{json,npz,log}`.

## Remaining gaps addressed

- Contact output now includes normal and persistent-anchor friction impulses,
  plus moments accumulated at their actual application points over the whole
  collision step. Output is the average world-frame wrench on body0 about its
  current COM. Patch friction is attributed to its first contact point; body
  pair sums preserve the physical wrench. Disabled reporting is compiled out.
- Contact rows now precede joint rows within each color group. This preserves
  the freshly solved drive velocity before integration, reducing mean crank
  error from about 13% to 0.23%. Each constraint still runs exactly once;
  paired impulses, copy ownership, and authored drive gains are unchanged.
- Test mode now checks sustained crank tracking as well as settled support
  stationarity, joint attachment and independently generated contact depths.
- OpenGL headless rendering passed through the normal example framework.

## Regression evidence

The final focused suite passed all 60 CPU/CUDA tests in 85.33 seconds, including
linear/angular momentum, static/dynamic contact wrenches, graph replay, legacy
contact sensors, temporal springs and Colibri checks
(`/tmp/colibri_commit_tests.log`). A final eight-test policy/wrench rerun also
passes after adding the zero-shape contact-buffer regression, bringing unique
focused coverage to 61 tests (`/tmp/colibri_final_force_edge.log`).

Kapla passed 600 frames at 46.3 FPS (`/tmp/colibri_commit_kapla.log`). G1 passed
500 policy steps / 10 seconds plus 100 timed steps, at 36.30 ms/policy step
including inference (`/tmp/colibri_commit_g1.{json,npz,log}`). Its final timed
base height was 0.772 m and its state remained finite. Both examples retain
their original soft-solver configuration.

The force-output regression fails before the fix with `NotImplementedError`
(`/tmp/colibri_force_before.log`). It verifies friction, moment, outer-step
scaling, graph replay and contact removal for static and dynamic endpoints.

The live crank-tracking regression fails with the previous contact ordering:
mean -3.2208 rad/s versus target -3.4907 rad/s after settling
(`/tmp/colibri_tracking_without_fix.log`). The final minute run passes.
The earlier spring-relaxation regression also failed before its fix in both
scalar and cooperative solves (`/tmp/colibri_spring_relax_without_fix.log`).

Source-style patch correlation, normal-load friction budgets and temporal
spring coefficients follow the local PhysX implementation. Common-point
contact impulses and paired joint wrenches preserve linear and angular
momentum. No added damping, base pinning, contact deletion, gain retuning or
extra support sweeps are used.

## Scope

This remains an experimental single-world rigid CUDA policy with the limits
in `newton/_src/solvers/phoenx/docs/CONTACT_TGS.md`. These measured checks do not
certify every gear tooth's engagement or arbitrary scene scales. The ordinary
soft solver remains the default for other examples.

Full-repository `uvx pre-commit run -a` was executed. It reports existing lint,
format, spelling and Warp-annotation issues outside this change. Its unrelated
automatic edits were restored. All hooks pass on the intended commit files.
