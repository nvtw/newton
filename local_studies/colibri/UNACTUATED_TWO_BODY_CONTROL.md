# Two-body control without the joint drive

This user-requested diagnostic retains FrameGround, Frame, and the authored
hinge. Only its drive dictionary is removed in process memory. The runner
asserts that every spring and damping gain is zero. Geometry, masses, friction,
gaps, 30 substeps per 120 Hz collision update, group size 4, batch size 2,
one biased sweep, and SOR 1 are unchanged. Flower geometry is disabled as in
the paired staged controls. Production files are unchanged.

Both 60-second runs pass geometric checks, but neither is fully stationary.

| Configuration | Translation, 10–60 s | Rotation, 10–60 s | Translation, 30–60 s |
| --- | ---: | ---: | ---: |
| Unactuated, production friction | 0.285295 mm | 0.172922 degrees | 0.125135 mm |
| Unactuated, total-normal candidate | 15.759355 mm | 3.931511 degrees | 0.579108 mm |

The authored actuated production case drifts 41.542 mm over 10–60 seconds.
Removing its drive substantially reduces motion, but does not eliminate it.
It also changes settling: the unactuated Frame falls to another pose. With the
total-normal candidate, the base ends 18.277 mm above its initial height and
about 14.22 degrees from its initial orientation. That run has a longer settling
transient, so its 10–60 second displacement is not purely steady creep.
Removing the drive is a diagnostic, not a proposed scene fix.

## Reproduction

Use the standard uv/interpreter invocation with:

    -m local_studies.colibri.unactuated_base_frame --body-count 2 --frames 3600 --substeps 30 --save-history --output PATH

The candidate wraps this runner with total_normal_friction. Artifacts:

- /tmp/colibri_base_frame_unactuated_original3600.json and associated files
- /tmp/colibri_base_frame_unactuated_totalnormal3600.json and associated files
- /tmp/colibri_unactuated_paired_late_motion.json

Independent native joint replay at the driven pose reproduces impulses within
4.5e-12 and gives zero immediate drive residual. Later contact rows reopen the
residual; copy averaging magnifies it. This implicates coupled finite-sweep
convergence, rather than an identified incorrect joint-block formula.
Evidence: /tmp/colibri_two_body_joint_replay_cpu.json.
Full contact, friction, and stationarity acceptance remains necessary.
