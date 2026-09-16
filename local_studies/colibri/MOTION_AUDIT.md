# Colibri driven-motion audit

`drive_motion.py` reconstructs each authored revolute joint's relative
orientation from body orientations and the two literal joint frames. It
unwraps the twist angle about the authored axis. Translation units do not
enter this measurement. This checks motion; it does not by itself establish
correct force transmission or energy balance.

## Analytical-normal slab8/envelope candidate

Input `/tmp/colibri_slab8_envelope_hash_b/states.npz`, 1200 post-step poses,
19.9833 seconds between the first and last saved pose:

- Crank: -10.33965 turns, mean -3.25101 rad/s, target -3.49066 rad/s.
- Every frame-to-frame crank speed remains negative, between -3.813 and
  -2.570 rad/s. No sampled crank stall.
- Frame-relative large gears: +5.184, -5.179, +5.173 turns.
- Head/bottom/tail cams: -1.298, -1.298, +1.293 turns.
- Left/right wing joint spans: 1.475 and 1.521 rad.

These observations show driven motion reaches the gears, cams and wings.
They are measured values, not independently established ideal gear ratios.
This run is the local candidate, not the public default.

Independent world angular-velocity projection gives mean crank speed
-3.36462 rad/s, with all samples negative (-3.761 to -2.722). This also
rejects a stalled crank, but differs from pose-derived interval averages;
endpoint velocity sampling and position corrections require separate
attribution before interpreting that difference as an error.

## Saved PhysX reference

Input `/tmp/colibri_physx_joint_trajectory.npz`, 1201 poses over 20 seconds:

- Crank: -8.88064 turns, mean -2.78994 rad/s.
- Frame-relative large gears: approximately +4.442, -4.442, +4.442 turns.
- Head/bottom/tail cams: -1.111, -1.002, +1.108 turns.
- Left/right wing joint spans: 1.475 and 1.523 rad.

The saved PhysX run also has increasing joint errors late in this interval
(see the joint-accuracy reports). Do not treat its entire trajectory as
ground truth or infer a solver bug solely from a trajectory difference.

## Reproduction and checks

Run with the existing environment:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.drive_motion /tmp/colibri_slab8_envelope_hash_b/states.npz --output /tmp/colibri_slab_motion_20s.json
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.drive_motion /tmp/colibri_physx_joint_trajectory.npz --output /tmp/colibri_physx_motion_20s.json
```

Quaternion multiplication/twist extraction was checked on a known 14-radian
rotation under a nontrivial shared parent orientation: agreement within
1e-14 radians, including unwrap across multiple revolutions.

## Public default, current analytical normals

`/tmp/colibri_public_motion_1200.json` and `.npz` record a fresh 1200-frame
run of the actual public example, with every original check passing. Peak
penetration is 0.21271 mm, anchor error 1.16513 mm, axis error 0.0079581 rad.
The history contains 1200 unique post-step poses; the final duplicate check
is deliberately excluded from motion sampling.

Crank motion is -10.76637 turns over 19.9833 seconds, mean -3.38517 rad/s.
All sampled interval speeds remain negative (-3.87687 to -2.66068 rad/s);
there is no crank stall. Left/right wing spans are 1.49395/1.54603 rad.
These numbers distinguish the public default from the local slab candidate.

Reproduce using the optional read-only history mode:

```sh
uv run --no-project /home/twidmer/Documents/git/newton/.venv/bin/python -m local_studies.colibri.check_public_analytic_gradient --frames 1200 --save-history --output /tmp/colibri_public_motion_1200.json
```

This correctness run overlapped other correctness work and is not FPS evidence.

The final public q/qd arrays match the prior no-history1200-frame analytical
normal run bit for bit (`/tmp/colibri_phoenx_analytic_gradient_public_1200.npz`).
History timestamps are unique and the last history pose exactly equals the
final saved pose. This checks the observed final trajectory; it does not
replace a full per-frame hash repeat.

## Current public-constructor slab candidate

The52.86FPS candidate uses canonical parallel preparation and differs from
the older slab prototype above. Its saved1200-frame history
`/tmp/colibri_public_slab8_hash_a/states.npz` gives10.34403 crank turns,
mean3.25238rad/s, with all interval speeds negative (2.32874 to3.76898rad/s
magnitudes). No sampled stall. Full per-joint report:
`/tmp/colibri_public_slab8_motion_20s.json`.
