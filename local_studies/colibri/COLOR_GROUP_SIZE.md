# Color-group size at the restored 30-substep setting

Conditions: 30 temporal steps per120Hz collision update, one biased iteration,
SOR1, no ether damping, exact source offsets, standard candidate admission,
canonical fixed-bound FP64 joint preparation. Only group size changes.

| Group size | Validation | Peak depth | Peak anchor |
|---:|---|---:|---:|
| 4 | 3600 frames /60 s PASS | 0.11076 mm | 0.52284 mm |
| 8 | Existing60 s PASS | 0.21826 mm | 0.6933 mm |
| 16 | 330-frame screen PASS | 0.09159 mm | 0.63853 mm |

An isolated paired330-frame comparison (30warmup,300measured) measured
65.4747FPS forsize4 and52.2254FPS forsize8. GPUcompute list was empty
before launch and otheragents held GPUwork. The size4 fullminute run is
separate stability evidence; its own overlapping timing is not accepted.

Artifacts:
- `/tmp/colibri_canonical_width4_30x1_isolated330.json`
- `/tmp/colibri_canonical_width8_30x1_isolated330.json`
- `/tmp/colibri_canonical_width4_30x1_3600.json`

The public example now uses group size four after the motion/repeat checks below.
Group4 has37partitions in a representative final graph. Current dispatch
launches32blocks, so someblocks process a second independentgroup serially.
A separate grid64 experiment is prepared to test that launch overhead without
changing partition membership or constraint order; no speed claim yet.

## Larger launch grid

An isolated paired 330-frame run measured 72.2468 FPS with 64 blocks versus
67.6841 FPS with 32 blocks (13.8415 versus 14.7745 ms per frame). Both passed
all physical checks. All seven saved arrays are byte-identical, including
contact headers whose sentinel NaNs require a byte comparison.

Artifacts:

- `/tmp/colibri_grid64_width4_30x1_330.json`
- `/tmp/colibri_grid32_width4_paired330.json`

This is a launch scheduling change at the restored 30-substep setting.
It does not alter constraint order, copy ownership, or arithmetic within a group.
The 64-block production integration is complete. The full 20-second repeat
below verifies its trajectory against the previous 32-block launch.

## Sustained motion

The public example with only group size overridden to four passed 1200
frames (20 simulated seconds). The crank completed 10.4239 turns, averaging
-3.27750 rad/s against the source target -3.49066 rad/s, with no stalled
samples. This verifies continued drive motion rather than a static lock.
Peak anchor error was 0.52285 mm, penetration 0.10453 mm, axis error
0.0123067 rad. Full pose history is saved for an independent repeat.
Artifact: `/tmp/colibri_width4_motion_repeat1.json` and `.npz`.

The integrated 64-block repeat passed the same 1200 frames. Final pose,
velocity, every saved pose in the 1200-frame history, timestamps, and labels
are byte-identical to the independent 32-block run. This supports repeatability
for this scene and duration, not a universal determinism proof.
Artifact: `/tmp/colibri_width4_motion_repeat2_grid64.json` and `.npz`.
The public example now selects group size four, retaining 30 substeps,
120 Hz collision updates, SOR 1, and no global damping.

Final direct public-example smoke check: 330 frames PASS with no constructor
or dispatch overrides. The report confirms group size four, 30 substeps,
and 120 Hz collision updates. Peak anchor 0.52285 mm, penetration 0.10453 mm.
Artifact: `/tmp/colibri_public_groups4_grid64_final330.json`.
Example Ruff and repository `git diff --check` passed; no commit made.

## Group size two: preliminary screen

A 330-frame screen at 30 substeps and 64 blocks passed, with maximum anchor
error 0.66504 mm. Its measured warm rate was 82.5993 FPS, but a fresh paired
isolated comparison and full-minute run are required before accepting a new
speed/stability result. The final graph has 78 groups and 383 copy slots.
The public default remains group size four.
Artifact: `/tmp/colibri_group2_grid64_30x1_screen330.json`.

Group size two is REJECTED by the full-minute test: after 3333 frames
(55.55 simulated seconds), TailMount/Tail_Feather_C axis error reaches
0.030495 rad, exceeding the unchanged 0.03 rad limit. Maximum anchor error
is 0.81256 mm and fresh penetration 0.88611 mm. The fast short screen was
insufficient. Public group size four remains unchanged.
Artifact: `/tmp/colibri_group2_grid64_30x1_3600.json`.

Group size three is also REJECTED: at frame2696 (44.933 seconds), fresh
penetration reaches 1.376 mm, exceeding the unchanged 1 mm limit. Joint
anchor error stays below 0.592 mm. Smaller groups are therefore not an
accepted optimization despite their short-screen speed. Group four remains
the accepted default.
Artifact: `/tmp/colibri_group3_grid64_30x1_3600.json`.

## Longer validation changes the acceptance scope

The group-four default passes 60 seconds but FAILS the requested five-minute
run at 67.9167 seconds. TailMount/TailPinion axis error jumps from0.000646 rad
at67.900 seconds to0.039944 rad one frame later (limit0.03). This is an abrupt
event, not gradual joint drift. The crank remains active before the failure.
Thus the previous one-minute acceptance must not be presented as long-term
robustness. Tail-event diagnosis is now the priority; no thresholds were relaxed.
Artifact: `/tmp/colibri_public_group4_grid64_18000.json` and `.npz`.
