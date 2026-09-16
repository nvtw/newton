# Controlled cause of the slow conditioned contact loop

The graph-only frozen replay uses identical saved inputs and substitutes
recorded exact intermediate values. All output arrays must match the original
kernel byte-for-byte. Resets are outside the timer. A captured one-kernel CUDA
graph excludes Python argument-packing time. The entire 331-frame capture
trajectory is byte-identical to the prior serial candidate.

## Measured counterfactuals

Biased phase, 67 visited points:

| Counterfactual | Original kernel | Substituted kernel | Saving |
| --- | ---: | ---: | ---: |
| Three contact row-velocity evaluations | 382.560 us | 80.368 us | 302.192 us |
| Impulse response/application | 382.560 us | 380.208 us | 2.352 us |
| Friction projection, preserving observable velocity evaluations in both variants | 381.984 us | 356.176 us | 25.808 us |

These are independent counterfactuals, not an additive partition of runtime.
The row-velocity calculation accounts for the dominant removable work in this
frozen kernel. Earlier emphasis on the per-point impulse response was incorrect.

Simply deleting friction projection also deletes its two tangent-velocity
inputs through compiler dead-code elimination. That misleading experiment
appeared to remove about 200 us even in a relaxation phase with zero actual
projection calls. Writing all three computed velocities to an observable sink
in BOTH kernels resolves the ambiguity: relaxation takes 333.136 us unchanged
versus 335.024 us with projection substituted, showing no projection saving.
All 16 variant/phase byte gates pass.

The underlying generic routine is `_contact_row_velocity` in
`articulations/maximal_contact_gs.py`. It traverses joint-coordinate branches
and evaluates FP64 scalar velocity expressions separately for the normal and
two tangent directions. The next implementation shares point-velocity work
across those three directions and tests an FP32 external-contact path, while
retaining cancellation-safe handling for internal contacts. No recorded-value
substitution is a usable simulation optimization.

## Other measured contributors to the cross-solver gap

- PhoenX's settled snapshot has 67 point blocks, 14 loaded; the SDK reports
  six normal contacts, five loaded, and two friction anchors at frame 600.
- PhoenX contact mobility refresh costs about 7–8 GPU ms/frame separately.
- Native PhoenX executes about 3,116 kernels/frame; PhysX executes 488.
- The D6-style scalar architecture measures 121 FPS in a short run but has
  unacceptable joint/base error. The block-joint alternative measures 94 FPS
  over 600 frames but creeps 62.67 um/s over seconds 5–10. Neither is accepted.

These findings explain measured pieces of the gap. They are not a claim that
all remaining full-scene cost, stability or creep issues have been solved.

Artifacts: `/tmp/colibri_point_sink_replay331.point_replay.json`,
`/tmp/colibri_point_sink_replay331.point_inputs_biased.{npz,json}`,
`/tmp/colibri_point_sink_replay331.point_inputs_relax.{npz,json}`.
See also `PHYSX_RUNTIME_PROFILE.md` for the actual SDK profiling and contact
reporting controls.
