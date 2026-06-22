# PhoenX G1 Solver Convergence Notes

This note records the current G1 solver-fidelity findings that matter for
PhoenX RL. The goal is sim-to-real-useful training, so these settings prioritize
physically credible drive/contact behavior over the fastest possible sample rate.

## Current Status (2026-06-22)

Physics parity is a diagnostic, not the final goal. Exact trajectory equality
against nanoG1 is less important than preserving the intended actuator/drive
behavior at the chosen PhoenX substep count: stiffness, damping, Coulomb
friction, armature, force limits, and solver-induced effective compliance. The
latest full 75M-sample training run with nanoG1/Puffer V-trace parity still
failed the quality gate (`battery_perf=0.302`, `battery_falls=393`) while
retaining acceptable speed (`train_seconds=184.3`, `total_wall_seconds=200.0`,
about 3.1x slower than the nanoG1 59 s reference). That points away from pure
trainer throughput as the main blocker, and toward actuator/contact response.

Fresh 20-step open-loop comparisons against nanoG1 host physics show the
current RL setting (`5x2`, 0.004 s physics dt, 2 position iterations, 1 velocity
iteration) diverges over only 0.4 s:

| action | setting | final base-pos error | base-z max error | joint-q traj RMS | joint-qd traj RMS |
| --- | --- | ---: | ---: | ---: | ---: |
| zero | `recipe_default` (`5x2`) | 0.0514 m | 0.0286 m | 0.0214 rad | 0.227 rad/s |
| zero | `phoenx_10x8` | 0.0125 m | 0.0048 m | 0.0062 rad | 0.081 rad/s |
| leg step, amp 0.2 | `recipe_default` (`5x2`) | 0.0543 m | 0.0291 m | 0.0244 rad | 0.298 rad/s |
| leg step, amp 0.2 | `phoenx_10x8` | 0.0179 m | 0.0059 m | 0.0080 rad | 0.119 rad/s |

The same leg-step drive-convergence sweep against a high-resolution PhoenX
reference (`20x8`, 0.001 s physics dt) shows most of the remaining fast-setting
error is solver/contact/drive convergence, not only a nanoG1-vs-PhoenX modeling
constant mismatch:

| setting | physics dt | iterations | base RMS vs ref | joint-q RMS vs ref | joint-qd RMS vs ref |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rl_current` (`5x2`) | 0.004 | 2 + 1 | 0.0227 | 0.0215 rad | 0.424 rad/s |
| `phoenx_5x4` | 0.004 | 4 + 1 | 0.0206 | 0.0158 rad | 0.240 rad/s |
| `phoenx_10x4` | 0.002 | 4 + 1 | 0.0059 | 0.0090 rad | 0.062 rad/s |
| `phoenx_10x8` | 0.002 | 8 + 2 | 0.0051 | 0.0082 rad | 0.053 rad/s |

The next physics target is therefore not to lower accuracy for speed. It is to
understand and reduce effective actuator-model drift in the fast `5x2`
formulation, especially Unitree PD stiffness/damping, passive friction, armature,
force limits, and foot-contact coupling.

Updated contact-support diagnostics now report per-foot contact counts for both
nanoG1 host physics and PhoenX, plus PhoenX per-foot normal/tangent impulse
sums. Gross foot support is present: zero-action runs match nanoG1 exactly at 4
contacts per foot for both `5x2` and `10x8`. Under the 0.2 leg-step target,
`10x8` still matches 4 contacts per foot, while `5x2` drops to mean contact
counts of 3.80 left and 3.85 right against nanoG1's 4.0/4.0, with contact-count
RMSE 0.418. This narrows the likely root cause to constraint/drive force
response and solver convergence, not a simple absent-contact bug. Candidate
remaining diagnostics: per-joint no-contact PD/friction/armature step response,
grounded hold-pose contact impulse traces, and a substep sweep that reports
tracking compliance against the configured Unitree gains and force limits.

A lifted drive-response sweep separates the drive model from foot contact.
With `--initial-base-z 3.0`, 20 policy steps, a 0.2 leg-step action, and the
nanoG1 `full` stepper, both solvers reported zero foot contacts. PhoenX `5x2`
ended with a target-error delta of 0.00131 rad RMS against nanoG1 and tracked
1.061 of the command versus nanoG1's 1.053; `10x8` was similarly close. With
nanoG1 `smooth` dynamics and PhoenX joint friction disabled, PhoenX overshot
slightly more (tracking ratio +0.032 to +0.041), so the current data does not
support a gross no-contact PD/friction/armature softness bug. The important
remaining question is how those same drives behave when coupled to grounded
contacts and solver compliance.

## nanoG1 Reference

nanoG1 v3 uses the following production physics and drive setup:

- Control frame: 0.02 s.
- Physics step: 0.004 s, decimation 5.
- Solver: Newton iterations 2, line-search iterations 3.
- Actions: first 12 leg actions only, action scale 0.25 rad.
- Leg stiffness: `(100, 100, 100, 150, 40, 40)` per side.
- Leg Unitree damping: `(2, 2, 2, 4, 2, 2)` per side.
- Passive `dof_damping`: `(2, 2, 2, 2, 1, 0.2)` per side.
- Effective zero-target-velocity damping on controlled legs:
  `(4, 4, 4, 6, 3, 2.2)` per side.

The reset pose in PhoenX matches nanoG1 host constants at the semantic level:
base height 0.785 m, identity base orientation, and the same 29 default joint
positions. Newton stores free-joint quaternions as `x, y, z, w`; nanoG1/MuJoCo
stores them as `w, x, y, z`, so parity tools must reorder only for comparison.

## PhoenX Parity Fixes

Two fixes were required before solver tuning was meaningful:

1. The G1 RL environment initialized and observed the free-joint quaternion in
   nanoG1/MuJoCo order. In Newton layout, `[1, 0, 0, 0]` is a 180-degree X
   rotation, which put the legs above the pelvis and left the robot airborne.
   The environment now initializes identity as `[0, 0, 0, 1]` and observes
   gravity with the same layout.
2. The raw Menagerie G1 foot point contacts produce zero ground contacts in
   PhoenX at reset. The default G1 recipe now uses
   `contact_geometry="nanog1_foot_boxes"`, which disables those imported foot
   point colliders and adds the two MuJoCo/nanoG1 foot contact boxes with local
   position `(0.04, 0.0, -0.029)`, half-extents `(0.09, 0.03, 0.008)`, and
   friction `0.6`. The raw `contact_geometry="mjcf"` mode remains available
   as a diagnostic.

PhoenX also folds nanoG1 passive `dof_damping` into all 29 G1 position-drive
`joint_target_kd` values. For these drives the target velocity is zero, so this
reproduces the same damping force contribution, `-damping * qd`. The generic
PhoenX model adapter forwards `joint_friction` into the ADBS Coulomb-friction
row. `SolverPhoenX` defaults to hard PhoenX Coulomb friction; the G1 recipe
uses `joint_friction_model="mujoco"` so imported `solreffriction` and
`solimpfriction` soften the row in the same spirit as nanoG1 host physics. The
G1 recipe authors nanoG1 `dof_frictionloss = 0.1` on every actuated joint.

## Open-Loop Parity

Benchmark command:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_open_loop_parity \
    --steps 20 --action-pattern leg_step --action-amplitude 0.2 --json-indent 0
```

Current results versus nanoG1 host physics:

| case | friction mode | setting | fall step | final base z | base-z max error | joint-q traj RMS | joint-qd traj RMS |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| zero action, 20 steps | none (`scale=0`) | `recipe_default` (`10x4`) | none | 0.760 m | 0.0205 m | 0.0262 rad | 2.02 rad/s |
| zero action, 20 steps | hard | `recipe_default` (`10x4`) | none | 0.763 m | 0.0182 m | 0.0187 rad | 6.20 rad/s |
| zero action, 20 steps | MuJoCo soft | `recipe_default` (`10x4`) | none | 0.763 m | 0.0182 m | 0.0258 rad | 4.52 rad/s |
| leg step, 20 steps | MuJoCo soft | `recipe_default` (`10x4`) | none | 0.758 m | 0.0168 m | 0.0232 rad | 4.48 rad/s |
| zero action, 20 steps | MuJoCo soft | `phoenx_10x8` | none | 0.762 m | 0.0187 m | 0.0237 rad | 3.10 rad/s |

Before the quaternion/contact fixes, the zero-action 5-step `10x4` base-z error
was about 0.049 m and the leg-step run fell at policy step 15. Hard joint
friction improves position parity but creates a large velocity mismatch. MuJoCo
soft friction is more faithful for the nanoG1 comparison and reduces that
velocity error, but it does not close the gap. Extra PhoenX iterations help,
which points to a remaining convergence/formulation gap rather than a pure RL
throughput issue.

## Convergence Sweep

Benchmark command:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_drive_convergence \
    --steps 20 --world-count 4 --action-pattern leg_step --action-amplitude 0.2 \
    --reference-setting phoenx_20x8 --json-indent 0
```

The 20-step horizon is 0.4 s at the 50 Hz policy rate. The reference is
`phoenx_20x8` with 0.001 s physics dt, 8 position iterations, and 2 velocity
iterations.

| setting | physics dt | iterations | fall fraction | min base height | joint-q RMS vs ref | target RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rl_current` (`5x2`) | 0.004 | 2 + 1 | 0.0 | 0.752 m | 0.0965 rad | 0.0552 rad |
| `phoenx_5x4` | 0.004 | 4 + 1 | 0.0 | 0.750 m | 0.0615 rad | 0.0477 rad |
| `phoenx_10x4` | 0.002 | 4 + 1 | 0.0 | 0.758 m | 0.0351 rad | 0.0533 rad |
| `phoenx_10x8` | 0.002 | 8 + 2 | 0.0 | 0.759 m | 0.0289 rad | 0.0497 rad |
| `phoenx_20x8` | 0.001 | 8 + 2 | 0.0 | 0.766 m | 0.0000 rad | 0.0709 rad |

The `10x4` setting remains the conservative open-loop fidelity diagnostic: it
materially improves trajectory agreement versus `5x2` without making the run as
expensive as the `20x8` reference. It is no longer the default RL training
setting because the stochastic early-policy distribution exposed a different
failure mode (fall-dominated rollouts), described below.

## Training Impact

With the grounded G1 setup (`contact_geometry="nanog1_foot_boxes"`) and
MuJoCo-soft nanoG1 joint friction enabled, short graph-leapfrog probes show two
different behaviors:

| setting | stochastic no-update rollouts | train-to-gate probe |
| --- | --- | --- |
| `5x2` (nanoG1 timing, no armature) | 64 worlds x 16-step rollouts stayed mostly non-terminal: first four rollout done means `0.000`, `0.054`, `0.034`, `0.037`. | 60 iterations measured 228k train env samples/s and 215k total env samples/s; gate failed at 15.7M samples (`battery_perf=0.289`, `battery_falls=94`). A full 75.2M-sample run also failed the gate before the armature fix. |
| `5x2` (nanoG1 timing + armature) | Same timing plus nanoG1 exported per-DOF armature in the PhoenX model. | compact 60-iteration probe measured 252k train env samples/s and 237k total env samples/s; gate still failed but improved to `battery_perf=0.388`, `battery_falls=27`, and finite leg-velocity diagnostics. |
| `10x4` (fidelity diagnostic) | same random policy became fall-dominated: first four rollout done means `0.115`, `0.536`, `0.724`, `0.757`. | prior 60-iteration probes degraded badly by iteration 60, with unstable velocity diagnostics. |

The heavier `10x4` setting is closer to the high-resolution PhoenX reference in
open-loop drive tests, but it moves the early stochastic training distribution
away from nanoG1 and makes PPO spend most samples on resets/falls. The default
RL recipe therefore uses `sim_substeps=5`, `solver_iterations=2`, and
`contact_geometry="nanog1_foot_boxes"`, matching nanoG1 production timing.
The model also carries nanoG1 damping, frictionloss, and armature; unit tests
compare those values directly against `web/g1_model_const.h`. Further solver
work should improve the constraint formulation so higher substep/iteration
counts do not change the training distribution this much.

## Current Decision

The default PhoenX G1 RL recipe uses `sim_substeps=5`, `solver_iterations=2`,
and `contact_geometry="nanog1_foot_boxes"`. Speed remains important, but
further optimization should preserve the grounded reset, active foot contacts,
and open-loop parity checks above.

Next likely checks:

1. Re-run train-to-gate with the fixed reset/contact setup.
2. Compare PhoenX reward/reset traces against nanoG1 for identical policy
   outputs and noisy reset seeds.
3. Profile the grounded-contact pipeline before optimizing solver kernels; the
   old profiles were collected with the physically invalid airborne reset.
