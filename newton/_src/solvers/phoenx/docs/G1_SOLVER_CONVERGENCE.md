# PhoenX G1 Solver Convergence Notes

This note records the current G1 solver-fidelity findings that matter for
PhoenX RL. The goal is sim-to-real-useful training, so these settings prioritize
physically credible drive/contact behavior over the fastest possible sample rate.

## Current Status (2026-06-22)

Update: a CUDA graph analytical stack-support regression exposed that one
velocity-relax sweep leaves a five-cube stack with persistent residual contact
velocity (`0.427 m/s` max in the measured fixture). Two velocity-relax sweeps
settled the same fixture to `0.046 m/s` and restored the analytical contact-force
test. On the full G1 open-loop leg-step comparison, this alone did not close the
nanoG1 gap; the dominant effect was still position/contact convergence. The G1
recipe therefore now defaults to `10` sim substeps, `8` position iterations, and
`2` velocity-relax iterations. In the 40-step grounded leg-step parity benchmark
against nanoG1 host physics, that default keeps `4.0/4.0` mean foot contacts, has
no fall, and ends at `0.012 rad` joint-q RMSE with about `4.7 mm` final base-z
error. The previous `8x4x2` recipe ended at about `0.043 rad` joint-q RMSE and
`18 mm` final base-z error, so it was too compliant for G1 training quality.

Current CUDA regression coverage for G1-relevant physics is now explicit:

- Normal support and stacking: `test_contact_force` checks static sphere/cube
  weights, two- and five-cube stacks, pyramid support, and momentum
  conservation. The five-cube stack is the regression that requires two
  velocity-relax sweeps.
- Tangential contact: `test_friction` checks kinetic deceleration/stop distance,
  static friction thresholds, circular friction-cone behavior, no drift at rest,
  friction-induced rotation, and inclined-plane behavior. The ramp checks use
  first principles directly: static hold for `tan(theta) <= mu`, and sliding
  acceleration `g * (sin(theta) - mu * cos(theta))` above the friction cone.
- Joint/friction/contact coupling: `test_contact_force` now also checks a
  compact three-body revolute "snowman" stack with +Z joint axes. Only the
  bottom box touches the floor, and its contact normal force must equal the
  combined weight of all connected bodies.
- Joint friction and drive/friction composition: `test_joint_friction` checks
  stiction against gravity, Coulomb spin-down, acceleration under constant
  torque, and the PD-drive deadband created by friction.
- G1 actuator metadata and force law: `test_g1_rl_training` checks nanoG1 gain,
  damping, armature, friction, force-limit imports, explicit clamped PD torque
  formulas for every G1 joint, passive damping subtraction, and the intentionally
  soft implicit-drive coefficients.
- G1 contact geometry: `test_g1_rl_training` checks the nanoG1 foot boxes, their
  transforms, half-extents, friction, collision/visibility flags, and CUDA graph
  contact generation.
- G1 foot friction material and threshold: `test_g1_rl_training` checks that
  both nanoG1 foot boxes and the G1 ground plane carry `shape_material_mu = 0.6`
  and runs a graph-captured standing G1 push test. The push test enlarges the
  foot boxes in X/Y for stability, applies equal horizontal forces to both feet,
  verifies sub-threshold stiction, and verifies breakaway above the Coulomb
  limit. PhoenX currently exposes one Coulomb sliding coefficient on shapes;
  that value is used as the static cone limit and dynamic sliding coefficient
  rather than separate static/dynamic material fields.
- G1 standing support balance: `test_g1_rl_training` now settles the RL MJCF G1
  model with strong position-hold gains inside CUDA graphs, verifies negligible
  pose drift over an additional captured window, and checks that the summed
  robot-ground normal force matches total body weight within 2%.

The current evidence says PhoenX contact physics is passing the analytical
checks we know how to write. A PhoenX/nanoG1 difference in contact manifolds or
solver response is therefore not, by itself, evidence that PhoenX contacts are
wrong. The new G1 standing force-balance regression is also important: the same
RL model's foot contacts can carry the full G1 weight when the posture is held
stiffly enough, so a zero-action collapse is not explained by missing ground
normal force alone. The remaining non-analytical gap is full grounded G1 support
coupling: foot contacts, tangential impulses, base motion, and articulated drives
interact in a way that has no compact closed-form solution. That is tracked by
the nanoG1 open-loop parity benchmark rather than by a unit test.

Latest current-default parity rerun (`10x8x2`, explicit torque, nanoG1 foot
boxes) gives this split: lifted 10-step leg-step with no contacts has
`0.00275 rad` joint-q trajectory RMSE and `0.00196 rad` final target-error delta
against nanoG1 smooth dynamics. Grounded 60-step zero action has `6.96 mm`
max base-z error and `0.0133 rad` joint-q trajectory RMSE. Grounded 60-step
leg-step has `10.85 mm` max base-z error, `0.0188 rad` joint-q trajectory RMSE,
and `0.0362 rad` final target-error delta. This keeps pointing at grounded
contact/support coupling, not a gross no-contact actuator formula bug. The long
zero-action stability test also now separates the pre-terminal no-reset window
from the training path: continuing a fallen foot-box-only model without reset is
outside the G1 RL contract, while the 120-step auto-reset graph path remains
finite.

Physics parity is a diagnostic, not the final goal. Exact trajectory equality
against nanoG1 is less important than preserving the intended actuator/drive
behavior at the chosen PhoenX substep count: stiffness, damping, Coulomb
friction, armature, force limits, and solver-induced effective compliance. If
these first-principles contact tests stay green, the next practical lever is to
tune G1 actuation, command curriculum, rewards, and initialization for PhoenX's
contact model while staying close to nanoG1 where that remains useful. The
latest full 75M-sample training run with nanoG1/Puffer V-trace parity still
failed the quality gate (`battery_perf=0.302`, `battery_falls=393`) while
retaining acceptable speed (`train_seconds=184.3`, `total_wall_seconds=200.0`,
about 3.1x slower than the nanoG1 59 s reference). That points away from pure
trainer throughput as the main blocker, and toward actuator/contact response.

Fresh 20-step open-loop comparisons against nanoG1 host physics showed the
historical fast RL setting (`5x2`, 0.004 s physics dt, 2 position iterations,
1 velocity iteration) diverging over only 0.4 s:

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
understand and reduce effective actuator/contact drift in fast low-iteration
formulations, especially Unitree PD stiffness/damping, passive friction,
armature, force limits, and foot-contact coupling. A 2026-06-22 lifted
per-joint stiffness check isolated one actuator issue: the old implicit PhoenX
constraint-drive path produced only about `0.25x` to `0.85x` of the analytical
`kp * (target - q) - kd * qd` torque at the default `8x4` training dt, plateauing
near `0.60x` mean even with many PGS iterations. The root cause is the intentional
implicit-Euler PD row conversion, not a target mapping bug: PhoenX prepares
`gamma = 1 / (dt * (kd + dt * kp_clamped))`,
`bias = C * kp_clamped / (kd + dt * kp_clamped)`, and
`M_soft = 1 / (M_inv + gamma)`, so the coefficient-level force ratio starts
near `M_soft / (dt * (kd + dt * kp_clamped))` instead of `1.0`. The revolute
drive boost is already at the configured `10x` cap and only limits
`kp_clamped`; it does not remove the implicit damping/effective-mass softness.
The G1 RL path now defaults to explicit clamped PD torques through
`control.joint_f`, with CUDA graph regressions checking all 29 explicit torques
and the implicit-drive coefficient softness against these formulas.

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

The PhoenX-only grounded convergence benchmark now accumulates support metrics
over every physics substep in the 0.02 s policy frame. Against the `20x8`
PhoenX reference on the same 20-step, 0.2 leg-step target, `5x2` averaged 3.83
contacts per foot/substep sample versus 3.98 for the reference, with frame normal
support impulse 0.626 lower and tangential impulse 1.92 lower. `5x4` restored
four contacts but still had similar tangential-impulse deficit. `10x8` reduced
normal-impulse error to 0.031 but still lagged tangential impulse by 0.806.
This points at grounded contact/friction coupling and solver convergence as the
current drive-related gap; no-contact PD/damping/armature response remains a
secondary suspect. A follow-up `5x8` run improved normal-impulse delta only from
-0.539 to -0.436 and tangential-impulse delta from -1.90 to -1.85, while `10x8`
cut those to -0.031 and -0.806. Smaller substeps therefore matter more than
extra iterations at the 0.004 s substep size. The same tangential support gap is
visible in a zero-action hold-pose sweep (`5x2`: -1.90, `10x8`: -1.01), so it
is not caused only by the leg-step action target. A follow-up tangent/load ratio
metric shows this is not simple friction-cone saturation: in the zero-action
hold-pose sweep, mean tangent/normal ratio was `0.31` for `5x2`, `0.25` for
`5x8`, `0.41` for `10x8`, and `0.54` for `20x8` with friction `0.6`. The coarse
settings are using less of the available friction load, which points at
substep-level tangential constraint convergence.

A post-merge diagnostic pass after merging
`origin/dev/tw4/contact_reduction_improvements` did not change the G1 foot-box
path measurements. It added graph-captured foot-contact diagnostics for
speculative rows, tangent bias, and high tangent-load usage. On the same leg-step
sweep, speculative contact fraction was `0.0` for `5x2`, `10x8`, and `20x8`;
active-normal fraction stayed near `0.99`; and active tangent fraction was `1.0`.
The gap is therefore not currently explained by disabled speculative rows or
missing active tangent rows. The strongest new signal is high tangent-load usage:
`5x2` reached the high tangent/load threshold on only `0.14` of active foot rows,
`10x8` reached `0.48`, and `20x8` reached `0.79`. Tangent bias was actually
higher at `5x2` (`0.137` frame-sum mean) than the `20x8` reference (`0.036`). A
follow-up friction-load diagnostic, using the same soft-contact normal debiasing
formula as the Coulomb projection, showed that `5x2` does lose some available
friction load (`5.53` versus `6.36` frame-sum mean; load ratio `0.869` versus
`0.924`). That explains part of the coarse-setting tangent deficit, but not all
of it: `10x8` matches the reference friction load almost exactly (`6.36`) while
still lagging tangent impulse by `0.806`. The next solver investigation should
therefore focus on tangential projection/convergence first, with normal-load
debiasing as a secondary coupling effect. The diagnostic timing includes
deliberate metric readbacks and is not a throughput benchmark.

## Action Parametrization and Startup Exploration

The G1 RL action interface intentionally uses residual joint-position targets,
not full-range joint-limit targets:

```text
target_q = default_joint_q + action_scale * clip(raw_action, -1, 1)
```

With the nanoG1/PhoenX default `action_scale = 0.25`, a saturated policy
action requests only a `0.25 rad` offset from the nominal standing pose before
the target is clamped to the actuator control range. This matches the common
legged-locomotion pattern used by legged_gym-style PD position control and by
Isaac Lab's default-offset joint-position action terms. It is deliberately not
the more aggressive mapping where `action = -1` means the joint lower limit and
`action = +1` means the joint upper limit. Full-range mapping would make early
random PPO exploration much more violent and is not the current G1 reference
recipe.

A 2026-06-22 graph-leapfrog startup probe with the default G1 PPO config
(`log_std_init = 0`, 1024 worlds, 64 rollout steps) measured clipped action RMS
`0.72`, action clip fraction `0.32-0.33`, and mean `log_std` close to `0` over
the first five updates. In physical target units, the initial clipped target RMS
is therefore about `0.18 rad`. Startup exploration is not too small; if PhoenX
needs a change here, the likely sweep is less aggressive or more structured
exploration (`log_std_init`, per-joint `action_scale`, optional tanh squashing,
or a short action-scale curriculum), not a switch to full actuator-range
mapping.

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

Three fixes were required before solver tuning was meaningful:

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
3. The G1 ground plane used to keep PhoenX's generic default friction `0.75`.
   PhoenX combines material friction by averaging by default, so the effective
   foot-floor coefficient was `(0.6 + 0.75) / 2 = 0.675`, while nanoG1's
   generated model authors the relevant foot-floor pair friction directly as
   `0.6`. The G1 recipe now sets the ground friction to `0.6`, making the
   averaged PhoenX contact coefficient match nanoG1 without changing global
   material-combine semantics.

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

Open-loop trace mode and historical nanoG1 host comparisons:

The benchmark now accepts `--trace-steps N` to emit per-control-step
records for early grounded divergence. A 6-step trace with the current default
setting and leg-symmetric action,

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_open_loop_parity \
    --steps 6 --trace-steps 6 --action-pattern leg_symmetric \
    --action-amplitude 0.6 --settings recipe_default --json-indent 2 --device cuda:0
```

shows that contact counts line up after the first two steps (`4/4` contacts per
foot in both engines), but PhoenX already sits lower (`-5.37 mm` base-z delta at
step 6) with `5.31 mm` base-XY delta and `0.0128 rad` joint-position RMSE. The
mean target-tracking-ratio delta is still small at step 6 (`+0.016`), while the
20-step run grows to about `+0.309`. This narrows the next check to grounded
support coupling over the next few policy frames: contact counts alone match,
but base support and effective drive response diverge before any fall.

The same benchmark now instruments nanoG1 host `force[]` rows for the foot
contacts and reports PhoenX final-substep impulse divided by PhoenX substep dt
as a force estimate. On the 20-step `recipe_default` leg-symmetric trace,
normal support agrees (`339.3 N` PhoenX estimate versus `339.8 N` nanoG1), but
tangential support does not (`131.8 N` PhoenX estimate versus `11.0 N` nanoG1).
The mean tangent/normal ratio is therefore `0.396` in PhoenX versus `0.0367` in
nanoG1. At step 6, after contact counts already match `4/4`, PhoenX reports a
support tangent/normal ratio of `0.460`, while nanoG1 reports `0.0107`. This
rules out a gross missing-normal-support issue for this trace and makes
tangential contact/friction projection the highest-priority grounded-physics
suspect.

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

A later no-reset zero-action G1 simulation exposed a more severe failure mode:
`5x2` can inject enough energy after ground impact to send the robot upward with
multi-million-scale joint velocities (`max_abs_qd` around `1.0e7` after 200
steps). The same 200-step probe stayed finite for `8x4` (`max_abs_qd` about
`0.88`), `10x4` (`0.70`), and `12x6` (`0.57`). The old `5x2` setting is useful
only as a historical throughput reference, not as a default for training or
visual no-reset simulation.

The model also carries nanoG1 damping, frictionloss, and armature; unit tests
compare those values directly against `web/g1_model_const.h`. Further solver
work should improve the constraint formulation so lower substep/iteration counts
become stable without sacrificing physical fidelity.

## Current Decision

The default PhoenX G1 RL recipe uses `sim_substeps=10`, `solver_iterations=8`,
`velocity_iterations=2`, and `contact_geometry="nanog1_foot_boxes"`. Speed
remains important, but the default must preserve grounded support fidelity and
survive contact without numerical blow-up. Pass `--sim-substeps 5 --solver-iterations 2` only to reproduce the old fast setting.

Next likely checks:

1. Re-run train-to-gate with the fixed reset/contact setup.
2. Compare PhoenX reward/reset traces against nanoG1 for identical policy
   outputs and noisy reset seeds.
3. Profile the grounded-contact pipeline before optimizing solver kernels; the
   old profiles were collected with the physically invalid airborne reset.
