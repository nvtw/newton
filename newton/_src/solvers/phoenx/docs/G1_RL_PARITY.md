# PhoenX G1 RL Parity Notes

This note tracks the current quality-facing parity work between PhoenX G1
training and nanoG1. The goal is to train a useful full-coordinate G1 walking
policy with the pure-Warp RL stack, while keeping the training path reusable for
other environments.

## Reference

The nanoG1 checkout at `/home/twidmer/Documents/git/nanoG1` points to the
PufferLib G1 fork `kingjulio8238/PufferLib`, branch `g1`, pinned at commit
`e3825cea`. The exact task source was fetched for inspection from that commit:

- `ocean/g1gpu/g1_gpu.cu` for observation, action, reward, termination, command
  resampling, and reset semantics.
- `ocean/g1gpu/g1_mirror.h` for the left/right mirror map.
- `ocean/g1gpu/g1phys/g1_staged_kernels.cuh` for Unitree PD drive semantics.

## Covered By CUDA Graph Tests

`test_g1_rl_training.py` now covers these high-risk nanoG1/PufferLib contracts:

- Recipe values, deploy constants, action masking, observation layout, and
  default G1 model constants.
- Continuous Gaussian actor log-probability and hand-written PPO actor loss /
  mean and log-std gradients.
- Value clipping, V-trace shifted rollout layout, PufferLib sample-variance
  advantage normalization, and priority replay probability / importance weights.
- Muon optimizer update semantics, transposed MLP weight layout, PufferNet linear
  layer layout, MinGRU equations, recurrent reset behavior, Puffer-style
  zero-state mirror target forward, save/load/resume, and graph-leapfrog smoke
  training.
- G1 v3 reward decomposition for a deterministic state, including actual
  actuator-force torque penalty, gait phase, contact/swing/hip/base-height
  shaping, action-rate penalty, dt scaling, done, and success metrics.
- Graph-captured G1 actuator-force gather using nanoG1's actuator-model formula,
  including force clamps and reset clearing so stale force cannot leak across
  episodes.

All of these tests run CUDA-only and use Warp CUDA graph capture.

## Current Difference List

- PhoenX G1 actuation now defaults to `actuation_model="explicit_torque"`:
  every substep computes the same actuator-model force signal nanoG1 writes to
  `g_af`, `clip(kp * (target - q) - kd * qd, force_range)`, and scatters it to
  `control.joint_f` so PhoenX consumes explicit generalized torques. The
  legacy `actuation_model="constraint_drive"` path remains available as a
  diagnostic for PhoenX implicit drive rows.
- A graph-captured 29-world regression perturbs one G1 actuator per world and
  checks every joint force against the analytical nanoG1/MuJoCo formula. This
  guards the likely actuator-quality blocker directly instead of relying on a
  training run to expose it.

- A 2026-06-22 model-contract audit found that PhoenX was importing Newton's
  cached Menagerie `unitree_g1/mjcf/g1_29dof.xml` body inertials (`35.112 kg`)
  while nanoG1 trains against a compiled MuJoCo Playground model exported in
  `web/g1_model_const.h` (`33.341 kg`). PhoenX G1 now pins body mass, COM, and
  full body-frame inertia to the nanoG1 header after MJCF import. The same audit
  found two imported joint-contract drifts: both hip-roll effort limits were
  `88 N*m` instead of nanoG1's `139 N*m`, and the right hip-roll joint range was
  sign-flipped. The G1 builder now overwrites all actuated joint ranges and
  effort limits from the nanoG1 constants.
- The high-level nanoG1 `recipe.py` does not list gait/base-height reward terms,
  but the pinned PufferLib G1 fork does define `G1_V3_W_CONTACT`,
  `G1_V3_W_SWING`, `G1_V3_W_HIP`, and `G1_V3_W_BASE_HEIGHT` in the CUDA task
  source. PhoenX keeps these terms in the default G1 v3 reward and tests them
  against the local pinned source when it is available. PhoenX keeps a
  configurable swing-foot-contact penalty for experiments, but the default is
  0.0 because the pinned nanoG1 CUDA reward has no such term.
- The local generic PufferLib checkout is branch `4.0` at `e90b58ed`, not the
  nanoG1 G1 fork. Parity work should use the nanoG1 recipe/deploy files plus
  the pinned fork source above.
- A 2026-06-22 isolation run replaced the pure-Warp PPO update with a
  PufferLib-style Torch learner while keeping PhoenX physics/env/rewards fixed.
  The learner uses native PufferNet layout, MinGRU, Puffer shifted V-trace,
  prioritized replay, Muon, and mirror loss. It still failed the 75M-sample
  gate, so the current walking blocker is unlikely to be only PhoenX Warp PPO
  implementation.
- The shipped `assets/nanoG1.bin` PufferNet policy can now be imported into a
  normal PhoenX PPO checkpoint. A zero-observation smoke matched the nanoG1 C
  inference shim to `1.2e-7` max absolute error, so this gives a verified
  PyTorch-free teacher/warm-start path.
- Short train-to-gate probes still fail the walking gate. The current evidence
  points to remaining simulator/reward/solver quality gaps and sample-efficiency
  issues, not just PPO math.
- The Python gate diagnostic now rotates root linear velocity with Newton's
  free-joint quaternion layout, `x, y, z, w`. The previous helper interpreted
  the same four numbers as `w, x, y, z`, corrupting the printed
  `mean_linear_velocity_error` diagnostics. The actual gate pass/fail metric was
  already computed on the device from `step_successes`, so this was a diagnostic
  bug, not a training-quality fix.

## 2026-06-22 MinGRU BPTT Fix

A PPO-side root cause was found in the pure-Warp PufferMinGRU sequence backward
kernel: the future recurrent gradient was added to the current output gradient
before computing the projection/highway path. In MinGRU, future recurrent
gradient should flow through the recurrent state into gate/candidate/previous
state terms, but it must not change the current projection or highway-input
gradients. A new CUDA graph regression test isolates a two-step sequence where
only the second output has upstream loss; the first step projection/highway
gradients must remain zero while hidden/gate gradients stay nonzero. The test
fails against the old formula and passes with the corrected routing.

The first quality probe after this fix is measurably better but still does not
solve walking. At 100 training iterations, rollout perf improved from the
previous post-reward-parity 0.618 to 0.683. The held-out gate improved from
battery_perf=0.579, battery_falls=3/24000, and forward 0.8 m/s velocity error
0.707 m/s to battery_perf=0.664, battery_falls=0/24000, and forward error
0.549 m/s. This is real PPO progress, but still below the walking gate.
Continuing the same run did not compound the gain: the 200, 300, and 400
checkpoints reached battery_perf=0.649, 0.658, and 0.678 respectively, with the
400 checkpoint at 1/24000 falls and forward error 0.590 m/s. Treat the 100-step
checkpoint as the best post-fix quality reference for now; the remaining blocker
is the longer-horizon training plateau, not this BPTT bug alone.

## Discrepancy Ledger

This table is the working order for quality bugs. New tuning or optimization
should land only after one row has a measured discrepancy and a regression test
or benchmark note.

| area | reference | PhoenX status | next action |
| --- | --- | --- | --- |
| PPO/V-trace replay math | PufferLib G1 fork at `e3825cea` | CUDA graph tests cover V-trace shifted rollout layout, priority weights, and whole-trajectory gather/scatter back to rollout buffers. | Keep as regression coverage; do not tune PPO until a new mismatch is proven. |
| Muon/network kernels | PufferLib source plus local finite-difference tests | Tests cover Muon update semantics, PufferNet linear layout, MinGRU equations, mirror maps, and manual PPO actor/value gradients. | Re-check precision/layout only if training probes show learner instability independent of physics. |
| Puffer learner isolation | PufferLib-style Torch PPO/V-trace/Muon learner | `bench_g1_train_puffer_torch` keeps PhoenX physics fixed and swaps only the RL learner. A 75.2M-sample run failed similarly to Warp PPO: `battery_perf=0.553`, `battery_falls=43`, forward-command perf `0.096`. | Treat remaining root cause as PhoenX env/physics/reward parity unless a more exact pinned native Puffer integration contradicts this result. |
| nanoG1 policy import | Shipped `assets/nanoG1.bin` plus `deploy/nanog1_policy.c` | `nanog1_import.py` converts the PufferNet binary into a PhoenX PPO checkpoint; imported Warp output matches the nanoG1 C shim to `1.2e-7` on a zero-observation smoke. The imported policy reaches only `battery_perf=0.700` on the full PhoenX gate and fails, so a known-good policy still degrades under PhoenX. | Use the imported policy as the next primary physics/env parity probe before changing PPO or adding reward terms. |
| Gate diagnostics | nanoG1-style command battery | Fixed Python velocity diagnostic quaternion order to Newton `xyzw`; graph-captured regression covers the case. | Re-run gates only when diagnostics are needed; do not treat this as quality progress. |
| Graph overlap / stale policy | Same recipe in eager and graph-leapfrog modes | A 60-iteration eager train-to-gate probe failed similarly to graph mode, so stream overlap/stale rollout policy is not the primary quality blocker. | Keep graph mode for throughput, but debug quality in the simpler eager path when possible. |
| G1 env/reward contract | `g1_gpu.cu`, `recipe.py`, deploy constants | Tests cover observation/action layout, actuator-force torque penalty, reward decomposition, gait/success terms, command constants, and graph recurrent reset behavior. The default now resets recurrent state at PPO rollout boundaries because the Warp buffer does not yet store initial hidden states for update-time replay. | Add targeted tests for command resampling/reset timing and done/bootstrap semantics before changing rewards. |
| Drive/contact physics | nanoG1 host physics plus first-principles drive tests | Explicit G1 actuator forces now match the analytical nanoG1/MuJoCo formula joint-by-joint; the old implicit drive-row path was effectively too compliant at training dt. G1 foot-ground friction now matches nanoG1 pair friction (`0.6`): the previous PhoenX ground `0.75` averaged with foot `0.6` to an unintended effective `0.675`. | Keep explicit torque as the default; use `constraint_drive` only for diagnostics, and re-run policy gates/open-loop parity after the friction fix. |
| End-to-end training quality | nanoG1 reaches a working policy in roughly the README time scale | Current 75M-sample PhoenX runs train in a few minutes but fail the gate. | Run full probes only after a ledger row changes; record before/after quality and throughput. |

## Current Measurement

The 2026-06-22 Puffer learner isolation run used:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.experimental.bench_g1_train_puffer_torch \
    --iterations 287 --world-count 4096 --rollout-steps 64 \
    --checkpoint-path /tmp/phoenx_g1_puffer_{iteration}.pt \
    --checkpoint-interval 287 --gate
```

It trained `75,235,328` samples in `263.9 s` at `285k` env samples/s and
failed the gate with `battery_perf=0.553`, `battery_falls=43`, and forward
`0.8 m/s` command tracking perf `0.096`. The policy learned stable standing
(`stand` perf `0.978`) and weak turning/lateral behavior, but not forward
walking. This matches the earlier pure-Warp PPO failure mode closely enough that
the next root-cause work should focus on PhoenX G1 environment/physics parity
rather than rewriting the PPO learner again.


The 2026-06-22 nanoG1 policy import used:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.experimental.nanog1_import \
    --device cuda:0 \
    --checkpoint /tmp/phoenx_nanog1_import_smoke.npz
```

A temporary C shim built from `deploy/nanog1_policy.c` and local PufferLib
`puffernet.h` matched the imported Warp PufferMinGRU checkpoint at `1.2e-7` max
absolute error for zero-observation inference and `4.8e-6` max error over a
32-step nonzero recurrent sequence. The full PhoenX gate with current stable
defaults (`sim_substeps=10`, `solver_iterations=8`, explicit torque,
foot-box contacts) failed but was much stronger than locally trained policies:
`battery_perf=0.700`, `battery_falls=134/24000`, `stand` perf `0.952`, and
`forward_0.8` perf `0.501`. Short 4800-sample sweeps showed `5x2` is unstable
(huge velocity diagnostics), `8x4` is stable but weaker (`battery_perf=0.652`),
`10x8` is best among tested settings (`0.730`), MJCF contact geometry did not
help (`0.712`), and the old constraint-drive path was similar (`0.728`). This
makes the imported policy the best near-term diagnostic for contact/drive/env
parity: a policy known to walk in nanoG1 still degrades in PhoenX before any PPO
fine-tuning is involved.

A follow-up parity fix changed G1 observation, reward, and gate diagnostics to
use body-frame root angular velocity, matching nanoG1's deployed IMU convention
and Newton's other robot-policy examples. The graph-captured observation and
reward decomposition tests now use non-identity root orientations to guard this
contract. The same short imported-policy gate at `10x8` remained essentially
unchanged after the fix (`battery_perf=0.732`, `battery_falls=21/4800`, previous
short baseline `0.730`), so this was a correctness fix rather than the root
cause of the remaining walking gap.

The latest full quality-facing probe before the stability default change used
4096 worlds, 64 rollout steps, the nanoG1-timed `5x2` recipe,
graph-leapfrog execution, and one 75.2M sample chunk. It completed training in
185.4 s and the whole train-save-reload gate run in 201.2 s, about 405.7k train
environment samples/s, but still failed the gate with `battery_perf=0.325` and
`battery_falls=372`. A 60-iteration eager probe also failed at 15.7M samples
with `battery_perf=0.321`, so the current blocker is quality/sample efficiency,
not only the two-stream graph schedule.

A 2026-06-22 sparse-command probe used 2048 worlds, 64 rollout steps, forward
commands in `[0.2, 0.6] m/s`, no lateral/yaw commands, `reward_mode=sparse_command`,
and graph-leapfrog execution. It trained 19.7M samples in 120 s at about 196k
environment samples/s. The iteration-50 checkpoint showed that PhoenX can learn
forward-ish motion quickly (`forward_0.8` perf `0.837`, `falls=5` in the short
diagnostic), but the command battery was still weak (`battery_perf=0.467`) and
later checkpoints regressed/generalized poorly. Treat sparse command reward as
a promising anti-standing lever, not a solved walking recipe.

A same-day dense-command probe removed the alive bonus (`w_alive=0.0`), raised
linear tracking to `w_track_lin=6.0`, and trained only forward commands in
`[0.2, 0.6] m/s` for 160 iterations at 2048 worlds. It reached 20.9M samples
in 111 s of training at about 189k samples/s and improved the short command
battery to `battery_perf=0.697`, with `forward_0.8` perf `0.688` and stand perf
`0.891`. It still failed due falls and weak lateral/yaw generalization, but it
is the best current anti-standing recipe and should be the next tuning baseline.

After changing the default to reset recurrent state at PPO rollout boundaries, a
full-command anti-standing run (`w_alive=0.0`, `w_track_lin=6.0`) reached the
nanoG1 sample budget without a walking-quality pass. At 75.0M samples it trained
for 327 s at `229k` env samples/s and reached `battery_perf=0.703`,
`battery_falls=30/3000`, `forward_0.8` perf `0.581`, and stand perf `0.939`.
Continuing the same checkpoint to 149.9M samples took another 328 s and did not
improve the short gate: `battery_perf=0.699`, `battery_falls=35/3000`. This
shows that the recurrent-state fix improves PPO consistency but the current
recipe plateaus; simply running longer for about ten minutes is not enough.

Sparse-target training now has a graph-compatible target-distance curriculum and
a target evaluator that separates target hits from walking quality. Target
success itself now requires balanced posture, which prevents the previous false
positive where a policy lunged through the target with large tilt. A strict
from-scratch run lost the sparse signal by iteration 40; a warm-start from the
previous non-strict sparse checkpoint retained clean 0.6 m success but still
failed at 1.0 m and 1.4 m. Current evidence therefore points to sparse task
exploration/curriculum or teacher warm-start needs after the false-success bug,
not a missing dense reward term.

A current imported-teacher fine-tune probe used the verified nanoG1 checkpoint
under PhoenX physics with `actor_lr=critic_lr=2e-4` and one PPO epoch. The first
60 iterations improved the full gate from the imported baseline
`battery_perf=0.707`, `battery_falls=122/24000` to `0.751`, `84/24000`.
Continuing the same run to iteration 120 plateaued at `0.749`, `77/24000`; a
higher `2e-3` learning-rate probe had unhealthy KL and clip fraction. A short
physics-knob sweep on the imported teacher suggested MJCF contacts might help,
but the full MJCF-contact gate was worse (`battery_perf=0.702`,
`battery_falls=106/24000`). Keep the foot-box contact default and treat the
teacher fine-tune as useful evidence, not a solved walking recipe: PPO can adapt
the teacher slightly, but a known-good nanoG1 policy still degrades under
PhoenX before optimization starts.

## 2026-06-22 Contract And Training Probes

A follow-up check found one real reward/metric convention bug: Newton free-joint
`joint_qd[0:3]` stores child-COM velocity, while nanoG1/MuJoCo command tracking
uses root body-origin velocity. G1 reward and gate diagnostics now subtract
`cross(root_angular_velocity, root_com_offset)` before rotating linear velocity
into the body frame. A CUDA graph regression drives nonzero root angular velocity
and fails if COM velocity is tracked directly. This is a correctness fix, but it
did not solve walking by itself: the imported nanoG1 teacher gate stayed near the
previous result (`battery_perf=0.702`, `battery_falls=123/24000`).

A PhoenX frame-convention audit on 2026-06-22 found no local-frame or COM-offset
bug in the Newton adapter path. PhoenX imports Newton `body_q` as the body-origin
pose, stores the body COM position internally, exports the body-origin pose by
subtracting `R * body_com`, and exports `body_qd[:3]` as COM linear velocity.
Targeted CUDA-graph regressions now cover both the raw offset-COM body path and
a FREE-root articulation with rotated parent frame, nonzero COM offset, and
nonzero angular velocity; the latter verifies PhoenX + `eval_ik` round-trips
parent-frame COM `joint_qd`. These checks narrow the remaining G1 walking issue
away from a simple PhoenX body-origin/COM or parent-frame `joint_qd` offset bug.



A 2026-06-22 post-model-parity training probe used the default dense G1 recipe
with pinned nanoG1 body inertials, joint ranges, and effort limits. A fresh
100-iteration run reached about `229k-233k` env samples/s and ended around
`reward=0.084`, rollout `perf=0.609`, `done=0.0035`, but failed the gate:
`battery_perf=0.570`, `battery_falls=20/24000`; the fast-forward command mostly
stood still (`mean_linear_velocity_error=0.802 m/s`). Continuing the same
checkpoint to 400 total iterations ended around `reward=0.092`, rollout
`perf=0.457`, `done=0.0023`, and failed the gate with `battery_perf=0.524`,
`battery_falls=29/24000`. The corrected model contract is necessary, but not
sufficient; the remaining failure still points to training stability and/or
physics-contact parity. The repeated very large value-loss spikes during this
run are the next learner-side discrepancy to isolate.

After disabling the non-nanoG1 default swing-foot-contact penalty and adding
mirror/qpos/bias parity guards, a fresh 100-iteration probe ended around
`reward=0.1005`, rollout `perf=0.618`, `done=0.0027`, and `228k-233k` samples/s.
The gate still failed, but early behavior improved: `battery_perf=0.579`,
`battery_falls=3/24000`, and the fast-forward command error dropped to
`0.707 m/s` from the previous `0.802 m/s`. This supports the parity-cleanup
direction but is not yet a successful walking policy.

The action/actuator interface was rechecked numerically against
`nanoG1/deploy/deploy_g1.py`: home pose, actuator target ranges, leg KP/KD,
`ACTION_SCALE=0.25`, and the 12-leg action mask match exactly. The remaining KD
difference is intentional: the explicit torque path applies Unitree derivative
damping only to the 12 leg actuators and keeps passive DoF damping separate, as
in the nanoG1 host code. Policies output normalized position-target deltas, not
torques and not linear interpolation over the full actuator range.

Network checks did not show a layer-normalization issue. The nanoG1/PufferNet
path is bias-free encoder -> 3x MinGRU -> bias-free decoder with no layer norm.
The cached Newton G1 policy is a plain ELU MLP with an identity normalizer.
PhoenX now exposes `--policy-network` and `--activation` so these can be tested
without source edits.

A small true-objective PBT pilot compared four seeded candidates for 15
iterations at 1024 worlds, using no-reset fixed-command progress scoring rather
than training reward. All candidates still fell in the 150-step evaluator; the
dense default had the least-bad score (`-2.34`), followed by the anti-standing
dense recipe (`-2.47`), conservative exploration (`-2.48`), and sparse-command
(`-2.49`). This short run is not a final ranking, but it confirms that reward
shaping alone is not an instant fix and that candidate recipes should be judged
by held-out displacement, survival, and falls before running full gates.

Fresh probes with the corrected velocity reward still collapse to standing or
low-motion behavior:

- Default dense nanoG1-style recipe, 4096 worlds, 300 iterations / 78.6M samples:
  training `perf` briefly reached `0.62` near iteration 60, then settled near
  `0.47`; the saved checkpoint gated at `battery_perf=0.552` with forward
  `0.8 m/s` perf `0.100` and stand perf `0.995`.
- Sparse-target probe, 180 iterations / 47.2M samples: target success stayed in
  the `1-3%` range and degraded late; it did not discover gait.
- MLP+ELU probe, 180 iterations / 47.2M samples: peaked near `perf=0.65` and
  then regressed like the recurrent policy, so the PufferMinGRU architecture is
  not the obvious blocker.
- Softer torso-penalty/lower-LR probe inspired by the local PufferLib fork
  (`w_ang_vel_xy=-0.05`, `w_orientation=-5`, `lr=0.002`, `gamma=0.995`,
  `gae_lambda=0.95`, `replay_ratio=1`, `entropy=1e-4`) was worse, ending near
  `perf=0.36`.
- Larger action authority (`action_scale=0.5`) was worse, ending near
  `perf=0.42`, so the default `0.25` action scale remains the best parity
  setting.

Open-loop parity now narrows the physics gap further. With the G1 held at
`z=2.0 m` for 20 control steps and nanoG1 using its smooth/no-contact stepper,
PhoenX free dynamics and explicit drives are close: base-position RMSE `3.8 mm`,
joint trajectory RMSE `0.0084 rad`, no contacts, and target-tracking-ratio delta
`0.037`. The same leg-symmetric target pattern from the normal grounded reset is
less close even over 20 steps (`base_pos_rmse=1.5 cm`,
`joint_q_traj_rmse=0.019 rad`, tracking-ratio delta `0.309`), and over 80 steps
the grounded falling trajectory diverges substantially. This makes grounded
contact/constraint response the current primary suspect; free-body dynamics and
explicit actuator math are not showing the same scale of mismatch.

Current interpretation: pure-Warp PPO has enough validation that it is unlikely
to be the only blocker, and the high-level policy/action contracts now look
correct. The strongest remaining hypothesis is a closed-loop PhoenX environment
or physics discrepancy that makes the nanoG1 gait less viable: contact impulse
behavior, tangential support, effective drive response under contact, or done /
bootstrap semantics. The imported teacher remains the best probe because it
walks in nanoG1 but degrades before PhoenX training starts.

## Next Checks

1. Add or tighten command/reset/done-bootstrap tests against the pinned nanoG1
   source so the environment contract is exhausted before reward tuning.
2. Compare grounded drive/contact response against first-principles expectations
   and nanoG1 traces, especially tangential support and effective stiffness at
   the stable `10x8` setting.
3. Use the imported teacher plus low-LR PPO only as an env/physics parity probe;
   do not keep extending it until another discrepancy is fixed.
4. Run longer train-to-gate checkpoints only after a concrete discrepancy is
   fixed, then compare learning curves and gate diagnostics before profiling for
   throughput.
