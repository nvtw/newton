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
- No-reset long-horizon evaluation exposed a finite numerical-explosion failure
  mode: a partially trained forward policy could reach enormous root positions
  before the height/upright fall test fired. PhoenX G1 now treats generous root
  position/velocity and joint position/velocity limits as environment
  invariants. These limits are not walking reward shaping; they terminate
  physically invalid simulation states so PPO cannot learn from corrupted
  trajectories.
- G1 foot-contact rewards now require accumulated normal contact impulse above
  a small threshold instead of using shape-pair presence alone. This avoids
  treating speculative/gap contacts as stance contacts for gait phase,
  feet-air-time, and foot-slide rewards. A CUDA graph regression creates foot
  shape pairs with zero impulse and verifies they do not count as active
  contacts.
- The Python gate diagnostic now rotates root linear velocity with Newton's
  free-joint quaternion layout, `x, y, z, w`. The previous helper interpreted
  the same four numbers as `w, x, y, z`, corrupting the printed
  `mean_linear_velocity_error` diagnostics. The actual gate pass/fail metric was
  already computed on the device from `step_successes`, so this was a diagnostic
  bug, not a training-quality fix.

## IsaacLab / PhysX Comparison

A 2026-06-23 IsaacLab pass checked the manager-based G1 velocity locomotion
recipe under PhysX because PhysX and PhoenX are both PGS-style rigid-contact
solvers. The relevant IsaacLab G1 task uses affine joint-position actions,
`default_joint + 0.5 * action`, not full joint-range mapping. Its flat-task
reward is velocity tracking plus yaw tracking, flat-orientation, action-rate,
joint-acceleration/torque penalties, positive biped feet-air-time, foot-slide,
ankle joint-limit, and joint-deviation penalties for hip yaw/roll, torso, arms,
and fingers. The IsaacLab PhoenX preset for this task uses 4 internal PhoenX
substeps, 8 PGS iterations, `velocity_readout="substep_end"`, and CUDA graphs.
IsaacLab currently disables the contact-force sensor for its PhoenX preset, so
its feet-air-time and feet-slide terms are dropped there.

PhoenX RL now has default-off, graph-captured equivalents for the reusable
IsaacLab terms that fit the current G1 environment: positive biped feet-air-time,
contact foot-slide, hip yaw/roll deviation, waist deviation, upper-body
deviation, leg joint acceleration, and ankle position-limit violation. Contact
terms reuse the existing G1 foot-contact scan, maintain preallocated
`(world, foot)` air/contact-time buffers, guard repeated `observe()` calls with a
per-world episode-step stamp, clear on graph reset, and are covered by CUDA graph
tests. Joint acceleration uses preallocated `(world, action)` previous-velocity
state plus a per-world episode-step stamp, so repeated graph-captured
`observe()` calls are idempotent. Public knobs are `--w-feet-air-time`,
`--feet-air-time-threshold`, `--w-feet-slide`, `--w-joint-deviation-hip`,
`--w-joint-deviation-waist`, `--w-joint-deviation-upper`,
`--w-joint-acc-legs`, and `--w-joint-pos-limit-ankle`.

Quality result: these terms are useful infrastructure but not the missing
walking lever by themselves. Replacing the nanoG1 phase-gait reward with a close
IsaacLab-style velocity/contact profile reached only `battery_perf=0.353` at
120 iterations and failed badly. Adding smaller IsaacLab contact terms on top of
the previous anti-standing nanoG1-style run reached `battery_perf=0.608` on the
standard 1000-step gate at 120 iterations (`162/24000` falls), roughly the same
plateau as the prior long anti-standing run. A 2026-06-23 full-body-action probe
with `controlled_action_count=29`, `action_scale=0.5`, and joint/contact
regularizers reached only reduced-gate `battery_perf=0.556` at 120 iterations
(`28/3600` falls). Keeping the nanoG1 12-leg action space with the same
regularizers reached reduced-gate `battery_perf=0.523` at 120 iterations
(`50/3600` falls). The current evidence says full-body IsaacLab-style actions and
these regularizer weights add exploration burden or over-regularize before they
solve walking. The next likely gap is still remaining physics/env mismatch,
reward-command curriculum behavior, or a teacher/open-loop parity issue, not the
absence of these reward terms.

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
| nanoG1 policy import | Shipped `assets/nanoG1.bin` plus `deploy/nanog1_policy.c` | `nanog1_import.py` converts the PufferNet binary into a PhoenX PPO checkpoint; imported Warp output matches the nanoG1 C shim to `1.2e-7` on a zero-observation smoke. On current defaults the imported policy reaches about `battery_perf=0.735` on the full PhoenX gate and fails, so a known-good policy still degrades under PhoenX. Low-LR PhoenX PPO fine-tuning improves this only slightly to `0.747`. | Use the imported policy and its small fine-tune gain as the primary physics/env parity probe; do not expect scalar reward tweaks alone to close the gap. |
| Gate diagnostics | nanoG1-style command battery | Fixed Python velocity diagnostic quaternion order to Newton `xyzw`; graph-captured regression covers the case. | Re-run gates only when diagnostics are needed; do not treat this as quality progress. |
| Graph overlap / stale policy | Same recipe in eager and graph-leapfrog modes | A 60-iteration eager train-to-gate probe failed similarly to graph mode, so stream overlap/stale rollout policy is not the primary quality blocker. | Keep graph mode for throughput, but debug quality in the simpler eager path when possible. |
| G1 env/reward contract | `g1_gpu.cu`, `recipe.py`, deploy constants | Tests cover observation/action layout, actuator-force torque penalty, reward decomposition, gait/success terms, command constants, and graph recurrent reset behavior. The default now resets recurrent state at PPO rollout boundaries because the Warp buffer does not yet store initial hidden states for update-time replay. | Add targeted tests for command resampling/reset timing and done/bootstrap semantics before changing rewards. |
| Drive/contact physics | nanoG1 host physics plus first-principles drive tests | Explicit G1 actuator forces now match the analytical nanoG1/MuJoCo formula joint-by-joint; the old implicit drive-row path was effectively too compliant at training dt. G1 foot-ground friction now matches nanoG1 pair friction (`0.6`): the previous PhoenX ground `0.75` averaged with foot `0.6` to an unintended effective `0.675`. | Keep explicit torque as the default; use `constraint_drive` only for diagnostics, and re-run policy gates/open-loop parity after the friction fix. |
| End-to-end training quality | nanoG1 reaches a working policy in roughly the README time scale | Velocity-command PhoenX runs still fail the nanoG1 gate, but dense target-conditioned PhoenX RL now trains a conservative G1 walking policy that reaches forward targets up to 2 m without falls in the target evaluator. | Extend the target horizon and then bridge back to sustained velocity-command walking; record before/after quality and throughput. |

## 2026-06-23 Target-Conditioned Walking

A clean target-conditioned G1 mode now exists as reward_mode=dense_target.
The policy observation reuses the existing target path: the first two command
slots carry the target XY vector in the robot body frame, while the third slot
remains the yaw-rate command. The reward combines target-directed progress and
target success with the same locomotion stability, gait/contact, action, torque,
power, and joint regularizers used elsewhere. This is different from the older
sparse_target mode, which was too sparse and repeatedly collapsed back to a
low-motion strategy.

The best current checkpoint is /tmp/phoenx_g1_dense_target_forwardcone_550.npz.
It was warm-started from the nanoG1 import/forward fine-tune path and trained
with a forward-cone target curriculum, PhoenX physics, and PhoenX PPO. Target
evaluation with ground_friction=0.4, radius 0.35 m, and 64 worlds reaches
0.6 m, 1.0 m, 1.4 m, 1.6 m, and 2.0 m forward targets with
success_fraction=1.0, fall_fraction=0.0, no height violations, and no tilt
violations. The 2.0 m case reaches success at about 161 policy steps on average,
with mean path length about 1.74 m and mean max tilt about 9 degrees. The same
policy does not yet solve a 3.0 m target: it falls in about 98% of worlds.

This is the first working end-to-end PhoenX + PhoenX RL G1 walking result, but
it is conservative target walking rather than a fast velocity-tracking gait. The
next quality step is extending the target horizon beyond 2 m or converting this
target-conditioned policy into a sustained velocity-command policy without
reintroducing the standing/stalling attractor.

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

A 2026-06-23 imported-teacher fine-tune probe used the verified nanoG1
checkpoint under PhoenX physics and the stable default foot-box contacts. On
current code, the freshly imported policy reached the standard gate at
`battery_perf=0.735`, `battery_falls=94/24000` and a reduced 300-step gate at
`0.751`, `12/3600`. Conservative PhoenX PPO fine-tuning with
`actor_lr=critic_lr=1e-4`, `train_epochs=1`, and `replay_ratio=1.0` improved the
reduced gate to `0.782`, `9/3600` at 60 iterations and the standard gate to
`0.747`, `90/24000`. Continuing to 120 iterations regressed on the reduced gate
(`0.773`, `10/3600`), while a `2e-4` learning rate, full-command curriculum, and
anti-standing reward variant were all weaker. A small material sweep on the
imported teacher gave reduced-gate `0.760` at ground friction `0.4`, `0.759` at
`0.5`, and `0.748` at `0.7`; fine-tuning at `0.4` reached only `0.777`, below
the default-friction fine-tune. Treat this as useful evidence but not a solved
walking recipe: PhoenX RL can adapt a known-good teacher slightly, then plateaus
well below the `0.90` gate. The remaining gap is likely an environment/physics
or teacher-distribution mismatch that PPO fine-tuning and simple material
changes do not remove.

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

Reward shaping may still be needed to make PhoenX policies prefer walking over
standing, but the current evidence argues against treating it as the only
blocker. A verified nanoG1 teacher policy and a PufferLib-style learner both
degrade under PhoenX before any new reward recipe can help, while open-loop
grounded traces show a large foot tangential-support mismatch. New reward or
PBT experiments should therefore be ranked by held-out fixed-command progress
and gate metrics, and should run only after each concrete physics/env mismatch
is recorded or ruled out.

A short fixed-command evaluator confirms that this objective is not blind to
walking-like behavior. At the 150-step `0.8 m/s` forward command, the imported
nanoG1 teacher under PhoenX reached `1.92 m` aligned displacement, `0.64 m/s`
mean aligned velocity, and `0.58` tracking perf before falling in some worlds.
A recent from-scratch PhoenX checkpoint moved backward on average (`-0.81 m`,
`-0.27 m/s`) with only `0.02` tracking perf. The objective therefore ranks the
more walking-like policy higher; the current blocker is producing that behavior
from scratch and keeping it stable under PhoenX physics.

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

The newest open-loop isolation keeps that hypothesis alive. With current stable
settings, PhoenX and nanoG1 agree on early foot contact counts and normal
support, but PhoenX produces much larger tangential foot-ground support even for
zero-action standing. The gap persists with PhoenX velocity relaxation disabled
and when nanoG1 is run at the same 10 x 0.002 s substep timing, so the next
root-cause work should inspect tangential contact projection/convergence rather
than adding more dense walking terms first.

The G1 foot/plane friction default is `0.6`, matching nanoG1's authored
foot-floor pair coefficient and staying in a plausible real-world range for
rubber-like feet on common floors. A lower `0.4` coefficient is also physically
plausible for less grippy contacts and is now exposed as a CLI sweep knob, but
it should be treated as a material/robustness experiment rather than a hidden
fix. In the current probes, `mu=0.4` reduced PhoenX tangent support and improved
the imported-teacher full gate from roughly `battery_perf=0.70` to `0.735`. A
PhoenX PPO fine-tune of the imported teacher at `mu=0.4` reached the best full
gate so far, `battery_perf=0.754` with `75/24000` battery falls, but it still
does not pass. From-scratch runs with the same material and extra shaping still
fall too often, so the current reliable path is teacher warm-start plus PhoenX
fine-tuning, not yet one-shot from-scratch training.

A new experimental `dense_sparse_command` reward mode keeps the dense nanoG1
tracking/stability terms and adds a boolean command-success bonus. CUDA graph
unit coverage verifies the bonus and keeps the original sparse-command mode
unchanged. The first short forward-only shaped run did not solve walking; it
reached only about `0.16` tracking perf on a 150-step forward evaluator after
120 iterations. A lower-exploration from-scratch run (`log_std_init=-0.7`,
`action_scale=0.18`, `mu=0.4`) was also not viable under held-out evaluation:
its saved checkpoints still fell in every 150-step forward world and topped out
near `0.19` tracking perf. Keep this mode available for compact reward/PBT
sweeps, but do not treat it as the current solution.

Two follow-up warm-start fine-tunes also failed to beat the best dense `mu=0.4`
teacher run. Adding the sparse command bonus to teacher fine-tuning reached only
`battery_perf=0.766` on the cheap 4800-sample gate. Raising yaw tracking weight
to `4.0` reduced yaw-rate RMS on the cheap gate (`0.67-0.70` versus roughly
`0.72`), but overall cheap-gate battery perf stayed below the previous best
(`0.760` versus `0.774`).

`w_command_progress` is now available as an explicit PhoenX-only bootstrap term
that rewards the command-aligned body-frame velocity projection. Its default is
zero, so the nanoG1 parity recipe is unchanged, and CUDA graph unit coverage
pins the scalar contribution when it is enabled. A short `mu=0.4` teacher
fine-tune with `reward_mode=dense_sparse_command`, `w_command_progress=2.0`,
and `w_sparse_command_success=2.0` was still flat: the cheap 4800-sample gate
reached about `battery_perf=0.785` and a 300-step `0.5 m/s` forward evaluator
still fell in roughly half the worlds. The next high-value work is therefore not
more scalar reward-weight nudging; it is either a stronger warm-start/imitation
stage, a proper curriculum/PBT pass that selects on held-out gate metrics, or
fixing the remaining contact/actuation mismatch.

## IsaacLab / PhysX G1 Check

The local IsaacLab G1 flat/rough velocity tasks are useful as a PhysX-near
reference, but they are not a drop-in nanoG1 replacement. The main flat G1
config uses PhysX by default and a PhoenX preset of `substeps=4`,
`solver_iterations=8`, `velocity_iterations=1`, and
`velocity_readout="substep_end"`. Its G1 asset uses 8/4 PhysX articulation
iterations, default-position actions with `scale=0.5`, ELU MLP PPO, and
4096 envs with 24 steps per rollout. The reward stack includes velocity
tracking, foot air-time, foot slide, ankle limit, joint-deviation, torque,
acceleration, and action-rate terms; on PhoenX inside IsaacLab, contact-sensor
terms are disabled there because the manager backend lacks a contact sensor.

Two direct PhoenX probes did not improve learning. An IsaacLab-style
all-29-joint action/reward/PPO probe with the existing nanoG1 observation
reached only `battery_perf=0.324` on the cheap 3600-sample gate after 80
iterations and fell `61/3600` samples. Adding an optional
`observation_mode="isaaclab_flat"` layout with body-frame base linear velocity
(no nanoG1 phase clock) made the same probe worse: `battery_perf=0.255`,
`90/3600` falls, high action clipping, and unstable value losses. This does
not disprove IsaacLab's recipe in its native manager/PhysX stack; it means that
blindly mixing IsaacLab observations/PPO with the current nanoG1-derived
PhoenX environment is not the current path to a walking policy.

Keep the optional IsaacLab-flat observation mode as an isolated experimental
knob with CUDA graph coverage. The practical takeaways for PhoenX remain: use
stable 8-iteration solver defaults, treat `default + scale * action` as a
valid SOTA action convention, keep contact/lift/slide rewards available for
controlled sweeps, and focus root-cause work on grounded contact/actuation
behavior rather than more unstructured reward transplants.

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

## Auto-Reset Metric Trap

A later diagnostic refined the contact interpretation above. The open-loop
``support_tangent`` metric sums per-contact tangent magnitudes. A direct
per-foot net wrench gather on the same zero-action standing trace showed much
smaller horizontal net forces, typically single-digit Newtons after the first
few policy steps while each foot carried roughly `130-225 N` vertically. The
large summed tangent metric is therefore mostly opposing shear inside each foot
contact patch. It is still useful as a convergence/friction diagnostic, but it
is not by itself evidence that PhoenX applies a large net horizontal ground
force to the robot.

From-scratch PPO probes exposed a more immediate RL failure mode. A
movement-biased run improved the auto-reset training/gate metric
(`battery_perf` rose from about `0.53` at 80 iterations to about `0.68` at 400
iterations), but no-reset forward evaluation showed the policy was only making
a short burst and then falling after about 52 policy steps. The auto-reset gate
counted the falls, but its average tracking metric was still too forgiving
because each reset produced another short burst.

A survival-biased continuation with a much stronger terminal penalty improved
no-reset mean survival from about `52` to about `578` steps on a 1000-step
forward evaluator, but tracking collapsed toward near-standing behavior. This
means the current blocker is not simply ignored terminal rewards. We need
checkpoint selection on held-out no-reset survival plus tracking, and a staged
command curriculum that preserves movement while increasing fall cost. Treat
auto-reset rollout `perf` as a training diagnostic only, not as proof of a
working walking policy.

CUDA graph RNG was audited at the same time. Rollout action sampling, command
sampling, reset noise, sparse target sampling, and PPO replay sampling all use
device seed counters in graph-captured code. Rollout collection advances the
action seed counter by `rollout_steps`, reset/command counters advance inside
the captured environment step path, and PPO replay advances its update counter
by `1_000_003` per update. Resumed chunks initialize counters from the saved
trainer iteration. A fresh run with the same `--seed` intentionally repeats the
same experiment; independent sweeps should pass different seeds. A CUDA graph
regression now verifies that repeated launches of the same captured PPO rollout
graph produce different stochastic actions and advance the seed counter.

## 2026-06-23 Phase-Gated From-Scratch Curricula

The experimental target curriculum runner now saves and evaluates every phase
checkpoint before chaining to the next phase. This exposed three concrete
failure modes that were previously easy to miss:

- Dense target progress was paid while the torso was outside the upright gate,
  so a falling forward lunge could get shaped progress reward. Target progress
  is now multiplied by the existing upright gate in sparse_target and
  dense_target modes, with a CUDA graph regression test.
- Resumed target phases initialized the target-distance curriculum counter from
  the saved PPO iteration. That made later phases begin near their endpoint
  distance instead of their phase start. `ConfigTrainG1PPO` now supports an
  explicit `target_curriculum_start_samples`; the curriculum runner resets it
  to zero for each phase while keeping policy weights.
- Later target phases trained only one distance per rollout. The G1 target
  sampler now has a graph-safe distance-range mode that samples per-world target
  distances from `[phase_start, current_distance]`, also covered by a CUDA graph
  test.

Measured result after these fixes: phase 0 from scratch reaches the 0.6 m target
with strict_success=1.0, fall_fraction=0.0, and max tilt below 11 degrees. The
next one-meter target phase still fails its gate: the best measured run kept
0.6 m strict_success=1.0 but had 1.0 m strict_success=0.0 and fall_fraction=1.0.
This says the phase mechanics are now honest, but the current target objective
still induces a lunge before it induces a sustained gait.

A separate velocity-command curriculum runner was added to train nanoG1-like
sustained walking before target steering. The first simple-forward run from
scratch (`0.15..0.45 m/s`, 220 PPO iterations) reached rollout tracking perf
around 0.69 but failed no-reset evaluation at the 0.3 m/s gate: fall_fraction=1.0,
mean_survival_steps=89.98/700, and mean_tracking_perf=0.096. This confirms that
short-horizon rollout reward is not enough; command phases must be judged by
no-reset survival/tracking gates before they are chained.

Command-tracking reward and sparse command success are now also multiplied by
the existing upright gate. A repeated simple-forward probe still failed the
no-reset command gate with fall_fraction=1.0, mean survival about 89/700 steps,
and tracking_perf=0.091. That narrows the issue: rewarding command tracking
while falling was wrong and is now regression-tested, but it was not the main
survival bottleneck.

The command runner now exposes `--reward-clip` and defaults to `4.0` so the
configured `w_termination=-4` is not clipped to `-1` during advantage
computation. The matched simple-forward probe with `reward_clip=4.0` still
failed the no-reset gate: fall_fraction=1.0, survival_fraction=0.154, and
tracking_perf=0.105. Preserving the terminal penalty is semantically cleaner,
but it is not the root cause of the G1 walking failure.

## 2026-06-23 Ant PPO Validation

An experimental `train_ant_phoenx_ppo.py` runner now trains the classic Ant
locomotion task using the same Warp PPO implementation and SolverPhoenX. It
parses `nv_ant.xml`, uses the existing Y-up Newton Ant convention, drives the
eight hinge DOFs with direct clamped torques, and evaluates checkpoints without
auto-reset. The runner includes an `--eval-only` mode for checkpoint scoring and
a CUDA graph regression test for the Ant step path.

Measured validation run (`2048` worlds, `64` rollout steps, ELU MLP
`128-64-32`, `log_std_init=-1.0`, `400` PPO iterations) trained a slow but real
locomotion policy. Rollout done rate dropped from `0.5245` at iteration 0 to
`0.0039` at iteration 399, and rollout reward rose from `-0.463` to `1.037` at
about `330k` samples/s after warmup. Deterministic no-reset eval over `300`
steps had fall_fraction=`0.234`, mean_survival_steps=`240.9/300`, alive-only
mean_forward_velocity=`0.167 m/s`, and mean_displacement_x=`0.81 m`.

This does not prove the G1 task is solved, but it is strong evidence that the
Warp PPO path can learn a contact-rich articulated locomotion policy with
PhoenX physics. The remaining G1 blocker is therefore more likely in the G1
robot/task/physics interface, curriculum, observations, or actuation details
than in a basic PPO implementation failure.

## 2026-06-23 Anymal Walk PPO Runner

A separate experimental `train_anymal_walk_phoenx_ppo.py` runner trains
Anymal C command walking with SolverPhoenX and the same Warp-only PPO stack. It
supports a single fixed command and a phased `--recipe forward` curriculum that
checkpoints between phases, evaluates without auto-reset, and gates on actual
walking quality: fall fraction, survival, body-forward velocity, velocity error,
and command-aligned tracking. The corresponding Anymal RL unit test captures
PPO action selection plus a PhoenX step in a CUDA graph, and now also verifies
that a stationary Anymal is not counted as successful for a nonzero velocity
command.

The first metric bug found here was that dense-command `successes` still
reported sparse target proximity. That made standing look good in logs. Dense
command mode now reports `velocity_tracking * yaw_tracking * upright *
command_aligned_speed_fraction`, while pure velocity and yaw tracking remain
separate eval metrics. Dense command rewards also include the configured fall
and energy penalties; those fields previously existed but only affected sparse
target mode.

A measured from-scratch forward-walking run used `512` worlds, `32` rollout
steps, ELU MLP `128-128-128`, log standard deviation `0.0`, Adam LR `1e-3`,
entropy `0.005`, five PPO epochs, and stable PhoenX defaults (`4` substeps,
`8` solver iterations). The run was chained manually through the same phase
checkpoints that `--recipe forward` automates:

| Phase | Command | Iterations | No-reset eval result |
| --- | ---: | ---: | --- |
| `warmup_forward` | `0.35 m/s` | `120` | pass; fall_fraction=`0.0`, vx=`0.393 m/s`, `|vx-cmd|=0.197`, quality=`0.432` |
| `walk_forward` | `0.65 m/s` | `110` | pass; fall_fraction=`0.0`, vx=`0.718 m/s`, `|vx-cmd|=0.131`, quality=`0.569` |
| `fast_efficient_forward` | `0.90 m/s` | `260` | pass; fall_fraction=`0.0`, vx=`0.914 m/s`, `|vx-cmd|=0.176`, quality=`0.657` |
| `disturbed_forward` | `0.90 m/s` | `180` | robustness phase with stochastic jitter and rare kicks; smoke-probed from checkpoint at `18` iterations with fall_fraction=`0.0`, vx=`0.993 m/s`, quality=`0.688` |

The final checkpoint from that probe is
`/tmp/phoenx_anymal_forward_probe_progress/checkpoint_02_fast_efficient_forward_490.npz`.
It is not a public artifact, but it proves the current PhoenX physics plus
PhoenX PPO stack can train a stable Anymal forward walker from scratch. The
recipe now also includes `disturbed_forward`, which continues from the fast
walker and applies small stochastic root-velocity jitter plus rare Bernoulli
kick impulses after a warmup (`p=0.003` per policy step, max `0.45 m/s` XY and
`0.35 rad/s` yaw). This follows the same principle as IsaacLab/RSL-style random
push events while remaining fully graph-capturable in Warp. Remaining gaps
before calling the task robust are omnidirectional command randomization,
longer no-reset eval horizons, stronger kick sweeps, and a cleaner
foot-contact/air-time term if we want IsaacLab-style gait regularization.

This also exposed a reusable-buffer bug in PPO: after reserving update-sized
network buffers, `act_reuse()` returned the full reserved action/value arrays
instead of views over the active observation batch. Small debug runs could then
produce `1024 x action_dim` actions for a smaller environment. `WarpMLP`,
`PufferMinGRUNet`, and `GaussianActor` now return active-batch views, with a
CUDA graph regression test that reserves eight rows and acts on a two-row batch.

The G1 lifted-drive regression now also covers all 29 joints. With the robot
lifted out of contact and a `0.05 rad` diagonal action command, the first 12
controlled leg joints reach more than `0.90x` of target within 25 policy steps
at the current default solver settings, and high-gain joints stay within
`0.80..1.20x`. The deliberately low-gain upper-body joints are much slower
(`0.18..0.40x`), matching their configured gains rather than indicating a gross
leg-drive responsiveness bug.

