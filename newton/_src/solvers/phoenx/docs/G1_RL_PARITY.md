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
| Drive/contact physics | nanoG1 host physics plus first-principles drive tests | Explicit G1 actuator forces now match the analytical nanoG1/MuJoCo formula joint-by-joint; the old implicit drive-row path was effectively too compliant at training dt. Grounded contact/friction coupling still differs. | Keep explicit torque as the default; use `constraint_drive` only for diagnostics, and continue contact/friction convergence checks at `8x4`. |
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

## Next Checks

1. Add or tighten command/reset/done-bootstrap tests against the pinned nanoG1
   source so the environment contract is exhausted before reward tuning.
2. Compare grounded drive/contact response against first-principles expectations
   and nanoG1 traces, especially tangential support and effective stiffness at
   the production `5x2` setting.
3. Run longer train-to-gate checkpoints only after a concrete discrepancy is
   fixed, then compare learning curves and gate diagnostics before profiling for
   throughput.
