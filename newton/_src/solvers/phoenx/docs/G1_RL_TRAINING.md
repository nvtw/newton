# PhoenX G1 RL Training Notes

This is the current PhoenX path for reproducing nanoG1-style G1 training without
adding a PyTorch dependency to Newton's Warp-only RL stack.

## Entry Points

- `newton.rl.EnvG1PhoenX`: vectorized Unitree G1 environment backed by
  `SolverPhoenX`.
- `newton.rl.train_g1_ppo`: Warp-only PPO training loop using the reusable
  `collect_ppo_rollout` helper.
- `newton.rl.capture_env_steps`: reusable CUDA graph-capture helper for vectorized
  environments.
- `python -m newton.rl train-g1-ppo`: CLI wrapper for quick runs.
- `python -m newton._src.solvers.phoenx.benchmarks.bench_g1_rl`: PhoenX G1
  env-step throughput benchmark with optional nanoG1 result ingestion.

## nanoG1 Parity Choices

The environment follows nanoG1's G1 v3 control surface where practical:

- MJCF asset: `unitree_g1/mjcf/g1_29dof.xml` with 36 coordinates, 35 DOFs,
  and 29 actions.
- Observation size: 98.
- Action size: 29.
- Control frame dt: 0.02 s.
- Physics dt: 0.004 s via `sim_substeps=5`.
- Primitive-only collision by default (`parse_meshes=False`), preserving the
  same G1 coordinates/DOFs/actions while avoiding generic mesh/SDF contact
  overflow in high-world-count RL runs.
- Leading 12 leg actions are controlled by default; remaining upper-body actions
  are masked to zero.
- Reward weights mirror the nanoG1 recipe for velocity tracking, base penalties,
  torque proxy, action-rate penalty, alive reward, and termination penalty.

## Current Benchmark Baseline

Measured on an NVIDIA RTX PRO 6000 Blackwell Workstation Edition with CUDA graph
replay and no learner in the measured loop:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_rl \
    --world-count 4096 --measure-replays 16 --warmup-steps 4
```

Result from this checkpoint:

- 763,732 env steps/s at 4096 worlds.
- 3,818,658 physics steps/s at 4096 worlds.
- 9.8 s setup time at 4096 worlds.

nanoG1 reports about 8.5M production physics steps/s and 7.25M matched physics
steps/s in its README/benchmark notes, so this PhoenX path is currently about
2.2x slower than nanoG1 production and about 1.9x slower than nanoG1 matched on
raw no-learner stepping. The benchmark can invoke the local nanoG1 checkout when
cloud execution is intended:

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_g1_rl --run-nanog1
```

That command shells out to:

```bash
cd /home/twidmer/Documents/git/nanoG1 && modal run bench/bench_nanog1.py --config production
```

## Known Gaps To nanoG1

- nanoG1 compiles a G1-specific CUDA engine through its pinned PufferLib fork;
  PhoenX currently runs through Newton's generic model, collision, broadphase,
  contact reduction, graph coloring, and solver paths.
- Full mesh collision is available with `parse_meshes=True`, but high-world-count
  RL runs should use primitive collision until the mesh/SDF reducer path is sized
  and optimized for thousands of humanoid worlds.
- nanoG1's Modal image installs `torch`, so using the same full PufferLib training
  stack is not currently a torch-free route even though the environment core is
  specialized CUDA.
- The Warp-only PPO loop is reusable but intentionally simple: it does not yet
  include nanoG1's V-trace/replay setup, Muon optimizer path, mirror loss, or
  PufferNet model stack.
- Environment stepping can be CUDA-graph replayed, but the full collect-policy-
  update loop is not captured end to end because actions and Warp Tape updates
  allocate intermediate arrays per rollout/update.
- Domain randomization and symmetry regularization still need to be added before
  sample efficiency can be compared meaningfully.

## Next Optimization Targets

1. Avoid generic replicated MJCF setup for high world counts; build or cache a
   compact fixed-topology G1 multi-world model path.
2. Remove avoidable broadphase/contact work for independent flat-ground G1 worlds.
3. Add mirror loss and nanoG1-style command/replay scheduling to Warp PPO.
4. Add a PufferLib interop path behind an optional dependency boundary if we want
   exact nanoG1 trainer compatibility.
