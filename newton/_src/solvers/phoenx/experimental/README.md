# PhoenX Experimental Tools

This directory is for PhoenX experiments that are useful during research but
are not part of the stable solver, RL API, examples, or benchmarks. Keep risky
PufferLib comparisons, prototype trainers, profiling sketches, and throwaway
training studies here until they either graduate into reusable PhoenX modules or
are removed.

Code here should still be readable and reproducible, but it may depend on local
checkouts or temporary assumptions. Document those assumptions at the top of the
file that needs them.

## G1 Training Search

bench_g1_pbt.py is a pure-Warp outer-loop probe inspired by Sample Factory PBT
design. It trains normal PhoenX PPO checkpoints, ranks them with the G1 quality
gate and/or fixed-command no-reset progress metrics, and mutates a small set of
PPO/reward/exploration knobs. The seeded candidates cover the current dense
default, the best anti-standing dense probe, conservative exploration, and
sparse-command reward. Keep this as an experimental research tool: the stable
user path should remain newton.rl.train_g1_ppo plus the single G1 recipe file.

nanog1_import.py imports the shipped nanoG1 PufferNet binary into a normal
PhoenX PPO checkpoint. This gives us a PyTorch-free teacher or warm-start policy
for imitation experiments while keeping the default PPO trainer unchanged.

train_g1_curriculum.py runs repeatable dense-target G1 curricula through the
normal PhoenX PPO API. `simple-target` is a short from-scratch probe and is not
expected to produce a perfect policy. `advanced-target` stages the task from
short forward targets to longer target-conditioned walking and guards against
the degenerate case where the initial target is already inside the sparse
success radius. The runner saves a normal PPO checkpoint and evaluates target
walking after every phase; by default it stops before chaining to the next phase
when the current phase fails its strict target success/fall/tilt gate. Each
phase resets its target-distance curriculum counter while preserving policy
weights, and samples per-world target distances across the current phase range
so later phases do not train only on the endpoint distance. Run it with:

```
uv run --extra dev -m newton._src.solvers.phoenx.experimental.train_g1_curriculum \
    --recipe advanced-target --output-dir /tmp/phoenx_g1_advanced --device cuda:0
```

To debug one phase at a time, resume from the previous phase checkpoint:

```
uv run --extra dev -m newton._src.solvers.phoenx.experimental.train_g1_curriculum \
    --recipe advanced-target --start-phase 1 --phase-count 1 \
    --resume-checkpoint /tmp/phoenx_g1_advanced/00_short_forward_targets_140.npz \
    --output-dir /tmp/phoenx_g1_advanced_phase1 --device cuda:0
```

train_g1_command_curriculum.py trains sustained velocity-command walking before
target steering. This is closer to the nanoG1 task shape and gives each phase a
no-reset command-following gate based on fall rate, survival, and tracking
performance. Run the simple forward-walking probe with:

```
uv run --extra dev -m newton._src.solvers.phoenx.experimental.train_g1_command_curriculum \
    --recipe simple-forward --output-dir /tmp/phoenx_g1_command_simple --device cuda:0
```

The useful imitation-learning direction from HumanCompatibleAI/imitation is a
behavior-cloning warm start from demonstrations or a teacher policy, followed by
the existing PPO fine-tune. GAIL/AIRL/preference learning are plausible later,
but they would add reward-model/discriminator complexity and should not enter
the default PhoenX RL path until a concrete benefit is measured.
