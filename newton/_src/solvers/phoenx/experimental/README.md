# PhoenX Experimental Tools

This directory is for PhoenX experiments that are useful during research but
are not part of the stable solver, RL API, examples, or benchmarks. Keep risky
PufferLib comparisons, profiling sketches, and throwaway research studies here until they either
graduate into reusable PhoenX modules/examples or are removed.

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

Scene-specific G1, Anymal, and Ant training entrypoints live under
`newton._src.solvers.phoenx.rl_training.examples`. Keep this directory for
research probes and local comparisons that are not ready as reusable examples.

The useful imitation-learning direction from HumanCompatibleAI/imitation is a
behavior-cloning warm start from demonstrations or a teacher policy, followed by
the existing PPO fine-tune. GAIL/AIRL/preference learning are plausible later,
but they would add reward-model/discriminator complexity and should not enter
the default PhoenX RL path until a concrete benefit is measured.
