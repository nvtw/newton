# PhoenX vs MuJoCo-Warp local benchmark

Hand-triggered sweep comparing
[`SolverPhoenX`](../solver.py) to
[`SolverMuJoCo`](../../../mujoco/solver_mujoco.py) on the current
machine's GPU. Designed for quick iteration, so intentionally:

- **Not a proper Newton benchmark.** Lives under the PhoenX solver
  subtree (`newton/_src/solvers/phoenx/benchmarks/`) and is not part
  of CI, not part of `python -m newton.tests`, not part of `asv`.
- **Single-GPU, local-only.** No SLURM, no remote machines.
- **Hand-triggered.** You decide whether to keep the results. Nothing
  is uploaded; `results/` is gitignored.

## Run it

```bash
# Default: g1_flat + h1_flat, num_worlds in {1024, 4096, 16384},
# substeps=4, solver_iter=8, 16 warmup + 64 measure frames.
# Takes about 5-10 min on an RTX PRO 6000.
python -m newton._src.solvers.phoenx.benchmarks.run_benchmarks

# Dylan Turpin's full sweep (capped to 32768 worlds for local runs).
python -m newton._src.solvers.phoenx.benchmarks.run_benchmarks --full-sweep

# Scope it: single scenario, single solver.
python -m newton._src.solvers.phoenx.benchmarks.run_benchmarks \
    --scenarios g1_flat --solvers phoenx --num-worlds 1024 4096

# Skip the GPU-lock reminder prompt (for unattended runs).
python -m newton._src.solvers.phoenx.benchmarks.run_benchmarks --yes

# Append to existing results instead of truncating.
python -m newton._src.solvers.phoenx.benchmarks.run_benchmarks --append
```

Results go to `results/points.jsonl`. One JSON row per
(scenario, solver, num_worlds, substeps, iterations) config.

## Lock your GPU clocks

Without clock locking, expect 2-5x run-to-run variance in
throughput numbers. The launcher prints a (very visible) reminder
and waits for you to hit Enter; if you're on Linux with sudo
access:

```bash
# Check the base clock first
nvidia-smi --query-gpu=clocks.base.graphics --format=csv,noheader,nounits

sudo nvidia-smi -pm 1
sudo nvidia-smi --lock-gpu-clocks=<base_clock_from_above>

# ... run benchmarks ...

sudo nvidia-smi --reset-gpu-clocks
```

Also helps: kill every other GPU process (browsers, IDE,
`nvidia-smi -L`'s compute apps table), and prefer a cold run (or
run a few frames first to warm the thermal envelope).

## View the dashboard

```bash
cd newton/_src/solvers/phoenx/benchmarks
python -m http.server 8000
# open http://localhost:8000/dashboard/
```

Chart.js reads `../results/points.jsonl` directly (relative to the
HTML). The dashboard shows per-scene env_fps, ms/world-step, and
gpu_used_gb plotted against `num_worlds` with PhoenX and MuJoCo
overlaid, plus the raw table for copy-paste.

## Render static PNG plots

For pasting into issues / slides / chat (no HTTP server):

```bash
python -m newton._src.solvers.phoenx.benchmarks.plot_bench
```

Writes one PNG per scenario into `results/plots/`. Uses the same
solver colours as the Chart.js dashboard, log-log axes for
env_fps / ms_per_step, and linear for gpu_used_gb. Needs
`matplotlib` (not in Newton's core deps; install with
`uv add matplotlib` or `pip install matplotlib`).

## Share an artefact

To show someone the results, grab:

- `results/points.jsonl` (the raw data)
- `dashboard/index.html` (the renderer)

Either tar them together, or push both to a branch and serve via
GitHub Pages. Nothing is wired up for you.

## Metrics

Per row:

- **env_fps**: `num_worlds * measure_frames / elapsed_s`. Throughput
  across all worlds. Dylan Turpin's nightly uses the same convention.
- **ms_per_step**: inverse of env_fps, in ms per single-world step.
  Easier to eyeball for regressions.
- **gpu_used_gb**: steady-state GPU memory occupancy (via
  `wp.get_device().total_memory - wp.get_device().free_memory`,
  measured after CUDA-graph capture). Approximate -- other processes
  on the GPU perturb it.
- **setup_gb**: GPU memory the scene allocated during setup (model
  build + solver construction). A proxy for "regression in how
  PhoenX / MuJoCo size internal buffers".

## Measurement loop

```
WARMUP   ─ N frames eagerly stepped, wp.synchronize_device after
CAPTURE  ─ one frame recorded into a CUDA graph
MEASURE  ─ M * wp.capture_launch, one wp.synchronize at the end
         ─ wall-clock around the M replays
```

16 warmup + 64 measure (Dylan's defaults). JIT compilation, the
contact sorter's lazy buffers, and any first-call allocations all
land in the warmup; measurement is just graph replays. This is
exactly the same shape as Dylan's nightly harness plus what the ASV
`bench_mujoco.FastBenchmark` does.

## Scenarios

- **`g1_flat`**: headless clone of `newton.examples.robot_g1`. 29-DoF
  humanoid on a ground plane, PD-position control,
  bounding-box-approximated mesh colliders.
- **`h1_flat`**: headless clone of `newton.examples.robot_h1`.
  Taller humanoid, similar setup.

To add a scene, drop a new module under `scenarios/` that exposes a
`build(num_worlds, solver_name, substeps, solver_iterations)`
returning a `runner.SceneHandle`, then register it in
`scenarios/__init__.py::SCENARIOS`.

## FeatherPGS?

Dylan Turpin's nightly compares FeatherPGS to MjWarp. FeatherPGS is
not in Newton's `main`; it lives on his `feather_pgs` branch. If you
want to include it here, check that branch out into a sibling
worktree and point the benchmark at it via `PYTHONPATH`. Not wired
up by default.
