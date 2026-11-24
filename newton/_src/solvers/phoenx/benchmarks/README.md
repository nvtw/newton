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

## Resting-container jitter ladder

The asset-free diagnostic in
[`experimental/bench_container_jitter.py`](experimental/bench_container_jitter.py)
provides reusable scenes for contact-solver comparisons:

| Case | Support | Contents |
|---|---|---|
| `fixed_1` | Fixed flat box | One cylinder |
| `plate_1` | Free 0.42 kg flat box | One cylinder |
| `plate_4` | Free 0.42 kg flat box | Four cylinders |
| `plate_12` | Free 0.42 kg flat box | Twelve cylinders |
| `box_12` | Free 0.42 kg box shell | Twelve cylinders |

Every cylinder weighs 0.57 kg. These controlled proxies require no private USD
assets. Support inertia follows its geometry. The shell has four walls and a
flat bottom; it does not reproduce the source factory's stepped carton floor.
`--support-mass` varies free-support mass while keeping geometry fixed and
scaling its inertia consistently. The default is 0.42 kg; fixed supports remain
static. Mass changes are diagnostic controls, not fixes for the factory scene.

```bash
uv run --no-sync -m newton._src.solvers.phoenx.benchmarks.experimental.bench_container_jitter \
    --output /tmp/container-jitter.json

uv run --no-sync -m newton._src.solvers.phoenx.benchmarks.experimental.bench_container_jitter \
    --case box_12 --iterations 64 --output /tmp/container-jitter-reference.json

# 120 Hz collision detection, 480 Hz physics:
uv run --no-sync -m newton._src.solvers.phoenx.benchmarks.experimental.bench_container_jitter \
    --case box_12 --substeps 2 --solver-substeps 4 --output /tmp/container-jitter-substeps.json

# Isolate a single heavy cylinder on a light support:
uv run --no-sync -m newton._src.solvers.phoenx.benchmarks.experimental.bench_container_jitter \
    --case plate_1 --support-mass 0.035 --output /tmp/container-jitter-light-plate.json
```

Defaults are three simulated seconds, measurement after one second, four
collision/solver substeps per 60 Hz frame, four main iterations and one relax
sweep per substep, requested 80 Hz contact frequency, sticky matching, and no
sleeping or mass splitting. JSON records the settings, effective soft-contact
coefficients, motion relative to the translating/rotating support, contact
counts, floor and base-top clearance, direct contents-to-floor force, support
escape, and diagnostic runtime. Negative clearance means penetration. Check
these alongside motion: soft contacts can let the carton sink until the world
floor supports its contents directly, producing misleadingly low jitter.
The base-top metric accounts for relative tilt but does not check wall penetration.
Runtime includes
readbacks and may include kernel compilation: do not interpret it as rendered
FPS or a clean solver-throughput benchmark.

`--substeps` controls collision passes per 60 Hz frame; `--solver-substeps`
controls physics steps between collision passes. Motion is sampled after each
collision/solver call, so changing collision frequency also changes measurement
cadence. Compare common sampling endpoints when isolating collision-rate effects;
high-frequency jitter can otherwise alias into different RMS values.

At the initial CPU baseline, relative speed RMS was approximately
0.0012/0.00013/3.66/44.75/21.26 mm/s in the order above. Difficulty is a structural
progression, not a promise of monotonically increasing jitter. The two single-body
controls settle; the loaded cases expose unresolved stability problems. Preserve
all settings when evaluating a solver patch and also check longer runs. Do not
accept quieter output caused by falling through the floor, escaping the support,
altered masses, sleeping, damping, or permanently constraining the contents.
Sliding, release, momentum, and contact-force tests remain separate acceptance
requirements. The benchmark reports measurements rather than declaring the
currently failing loaded cases successful.

## Run the robot throughput sweep

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

# Compare generalized- and maximal-coordinate PhoenX against MuJoCo Warp.
python -m newton._src.solvers.phoenx.benchmarks.run_benchmarks \
    --scenarios feather_pgs_g1_flat \
    --solvers phoenx_reduced phoenx_maximal mujoco \
    --num-worlds 1024 4096 16384 32768

# Reproduce the FeatherPGS G1 sweep (MjWarp still uses 100/50 iterations).
python -m newton._src.solvers.phoenx.benchmarks.run_benchmarks \
    --scenarios feather_pgs_g1_flat \
    --solvers phoenx_reduced phoenx_maximal mujoco \
    --num-worlds 1024 2048 4096 8192 16384 32768 65536 \
    --substeps 2 --solver-iterations 8

# Run H1 with enough MjWarp contact capacity to avoid dropping contacts.
python -m newton._src.solvers.phoenx.benchmarks.run_benchmarks \
    --scenarios feather_pgs_h1_tabletop \
    --solvers phoenx_maximal mujoco_sized \
    --num-worlds 1024 4096 \
    --substeps 4 --solver-iterations 8

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

## High-mass-ratio convergence

The maximal-coordinate convergence frontier uses a 100:1 heavy-on-light
contact stack. It averages supported loads over a steady window and records
RMS velocity, separation, and lateral drift, because one final-frame impulse
sample can hide oscillatory under-convergence.

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_high_mass_ratio
```

Use `--mass-ratios` and `--settings SUBSTEPSxITERATIONS` to widen the sweep.
Every physics frame is CUDA-graph captured; diagnostic readback is excluded
from the separately reported graph-replay throughput.

## Direct joint-equality convergence

The maximal-coordinate joint benchmark measures the production mechanism-wide
direct solve at the default five-substep, two-inequality-iteration work point.
It reports anchor error, graph-replay FPS, and both the original and diagonally
equilibrated mechanism-matrix condition numbers. Production PhoenX no longer
contains a PGS equality mode; compare historical commits explicitly when a PGS
baseline is needed.

```bash
uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_direct_equality

uv run --extra dev -m newton._src.solvers.phoenx.benchmarks.bench_direct_equality \
    --lengths 12 26 --mass-ratios 1 10000 1000000 \
    --output /tmp/phoenx-direct-equality.json
```

Add `--heterogeneous 32x2 8x8 2x26` to measure one world containing
32 two-link, eight eight-link, and two 26-link mechanisms. The result includes
the compact matrix-entry task count and the former rectangular launch size.

Compilation, diagnostics, and condition-number readback are outside the
timing window.

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

16 warmup + 64 measure (Dylan's defaults). Each simulation substep runs
collision detection followed by one solver step; PhoenX uses one internal
substep so its outer cadence matches MuJoCo Warp. JIT compilation, the
contact sorter's lazy buffers, and any first-call allocations all
land in the warmup; measurement is just graph replays. This is
exactly the same shape as Dylan's nightly harness plus what the ASV
`bench_mujoco.FastBenchmark` does.

## Scenarios

- **`feather_pgs_g1_flat`** and **`feather_pgs_h1_tabletop`**:
  model definitions copied from Dylan Turpin's FeatherPGS nightly worker at
  commit `af928ab3`. These are the scenarios to use when comparing against
  the published FeatherPGS results. The current MjWarp API no longer exposes
  the old worker's `ls_parallel` option; all other explicit MjWarp settings
  are retained, including 100 Newton and 50 line-search iterations.
  The copied H1 `nconmax=128` setting overflows with current MjWarp (which
  raises it to 204 internally but needs more than 256 in this scene). Use
  **`mujoco_sized`** for the separately labeled `nconmax=320` variant when
  useful-work validation matters; plain **`mujoco`** retains Dylan's exact
  legacy setting for reproduction.
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
