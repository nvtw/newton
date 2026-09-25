# PhoenX scene autotuner (experimental)

The tuner takes a Python factory that returns a finalized model and a fixed
physical setup. The factory may build any number of worlds. It creates fresh
states, contacts, and a solver for every trial. It varies solver layout,
substeps, position iterations, multi-world scheduler, mass splitting, and the
regular-color cap. Thorough mode also checks multi-world threads per world.
Material parameters, joint mode, collision update frequency, and one velocity
iteration remain fixed.

Start with the included two-world smoke scene:

```console
uv run --extra examples -m newton.examples.phoenx.tune_phoenx \
  newton.examples.phoenx.tune_phoenx_demo:make_scene --mode default
```

For a real scene, write an importable `make_scene()` function returning
`TuningScene(model=builder.finalize(), frame_dt=1/60,
solver_options={"substeps": 8, "solver_iterations": 4,
"velocity_iterations": 1, ...})`. Put all model construction, per-world
replication, collision filters, material choices, joint mode, and solver
options other than the tuned settings in that function. Set
`collision_updates=2` for 120 Hz collision refresh with a 60 Hz frame. If your
scene uses a custom collision pipeline, provide `pipeline_factory` returning
a new pipeline with sticky contact matching. If controls or forces change over
time, provide `before_update(state, control, update_index, update_dt)`.
It runs before collision detection on every update. The callback must apply
the same deterministic inputs in every trial.

`--mode fast` runs 60 frames per candidate with a 20-second soft budget and
checks the layout, mass splitting on/off, and one-axis work reductions.
`default` runs 120 frames with a 90-second budget and also checks combined
reductions, multi-world schedulers, and regular-color caps near the supplied
value (12 by default). `thorough` runs 240 frames with a five-minute budget
and explores quarter budgets, substep/iteration tradeoffs, and 8/16/32 threads
per world. `--frames` and
`--time-budget-s` override these defaults. The budget estimate uses the
reference FPS and each candidate's work ratio. Compilation can make the
actual run exceed the soft budget; building the scene is not counted. This
is a staged search, not a claim of global optimality.

The color cap matters only with mass splitting enabled; it controls how many
regular graph colors are retained before the mass-split overflow partition.
The tuner does not toggle mass splitting when another fixed option requires
it on or off, such as temporal TGS, patch friction, reduced-coordinate joints,
or grouped colors. Unsupported candidate combinations are reported and skipped.
Other knobs such as joint mode, contact friction model, collision frequency,
mass-splitting batch size, color-group size, contact chunking, and prepare
refresh stride remain scene-defined. Several change the physical or numerical
method, so the tuner does not silently mix them into a speed comparison.

The output table gives measured simulation FPS (without rendering), maximum
joint anchor translation, maximum constrained angular error, and maximum
freshly detected rigid contact penetration. Measurements are sampled every
`--sample-stride` frames; lower strides catch shorter spikes but add cost.
The first few frames are excluded from timing to avoid compilation bias, but
included in error measurements. The default mode's 120-frame horizon is two
seconds at 60 Hz. Increase `--frames` for slowly developing failures.

The first trial is the supplied reference. By default, a candidate passes if
each error stays within 10% of that reference plus a small numerical floor
(0.1 mm for lengths, 0.1 degree for angles). Absolute caps are combined with
these reference limits by taking the stricter value. Use `--relative-slack`
to allow a larger relative degradation when appropriate.
**A bad reference does not become physically correct through autotuning.**
For meaningful acceptance, supply scene-specific caps, for example
`--max-joint-mm 1 --max-joint-deg 0.5 --max-penetration-mm 3`. No setting is
recommended if all candidates violate a cap. Optional `extra_metrics(state,
contacts)` may return additional nonnegative error measures, such as chain
sag, motor tracking error, or interpenetration of a specific pair. Rod and
distance joints are reported as unscored because their allowed deformation
cannot be inferred from anchor alignment alone; use an extra metric for them.
The command supports `--json results.json` for scripts.

Single-world scheduling can solve several independent worlds; the tuner tries
it alongside the multi-world schedulers when the model has more than one world.
For a one-world scene it does not generate multi-world candidates. FPS is the end-to-end simulation
rate for direct Python stepping, including collision detection and solver
dispatch, with rendering and metric sampling excluded. It is not a prediction
of GUI FPS or of CUDA-graph throughput.
