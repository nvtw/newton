# PhoenX autotuner field check (2026-09-25)

The real-scene factories in `tune_phoenx_scenes.py` were run on an NVIDIA RTX
PRO 6000 Blackwell. Each candidate starts from the same model pose and uses
the example's materials, drives, collision pipeline, and contact refresh
frequency. Results below use a 60 Hz frame and sample maximum errors every
10 frames. FPS is simulation-only direct Python stepping; the examples use
CUDA graphs and rendering, so these figures are not GUI frame rates.

| Scene (duration) | Authored setting | Best passing alternative | FPS old → alternative | Max joint translation, mm | Max joint rotation, deg | Max penetration, mm | Recommendation |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| BikeTransmission (6 s) | 8 substeps, 4 iterations | 8 substeps, 2 iterations | 17.89 → 18.54 (+3.6%) | 0.149 → 0.156 | 0.0043 → 0.0044 | 1.655 → 1.247 | Keep authored setting; gain below 5% |
| Caterpillar (3 s) | 5 substeps, 2 iterations | 5 substeps, 1 iteration | 31.64 → 32.34 (+2.2%) | 1.622 → 1.625 | 0.0019 → 0.0021 | 53.469 → 53.067 | Keep authored setting; gain below 5% |
| Colibri, 4 worlds (3 s) | 24 substeps, 1 iteration, 8 colors | Same work, 12 colors | 53.48 → 53.86 (+0.7%) | 0.363 → 0.363 | 0.167 → 0.167 | 0.182 → 0.182 | Keep authored setting; color-cap result is timing noise scale |
| G1, 16 worlds (3 s) | 1 internal substep, 2 iterations | 1 internal substep, 1 iteration | 49.38 → 58.45 (+18.4%) | 0.0043 → 0.0048 | 0.00043 → 0.00053 | 0.905 → 0.865 | Try 1 iteration in the G1 workload |

The G1 workload makes five collision/solver calls per frame; the other three
make two. These are bounded searches, not globally optimal settings. The 5%
minimum-gain rule prevents the tuner from suggesting the small Bike,
Caterpillar, and Colibri gains as changes to their defaults. The G1 result is
promising for this flat-ground scene, but its graph-captured training workload
and behaviors beyond three seconds still need validation before changing that
example's default.

The quality gate correctly rejected larger apparent speedups: halving Bike
substeps caused 20.3 mm joint error and 8.2 mm penetration; halving
Caterpillar substeps caused 818 mm penetration; halving Colibri substeps
increased joint translation from 0.363 to 0.653 mm and rotation from 0.167 to
0.429 degrees. Enabling mass splitting on the G1 trial produced nonfinite
state and was rejected. The Caterpillar's approximately 53 mm measured
penetration is present in the authored reference and should not be interpreted
as a target quality level for other scenes.

To reproduce a run:

```console
uv run --extra examples -m newton.examples.phoenx.tune_phoenx \
  newton.examples.phoenx.tune_phoenx_scenes:make_g1_scene \
  --mode fast --frames 180 --sample-stride 10 --time-budget-s 120
```

Replace the factory with `make_bike_scene`, `make_caterpillar_scene`, or
`make_colibri_scene`. Use `--json path.json` for the complete candidate table.
The copyrighted mechanism OBJ assets stay in the local asset directory and
are not part of this report or the repository.
