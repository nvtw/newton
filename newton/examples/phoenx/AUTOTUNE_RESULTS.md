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

## Additional scenes (2026-09-25)

The same GPU and direct-stepping method were used for two clocks and a
larger G1 batch. Angular drag and authored drives were active in both clocks.

| Scene (duration) | Authored → alternative | FPS | Joint translation, mm | Joint rotation, deg | Penetration, mm | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| AnalogDigitalClock (2 s) | 16×6 → 8×6 | 46.34 → 77.25 | 1.283 → 2.039 | 0.033 → 0.084 | 0.086 → 0.195 | Reject; joint and contact errors rise too far |
| TourbillonClock (6 s) | 4×4 → 2×4 | 56.55 → 90.43 | 0.029 → 0.008 | 0.011 → 0.015 | 2.589 → 2.636 | Reject; driven gear motion changes substantially |
| G1, 64 worlds (2 s) | 1×2 → 1×1 | 48.85 → 57.74 | 0.0043 → 0.0048 | 0.00043 → 0.00053 | 0.905 → 0.865 | Candidate: 1 iteration (+18.2%) |

Here `substeps×iterations` is per contact refresh. Both clocks refresh
contacts at 120 Hz; G1 makes five refreshes per 60 Hz frame. Analog clock
half-substeps and half-iterations both failed the quality gate. The
Tourbillon half-substep result initially passed the geometric gate, but its
driven joint's mean absolute relative angular speed over the last half of
the run changed from 0.058 to 2.861 rad/s. The new motion-observable gate
rejects it with a 0.5 rad/s tolerance. The authored drive target is about
14 rad/s in magnitude, so the clock's actual motion merits a separate
physical-behavior investigation; matching its reference is not proof of
correct clock operation. Iteration counts below Tourbillon's four required
direct-projection passes are now skipped instead of run as invalid trials.

The G1 result corroborates the 16-world result at a larger batch size, but
neither flat-ground run includes walking control, policy observations, or the
training environment's CUDA graph. Kapla Tower uses the lower-level
`PhoenXWorld` stepping path; the current `SolverPhoenX` tuner cannot make a
like-for-like comparison to that example.
