# PhoenX autotuner coverage audit

The settings below are the authored defaults or explicit solver choices in the
current examples. They are scene baselines, not universally optimal values.

| Workload | Explicit choices that matter | Autotuner coverage |
| --- | --- | --- |
| Colibri | TGS, single-world, 24 substeps per contact refresh, one position iteration, grouped mass splitting (group 2, batch 2, cap 8), parallel contact prepare, final-substep velocity relaxation, prepare stride 1 | Work counts and cap; grouped batch/size and prepare policy in thorough mode. TGS requirements prevent incompatible layout or mass toggles. |
| Caterpillar | Maximal direct joints, single-world, 5 substeps and 2 iterations per 120 Hz contact refresh, grouped splitting (2/2/8), parallel prepare, contact chunks of 64 | Work counts, layout where valid, cap; group, batch, prepare, and chunk trials in thorough mode. |
| BikeTransmission | Maximal direct joints, single-world, 8 substeps at 120 Hz and 4 iterations, grouped splitting (2/2/8), contact chunks of 64, body-pair grouping, configurable direct joint projection passes | Work counts, cap, group, batch, and chunks. Body-pair grouping and direct passes remain fixed. |
| AnalogDigitalClock | Maximal PGS joints, single-world, 16 substeps per 120 Hz refresh and 6 iterations, grouped splitting (2/2/8), parallel prepare, chunks of 64, configurable joint refinement | Work counts, cap, group, batch, prepare, and chunks. Joint refinement remains fixed. |
| G1 example | 5 internal substeps, 2 position and 2 velocity iterations; maximal direct or reduced joint mode | Work counts; authored velocity count and joint mode stay fixed. |
| G1 RL recipe | 8,192 worlds, 3 outer physics steps per policy step, one internal solver substep, 2 position/1 velocity iterations, reduced joints, patch friction, auto lane width/scheduler/prepare stride | Lane width/scheduler and prepare stride are candidates, but the generic runner does not reproduce the environment step or graph-captured throughput. |
| ANYmal RL recipe | 1,024 worlds, 4 outer physics steps, one internal solver substep, 8 position/1 velocity iterations, maximal direct joints; collision can run once per policy frame with partition reuse | Generic runner does not reproduce its collision reuse, control, reward, reset, or CUDA graph. |
| Kapla tower | Direct `PhoenXWorld`, single-world, 4 substeps, 10 iterations with splitting, cap 9, batch 1, unrolled splitting, endpoint-owner partitioner, colored contact storage, grid capped by GPU SM count | Mass, cap, batch, and work counts are generic candidates; this direct-world example needs an adapter before it can use `TuningScene`. |

The largest remaining *safe performance* candidates are Kapla's unrolled
splitting, partitioner and colored-contact layout, plus bike body-pair
grouping. These have feature preconditions and kernel-compilation costs, so
they should be screened only after the basic search has a passing winner.
Joint mode, solver scheme, friction model, collision frequency, speculative
contact policy, joint refinement, and projection passes can materially change
the numerical method; they require scene-specific quality checks. For RL,
graph-captured environment-step throughput and task gates are separate
objectives from this direct-stepping physics FPS.
