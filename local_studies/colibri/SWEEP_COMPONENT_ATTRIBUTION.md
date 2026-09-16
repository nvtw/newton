# Frozen ordered-sweep attribution

This diagnostic attributes cost, not physical accuracy. It captures the first
biased iterate after 330 public Colibri frames, restores all 355 device arrays
before every replay, and never continues an ablated output as a trajectory.
The original color ordering, barriers, and traversal remain in every variant.
Each variant repeats its own output byte-exactly.

## Measurement

On the RTX PRO 6000 Blackwell, 100 isolated replays (last 90 reported):

| Callback enabled | CUDA-event mean | NSYS kernel mean | Registers/thread | Local memory/thread |
| --- | ---: | ---: | ---: | ---: |
| Both | 75.77 us | 73.42 us | 146 | 0 |
| Joints only | 57.34 us | 56.09 us | 128 | 0 |
| Contacts only | 20.48 us | 17.95 us | 140 | 0 |

The graph contains 38 joint rows, 450 contact columns and 2,582 contact
points (at most six per column), across 174 colors grouped four at a time.
All variants launch 64 blocks of 32 threads. CUDA-event measurements include
event-boundary overhead; NSYS numbers come from a separate instrumented replay.

Ablations retain empty traversals: joints-only still visits all 450 contact
columns and contacts-only visits all joint rows. Their costs are not additive.
Different branch compilation also changes register allocation. This first
biased sweep is one frozen sample, not an estimate of every temporal sweep.

## Interpretation and next experiment

The joint callback is the main critical-path cost in this sample despite its
small row count. There is no measured register spilling. The register reduction
from 146 to 128 alone does not establish an occupancy bottleneck, especially
with only one warp per block and 44 active groups.

The bilateral callback forms its RHS with FP64 wrench/twist dots, executes a
dependent FP64 triangular solve, then accumulates and applies FP32 impulses.
It stores pair velocities before the inequality callback reloads them.
However, the previous exact inactive-inequality shortcut and fixed-loop
bilateral expansion each changed total runtime by less than 0.2%; neither was
promoted. Repeating those optimizations is not justified.

A useful next bounded attribution is to separate bilateral RHS construction,
factor solve, and impulse application using the same restored inputs. This
would distinguish dependent arithmetic from scattered row-data loads before
designing a cooperative per-joint solve. Such a solve must preserve operation
order and all finite-drive semantics to qualify as an exact optimization.

Separate joint/contact launches per color are not presently justified:
they could reduce static register use, but add many launches/barriers and
must preserve ordering across colors within every group. Simply applying
all joints before all contacts would change finite-iteration physics.

## Reproduction

Run with the repository interpreter:

    python -m local_studies.colibri.profile_sweep_components

For resources:

    nsys profile --trace=cuda --cuda-graph-trace=node       --capture-range=cudaProfilerApi --capture-range-end=stop       -o /tmp/colibri_sweep_components_profile       python -m local_studies.colibri.profile_sweep_components --profile

Artifacts: /tmp/colibri_sweep_components_uninstrumented.json,
 /tmp/colibri_sweep_components_summary.json, and the matching NSYS report/SQLite.
No canonical solver code changed.

## Bilateral prefix follow-up

The same restored-world replay now supports --bilateral-prefixes. These are
cumulative source prefixes: RHS construction, RHS plus triangular solve,
and the full bilateral callback. Truncated prefixes write their result to
FP64 scratch, preventing dead-code elimination. Scratch is also restored
and its repeated outputs are byte-exact.

| Prefix | Event mean | NSYS kernel mean | Registers/thread | Local memory |
| --- | ---: | ---: | ---: | ---: |
| RHS | 32.79 us | 31.31 us | 74 | 0 |
| RHS + solve | 45.04 us | 43.66 us | 80 | 0 |
| Full bilateral | 53.18 us | 50.86 us | 94 | 0 |

The prefix measurements retain empty contact traversal and color barriers.
Truncation removes body writes, so subsequent joints see the original
frozen velocities; these are cost probes, not staged execution of an
equivalent physical sweep. Added scratch stores and compiler differences
also prevent treating differences as exact isolated stage costs.

Nevertheless, RHS/body-data gathering already occupies a substantial
fraction of the bilateral cost. The triangular solve adds about 12 us;
impulse application adds roughly 7-8 us. Division latency alone cannot
explain the entire joint callback. No prefix spills. Parent is separately
testing prepared reciprocal diagonals with physical validation.

An initial INVALID prefix capture reused a Warp function name, causing
all three variants to resolve to the same module hash. Its approximately
53 us equal timings must not be used. The corrected version assigns unique
callback names, produces three distinct module hashes, and has different
observable RHS/solution scratch norms. The valid NSYS artifact is
/tmp/colibri_bilateral_prefixes_valid_profile.nsys-rep; the earlier
/tmp/colibri_bilateral_prefixes_profile.nsys-rep is invalid.
Uninstrumented corrected measurements are in
/tmp/colibri_bilateral_prefixes_uninstrumented.json.
