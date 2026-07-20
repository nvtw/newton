# PhoenX Mini Throughput Experiments

## M1 — consume inverse mass from packed velocity (2026-07-19)

Parent: `5650e6a7`. Hardware and roofline denominators are those below. The
candidate keeps the prepared constraint layout unchanged, but reads inverse
mass from the already-loaded `linear_velocity.w` instead of `arm_mass.w` in
all packed contact and revolute solves.

Robot scene, sticky matching, eight bodies/world, one substep, four PGS
iterations. An alternating source bracket used 3,000 replays at 8K worlds and
1,000 replays at 32K worlds. Each median combines two runs:

| Worlds | Control | Candidate | Throughput | Useful bandwidth | Sequential / random-vec4 roofline |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 360.537 us | **359.150 us** | **+0.386%** | 418.81 -> **420.42 GB/s** | 28.12 / 40.39% -> **28.23 / 40.55%** |
| 32,768 | 1.33277 ms | **1.31839 ms** | **+1.091%** | 453.18 -> **458.12 GB/s** | 30.43 / 43.71% -> **30.76 / 44.19%** |

Both scales had finite state and zero gather/color overflow. The three focused
mini unittests pass. A separate deterministic 64-world robot run produced
bit-identical poses and velocities after 60 steps. This qualifies the packed
fourth lane as a small real win. It does not prove that narrowing the prepared
arm arrays is faster; that alignment/traffic tradeoff is the next isolated
experiment.

## M2 — narrow prepared lever arms (rejected, 2026-07-19)

Changing prepared arm/mass records from aligned vec4 to vec3 removed eight
bytes/constraint. It was neutral at 8K and regressed the 32K robot median from
1.31215 to 1.33605 ms (**-1.79% throughput**). Twelve-byte gather alignment
cost more than the saved traffic. Keep aligned vec4 records.

## M3 — subwarp world packing (accepted, 2026-07-19)

The robot graph uses at most five constraints/color, so one warp/world wastes
most lanes. M3 permits logical 8- and 16-thread worlds, launches logical widths
up to 32 in 128-thread physical blocks, and uses masked warp barriers between
PGS colors. At width eight this packs four worlds/warp and 16 worlds/block.

Alternating exact-source bracket, sticky robot scene:

| Worlds | M1 width 32 | M3 width 8 | Throughput | Useful bandwidth | Sequential / random-vec4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 354.180 us | **295.708 us** | **+19.77%** | 426.32 -> **510.63 GB/s** | 28.63 / 41.12% -> **34.28 / 49.25%** |
| 32,768 | 1.31769 ms | **1.19879 ms** | **+9.92%** | 458.37 -> **503.83 GB/s** | 30.78 / 44.21% -> **33.83 / 48.59%** |

Width 16 was intermediate (316.49 us at 8K). Packing two or four 32-thread
worlds/block helped slightly; eight regressed. Contact-only width eight was
about 1.4% slower, so M3 remains an explicit topology experiment rather than a
universal default. Three mini tests pass, explicit width-eight runs have zero
overflow, and width 32 versus width eight produced bit-identical poses and
velocities after 60 deterministic steps.

## M4 — per-body color bitmasks (accepted, 2026-07-19)

The serial greedy colorer stored one int owner for every body/color pair and
cleared that plane every substep. M4 stores the same used-color set as one
uint64/body. Scratch and reset traffic fall 32x (64 MiB to 2 MiB at 32K
eight-body worlds), while smallest-free-color order is unchanged.

Alternating exact-source bracket using M3 width eight:

| Worlds | M3 owner plane | M4 bitmask | Throughput | Useful bandwidth | Sequential / random-vec4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 294.834 us | **270.364 us** | **+9.05%** | 512.14 -> **558.00 GB/s** | 34.39 / 49.40% -> **37.50 / 53.87%** |
| 32,768 | 1.19845 ms | **1.13645 ms** | **+5.46%** | 503.99 -> **531.46 GB/s** | 33.84 / 48.61% -> **35.69 / 51.26%** |

The 8K contact-only stack median also improves from 764.20 to 704.06 us
(**+8.54%**), with under 0.4% contact-count variation. Three mini tests pass;
color buckets/counts, poses, and velocities are bit-identical after 60 robot
steps. An 8K profile attributes 23.9% of kernel time to solve, 13.2% to CUB
contact sorts, 11.3% to prepare, and 6.4% to coloring. Sorting is now the
largest non-solver target.

## M5 — one lane per world (rejected, 2026-07-19)

M5 tested whether transposing execution across thousands of worlds would make
corresponding contact/body vec4 loads coalesce. The existing `serial_world`
path changes constraint order; a temporary `striped_world` kernel preserved
the color-major prepared rows and PGS color order while changing only thread
mapping. Both assign one CUDA lane to each world.

Matched 32,768-world above-L2 runs, 200 captured replays:

| Layout | Frame | Constraint iterations/s | Useful bandwidth | Sequential / random-vec4 / FP32 |
| --- | ---: | ---: | ---: | ---: |
| Colored subwarp | **1.949 ms** | **2.139 B** | **752.8 GB/s** | **50.6 / 72.6 / 1.10%** |
| Serial world order | 2.281 ms | 1.844 B | 649.0 GB/s | 43.6 / 62.6 / 0.94% |
| Color-preserving striped | 2.500 ms | 1.668 B | 587.2 GB/s | 39.4 / 56.6 / 0.85% |

World striping loses 28.3% frame throughput and 22.0% normalized constraint
throughput. Contact manifolds are not aligned enough across worlds to recover
the lost within-color parallelism. The striped prototype was removed. Future
coalescing experiments must retain subwarp parallelism.


## M6 - deterministic direct-to-sorter collision output (accepted, 2026-07-19)

M6 corrects the benchmark methodology: mini now always enables deterministic
collision sorting. Earlier max-mode measurements did not pay that required
cost and remain useful only for comparing their isolated solver variants.

A 32K deterministic baseline took 2.27171 ms. Narrow phase wrote the canonical
contact arrays, then sorting backed up 13 columns and gathered the permutation.
M6 writes the unsorted record directly into the sorter's existing source
storage and gathers once. Contact matching owns separate prior midpoint/normal
buffers (+24 bytes/contact only when matching is enabled), so the collision
pipeline retains one general canonical deterministic output order.

Three clean 200-replay candidates were 2.18414, 2.16529, and 2.17127 ms:
**2.17127 ms median, +4.63% throughput**. Useful-work bandwidth changes from
636.4 to **665.8 GB/s**, or 42.7 to **44.7%** of sequential peak and 61.4 to
**64.2%** of random-vec4 peak. Estimated FP32 use is 0.97% of peak. An 8K
deterministic run takes 635.77 us and reaches 650.5 GB/s (43.7% sequential,
62.7% random-vec4).

Nsight confirms the full-record backup kernel is absent. The new kernel-time
shares are solve 34.7%, prepare 15.8%, narrow phase 14.7%, canonical gather
11.7%, coloring 5.3%, and radix sort 4.6%. Forty matching tests pass on CPU and
CUDA; five deterministic graph-capture tests, including 500-step latest and
sticky runs, pass on CUDA.



## M7 - world-major deterministic scheduling and interleaved rows (accepted, 2026-07-19)

M7 uses the required canonical collision sort as solver preparation. Primitive
multi-world keys are now ordered by world, local shape pair, and subkey, so
contacts against shared ground remain in the same run as local contacts.
Mini marks run boundaries, appends construction-time stable revolute lists,
and colors canonical constraints without atomic slot allocation. Multiple
runs are diagnosed as overflow rather than solved nondeterministically.

Contact-only prepared rows are tiled across adjacent worlds. The qualified
16-lane layout places two logical worlds in each warp, interleaves their
corresponding vec4 rows, and uses independent masked barriers between PGS
colors. It preserves per-world constraint, color, and arithmetic order.
Mixed revolute/contact worlds use the same deterministic schedule and the
16-lane default, without a separate solver mode.

The new --fixed-state benchmark restores identical state inside each graph
replay. Exact-source M6/M7 brackets therefore process identical contacts:

| Workload | M6 | M7 | Throughput | Useful bandwidth |
| :--- | ---: | ---: | ---: | ---: |
| 8K stack, 262,144 contacts | 525.53 us | **414.93 us** | **+26.7%** | 702.3 -> **889.5 GB/s** |
| 32K stack, 1,048,576 contacts | 1.67731 ms | **1.38237 ms** | **+21.3%** | 880.2 -> **1,068.0 GB/s** |
| 8K robot, 32,768 contacts + 65,536 joints | 299.16 us | **257.81 us** | **+16.0%** | 504.7 -> **585.7 GB/s** |

The 32K medians combine three 200-replay runs per revision. M7 reaches **71.7%
of sequential DRAM**, **103.0% of random-vec4 gather**, and 1.55% of FP32
peak. Exceeding the random-vec4 calibration is evidence that interleaving made
the hot loads more coalesced; it is still below sequential DRAM.

Matched Nsight medians show solve falling 575.4 to **322.7 us** (-43.9%) and
prepare 249.6 to **178.1 us** (-28.6%). Canonical ordering also reduces the
fixed stack from 12 to 8 greedy colors. Deterministic run marking/gather costs
21.5 us versus the old atomic gather's 15.6 us; the 5.9 us cost is dominated
by the hot-loop savings.

Five mini tests pass. The 32- and 16-lane paths produce bit-identical poses and
velocities after 30 deterministic steps; mixed schedules repeat bit-identically
across ten rebuilds. Stack state stays finite and revolute anchor error remains
bounded, with zero gather/color overflow.


This is an append-only scientific ledger. Failed qualifications and rejected
results remain visible.

## Fixed method

Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB). Software:
Warp 1.15.0.dev20260626, CUDA toolkit 12.9, driver 13.1.

The workload uses Newton CollisionPipeline, eight free boxes/world on shared
ground, one substep, and four colored PGS iterations. Captured-graph
end-to-end time includes collision, scheduling, integration, solve, and state
copy. Acceptance requires finite state and zero gather/color overflow.

Benchmark scales are 4,096 worlds (cache/latency), 16,384 worlds (large RL),
and 32,768 worlds (deliberately above-L2 streaming). A roughly 30-body
revolute/contact workload is still required for robot-training generalization.

Canonical command:

    uv run -m newton._src.solvers.phoenx.mini.benchmark       --worlds 4096 --bodies-per-world 8 --substeps 1 --iterations 4       --settle-steps 30 --warmup 20 --replays 200

## Rooflines and utilization percentages

Accepted 512 MiB-per-array above-cache calibration:

- Sequential float bandwidth: **1,489.14 GB/s** best.
- Random scalar gather: **609.60 GB/s** best.
- Random vec4 gather: **1,036.82 GB/s** best.
- Independent FP32 FMA: **87.810 TFLOP/s** best.
- vec3 dot/cross: **82.496 TFLOP/s** best.

The 8 MiB quick calibration is rejected as a denominator because cache made
logical bandwidth exceed DRAM bandwidth.

GPU counters are privilege-blocked (ERR_NVGPUCTRPERM). The benchmark therefore
emits conservative algorithmic percentages, not DRAM-counter measurements.
C0/C1 use 348 unique bytes and about 1,100 FP32 operations/contact-iteration.
C2 uses 352 unique bytes and about 450 FP32 operations/contact-iteration after
moving geometry, tangent, and effective-mass work into one preparation pass.
Repeated source-level reloads are excluded because the compiler or L2 may
remove them. Sequential DRAM is the absolute memory denominator; random vec4
gather is the closest calibrated access-pattern denominator for C2. Values
against random scalar gather may exceed 100% after coalescing and are retained
only as a diagnostic that the access pattern changed.

## C0 — full-list color scan baseline

One block/world; matrix-free normal plus two box-friction rows/contact. A
parallel atomic coloring assigned body-disjoint colors. The solve scanned the
full constraint list for each of 32 colors. CUDA smoke passed.

At 4,096 worlds and block size 128, trials were 999.51, 1,007.40, and 1,005.15
us/frame: **1,005.15 us median**, 4.075 M world-steps/s, 32.60 M body-steps/s,
and about 524 M contact-iterations/s. Nsight attributed 89.5% of kernel time to
solve_worlds_kernel (867.0 us median); coloring was 28.9 us, narrow phase about
48 us, and integration under 5 us.

### C0a — block-size sweep, accepted

| Threads/world | Frame time [us] | World-steps/s | vs block 128 |
| ---: | ---: | ---: | ---: |
| 32 | **880.20 median** | **4.654 M** | **1.142x** |
| 64 | 1,076.16 | 3.806 M | 0.934x |
| 128 | 1,005.15 median | 4.075 M | 1.000x |
| 256 | 1,404.19 | 2.917 M | 0.716x |

One warp/world wins because the scene has at most 39 constraints/world. The
32-thread default is a 12.4% end-to-end improvement.

First 16K/32K C0a attempts are rejected: an accidental harness edit lowered
max_constraints_per_world from 128 to 32 and dropped constraints. It was
diagnosed and corrected before C1 acceptance.

## C1 — compact deterministic color buckets, accepted

The O(colors x constraints) scan became fixed-capacity color-major buckets.
Transient atomic color claims became one deterministic greedy-coloring
thread/world. Thousands of worlds retain GPU parallelism while each tiny graph
colors serially without atomics. The ceiling is 64 colors, but solve visits only
the actual per-world color count.

An intermediate atomic-bucket result around 672 us at 4K is rejected because it
dropped 6k constraints through transient color contention. Diagnostic results
using the accidental 32-constraint world cap are also rejected.

Corrected zero-overflow results:

| Worlds | Frame time | World-steps/s | Body-steps/s | Contact-iters/s |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | **592.23 us median** | **6.916 M** | **55.33 M** | **0.955 B** |
| 16,384 | 1.749 ms | 9.367 M | 74.93 M | 1.422 B |
| 32,768 | 3.434 ms | 9.541 M | 76.33 M | 1.445 B |

The 4K trials were 588.27, 599.50, and 592.23 us. C1 is **1.486x C0a
end-to-end** (32.7% less time).

For 32K above-L2, the mandatory model gives **503 GB/s**, or **33.8% of
sequential peak** and **82.5% of random-scalar peak**. Estimated compute is
**1.59 TFLOP/s**, or **1.81% of measured FP32 peak**. This indicates a
scattered-access/latency workload, not a compute-throughput workload.

## PhoenX comparison and contact-matching control

PhoenX requires sticky contact matching. The fair comparison therefore enables
sticky matching for both solvers, while the maximum-throughput mini mode
disables it because mini does not consume match indices or warm-start state.

At 4K worlds with sticky matching, C2 is 503.06 us versus PhoenX's 599.43 us
median: **1.19x world throughput**. At 32K, C2 is 2.866 ms versus PhoenX's
3.890 ms: **1.36x world throughput**. These runs use the same model, collision
pipeline options, point friction, substep count, and PGS iteration count.

Disabling unused matching is a separate accepted throughput choice. At 4K,
three C2 trials were 419.02, 409.47, and 409.80 us, or **409.80 us median**.
That is **1.46x PhoenX world throughput**. At 32K above-L2, C2 reaches:

- **2.119 ms/frame**
- **15.463 M world-steps/s**
- **123.707 M body-steps/s**
- **2.344 B contact-iterations/s**
- **825.22 GB/s** algorithmic minimum
- **55.42%** of sequential DRAM and **79.59%** of random-vec4 gather
- **1.055 estimated TFLOP/s**, **1.20%** of measured FP32 peak
- zero gather/color overflow

This is **1.84x PhoenX world throughput** (45.5% less end-to-end frame time).
The comparison is intentionally reported both ways: sticky isolates solver
architecture; disabled matching describes the fastest useful mini pipeline.

Nsight on 32K max mode reports medians of 816.6 us for packed solve, 392.5 us
for contact preparation, 335.8 us for GJK/MPR narrow phase, and 194.2 us for
coloring. The solve is now 41% of kernel time rather than C1's 57% at 2.16 ms.

## C2 — color-major vec4 preparation, accepted

C2 keeps graph-colored PGS but moves repeated work out of the iteration loop.
It computes world inverse inertia once/body, prepares body IDs, arms, normal,
tangent, bias, friction, and three effective masses once/contact, and stores
aligned color-major vec4 streams. The hot loop loads linear/angular velocities
once, solves normal and two friction axes in registers, then stores each body
once. Revolute worlds retain the C1 five-row fallback.

At 32K with sticky matching, solve falls from C1's 2.16 ms profiler median to
**724.9 us**, a **2.98x hot-kernel speedup**. Preparation costs 327.3 us.
End-to-end C2 is 2.866 ms versus the matching C1 result of 4.018 ms, a 28.7%
reduction. Max mode reaches the numbers above.

Correctness is guarded by CUDA unittests for a finite multi-world stack,
body-disjoint color buckets, zero overflow, and bounded revolute-anchor drift.

## Robot-like revolute/contact workload

The capability workload is a 30-link revolute chain/world, forward-kinematics
initialized with its last link penetrating the ground by 2 cm. Adjacent and
non-adjacent self-collision are disabled with Newton collision groups; ground
contact remains enabled. This avoids a pathological collapsing pile while
retaining 30 revolute constraints and about 15 ground contacts/world. At 8,192
worlds the 245,760 bodies, joints, schedule, and contact data exceed L2.

Several pre-FK/self-collision runs are rejected: contact buffers from 256 to
1,024 contacts/world overflowed as all links began coincident and collapsed.
No numbers from those runs are used.

Qualified R0 (old C1-style mixed fallback):

| Worlds | Frame time | World-steps/s | Scalar rows/s | Seq. DRAM model |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 497.84 us | 8.228 M | 6.412 B | 32.7% |
| 8,192 | 892.41 us | 9.180 M | 7.154 B | 36.5% |

PhoenX on the identical scene takes 690.98 us at 4K and 1.163 ms at 8K
(sticky matching, required by PhoenX).

## C4 — packed mixed contact/revolute colors, accepted

Any revolute previously sent the whole world through C1. C4 precomputes the
five revolute row directions, biases, effective masses, arms, and motor impulse
once/substep into aligned color-major vec4 streams. Each constraint loads both
body velocities once, solves all five rows in registers, and writes each body
once. Contacts in the same colors use the C2 packed path.

| Worlds | Matching | Frame time | World-steps/s | Scalar rows/s |
| ---: | :--- | ---: | ---: | ---: |
| 4,096 | disabled | **403.11 us** | **10.161 M** | **7.917 B** |
| 8,192 | disabled | **723.48 us** | **11.323 M** | **8.824 B** |
| 8,192 | sticky | 834.67 us | 9.815 M | 7.649 B |
| 8,192 PhoenX | sticky | 1.163 ms | 7.043 M | 4.738 B |

C4 reduces the qualified 8K fallback time by 18.9%. Max mode is **1.61x
PhoenX world throughput** and **1.86x scalar-row throughput**. With sticky
matching on both, C4 remains **1.39x faster in world throughput** and **1.61x
in scalar-row throughput**.

The 8K max-mode algorithmic model is **781.66 GB/s**, **52.49% of sequential
DRAM**, **75.39% of random vec4 gather**, and **1.28% of FP32 peak**. Hardware
counters remain unavailable. Nsight medians on a shorter trace are 148.4 us
for mixed solve, 102.9 us for mixed prepare, 120.2 us for NxN broad phase, and
60.3 us for coloring. Mixed solve is only 26.6% of kernel time, so further
end-to-end gains require schedule/collision improvements as well as row math.

## C4a — measured schedule footprint, accepted

The general solver retains a 64-color safety ceiling, but benchmark workloads
should not allocate and clear planes they cannot use. The observed maxima are
15 colors for the stack and 6 for the robot. Explicit ceilings of 16 and 8
retain zero overflow. Packed paths also stopped clearing three legacy scalar
impulse arrays that they never read.

| Workload | Colors | Frame time | World-steps/s | Seq. DRAM | Vec4 gather |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 32K stack | 64 | 2.119 ms | 15.463 M | 55.42% | 79.59% |
| 32K stack | 24 | 2.034 ms | 16.110 M | 57.74% | 82.93% |
| 32K stack | 16 | **1.962 ms** | **16.698 M** | **59.85%** | **85.96%** |
| 8K robot | 64 | 723.48 us | 11.323 M | 52.49% | 75.39% |
| 8K robot | 8 | **690.15 us** | **11.870 M** | **55.05%** | **79.06%** |

The tuned stack is **1.98x PhoenX end-to-end world throughput**. The tuned
robot is **1.69x PhoenX world throughput** and **1.95x scalar-row throughput**.
These ceilings are scene qualifications, not safe universal defaults; overflow
diagnostics remain mandatory.

## C3 rejected variants

### Shared tile body cache

A Warp dynamic tile-extract/scatter specialization for retaining each tiny
world's body state in shared memory did not finish compiling within 180 s.
It is opt-in and rejected as a default. A lower-level warp-shuffle or native
CUDA version may still be worth testing.

### One thread per world

This removes graph coloring and runs serial PGS inside each world, using 32
worlds/warp. It is correct but loses parallelism and creates a long dependency
chain. At 4K it takes 1.229 ms versus C2's 0.410 ms. At 32K it takes 2.462 ms
versus C2's 2.119 ms, a 16.2% regression even after enough worlds are present
to occupy the GPU. Graph-colored C2 remains accepted.

## Next isolated checkpoints

1. Test a hybrid matrix-free C5 that reconstructs tangent/effective mass in
   the hot loop to trade cheap FLOPs for fewer stream bytes/contact.
2. Transfer C5 to full PhoenX with exact packed endpoint keys and safe
   CUDA-graph conditional boundaries.
3. Replace the rejected dynamic shared tile with a warp-shuffle or native CUDA
   body cache only if compile time and register count remain bounded.
4. Add warm start as a separate convergence/throughput checkpoint; do not pay
   sticky-matching cost without consuming its output.
5. Measure convergence-to-error, not only equal iteration count, once warm
   start is available.

A capability remains optional when it regresses throughput unless correctness
or convergence repays the cost.



## C5 - reuse unchanged constraint topology, accepted experiment

Deterministic contact sorting often changes contact points while preserving the
ordered shape-pair stream and therefore the constraint graph. C5 compares that
stream with the previous frame, then uses a CUDA graph conditional to retain
the existing world buckets and coloring when the count and every shape pair
are unchanged. Any topology change takes the original full rebuild. The
predicate is exact, scene-independent, and disabled by default while the
stronger endpoint-key form is transferred to full PhoenX.

RTX PRO 6000 exact-source medians, 32K worlds, sticky matching, one substep,
four iterations:

| Workload | Rebuild every frame | C5 reuse | Throughput gain | C5 useful bandwidth |
| --- | ---: | ---: | ---: | ---: |
| Stack, 1,048,576 contacts | 1.667 ms | **1.572 ms** | **+6.0%** | 939.46 GB/s (63.1% sequential, 90.6% random-vec4) |
| Robot, 131,072 contacts + 262,144 revolute | 1.116 ms | **1.038 ms** | **+7.5%** | 581.99 GB/s (39.1% sequential, 56.1% random-vec4) |

The exact comparison costs 13.9 us; predicate setup and conditional dispatch
bring total detection to 17.3 us. Stable frames omit the 85.3 us color, 14.4 us
world-run mark, and 8.5 us gather kernels. A 40-step evolving stack is bitwise
identical with reuse enabled and disabled, proving both stable reuse and dirty
fallback. The full PhoenX transfer should compare packed graph endpoints, not
shape ids, so joints, live mass/node changes, and compound rigid contacts share
one invalidation rule.

## F23 - reuse stable rigid coloring in full PhoenX

Full PhoenX now compares one packed 64-bit graph-endpoint key per active rigid
constraint while projecting constraints. If the count and every endpoint pair
match, a CUDA graph conditional retains the deterministic world buckets and
greedy coloring. Any change executes the original build. The optimization is
disabled for single-world, deformable, nonstandard, unsupported-conditional,
and greedy-overflow fallback paths.

RTX PRO 6000 exact-source fixed-state results, sticky matching, one substep,
four iterations:

| Workload | F22 rebuild | F23 reuse | Throughput gain | F23 useful bandwidth |
| --- | ---: | ---: | ---: | ---: |
| 32K worlds x 8-body stack | 2.081 ms | **1.992 ms** | **+4.5%** | 740.5 GB/s (49.7% sequential, 71.4% random-vec4) |
| 8K worlds x 30-link robot | 896.7 us | **791.7 us** | **+13.3%** | 555.0 GB/s (37.3% sequential, 53.5% random-vec4) |

Each table entry is the median of two interleaved candidate/control samples.
A 4K-world stack spot check improved 361.46 to 325.49 us (+11.1%). The smaller
case benefits more because skipped scheduling has a larger fixed-cost share.

A captured-graph regression forces both the dirty and stable branches while
comparing every evolving pose and velocity bitwise against rebuilding every
frame. Risk-weighted validation also covers deterministic replay, compound
contacts, contact isolation, basic joint types, cloth fallback, and randomized
partitioner stress. These passed as 31 isolated unittests. One combined-process
run passed 21 tests before a host allocator crash; isolated reruns passed, so
large mixed-module runs should remain separate until that test-harness issue is
understood.

This does not mean PhoenX uses 100% of DRAM bandwidth. The percentages are a
useful-physics lower bound, not hardware counters; collision, matching, history,
and preparation remain outside it. F23 removes about 90 us of stable-frame
schedule work and is most valuable where topology persists.

## F24 - contact-frame and validator experiments, rejected

RTX PRO 6000 controlled experiments found no safe full-solver win:

- Rebuilding the tangent deterministically and deleting frame history improved
  the fixed 32K stack 3.4%, but half-scale stack speed reached 0.815 m/s.
- Delaying body-velocity loads for reused tangents reduced warm-start gather
  152.11 to 150.75 us (0.9%) but left the frame neutral (1.991 vs 1.987 ms).
- Above-L2 frame codec medians for 4.19M frames were 259.82 us direct,
  199.88 us oct4, and 182.54 us quaternion-xyz. Oct4 error stayed below
  7.6e-7, yet full-solver half-scale drift reached 0.049 m; orthogonalization
  worsened maximum speed to 1.464 m/s. Quaternion-xyz error reached 1.02e-3.
- A bit-exact lean endpoint validator was neutral on the 32K stack (1.994 vs
  1.991 ms) and regressed the mixed robot (788.44 vs 779.11 us). Moving the
  two world scans inside the dirty conditional is unsupported because their
  child graph contains allocation nodes.

No solver code was retained. Previous history remains normal+tangent1; tangent2
is reconstructed by cross product. The next contact-staging decision requires
the requested Nsight Compute counters, not more source-level load guesses.

## F25 - mini-style cross-frame cold start, rejected

Cold-starting impulses with a fresh velocity-aligned tangent passed 28 scale,
friction, and force tests. It removed 94.3 us of fixed-state history work and
improved the 32K stack 1.990 to 1.897 ms (+4.9%). On the evolving stack it
produced 1,114,112 instead of 1,048,576 contacts and took 2.106 instead of
2.086 ms. Retaining only the normal impulse still produced 1,114,112 contacts
and took 2.143 ms, despite a 3.5% fixed-state gain. The mixed robot was neutral.

Cross-frame friction-frame continuity reduces downstream contact work enough
to repay its staging cost. Full normal+tangent1 history remains retained.


## F26 - hardware counters and layout controls

Nsight Compute on the fixed 32K stack identified different bottlenecks:

| Kernel | Time | DRAM | DRAM peak | SM peak | Excess sectors |
| --- | ---: | ---: | ---: | ---: | ---: |
| deterministic collision gather | 204.9 us | 1.10 TB/s | 64.3% | 7.6% | 75% |
| contact warm-start gather | 132.5 us | 1.06 TB/s | 61.8% | 47.6% | 25% |
| contact history copy | 62.3 us | 952 GB/s | 55.8% | 60.4% | negligible |
| fused prepare + PGS | 555.2 us | 386 GB/s | 22.5% | 19.6% | 85% |

The fused kernel uses 168 registers/thread and 25% occupancy, with 93.4% L2
hit rate. Its excess sectors are therefore chiefly scattered/cache traffic,
not proof that the full frame is close to DRAM peak. The collision gather is
Newton’s general deterministic sort permutation, not PhoenX row gathering.

Controlled full-solver experiments were rejected:

- Disabling Warp grid stride in multi-world fast-tail was neutral and unsafe
  when the physical grid is capped. Single-world persistent kernels already
  use `grid_stride=False`.
- Splitting fused prepare and PGS cost 360.0 versus 313.6 us (+14.8% kernel
  time) and regressed the frame 2.4%.
- World-major deterministic sorting reduced world runs but regressed the frame
  2.7%; collision gather alone grew 23.3 us.
- True contact-major hot-row AoS recovered the initial plane-major vec4 loss,
  but remained neutral: 1.9948 versus 1.9917 ms over 400 replays (-0.15%).
- Symmetric compile-time contact-only dispatch unexpectedly regressed 2.1004
  versus 1.9825 ms (-5.6%); retaining the unified runtime dispatch generates
  the faster CUDA on this Warp/NVRTC toolchain.

No solver code was retained. A current 2x2x16 single-world tower sweep found
the shipped 256-row fused-tail threshold best at 477.6 us/solve; disabling the
tail took 480.0 us. Large single-world tuning remains an explicit guardrail,
but this launch threshold is already on its performance plateau.

## J0 - deterministic gather-Jacobi, rejected

A contact kernel wrote two endpoint velocity deltas and a deterministic body
CSR gather applied them. This removed coloring from the solve and exposed all
contacts concurrently, but added 64 B/contact/iteration of delta traffic plus
a second streaming pass. At four iterations:

| Workload | Colored PGS | Gather-Jacobi | Change | Jacobi useful roofline |
| --- | ---: | ---: | ---: | ---: |
| 4K fixed stack | 238.2 us | 313.8 us | -24.1% throughput | 39.5% sequential / 56.7% random-vec4 |
| 32K above-L2 stack | 1.568 ms | 1.978 ms | -20.7% throughput | 50.1% sequential / 72.0% random-vec4 |

Jacobi would also need parity qualification and likely more sweeps to match PGS
convergence. The prototype was removed. Full parallelism is not a win when it
materializes endpoint deltas; future algorithm changes must keep body updates
on chip or remove another pass.

## J1 - fewer over-relaxed PGS sweeps, rejected

Full PhoenX already exposes SOR. On the fixed 32K above-L2 stack, three sweeps
at SOR 1.2 took 1.952 ms versus 1.985 ms for four sweeps at SOR 1.0: only 1.66%
more frame throughput. This is the upper bound before proving equal convergence.
No code was retained. PGS arithmetic is too small a frame fraction for sweep
count tuning to deliver a high-single-digit win; target staging passes instead.

## J2 - AoS deterministic contact staging, rejected

The deterministic narrow phase wrote one 112-byte packed record/contact, and
the radix permutation gathered it into canonical SoA. Public contacts and
matching stayed unchanged. On the fixed 32K stack it took 2.153 ms versus
2.013 ms for an adjacent F23 control: -6.5% throughput. Packing makes same-field
warp accesses stride by the record size; per-thread locality did not repay lost
coalescing and 20 extra padding bytes/contact. The prototype was removed. A
large gather win now requires avoiding or fusing materialization, not AoS.

## J3 - fuse deterministic gather with matching, rejected

The existing SoA source and canonical output were retained. Match pass one
consumed each permuted source record inside the gather kernel, eliminating its
subsequent canonical geometry reread. On the fixed 32K stack the fused frame
was 2.016 ms versus 2.013 ms for the adjacent F23 control (-0.17%). The larger
kernel offset the removed read/launch. The prototype was removed. Materialization
must be skipped, not burdened with additional matching work.


## J4 - direct valid-run contact compaction, accepted in full

Matched graph-node profiles showed that full PhoenX's fused prepare+PGS kernel
was already faster than mini prepare+solve (319 versus 494 us). The remaining
gap was full-only contact staging: 149 us warm-start gather, 55 us history copy,
and roughly 130--170 us of pair/column ingest, element projection, run building,
and scans. Shared collision kernels were matched. This ended solver-kernel
tuning and moved the experiment to contact ingest.

The mini benchmark replaces two full-capacity scans and intermediate pair
arrays with: mark only valid sorted shape-pair boundaries, one inclusive scan,
then materialize run metadata and the contact-to-column map directly. RTX PRO
6000 isolated results, four points/pair:

| Contacts | Two-stage compaction | Direct compaction | Speedup |
| ---: | ---: | ---: | ---: |
| 1,048,576 | 27.28 us | 19.19 us | 1.42x |
| 4,194,304 | 87.02 us | 45.96 us | 1.89x |

At eight points/pair the 1M-contact speedup remains 1.35x (29.11 to 21.54 us),
so the boundary thread's short forward walk is not limited to four-point box
manifolds.

Transferred to full PhoenX, the normal shape-pair path now compacts directly.
The old two-stage path remains only for optional compound body-pair grouping.
Matched F23-control results:

| Workload | F23 control | Direct runs | Gain | Direct useful bandwidth |
| --- | ---: | ---: | ---: | ---: |
| 32K x 8 fixed stack, above L2 | 1.996 ms | **1.898 ms** | **+5.2%** | 778.0 GB/s (52.2% sequential, 75.0% random-vec4) |
| 8K x 8 evolving stack | 595.95 us | **579.62 us** | **+2.8%** | 736.3 GB/s (49.4% sequential, 71.0% random-vec4) |
| 512-body single-world stack | 895.92 us | **876.79 us** | **+2.2%** | 3.29 GB/s useful-work lower bound |

The evolving control and candidate finish with the same 303,104 contacts.
Determinism (6 tests), compound fallback (5), ingest edge cases (4), and mixed
joint/material behavior (18) pass. The broad multi-world module was stopped
after producing no progress for 60 seconds; the 32K and evolving 8K workloads
exercise the changed multi-world path directly.

## J5 - defer contact-history permutation, rejected

Keeping history in the previous canonical order and applying the match
permutation at its consumer avoids a copy only in appearance: it turns the
next read into a gather. A 1M-contact mini probe covering the nine retained
history floats was neutral for identity matches (18.804 versus 18.776 us,
1.002x) and slightly slower for within-manifold reorder (19.715 versus
19.791 us, 0.996x). Fixed contacts were 100% identity, but an evolving 8K
stack was 0% identity, so an identity shortcut would overfit stable topology.
No code was retained.

## J6 - fuse contact-column packing into run materialization, rejected

The direct-run boundary thread also wrote the contact-column header, removing
the 37 us column-pack kernel. It improved the fixed 32K stack only 0.88%
(1.9133 to 1.8965 ms) and regressed the evolving 8K stack 1.0% (553.76 to
559.30 us), with identical final contact counts. Concentrating material and
header gathers in the serial boundary thread loses enough parallel memory
latency hiding to erase the saved pass when contacts change. No code was
retained.

## J7 - replace pipeline sticky replay with PhoenX history, rejected

Newton's deterministic latest matching measures the upper bound from
removing sticky replay: fixed 32K improved 1.891 to 1.722 ms (+9.8%), and an
above-L2 evolving 32K stack improved 1.984 to 1.891 ms (+4.9%). Native PhoenX
anchor carry with 1 mm, 10 mm, match-only, and the exact fresh-gap gate all
changed the settled topology (1.11--1.28M versus 1.05M contacts). Sticky also
carries its own matching history forward; copying only solver anchors is not
equivalent. The prototype was removed.

## J8 - cache sticky replay's fresh-gap gate, rejected

The match kernel already transforms both current contact points to world
space. Caching its exact penetration gap made replay a pure history copy:
replay fell 77.1 to 55.9 us, while match rose 73.4 to 78.3 us. The fixed 32K
frame improved only 1.915 to 1.901 ms (+0.8%); dynamic gain is smaller. One
extra float/contact and API plumbing are not worthwhile. The prototype was
removed. A larger sticky win must eliminate duplicated history fields/passes,
not shift arithmetic between kernels.

## J9 - canonical sticky overlay, accepted in mini

Sticky matching stores five vec3 history fields even though the previous
canonical contact buffer already contains them. A first pre-gather replay
wrote through the random sort permutation: 1.38x faster at 1M contacts, but
3.1x slower at 4M once random writes escaped L2. It was rejected.

The accepted layout reuses that history allocation as a canonical overlay.
Matched rows read the prior canonical buffer; fresh rows gather sort scratch;
both write the overlay coalesced. The final gather then reads the overlay
coalesced. This can replace sticky replay plus sticky save without extra
storage. Isolated RTX PRO 6000 results:

| Contacts | Fresh rows | Current lifecycle | Canonical overlay | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1,048,576 | 0% | 247.86 us | 57.28 us | 4.33x |
| 4,194,304 | 0% | 2,309.80 us | 721.09 us | 3.20x |
| 4,194,304 | 10% | 2,580.80 us | 860.55 us | 3.00x |
| 4,194,304 | 50% | 2,508.59 us | 1,406.98 us | 1.78x |
| 4,194,304 | 100% | 1,954.62 us | 1,954.87 us | 1.00x |

At 4M matched contacts the overlay moves a 1.396 TB/s logical rate, 93.7% of
measured sequential DRAM. It remains neutral when every row is fresh, so it
does not rely on stable topology for safety. Transfer requires matching before
canonical gather and retaining only key/midpoint/normal matcher history.

## J10 - canonical sticky overlay in full PhoenX

The mini layout transferred without a PhoenX-specific solver path. Sticky
matching now builds one coalesced canonical geometry overlay before the
existing deterministic gather. The old full-record sticky save and replay
path was removed. Alternating contact buffers remain supported.

Adjacent exact-source RTX PRO 6000 results, one substep, four iterations:

| Workload | Control | Overlay | Throughput gain | Overlay useful roofline |
| --- | ---: | ---: | ---: | ---: |
| 32K x 8 fixed stack, 1,048,576 contacts | 2.004 ms | **1.908 ms** | **+5.0%** | 770--777 GB/s; 51.7--52.2% sequential, 74.3--75.0% random-vec4, 1.1% FP32 |
| 32K x 8 evolving stack, 655,360 contacts | 2.180 ms | **2.136 ms** | **+2.1%** | 432.1 GB/s; 29.0% sequential, 41.7% random-vec4, 0.63% FP32 |
| 512-body single-world stack | 277.02 us | 278.37 us | -0.5% (noise guardrail) | useful-work lower bound |

The changing-state runs finish with identical contact counts. The default
midpoint breaking threshold remains 0.5 mm; sticky geometry is reused only
while the fresh narrow-phase gap is non-positive. All 40 matching tests and
five deterministic pipeline tests pass, including 500-step sticky CUDA graph
execution and alternating contact buffers.

## J11 - compact deterministic warmstart frame in full PhoenX

Matched sticky contacts already reuse the previous canonical normal and anchors.
PhoenX therefore reconstructs the tangent deterministically from the normal and
retains only three impulse scalars per contact. The six-float frame-history
buffer and its copy traffic are removed. Primitive contacts also avoid redundant
body/velocity gathers; mesh, height-field, tetrahedron, and repeated-generation
safeguards retain their gap calculation. The sticky fresh-gap decision moves
into matching, where both world points already exist. No solver path was added.

RTX PRO 6000, 32K x 8 fixed stack, 1,048,576 contacts, adjacent 300-frame runs:

| Metric | J10 control | Compact frame | Change |
| --- | ---: | ---: | ---: |
| Frame | 1.9150 ms | **1.8097 ms** | **-5.5%** |
| Throughput | 2.190B constraints/s | **2.318B/s** | **+5.8%** |
| Useful bandwidth | 771.0 GB/s | **815.8 GB/s** | **+5.8%** |
| Sequential / random-vec4 roof | 51.8% / 74.4% | **54.8% / 78.7%** | +3.0 / +4.3 points |

Nsys medians show the source: warmstart gather 146.1 to 100.0 us (-32%),
history copy 56.9 to 30.5 us (-46%), and overlay 99.0 to 55.5 us (-44%).
Matching rises 77.8 to 84.5 us (+9%) because it absorbs the exact gap gate.
The 512-body single-world guardrail is 619.75 to 615.85 us (+0.6%) with only
2,048 contacts. Evolving trajectories are not used for an A/B claim because the
deterministic tangent basis changes their final contact count.

A proposed 1 mm sticky separation tolerance destabilized the half-scale stack
(1.96 m/s versus the 0.1 m/s limit) and was rejected. The original zero-gap
rule passes all four scale/tower stability tests. Matching (40), friction and
contact behavior (44), determinism/isolation (7), and bunny mesh (1) tests pass.

## J12 - vec4-packed contact impulses, rejected

Mini stores the three contact impulses in one vec4. Applying that layout to the
full shared contact container regressed the fixed 32K stack to 1.836 ms when
used through scalar compatibility accessors. Aggregate loads/stores recovered
the loss and measured 1.793 versus 1.810 ms (+0.9%), below the representation
change threshold. Nsys showed fused prepare/PGS improving 328.6 to 318.8 us
(-3.0%), but warmstart rose 100.0 to 102.5 us and history copy 30.5 to 34.1 us
because their streaming records grew from 12 to 16 bytes. The prototype was
removed. Vec4 packing is valuable for larger always-coaccessed groups (F11),
not automatically for every three-scalar record.

## J13 - select sticky history during canonical gather, accepted

Sticky matching previously materialized five canonical geometry arrays and the
full gather immediately read them back. Selecting matched history directly in
the deterministic gather removes that overlay kernel and 90 net lines. Sticky
history now lives in matcher-owned arrays, avoiding races when callers reuse a
single Contacts buffer.

On the fixed 32K-world, 1,048,576-contact workload, full PhoenX improved from
1.810 to 1.772 ms (+2.1% throughput): 833 GB/s useful lower-bound bandwidth,
56.0% of the 1,489 GB/s sequential roof and 80.4% of the 1,037 GB/s random-vec4
roof. A second profile measured 1.756 ms, 56.5% and 81.1% respectively. Nsys
showed gather falling 127.6 to 98.8 us and removed the 55.5 us overlay; the
wider state save rose 34.5 to 97.2 us, a net 21 us lifecycle reduction.

The evolving 32K workload retained exactly 819,200 final contacts and measured
2.093 versus 2.088 ms (-0.3%, noise). The dense 512-body, 2x2 Kapla guardrail
reduced profiled GPU work about 0.4%. Matching (40) and broad physics,
determinism, isolation, and mesh-contact tests (54) pass.


## J14 - persistent active-count history save, rejected

A 4-block/SM persistent save raised the 1M-contact save kernel from 97.2 to
105.0 us and measured 1.783 versus 1.780 ms end to end. The broad streaming
grid needs its latency hiding; active-count gating already removes tail traffic.

## J15 - precomputed contact midpoint/gap stream, rejected

Streaming a narrow-phase midpoint/gap into matching removed duplicate body
transforms: matching fell 84.2 to 64.1 us. The extra collision output and cache
pollution raised GJK 200.8 to 213.8 us, gather 98.8 to 112.8 us, and save 97.2
to 104.0 us. The frame regressed from 1.7800 to 1.7844 ms. Recompute is cheaper
than another full contact stream.

## J16 - narrower graph metadata, rejected

Two-body-only graph elements improved element projection 37.25 to 34.91 us and
run merge 22.50 to 19.74 us, but coloring rose 50.59 to 56.29 us and run marking
24.19 to 28.96 us; the frame was neutral. Packing world id into the existing
family word halved merge (22.50 to 10.82 us) but raised coloring to 56.99 us and
marking to 28.38 us, for only a noisy 0.5% frame change. Generic graph width is
not the next representation lever.

## J17 - moderate fused-PGS register cap, rejected

The contact-only fused kernel uses 168 registers/thread. A moderate
`launch_bounds=(256, 2)` cap (about 128 registers, unlike the earlier severe
64-register G1 control) measured 1.800 versus 1.780 ms on the fixed 32K-world
workload (-1.1% throughput). The spill/latency cost still exceeds added
occupancy; no cap remains.

## J18 - conditional coloring scans, rejected

An evolving 32K-world run reached 1,212,416 contacts while reporting 0 topology
rebuilds across 120 frames: contact geometry/count changed, but the deterministic
body-pair graph remained reusable. PhoenX nevertheless executes two global
world-stream scans before its existing CUDA conditional. The trace measures
about 25.5 us per CUB scan plus 24.2 us for count/mark, roughly 75 us/frame.
Moving CUB scans directly into the conditional is illegal because their graph
contains allocation nodes. An allocation-free hierarchical tile scan was exact
at 1M, 4M, and non-power-of-two sizes. At 1M it took 17.20 us versus CUB at
8.10 us; a true conditional took 22.95 us and a skipped conditional 5.29 us.
At 4M it took 43.05 versus 12.29 us. The projected stable-topology frame gain
is only about 2%, while rebuild-heavy scenes regress. The prototype was removed.


## J19 - segmented per-world contact sort, not transferred

A fixed-capacity per-world bin, 64-element tile sort, and stable compaction
produced exactly the same keys/source indices as the global 26-bit radix sort.
At 32K worlds, 1,048,576 live contacts, and 2,097,152 slots it took 69.35 us
versus 110.61 us (1.59x). Nsys attributes 31.94 us to atomic binning, 24.88 us
to tile sort, 6.40 us to compaction, and 3.46 us to the world-count scan.
The isolated saving projects to only 2--3% of the full frame and requires the
world-major canonical order that previously raised the full collision gather
23.3 us and regressed the frame 2.7%. The extra collision path is not justified;
the prototype was removed.

## J20 - warp-local in-place sticky lifecycle, implementation rejected

A one-warp/world design can theoretically load previous canonical geometry,
match 32 current contacts, overwrite the same history safely after warp sync,
and thereby replace global match, gather, and save passes. Both a fully
unrolled shuffle implementation and a runtime-loop version compiled for more
than two minutes in Warp/NVRTC. This form fails compile-time and code-size
requirements before runtime qualification. It was removed. The architectural
idea remains valid only if a future tiled implementation generates a much
smaller kernel.

## J21 - canonical anchors instead of solver copies, accepted

PhoenX now reads rigid local points from canonical `Contacts` and removes six
duplicate float planes from its contact container. Persistent rows fall 18 to
12 generally and 12 to 6 for rigid color-packed rows. No solver path was added.

RTX PRO 6000, one substep/four iterations:

| Workload | J13 | J21 | Gain | J21 roofline |
| --- | ---: | ---: | ---: | ---: |
| 32K x 8 fixed, 1,048,576 contacts | 1.697 ms | **1.661 ms** | **+2.2%** | 59.7% sequential, 85.7% random-vec4 |
| 32K x 8 evolving, 1,048,576 contacts | 2.009 ms | **1.981 ms** | **+1.4%** | 50.1% sequential, 71.9% random-vec4 |
| 8K x 8 evolving, 278,528 contacts | 531.14 us | 536.66 us | -1.0% | 49.1% sequential, 70.5% random-vec4 |
| 11K-body Kapla, 6x10 | 12.611 ms | **12.289 ms** | **+2.6%** | 57.5% sequential, 82.5% random-vec4 |

Four colored-row, 38 rigid/friction/simple, one reduced-contact, and the full
60-frame cloth-on-box settling test pass. The small evolving loss is accepted
for the broad above-L2 wins, 24 MiB less state per 1M contacts, and removal of
the duplicate anchor API.

## J22 - device-phase sticky-history ping-pong, rejected

Mini compared the current random gather + streaming save against graph-safe
double buffering, both as two arrays and as one 2N array indexed by a device
phase. At 1,048,576 contacts:

| Fresh contacts | Current | 2N offset ping-pong | Result |
| ---: | ---: | ---: | ---: |
| 0% | 150.85 us | 336.15 us | -55.1% |
| 10% | 181.85 us | 332.64 us | -45.3% |
| 100% | 274.32 us | **265.76 us** | +3.2% |

Fusing random history reads with seven streaming output fields destroys memory
latency hiding. The all-fresh gain projects to only 0.5% of the full frame and
overfits contact churn. Keep random gather and streaming publication separate;
the prototype was removed.

## J23 - exact-key sticky fast lane, rejected before implementation

Four post-settle evolving 8K-stack frames had exact prior keys for 100% of
contacts, but only 13--21% passed sticky spatial/gap gates; fixed input was
100% matched. Separated rows already exit before pair search, and one evolving
frame mapped only 43% of accepted matches to the identical sub-key. An exact
key shortcut mainly specializes fixed topology and cannot remove matching or
fallback work in changing scenes. No solver code was written.

## J24 - matcher-state ping-pong, transfer rejected

A 2N device-phase buffer for only key/midpoint/normal beat separate match+save
at 4M contacts: 1.17x fully matched and 1.25x at 80% fresh, reaching 94% and
80% of sequential peak. A full transfer improved fixed 32K input about 1%, but
changed an evolving trajectory to 1.343M contacts versus 1.049M. Cause: match
sees fresh geometry, while saved matcher state must reflect geometry selected
by sticky canonical gather. Publishing after selection either repeats the save
or fuses into the random gather rejected by J22. The prototype was removed.

## J25 - reconstruct contact tangent from normal, rejected

The tangent is a deterministic function of the stored normal. Removing its
three float planes reduced rigid colored rows 6 to 3 and saved 12 MiB per 1M
contacts. A 32K changing-stack prototype improved 1.948 to 1.901 ms (+2.5%),
but Kapla regressed 12.222 to 12.797 ms (-4.5%) even after callers reused their
loaded normal. Its persistent PGS loop keeps the stored tangent cache-friendly;
reconstruction ALU repeats every sweep. Do not compress/decode the frame in the
hot solver or add a workload-specific layout. The prototype was removed.

## J26 - fuse rigid column pack and coloring projection, rejected

A rigid-only kernel replaced the separate contact-column pack and contact tail
of element projection. At 1,048,576 fixed contacts, Nsys measured 39.23 + 37.87
us control versus 55.68 us fused, saving 21.42 us locally. The paired 800-frame
result improved only 1.64650 to 1.64181 ms (+0.29%): 60.22% to 60.39% of
sequential peak and 86.48% to 86.73% of random-vec4 peak. The extra orchestration
and rigid-only kernel are not justified by the frame result; removed.

## J27 - shared-memory warp-local sticky lifecycle, rejected

A native CUDA runtime-loop implementation fixed J20 compile time (234--246 ms)
and fused deterministic 32-contact/world match, gather, and history publication.
At 32K worlds/1,048,576 contacts with 20% fresh points, the production-like
four-point manifold test was exact but took 175.05 us versus 171.68 us for
three tuned kernels (-1.9%). Block sweeps from 256 to 32 threads did not beat a
256-thread control. The fused kernel reached 1,126 GB/s useful traffic, 75.6%
of sequential and 108.6% of random-vec4 peak. Loading every possible history
row into shared memory costs more than selected random reads; removed.

## J28 - shared body working set plus interleaved rows, accepted in mini

A single native subwarp kernel supports 4/8/16/32-lane worlds. It loads packed
linear velocity/inverse mass, angular velocity, and three inverse-inertia rows
once per body, solves every color/iteration in shared memory, then writes each
body once. Constraint rows retain the interleaved-world layout so active lanes
across a warp touch the same sectors. The superseded 157-line tiled prototype
was removed. Conditional graph nodes are not involved.

RTX PRO 6000, 1,048,576 contacts, one substep/four iterations, above L2:

| Bodies/world | Global body state | Shared + interleaved | Gain | Sequential | Random vec4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 1.489 ms | **1.325 ms** | **+12.4%** | 74.8% | 107.5% |
| 8 | 1.463 ms | **1.283 ms** | **+14.0%** | 77.3% | 111.0% |
| 16 | 1.565 ms | **1.375 ms** | **+13.8%** | 72.1% | 103.6% |
| 32 | 1.608 ms | **1.475 ms** | **+9.0%** | 67.2% | 96.5% |

At 32K changing worlds, both paths ended at 1,245,184 contacts, 38
constraints/world, 10 colors, and zero overflow. Time improved 1.930 to 1.756
ms (+9.9%), reaching 67.0% sequential and 96.3% random-vec4 bandwidth. Nsys
measured the fixed 8-body PGS median at 309 us global versus 147 us cached
(+52.4% kernel throughput). Two cached runs stayed bitwise identical for 120
frames; first-step parity passed at all four world sizes. Caching velocity only
was 1.9% slower than caching velocity plus inertia. Transfer the complete body
working set and interleaved colored rows to full PhoenX; do not enable a cache
with unused logical lanes.
