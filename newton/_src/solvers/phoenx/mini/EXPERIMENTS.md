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
2. Reuse the static revolute topology and its coloring across frames; only
   dynamic contacts should be gathered and colored.
3. Replace the rejected dynamic shared tile with a warp-shuffle or native CUDA
   body cache only if compile time and register count remain bounded.
4. Add warm start as a separate convergence/throughput checkpoint; do not pay
   sticky-matching cost without consuming its output.
5. Measure convergence-to-error, not only equal iteration count, once warm
   start is available.

A capability remains optional when it regresses throughput unless correctness
or convergence repays the cost.
