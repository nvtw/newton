# Full PhoenX Throughput Experiments

This is an append-only ledger for changes to the full PhoenX solver. Results
must use the same Newton collision pipeline, scene, contact policy, substeps,
and iteration count as the comparison run. Kernel-only improvements and
end-to-end improvements are reported separately.

## Hardware and rooflines

Hardware: NVIDIA RTX PRO 6000 Blackwell Workstation Edition (96 GB). Software:
Warp 1.15.0.dev20260626, CUDA toolkit 12.9, driver 13.1.

The accepted above-cache calibration from the mini experiments is:

- Sequential float bandwidth: **1,489.14 GB/s**.
- Random scalar gather: **609.60 GB/s**.
- Random vec4 gather: **1,036.82 GB/s**.
- Independent FP32 FMA: **87.810 TFLOP/s**.
- vec3 dot/cross: **82.496 TFLOP/s**.

Hardware performance counters remain unavailable because the driver reports
`ERR_NVGPUCTRPERM`. Roofline percentages are therefore algorithmic lower-bound
models unless explicitly identified as profiler counters.

## F0 — full-solver baseline

Workload: 32,768 worlds, eight free boxes per world on shared ground, one
substep, four PGS iterations, point friction, sticky contact matching, and 200
captured-graph replays after 30 settle and 20 warm-up steps.

- Frame time: **4.025 ms**.
- World-steps/s: **8.140 M**.
- Body-steps/s: **65.122 M**.
- Final contacts: 1,030,013.
- Comparable mini C2 sticky time: **2.866 ms**.
- Full PhoenX is **1.40x slower** than mini C2 sticky end to end.

The full solver does not yet have an accepted logical-byte model. Reporting a
memory-throughput percentage using the mini byte count would be misleading
because the full path performs additional warm-start, ingest, and scheduling
traffic.

## F1 — consume prepared contact lever arms

The multi-world iterate recomputed two quaternion rotations and loaded body
poses, local anchors, and margins during every contact sweep even though rigid
contact preparation already stored the two world-space lever arms in the
derived contact stream. F1 consumes those prepared values. It introduces no
new allocation, solver path, or feature gate.

A controlled 30-replay Nsight Systems A/B used the F0 workload. Final contact
counts were 1,176,460 before and 1,176,795 after, a 0.03% difference.

| Variant | Fused prepare/solve median | Change |
| :--- | ---: | ---: |
| Recompute lever arms per sweep | 669.58 us | baseline |
| Consume prepared lever arms | **573.37 us** | **-14.4%** |

The profiler-run end-to-end time remained about 3.880 ms and was dominated by
other kernels and variable collision/sort work. A separate 200-replay run was
3.858 ms, but it had 4.3% fewer final contacts than F0 and is rejected as a
clean end-to-end speedup claim. F1 is accepted as a hot-kernel improvement;
its end-to-end gain is not yet statistically resolved.

The next target is common contact scheduling and state maintenance. In the F1
trace, median times included 528.4 us for per-world coloring, 267.6 us for
contact matching, 208.6 us for warm-start gather, 140.0 us for current-to-prev
state copy, and 130.1 us for sorted-state save. These costs explain most of the
remaining gap to mini PhoenX.


## F2 — one deterministic greedy thread per world

The block-cooperative Jones-Plassmann colorer used 64 GPU threads, MIS rounds,
block barriers, and atomic histogram/scatter operations for every small world.
Mini C1 demonstrated that tens of thousands of RL worlds provide the required
parallelism and that each tiny graph is cheaper to color serially. F2 therefore
uses one deterministic smallest-free-color thread per world. It retains the
same generic 1--8 endpoint interaction representation and family-grouped CSR;
it is not restricted to contacts or a particular joint type.

The serial owner initializes only color buckets it actually uses. Two
`[world, 64]` histogram/cursor scratch planes and the unused priority input were
removed. An attempted exact-priority selection variant is rejected: its
quadratic local search regressed the 8K robot frame to 1.355 ms.

Controlled 32K stack Nsight runs used 30 captured replays. The old and new
final contact counts were 1,176,795 and 1,146,880, a 2.5% workload difference.

| Metric | F1 block/MIS | F2 serial greedy | Change |
| :--- | ---: | ---: | ---: |
| Coloring kernel median | 528.41 us | **83.20 us** | **-84.3%** |
| Captured frame | 3.880 ms | **3.211 ms** | **-17.2%** |
| World-steps/s | 8.445 M | **10.206 M** | **+20.9%** |
| Logical minimum bandwidth | 426.9 GB/s | **503.0 GB/s** | **+17.8%** |
| Sequential DRAM roofline | 28.7% | **33.8%** | +5.1 points |
| Random vec4 roofline | 41.2% | **48.5%** | +7.3 points |

The bandwidth model deliberately reuses the mini C2 lower bound of 352 unique
bytes/contact-iteration. It understates full-PhoenX traffic from matching,
warm-start state, ingest, and sorting and is not a hardware-counter reading.
Estimated contact-row compute is 0.643 TFLOP/s, or 0.73% of measured FP32 FMA.

The longer 200-replay 32K run reaches **3.086 ms** and **10.618 M
world-steps/s**, 1.30x the F0 baseline. Its final settled manifold has fewer
contacts than F0, so the controlled profiler comparison above is the accepted
performance claim. It is 7.7% slower than mini C2 sticky at 2.866 ms, but that
cross-solver comparison also ends with different manifolds.

Physics qualification passed 11 multi-world/color/contact/stacking/friction
tests plus 41 tests covering mixed joint modes, basic joints, chain
convergence, prismatic, ball-socket, and fixed constraints. The deterministic
color order changes floating-point trajectories, as any PGS reorder does; the
throughput result is accepted only with the preceding behavioral coverage.

## F3 — direct endpoint color ownership

F2 still rebuilt a global adjacency CSR every step and then made each serial
world owner walk that scattered CSR to discover forbidden colors. Mini colors
directly from endpoint ownership. F3 applies that exact idea to the full
solver: one 64-bit color mask is stored per unified body/particle node, and the
world owner ORs the masks of an element's 1--8 endpoints. It clears only nodes
touched by that world and stamps the selected bit back to those endpoints.
The representation remains generic across all constraint families.

The direct multi-world greedy path no longer launches adjacency clear, count,
scan, store, or incremental-loop initialization. The single-world algorithms
still build adjacency. If a multi-world graph exceeds 64 colors, the existing
JP fallback first rebuilds adjacency and remains correct rather than consuming
stale data.

A controlled 8K robot A/B toggled only the now-unnecessary adjacency reset.
Both variants used direct-mask coloring, 500 captured replays, sticky matching,
32,768 final contacts, and 65,536 revolute constraints:

| Variant | Frame | World-steps/s | Change |
| :--- | ---: | ---: | ---: |
| Rebuild unused adjacency | 535.20 us | 15.306 M | baseline |
| Direct ownership only | **515.88 us** | **15.880 M** | **+3.75%** |

The accepted full result represents 292.70 GB/s of useful prepared-row work:
**19.66%** of sequential bandwidth, **28.23%** of random-vec4 bandwidth, and
**48.01%** of random-scalar bandwidth. Estimated useful row compute is 0.419
TFLOP/s, or **0.477%** of FP32 FMA peak. These are lower bounds: full-PhoenX
matching, ingest, sorting, and scheduling bytes are deliberately excluded.

On the identical current harness, tuned mini C4 sticky takes 344.65 us and
reaches 438.11 GB/s, 29.42% of sequential bandwidth, 42.26% of random-vec4
bandwidth, and 0.715% of FP32 peak. Full PhoenX is now 1.50x slower on this
stable mixed contact/revolute scene; its remaining gap is no longer adjacency
construction.

The 32K above-L2 stack takes 2.971 ms over the standard 200 replays, or 11.028
M world-steps/s. Its final 688,128-contact manifold gives a 326.09 GB/s useful
work lower bound, 21.90% of sequential bandwidth, 31.45% of random-vec4
bandwidth, and 0.475% of FP32 peak. A same-harness mini run takes 2.731 ms but
ends with 1,026,090 contacts. Frame time and processed-contact rate are both
reported because reordered PGS trajectories make the settled manifolds
non-identical; the 9% frame-time gap alone is not a clean solver-throughput
comparison.

Nsight confirms the direct run contains element rebuilding, world bucketing,
coloring, contact preparation, and the complete PGS solve, while the unused
adjacency kernels are absent. Physics qualification passes 52 selected tests
covering multi-world ordering, colored contacts, stacking/friction, mixed joint
modes, chain convergence, prismatic, ball-socket, and fixed constraints.

## F4 — scheduler-shaped color output

The shared greedy colorer always built three family sub-prefixes per color even
when the selected block-world scheduler consumed only the color prefix. F4
keeps one source algorithm and one data representation but compiles its family
grouping choice through a cached kernel factory. Fast-tail retains joint,
contact, and deformable family grouping. Block-world writes one stable bucket
per color and avoids the unused family loads, counters, prefix loops, and
scatter arithmetic.

The controlled 8K mixed contact/revolute profile keeps the fused solve
unchanged at 190.0 us and reduces coloring from 35.58 us to **24.61 us**
(**-30.8%**). A 1,000-replay stable-manifold run improves:

| Metric | F3 | F4 | Change |
| :--- | ---: | ---: | ---: |
| Frame | 515.88 us | **506.30 us** | **-1.86%** |
| World-steps/s | 15.880 M | **16.180 M** | **+1.89%** |
| Useful-work bandwidth | 292.70 GB/s | **298.23 GB/s** | **+1.89%** |
| Sequential bandwidth roofline | 19.66% | **20.03%** | +0.37 points |
| Random-vec4 roofline | 28.23% | **28.76%** | +0.53 points |
| FP32 FMA roofline | 0.477% | **0.486%** | +0.009 points |

The same current mini C4 sticky result is 344.65 us, so full PhoenX is now
1.47x slower on this stable workload. The remaining gap is dominated by the
physically richer full prepare/solve and state pipeline, not coloring.

The compile-time family-grouped 32K fast-tail run takes 2.986 ms with the same
688,128 final contacts as F3's 2.971 ms. Its 0.5% difference is within observed
run variance; useful-work lower bounds are 324.44 GB/s, 21.79% of sequential
bandwidth, 31.29% of random-vec4 bandwidth, and 0.472% of FP32 peak.

Two related hypotheses are rejected and leave no code behind. Removing the
rigid joint iterate's redundant-looking access-mode checks regressed the stable
robot from 515.88 us to 521.32 us, likely through less favorable generated-code
scheduling. Lazy per-node color epochs removed the mask-clear pass but added
8 bytes per node and measured 516.13 us, statistically neutral. A first runtime
family-group flag sped up block-world but repeatedly regressed fast-tail to
2.996--3.000 ms; the accepted cached factory removes that runtime branch.

Physics qualification passes 53 selected tests spanning both scheduler
variants, multi-world ordering, colored contacts, stacking/friction, mixed
joint modes, chain convergence, prismatic, ball-socket, and fixed constraints.


## F5 — phase-split control (rejected)

Mini compiles preparation and PGS iteration as separate kernels, while the
full block-world scheduler fuses both dispatchers. A generic compile-time phase
prototype reused the existing joint/contact implementations and launched one
prepare kernel followed by one iterate kernel. This tested register pressure
and instruction-cache footprint without introducing a joint-specific solver.

Two 3,000-replay runs per variant on the stable 8K mixed robot measured:

| Variant | Runs | Median | Change |
| :--- | ---: | ---: | ---: |
| Fused prepare + iterate | 511.88, 513.66 us | 512.77 us | baseline |
| Separate prepare / iterate | 510.94, 512.82 us | 511.88 us | -0.17% |

The difference is below run variance. Any smaller phase kernels are offset by
the extra graph launch, so this hypothesis is rejected and leaves no solver
path behind. The result also narrows the representation work: an Nsight range
on the same workload attributes about 194 us to full prepare/iterate versus
about 146 us for mini preparation plus solve, while full world bucketing
performs two additional radix passes costing about 21 us/frame. Removing or
shrinking actual state and scheduling traffic is more promising than
rearranging the same dispatcher code.


## F6 — stable monotone-run world bucketing

The unified CID stream is already piecewise world-monotone: model constraint
families are world-grouped and Newton contacts arrive sorted by shape pair. The
stable global world sort nevertheless performed two additional CUB radix
passes on the 8K robot and moved the entire active stream through ping-pong
buffers. F6 marks every world-id decrease, scans those run flags, and has one
thread per world stable-merge the monotone runs with binary searches. Original
CID order is preserved exactly within each world; inactive elements are
excluded correctly and arbitrarily interleaved input remains supported. No
atomic scatter or joint-specific path is
introduced. Existing two-capacity radix scratch is reused for flags, run ids,
run starts, and output, so allocation size does not grow.

A detached worktree at commit `11899d34` provided an immediate source-bracketed
control under the same GPU clock state. Both variants used 3,000 captured
replays, sticky matching, 32,768 contact points, 65,536 revolute constraints,
one substep, and four iterations:

| Metric | F4/S2 radix control | F6 run merge | Change |
| :--- | ---: | ---: | ---: |
| Frame | 498.96 us | **474.57 us** | **-4.89%** |
| World-steps/s | 16.418 M | **17.262 M** | **+5.14%** |
| Useful-work bandwidth | 302.62 GB/s | **318.17 GB/s** | **+5.14%** |
| Sequential bandwidth roofline | 20.32% | **21.37%** | +1.05 points |
| Random-vec4 roofline | 29.19% | **30.69%** | +1.50 points |
| FP32 FMA roofline | 0.494% | **0.519%** | +0.025 points |

The benchmark now reports `world_id_runs`; this robot stream has two runs
(joints and contacts). Against mini C4 at 344.65 us, full PhoenX narrows from
about 1.45x in the bracketed control to **1.38x**.

The same immediate bracket on the 32K above-L2 stack retained exactly 688,128
contact points:

| Metric | Radix control | F6 run merge | Change |
| :--- | ---: | ---: | ---: |
| Frame | 2.995 ms | **2.756 ms** | **-7.98%** |
| World-steps/s | 10.942 M | **11.891 M** | **+8.67%** |
| Useful-work bandwidth | 323.53 GB/s | **351.58 GB/s** | **+8.67%** |
| Sequential bandwidth roofline | 21.73% | **23.61%** | +1.88 points |
| Random-vec4 roofline | 31.20% | **33.91%** | +2.71 points |
| FP32 FMA roofline | 0.471% | **0.512%** | +0.041 points |

Nsight confirms the two world-bucketing radix passes disappear; the three
remaining radix passes belong to Newton contact sorting/matching. A follow-on
fused the 19.04 us run merge into the 24.23 us serial colorer, but the fused
kernel measured 43.01 us. It rearranged rather than removed work, so the
prototype is rejected and the simpler shared merge remains.

Qualification passes an adversarial eight-run stable-order test, multi-world
cloth particle ownership, world isolation, a 1,024-world stress scene,
cached-prepare contacts, and representative ball-socket, revolute-limit, and
prismatic-drive/limit tests. The adversarial test compares the exact output
against a CPU stable world sort rather than merely checking finite physics.


## F7 — sparse optional revolute rows

Matched CUDA-graph-node profiles isolate the remaining mixed-workload solver
gap. At 8K worlds, full PhoenX's fused prepare/iterate kernel had a 177.01 us
median, while mini's packed prepare and solve medians summed to 146.13 us
(41.87 + 104.26 us). A contact-only control then showed that full PhoenX was
already slightly faster per solved contact: 1.582 versus 1.548 billion
constraint-iterations/s. The actionable gap was therefore the richer joint
path, not contact packing.

The register-cached revolute loop read drive coefficients, accumulated drive
impulse, friction state, inverse axial mass, and the world hinge axis for every
joint. It also evaluated the axial relative velocity and applied two
inverse-inertia matrix products every sweep. Those operations are physically
inactive when drive mode is off, no limit is clamped, and axial friction is
zero. F7 derives one data-driven activity condition from the existing union
fields, loads optional payload only for active rows, and skips the zero axial
solve for passive hinges. There is no new solver variant, joint-specific
sidecar, allocation, or change to the coupled five-row hinge lock. Driven,
limited, and frictional hinges retain the original equations.

An immediate alternating source bracket at 8K worlds used 3,000 graph replays,
32,768 contacts, 65,536 revolute constraints, one substep, and four PGS
iterations:

| Metric | F6 control | F7 sparse rows | Change |
| :--- | ---: | ---: | ---: |
| Frame | 474.58 us | **470.00 us** | **-0.97%** |
| World-steps/s | 17.261 M | **17.430 M** | **+0.98%** |
| Useful-work bandwidth | 318.16 GB/s | **321.27 GB/s** | **+0.98%** |
| Sequential bandwidth roofline | 21.37% | **21.57%** | +0.21 points |
| Random-vec4 roofline | 30.69% | **30.99%** | +0.30 points |
| FP32 FMA roofline | 0.519% | **0.524%** | +0.005 points |

The final-source node profile reduces the fused solver median from 177.01 to
**172.67 us (-2.45%)**. Unrelated coloring and run-merge medians remain within
0.1%, locating the change in the intended kernel. Against the recorded mini C4
reference at 344.65 us, full PhoenX narrows from 1.38x to about **1.36x**.

At 32K mixed worlds, the prepared joint and body streams exceed L2. A final
1,000-replay source bracket keeps exactly 131,072 contacts and 262,144 joints:

| Metric | F6 control | F7 sparse rows | Change |
| :--- | ---: | ---: | ---: |
| Frame | 1.639 ms | **1.601 ms** | **-2.29%** |
| World-steps/s | 19.994 M | **20.463 M** | **+2.35%** |
| Useful-work bandwidth | 368.52 GB/s | **377.17 GB/s** | **+2.35%** |
| Sequential bandwidth roofline | 24.75% | **25.33%** | +0.58 points |
| Random-vec4 roofline | 35.54% | **36.38%** | +0.84 points |
| FP32 FMA roofline | 0.601% | **0.615%** | +0.014 points |

Physics qualification passes 13 tests covering the multi-world G1 driven-joint
parity case across fast-tail and both block-world sizes, joint drive/friction
composition, revolute limits, passive hinges, cached prepare, and coupled-chain
convergence. The exact G1 active-drive test also passes again after the final
axis-load hoist. The representation lesson is to keep mode payload in one
shared union but make optional row traffic proportional to active features;
allocating or reading the full union in every PGS iteration defeats the compact
layout.


## F8 — final body-state export fusion (rejected)

The default Newton state export followed PhoenX's final per-body damping,
world-inertia refresh, and force clear. A prototype extended that existing
kernel to optionally write ``body_q`` and ``body_qd`` after the final updates,
then removed the standalone export launch. Direct ``PhoenXWorld`` callers used
the same kernel with export disabled; finite-difference and substep-average
readouts retained their existing exporters.

An immediate source bracket measured 482.76 -> 480.42 us at 8K mixed worlds
(**+0.49% throughput**) and 1.5999 -> 1.5969 ms at 32K worlds (**+0.19%**).
The saved launch and shared body reads are too small relative to the collision,
scheduling, and PGS pipeline. The implementation required three internal step
arguments, two graph-stable placeholder allocations, and alternate-readout
routing, so it fails the no-bloat threshold. The prototype is fully reverted.
The next representation change must target joint-row or scheduling traffic
that scales with constraints rather than a once-per-frame body export.


## F9 - cooperative merge and basis reconstruction (rejected)

Two small traffic controls were measured and fully reverted. First, the
monotone-run world merge assigned 2, 4, 8, or 16 cooperative lanes to each
world while preserving exact stable CID order. Their 8K kernel medians were
18.02, 17.79, 17.92, and 18.24 us versus 19.15 us for one lane. Four lanes
therefore saved 7.1% inside the merge kernel, but duplicated binary searches
erased it end to end: 480.41 -> 480.74 us at 8K and 1.5976 -> 1.6004 ms at
32K. The scalar owner remains simpler and faster at frame scope.

Second, the revolute loop reconstructed its deterministic tangent basis from
the cached world axis, replacing six tangent scalar loads with three axis
loads. Cheap-looking arithmetic still increased register/instruction pressure:
the immediate 8K bracket regressed 479.21 -> 486.71 us (**-1.54%
throughput**). This prototype is also fully reverted. The controls reinforce
that useful traffic must be removed without adding per-thread setup to the
already register-heavy joint loop.


## F10 - family-aliased mutable joint state

The unified joint record mixed read-mostly prepared geometry/Schur data with
nine mutable warm-start dwords: two shared vec3 impulses plus drive, limit, and
friction scalars. PhoenX already allocated one generic 12-dword multiplier
sidecar per constraint for family-aliased mutable state. F10 uses its exact
capacity for joint impulses: shared anchors at slots 0--5, the mode-exclusive
third anchor at 6--8, and axial drive/limit/friction at 9--11. No array,
constraint-type sidecar, feature flag, or solver path is added.

All joint modes and both maximal-tree projectors now use the same generic
accessors. Removing the dead fields shrinks the unified joint record from 126
to **117 dwords (468 B), -7.14%**. More importantly, the hot loop no longer
mixes stores to the large prepared record: mutable state has a compact base
pointer and family-local working set.

An immediate source bracket used the committed F7 worktree as control, 3,000
captured replays, 32,768 contacts, 65,536 revolute constraints, one substep,
and four PGS iterations:

| Metric | F7 control | F10 compact state | Change |
| :--- | ---: | ---: | ---: |
| Frame | 480.339 us | **465.280 us** | **-3.14%** |
| World-steps/s | 17.055 M | **17.607 M** | **+3.24%** |
| Useful-work bandwidth | 314.35 GB/s | **324.52 GB/s** | **+3.24%** |
| Sequential bandwidth roofline | 21.11% | **21.79%** | +0.68 points |
| Random-vec4 roofline | 30.32% | **31.30%** | +0.98 points |
| Random-scalar roofline | 51.57% | **53.24%** | +1.67 points |
| FP32 FMA roofline | 0.513% | **0.529%** | +0.016 points |

The final graph-node profile reduces fused prepare/iterate from F7's 172.67 us
to **168.70 us (-2.30%)** while coloring remains 24.35 us. A clock-adjacent
current mini/full pair measures 360.28 and 468.07 us, respectively, narrowing
the stable mixed-workload gap to about **1.30x**.

At 32K worlds the immediate result is neutral within observed clock variance:
1.5961 ms control versus 1.5985 ms compact state (-0.15% throughput). The
working set already exceeds L2, and the sidecar does not reduce the number of
hot multiplier loads/stores; F10 is accepted for the clear 8K/compiler-cache
win, smaller representation, and absence of an above-L2 material regression,
not as a 32K speedup claim.

Qualification passes 62 selected tests covering every revolute/prismatic
drive and limit mode, joint friction/stiction, ball, fixed, cable, a 30-link
chain, general and specialized maximal-tree projection, multi-world loops, and
the G1 drive-coefficient graph check. The existing poisoned G1 reset test still
fails after the subsequent step because body 0's deliberately poisoned angular
velocity remains nonfinite; the committed F7 control fails identically. Its
new sidecar checks are finite immediately after reset and after the step, so
that pre-existing reduced-mode body-reset defect is recorded but is not
attributed to F10.


## F11 - vec4-packed generic multiplier sidecar

F10 still stored its twelve mutable multiplier dwords as twelve independent
scalar planes. A joint impulse vector therefore issued three plane loads and
stores even though its components are always consumed together. F11 changes
the same generic 48-byte sidecar into three vec4 planes. Each joint maps an
impulse to xyz and its correlated axial scalar to w: drive with the first
impulse, limit with the second, and friction with the mode-exclusive third.
Scalar deformable families retain logical dword offsets through generic
read-modify-write accessors. No bytes, arrays, feature flags, or solver paths
are added.

An alternating source bracket against detached F10 used the stable robot
workload, sticky matching, one substep, four PGS iterations, and 3,000 replays
at 8K worlds. Medians combine two candidate and two control runs:

| Metric | F10 scalar planes | F11 vec4 planes | Change |
| :--- | ---: | ---: | ---: |
| Frame | 488.89 us | **442.53 us** | **-9.48%** |
| World-steps/s | 16.756 M | **18.512 M** | **+10.48%** |
| Useful-work bandwidth | 308.85 GB/s | **341.21 GB/s** | **+10.48%** |
| Sequential bandwidth roofline | 20.74% | **22.91%** | +2.17 points |
| Random-vec4 roofline | 29.79% | **32.91%** | +3.12 points |
| FP32 FMA roofline | 0.504% | **0.557%** | +0.053 points |

The above-L2 32K bracket uses 1,000 replays, 131,072 contacts, and 262,144
revolute constraints. Two candidate runs surround the control:

| Metric | F10 scalar planes | F11 vec4 planes | Change |
| :--- | ---: | ---: | ---: |
| Frame | 1.595 ms | **1.282 ms** | **-19.62%** |
| World-steps/s | 20.542 M | **25.558 M** | **+24.42%** |
| Useful-work bandwidth | 378.63 GB/s | **471.08 GB/s** | **+24.42%** |
| Sequential bandwidth roofline | 25.43% | **31.63%** | +6.20 points |
| Random-vec4 roofline | 36.52% | **45.44%** | +8.92 points |
| FP32 FMA roofline | 0.618% | **0.768%** | +0.150 points |

The unchanged byte count and stronger above-L2 result identify memory
instruction shape and coalescing, rather than cache footprint, as the cause.
With identical sticky contact matching, current mini measures 368.34 us at 8K
and 1.337 ms at 32K. Full PhoenX remains 20.1% slower in the cache-resident
case, where general scheduling has a fixed cost, but is **4.1% faster than
mini above L2**. With mini's default disabled matching it measures 1.193 ms,
leaving full 7.5% slower while doing the additional production work.

Contact-only Kapla is intentionally unaffected. A source bracket gives 79.00
-> 79.24 FPS (+0.3%) for 1x1 and 33.04 -> 33.50 FPS (+1.4%) for 2x2, with
bit-identical drift, extrema, contact counts, and colors. The 2x2 candidate
still reaches 1,411.31 GB/s, **94.77% of sequential bandwidth**.

Qualification passes 96 sampled tests spanning all joint modes, drives,
limits, friction, fixed and cable joints, scalar and block deformables, cloth,
maximal and general loop projectors, and the G1 implicit-drive graph check.
The known G1 poisoned-reset test again fails only after stepping because its
poisoned body angular velocity survives reduced-mode reset; all three packed
multiplier groups are finite, and detached F10 has the identical body failure.
The experiment's central lesson is that equal byte counts are not equal memory
systems: grouping always-coaccessed scalars into aligned vector transactions
can matter more than shrinking a schema.


## S0 — large single-world Kapla baseline

The F2--F4 changes target independent-world ownership and block-world output;
they are not expected to improve one large connected island. A separate
single-world baseline now prevents multi-world wins from hiding a Kapla
regression. The production-style case has 6 substeps, 10 PGS iterations, two
physics ticks per reported frame, mass splitting, point friction, and the
Newton SAP/narrow-phase/matching pipeline.

| Scene | Bodies | Final points / columns | Colors | Frame | Useful-work bandwidth | Sequential roofline | Random-vec4 roofline | FP32 roofline |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Canonical 1x1 | 11,341 | 250,847 / 72,022 | 9 | 12.948 ms (77.23 FPS) | 818.32 GB/s | 54.95% | 78.93% | 1.19% |
| Above-L2 2x2 | 45,361 | 997,472 / 287,150 | 9 | 29.957 ms (33.38 FPS) | 1,406.47 GB/s | **94.45%** | 135.65% | 2.05% |

These are algorithmic useful-work estimates, not GPU-counter readings: 352
bytes and 450 FLOP per final contact-point iteration, using the final contact
count as an approximation for the measured trajectory. Exceeding the
random-vec4 microbenchmark is possible because contact columns reuse body
state and mass-splitting slots across several points. The 2x2 sequential
percentage is the meaningful upper comparison: at 94.45%, wholesale extra
row traffic cannot plausibly help.

Nsight Systems shows why the recent multi-world coloring work does not move
Kapla. On 1x1, persistent prepare/iterate/relax plus mass-splitting averaging
consume about 71% of GPU time; coloring histogram/scatter and speculative
coloring are about 1%. On 2x2 the solver kernels are about 58%, SAP 8.6%,
narrow phase 6.3%, radix sorting 5.7%, and contact matching 2.1%. Collision and
sorting therefore become material as the working set leaves L2.

A controlled persistent-grid sweep at 1x1 tested 4, 6, 8, 10, and 12
one-warp blocks per SM. Four blocks measured 75.12 FPS; 6--12 blocks were flat
around 76.7--76.9 FPS. The existing 8-block launch lies on the plateau, so no
scene-size launch heuristic is added. The next accepted single-world change
must improve both 1x1 and 2x2, preserve the contact trajectory within the
physics tolerances, and report both frame time and useful work.


## S1 — color-ordered contact locality control

A source-identical A/B toggled the existing color-ordered contact headers and
rows while keeping mass splitting, PGS iteration count, collision, and final
contact manifolds fixed. This measures the single-world representation lever
rather than attributing old locality work to F2--F4.

| Scene | Canonical contact order | Color order | Change |
| :--- | ---: | ---: | ---: |
| Kapla 1x1 | 67.19 FPS | **76.74 FPS** | **+14.2%** |
| Kapla 2x2 above L2 | 21.37 FPS | **33.16 FPS** | **+55.2%** |

The 2x2 useful-work rate rises from about 900.5 to 1,397.2 GB/s, or 60.5% to
93.8% of the measured sequential bandwidth ceiling. This is the strongest
large-single-world evidence so far: color-slot packing converts scattered PGS
row access into a near-streaming workload, and its advantage grows rather than
shrinks when the working set leaves L2. Future full-solver representations
should extend this one generic packed stream instead of adding joint-type or
scene-specific solve paths.

A follow-on removed the per-point row-to-column slot map. Packed headers keep
both canonical and colored first-contact indices, so in principle the solver
header could remain colored between frames. The change removed a 4 MB map and
about 20 MB/frame of map write/read traffic in 2x2. A bracketed source A/B,
however, measured 1x1 76.82 -> 76.36 FPS (-0.6%) and 2x2 33.08 -> 33.25 FPS
(+0.5%). The trade is below the noise floor and fails the requirement that
both cache-resident and above-L2 scenes improve. The prototype is rejected and
leaves no solver code behind.


## S2 — eliminate unreachable contact-owner loads

The single-sweep contact iterate read ``articulation_owner[cid]`` before every
column solve. That guard was unreachable work: the single-world and fast-tail
factory flags compile classic contacts out when a reduced articulation owns
the contact pipeline, while block-world/multi-sweep dispatch uses a different
entry point that retains its explicit ownership guard. The only callers of the
single-sweep wrappers are the shared single-world dispatcher functions, so the
load and branch can be removed once from the common contact implementation;
no joint-type or scene-specific path is introduced.

A source-bracketed Kapla measurement with identical contact counts gives:

| Scene | S1 | S2 | Change | S2 useful-work bandwidth |
| :--- | ---: | ---: | ---: | ---: |
| Kapla 1x1 | 76.82 FPS | **78.92 FPS** | **+2.7%** | 835.20 GB/s (56.1% sequential) |
| Kapla 2x2 above L2 | 33.08 FPS | **34.40 FPS** | **+4.0%** | 1,449.55 GB/s (**97.3% sequential**) |

The 2x2 estimate is now within 2.7% of the independently measured sequential
bandwidth ceiling. Its estimated useful contact compute is 1.85 TFLOP/s, or
2.11% of FP32 peak; the workload remains bandwidth-shaped. Nsight confirms the
dominant persistent kernel falls from about 5.75 to 5.50 us per launch (-4.3%)
and another sweep variant from 5.94 to 5.30 us (-10.8%); collision kernels are
unchanged.

Physics qualification passes 12 CUDA-graph tests: color-ordered row round-trip
and stack parity, the long reduced Kapla regression, maximal-contact fallback,
world-serial mixed contacts, and both cloth/reduced ownership-disjointness
cases. Those mixed tests directly exercise the ownership invariant rather than
only checking a rigid tower for finite values.
