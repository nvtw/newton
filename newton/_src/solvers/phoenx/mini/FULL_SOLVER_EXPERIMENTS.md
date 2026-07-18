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
