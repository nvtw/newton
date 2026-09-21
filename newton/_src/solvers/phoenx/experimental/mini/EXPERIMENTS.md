# PhoenX Mini Throughput Experiments

This is the retained decision ledger for the isolated rigid-body mini solver.
Numbers are hardware- and harness-specific; rerun the benchmark before using a
result to justify full-solver changes.

## Method

Use identical generated worlds, contacts, iterations, warm-up, graph replay,
and validation. Report frame time plus processed worlds/bodies/constraints.
Roofline percentages are lower-bound models unless hardware counters are named.
A mini win is evidence, not authorization to transfer it to full PhoenX.

## Retained calibration

RTX PRO 6000 Blackwell, Warp 1.15 development build, CUDA 12.9:

- sequential float bandwidth: 1,489.14 GB/s;
- random scalar gather: 609.60 GB/s;
- random vec4 gather: 1,036.82 GB/s;
- independent FP32 FMA: 87.810 TFLOP/s;
- vec3 dot/cross: 82.496 TFLOP/s.

## Accepted findings

| ID | Finding |
| --- | --- |
| M1 | Consume inverse mass from packed velocity when already resident. |
| M3 | Subwarp world packing helps many small uniform worlds. |
| M4 | Per-body color bitmasks beat rebuilt adjacency for small graphs. |
| M6 | Deterministic direct-to-sorter collision output removes staging. |
| M7 | World-major deterministic scheduling plus interleaved rows is the useful combination. |
| C0a | Block size is workload-dependent; keep a measured selection policy. |
| C1 | Compact deterministic color buckets beat full-list scans. |
| C2 | Color-major vec4 preparation helps the dense mini layout. |
| C4/C4a | Packed mixed contact/revolute colors work; account for schedule footprint. |
| C5 | Stable-topology reuse establishes an upper bound, not a default full-solver policy. |
| J4 | Direct valid-run contact compaction transferred successfully. |
| J9/J13 | Canonical sticky overlay and selection during gather are useful. |
| J21 | Canonical anchors avoid redundant solver copies. |
| J28 | Shared body working set plus interleaved rows wins in the dense mini case. |
| J33/J34 | A conservative hybrid scheduler needs an automatic workload policy. |
| J36 | Lazy warm-start anchors are safe and useful. |
| J39/J40 | Compact sticky offsets and bounded staging grids reduce fixed overhead. |
| J43 | Conditional multi-world coloring is useful when topology is unchanged. |

## Rejected findings

| IDs | Reason |
| --- | --- |
| M2, J52 | Narrow/FP16 prepared rows lost precision or conversion cost exceeded traffic savings. |
| M5, C3 | One thread/lane per world underused the machine or serialized too much work. |
| J0, J1, J29, J35 | Jacobi/fewer-sweep/hybrid/classic-PGS variants failed matched quality or throughput. |
| J2, J3, J41, J42, J44 | Contact staging/compaction fusion increased sorting, synchronization, or scattered traffic. |
| J5--J8, J14, J20, J22--J25, J34 | Sticky-history shortcuts added state, branches, or synchronization without a net win. |
| J10--J12 | Full-solver transfer or packed-impulse variants did not preserve the mini benefit. |
| J15--J18 | Precomputed metadata and conditional scans saved too little work. |
| J19 | Segmented sort did not transfer to the production pipeline. |
| J26, J30, J37 | Fusing preparation/coloring/iteration enlarged live state and regressed. |
| J27, J31, J32, J51 | Shared/register body caches fail on generic sparse working sets or register pressure. |
| J38, J45, J47 | Extra vec4 grouping/payload specialization remained mini-only or regressed full PhoenX. |
| J46, J49, J50 | Proxy/remap approaches did not predict or improve the full mixed workload. |
| J48 | Incremental recoloring had insufficient headroom. |
| J53, J54 | Dense active-set and patch-friction controls did not justify replacement. |
| J55 | Recomputing mass to reduce row loads lost. |
| J57 | Transfer audit rejected remaining dense-mini assumptions for solver corners. |

## Current conclusion

The mini solver demonstrates the ceiling for dense, regular, many-world rigid
workloads. Full PhoenX is dominated by generic contact history, heterogeneous
constraints, sparse articulation traversal, and validation requirements. The
most transferable principles are deterministic compact scheduling, direct
endpoint ownership, bounded metadata, and avoiding redundant state copies—not
blanket vectorization or shared-memory caching.

## Complete experiment ledger

Every original experiment remains represented here. “Accepted” can mean useful
in mini only; full-solver transfer requires a separate result.

| ID | Outcome and retained reason |
| --- | --- |
| M1 | Accepted: reuse inverse mass already packed with velocity. |
| M2 | Rejected: narrower lever arms did not repay conversion/precision cost. |
| M3 | Accepted: subwarp packing improves many small worlds. |
| M4 | Accepted: endpoint color masks remove adjacency construction. |
| M5 | Rejected: one lane per world serialized too much work. |
| M6 | Accepted: deterministic direct collision output removes staging. |
| M7 | Accepted: world-major scheduling and interleaved rows must be evaluated together. |
| C0/C0a | Baseline full-list scan; retain the measured block-size policy, not one universal block size. |
| C1 | Accepted: compact deterministic color buckets avoid scanning inactive rows. |
| C2 | Accepted in dense mini: color-major vec4 preparation improves regular row access. |
| C3 | Rejected: shared tile body cache and one-thread-per-world variants lost. |
| C4/C4a | Accepted: packed mixed contact/revolute colors; include schedule storage in byte models. |
| C5 | Accepted as an upper bound for unchanged topology, not a generic production assumption. |
| F23 | Stable rigid-color reuse transferred only under a strict full-solver predicate. |
| F24 | Rejected: contact-frame/validator changes did not produce a clean gain. |
| F25 | Rejected: cross-frame cold start changed behavior or lost end to end. |
| F26 | Diagnostic only: hardware counters/layout controls define how later claims are measured. |
| J0 | Rejected: deterministic gather-Jacobi did not match PGS quality/performance. |
| J1 | Rejected: fewer over-relaxed sweeps failed the matched-quality gate. |
| J2 | Rejected: AoS deterministic contact staging added traffic. |
| J3 | Rejected: gathering/matching fusion enlarged the critical kernel. |
| J4 | Accepted in full: direct valid-run compaction removes an intermediate pass. |
| J5 | Rejected: deferring history permutation moved rather than removed work. |
| J6 | Rejected: packing during run materialization increased live state. |
| J7 | Rejected: replacing pipeline sticky replay changed lifecycle assumptions. |
| J8 | Rejected: caching the fresh-gap gate did not amortize state traffic. |
| J9 | Accepted in mini: canonical sticky overlay removes duplicate representation. |
| J10 | Full transfer required separate qualification; mini acceptance was insufficient. |
| J11 | Compact deterministic warm-start framing was evaluated as a full-path control. |
| J12 | Rejected: vec4-packed impulses added packing without enough request reduction. |
| J13 | Accepted: choose sticky history during canonical gather. |
| J14 | Rejected: persistent active-count save added state for little work. |
| J15 | Rejected: midpoint/gap precompute added a stream read. |
| J16 | Rejected: narrower graph metadata was not on the limiting path. |
| J17 | Rejected: register caps caused spills or reduced useful occupancy. |
| J18 | Rejected: conditional coloring scans added branches for insufficient savings. |
| J19 | Not transferred: segmented per-world sorting did not fit production layout. |
| J20 | Rejected: in-place warp-local sticky lifecycle was too complex and synchronization-heavy. |
| J21 | Accepted: canonical anchors replace redundant solver copies. |
| J22 | Rejected: device-phase sticky ping-pong added synchronization/state. |
| J23 | Rejected before implementation: exact-key fast lane lacked sufficient hit-rate headroom. |
| J24 | Rejected transfer: matcher ping-pong did not simplify the production lifecycle. |
| J25 | Rejected: tangent reconstruction saved storage but added unstable/repeated work. |
| J26 | Rejected: column-pack/color-projection fusion increased live state. |
| J27 | Rejected: shared-memory sticky lifecycle did not fit generic working sets. |
| J28 | Accepted in mini: shared body working set plus interleaved rows; not a generic full result. |
| J29 | Hybrid mass-splitting tail must be judged at matched convergence, not sweep count. |
| J30 | Rejected: splitting full prepare/iterate added boundaries and traffic. |
| J31 | Rejected: two register-resident sweeps exceeded the register budget. |
| J32 | Rejected: generic shared body cache lost on sparse/heterogeneous worlds. |
| J33 | Accepted for the 45k-body single-world regime: conservative hybrid scheduling. |
| J34 | Corrected: automatic policy must use workload shape and preserve conservative fallback. |
| J35 | Rejected: classic PGS color order changed convergence or lost throughput. |
| J36 | Accepted: defer warm-start anchor work until needed. |
| J37 | Rejected: fused column/graph staging enlarged live state. |
| J38 | Retained in mini only: vec4 inverse-mass lane did not establish a full-solver win. |
| J39 | Accepted: compact sticky offset history reduces fixed metadata. |
| J40 | Accepted: bound staging grids to active capacity. |
| J41 | Rejected: pair-first generation did not beat the canonical collision/sort pipeline. |
| J42 | Rejected: packed vec4 sort source added conversion/footprint. |
| J43 | Accepted: skip multi-world recoloring only under a validated unchanged-topology predicate. |
| J44 | Rejected: tiled compaction synchronization outweighed saved passes. |
| J45 | Bounded experiment: main-constraint vec4 grouping helps only sufficiently dense layouts. |
| J46 | Control result: full/mini crossover depends on mixed-rigid density and generic pipeline overhead. |
| J47 | Rejected: rigid-only copy payload specialization saved too little. |
| J48 | Rejected: incremental recoloring upper bound was insufficient. |
| J49 | Rejected: unified mixed-frame proxy did not predict full behavior. |
| J50 | Rejected: solver-local body remapping added conversion/indirection. |
| J51 | Rejected: warp-distributed register cache lost to sparse accesses/register pressure. |
| J52 | Rejected: FP16 immutable rows did not meet precision/performance requirements. |
| J53 | Rejected: dense active-set solve upper bound did not justify replacement. |
| J54 | Rejected: full patch-friction control showed no transferable mini advantage. |
| J55 | Rejected: recomputing mass to avoid inertia loads lost. |
| J56 | Integrity audit: compare active work, contacts, warm-up, capture mode, and validation before accepting timings. |
| J57 | Rejected transfer: remaining mini assumptions failed representative solver corners. |

## Benchmark controls retained from the original log

- The fixed method separates settle, warm-up, and measured graph replays.
- Report logical useful bytes separately from actual hardware-counter traffic.
- Compare PhoenX and mini only on matched contact generation/history semantics;
  different final manifolds invalidate simple frame-time ratios.
- Robot-like mixed revolute/contact workloads and dense contact-only workloads
  require separate scheduler conclusions.
