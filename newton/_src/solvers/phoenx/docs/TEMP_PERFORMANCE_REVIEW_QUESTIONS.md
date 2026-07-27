# Temporary PhoenX performance review questions

Answers: [TEMP_PERFORMANCE_REVIEW_ANSWERS.md](TEMP_PERFORMANCE_REVIEW_ANSWERS.md).

This is a request for an independent technical review. Please reason from first principles and current code, not historical benchmark headlines. Concrete designs, failure modes, and the smallest decisive experiments are more useful than broad optimization lists.

## Objective and non-negotiable constraints

The end goal is minimum wall-clock time to a robust G1 policy, while improving PhoenX generally (including large single-world scenes such as Kapla).

- Preserve the hard Hertz contact model and high-fidelity dynamics.
- Preserve deterministic replay for identical seeds.
- Fix root causes of correctness problems and add regression tests.
- Do not specialize behavior to one scene, add heuristic/if-elif sprawl, or change the solver fixed point merely to win a benchmark.
- General static kernel factories or statistically justified automatic policies are acceptable.
- Avoid code bloat unless a large architectural win clearly justifies it.
- Compare against analytical physics anchors and validate other solver paths when a shared bug is found.

Relevant code is under `newton/_src/solvers/phoenx/`. The current branch is `twidmer/phoenx-bandwidth-wins`.

## Current evidence

Hardware for the measurements below: NVIDIA RTX PRO 6000 Blackwell, 188 SMs; Warp 1.16 development build.

### Large single-world Kapla

A fresh 100-frame Nsight Systems capture of the production 11,340-brick scene attributes GPU kernel time as follows:

- 51.7%: persistent PGS iterate, 108,000 launches, average about 5.50 us.
- 66.2%: prepare + iterate + relax variants of the persistent PGS kernel.
- 6.3%: mass-splitting average/broadcast.
- 6.4%: broad + narrow phase combined.
- The 108,000 iterate launches are nine colors x ten iterations x 1,200 substeps.

A focused Nsight Compute capture of the intended production iterate kernel reports:

- launch: 1,504 blocks x 32 threads (eight one-warp blocks per SM);
- 101 registers/thread, no spilling;
- 0.50 waves/SM, 5.1-6.2% achieved occupancy;
- about 1.5 active but only 0.09 eligible warps per scheduler;
- about 91% of scheduler cycles have no eligible warp;
- roughly 60% of cycles between issued instructions are long-scoreboard stalls;
- 77-78% of global sectors are reported as excessive/uncoalesced;
- only 15-17% DRAM throughput and 2.5-2.7% SM throughput;
- approximately 78% L1 hit rate and 54% L2 hit rate.

The scene already uses:

- color-packed contact headers and per-contact solve rows;
- deterministic within-color sorting by family and minimum body id;
- cached mass-splitting slot/count lookup;
- symmetric six-float world inverse inertia;
- a measured eight-blocks/SM persistent grid;
- fixed unrolled nine-color dispatch for mass splitting.

The remaining scattered accesses are mainly mutable body/copy-state endpoints. A physical constraint-row reorder was historically only about +0.5% for a one-world tower. Forced 128-bit `vec4` layouts regressed every tested PhoenX layout. Experimental Warp `array_noalias` was correct, but only about +1% end to end on G1 and does not make scattered addresses coalesced.

A new bitwise-exact isolated probe gave eight lanes to each 1-5 point contact manifold so lanes could preload six `vec3` row streams while lane 0 retained sequential Gauss-Seidel order. With the real 1,504-warp launch geometry and Kapla's measured manifold distribution, scalar and cooperative versions were neutral: 4.095 vs 4.099 us. Kapla's distribution after settling was mean 3.49 points/column, with 69% of columns containing four or five points.

Both software and hardware grid-barrier megakernels have now been rejected. A direct CUDA cooperative-kernel probe was bitwise correct, but nine captured phase launches took 12.65 us versus 38.54 us for one cooperative launch with grid barriers; at 90 phases the comparison was 112.65 versus 371.27 us. The software atomic-barrier version was slower still. The launch count is conspicuous, but replacing launches with global barriers is not the missing lever on this hardware.

Hot implementation areas:

- `solver_phoenx_kernels.py:_make_singleworld_persistent_kernel`
- `constraints/constraint_contact_cloth.py:_make_contact_iterate_at`
- `mass_splitting/kernels.py:_make_average_and_broadcast_kernel`
- `graph_coloring/graph_coloring_incremental.py:_sort_csr_by_body_locality`
- `dispatch/single_world_mass_splitting_unrolled.py`

### Reduced robot fleets and training

Retained general wins include cooperative reduced-contact row construction, lane-local patch application, warp-shuffle scalar broadcasts, dead-store removal in articulated advance, compact symmetric inertia, and a single-pass hard-Hertz contact gather. These preserve tested trajectories and typically produce low-single-digit end-to-end gains.

The reduced G1 hot path still contains dependent ABA/factor traversals and scattered contact-response work. Prior attempts that lost include depth-major factor scratch (+0.35% only), globally coalesced response transposes (about -6.6%), forced vector response layouts, register caps, shared response slabs, and matrix-free per-sweep tree solves.

PhoenXRL uses cuBLAS for large dense contractions and custom Warp kernels elsewhere. A warp-shuffle gradient reduction improved its isolated kernel 8-14% but the full graph PPO update only about 0.6%. The important metric remains end-to-end wall-to-policy, including rollout, learner, and sample efficiency; nanoG1 may differ in contact model, objective details, and stochastic seed behavior, so throughput and learning quality must be separated.

## Prioritized review questions

### 1. What architecture could reduce Kapla's tiny colored-kernel latency floor?

Hardware cooperative grid synchronization is already decisively slower than captured launches. Is there another architecture that can reduce the 108,000-launch/latency burden without simply replacing launches with more expensive global barriers?

A useful answer should address:

- how to fuse nine colors, ten solver iterations, and the required mass-splitting average/broadcast without changing arithmetic order or convergence;
- CUDA Graph capture/replay and determinism;
- whether device-side graph launch, conditional graph nodes, or a different solver decomposition can help;
- a small synthetic experiment that can decisively reject the idea before duplicating the production solver.

Please do not recommend cooperative launch unless you can explain why the measured 3.0-3.3x regression would disappear in the real kernel.

### 2. Can mutable body/copy-state access be made coalesced without adding more traffic than it saves?

The contact rows themselves are packed; body endpoints remain graph-scattered and are updated after every color. Is there a representation or schedule that preserves exact colored Gauss-Seidel semantics while making endpoint state adjacent?

Please evaluate ideas such as:

- packing immutable per-endpoint properties (inverse mass and symmetric inverse inertia) into the already color-packed contact columns once per substep, amortized across solver iterations;
- color-local endpoint state followed by deterministic transitions between colors;
- alternative copy-state indexing or edge/vertex renumbering;
- persistent on-chip body caches across colors;
- graph partitioning into resident tiles with explicit boundary reconciliation;
- an endpoint-oriented rather than constraint-oriented solver schedule.

For each design, account for synchronization, duplicate state, gather/scatter traffic, dynamic contacts, and deterministic ordering. Explain why the design is general rather than Kapla-specific.

### 3. Is there a better way to expose parallelism inside contact manifolds?

The simple eight-lane preload/shuffle proxy was neutral. Does that convincingly reject cooperative manifolds, or is the proxy missing a crucial effect of the real 101-register, long-scoreboard kernel?

Could warp specialization, staged asynchronous copies, segmented subgroups, or a projection/update refactor hide latency while preserving the exact sequential point order and lambda updates? Please propose a decisive isolated test before a large integration.

### 4. What is the highest-value remaining reduced-coordinate G1 optimization?

Given that dependent ABA traversal and sparse response access dominate, what general dataflow change could create more memory-level parallelism or reuse without changing dynamics?

Please distinguish:

- irreducible tree dependency latency;
- avoidable repeated topology/factor loads;
- useful world interleaving or batched traversals;
- data that can safely remain resident across contact rows/substeps;
- changes that merely move traffic into staging kernels.

Identify the exact kernels/data structures to inspect and the counters that would validate the hypothesis.

### 5. How should PhoenXRL close any remaining learner/rollout pipeline gap without overfitting?

Please inspect the current learner before assuming old timing numbers are valid. Which operations should remain cuBLAS, which common patterns merit generated fused Warp kernels or handwritten backward kernels, and where is synchronization or launch overhead still material?

Also address whether rollout and learner can overlap more without breaking deterministic seeded runs, and how to measure wall-to-quality across seeds with confidence rather than optimizing one lucky learning curve.

### 6. What experiment sequence maximizes information per engineering hour?

Please give a short ranked sequence of experiments, with:

- expected upside and why;
- implementation cost;
- decisive accept/reject metric;
- correctness/determinism gates;
- what result would change the next decision.

Prefer two or three architectural experiments over many microoptimizations.
