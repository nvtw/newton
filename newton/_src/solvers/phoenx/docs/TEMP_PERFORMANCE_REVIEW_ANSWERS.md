# Response to the PhoenX performance review questions

Independent review, 2026-07-27, branch `twidmer/phoenx-bandwidth-wins`.

**Method and disclosure.** Nothing was executed on the GPU for this review (the
device was in use by other benchmarks). Every quantitative statement below is
either (a) reproduced from the numbers in `TEMP_PERFORMANCE_REVIEW_QUESTIONS.md`
and `PERF_NOTES.md`, or (b) derived arithmetically from those numbers plus the
current source. Claims of type (b) are marked *[derived]* and each one carries
the cheapest measurement that would falsify it. Nothing here is a measured
result.

---

## 0. The single most important thing in this document

The Kapla iterate kernel is not bandwidth-bound, not register-bound, and not
occupancy-bound in the usual sense. It is **memory-latency-bound while running
at roughly one work item per thread with ~2.4x of the register file unused**.
The profile numbers say this quite directly, and the code says why.

`_singleworld_total_threads` (`solver_phoenx.py:510`) sizes the persistent grid
from *constraint capacity*, not from per-colour work:

```python
capacity_blocks = (constraint_capacity + block_dim - 1) // block_dim   # block_dim = 32
max_blocks_limit = sm_count * (12 if colored_contact_headers else 4)   # 188 * 12 = 2256
num_blocks = max(32, min(capacity_blocks, max_blocks_limit))
```

The reported launch is 1,504 blocks. 1,504 < 2,256, so the grid is
**capacity-limited**: `constraint_capacity ≈ 48.1k`, and the grid is exactly one
thread per constraint *slot*. But the kernel body iterates over one colour:

```python
for t in range(thread_start, count, stride):   # count = size of THIS colour
```

With 8 colored mass-splitting partitions plus overflow, `count` is on the order
of `capacity / 9` ≈ 5k, against 48,128 threads. *[derived]* **Roughly 85–90% of
the launched threads take the early-exit path in every one of the 108,000
launches, and essentially every working thread processes exactly one item.**

Three profile numbers corroborate this and nothing else explains all three:

| Reported | Consistent with |
| --- | --- |
| 0.50 waves/SM, 8 one-warp blocks/SM resident | whole grid resident in half a wave |
| theoretical 12.5% occupancy (8/64 warps) vs **achieved 5.1–6.2%** | ~half the resident warps exit before doing work; time-averaged occupancy collapses |
| **0.09 eligible warps/scheduler, 91% no-eligible cycles, 60% long-scoreboard** | 2 warps/scheduler, each with ~1 outstanding load, no second item to overlap |

And the register arithmetic *[derived]*: 101 registers rounds to ~104/thread =
3,328/warp. At 65,536 registers/SM that permits **~19 warps/SM**. The launch
provides **8**. To stay at 8 warps/SM you could spend up to ~240 registers per
thread. There is roughly **140 registers/thread of unused budget**, and the
kernel is stalling on memory.

Every rejected Kapla experiment in `PERF_NOTES.md` — register caps, occupancy
hunting, wider vector loads, coalescing transposes, cooperative manifolds —
spends effort on *bytes* or on *occupancy*. The evidence says the scarce
resource is **outstanding independent memory requests per thread**. That is the
axis nothing has been tried on, and it is the axis with free register budget.

Concretely: right-size the grid so each thread owns several colour items, then
software-pipeline across items. Details in §1.3. Both halves are **bitwise
exact** (see §1.5), so the correctness gate is trajectory equality, not a
quality screen.

---

## 1. Removing Kapla's launch/latency floor

### 1.1 Cooperative launch will not win. Reject it without building it.

`cooperative_groups::this_grid().sync()` is not a hardware grid barrier. On all
current NVIDIA architectures it is compiled to essentially what your rejected
software barrier already did: an arrive-count atomic on a global barrier object,
a spin on that counter, and `__threadfence()`/`membar.gl` on both sides. What
cooperative launch adds is a *residency guarantee* (the runtime refuses to launch
a grid that cannot be co-resident), which makes the barrier **correct**, not
**fast**.

So the prior rejection already measured the thing being proposed. The software
barrier lost on synchronisation and residency cost; the hardware-blessed version
has the same synchronisation cost and *strictly tighter* residency constraints
(at 101 registers you would be capped near 19 warps/SM ≈ 3,572 warps total, and
you would have to hold that grid across all nine colours, ten iterations and the
mass-splitting average/broadcast — which currently runs as separate kernels over
a different index space, i.e. per unified node, not per constraint).

Additional blockers, each independently sufficient:

- **Warp has no cooperative launch path.** You would need `cuLaunchCooperativeKernel`
  plumbed through `warp.context.Launch`, a `-rdc=true`-style codegen change for
  the CG headers, plus a `wp.grid_sync()` builtin. That is real runtime surface
  area for a change whose expected value is ≈ 0.
- **CUDA Graphs.** Cooperative kernels are capturable, but the residency
  requirement interacts badly with graph node scheduling and you lose the ability
  to run anything concurrently with the solver in the same stream.
- **Index-space mismatch.** Average/broadcast is per *node*
  (`copy_state.section_end.shape[0]`), iterate is per *constraint slot*. Fusing
  them means one grid running two different decompositions with a grid barrier
  between — the grid must be sized for the max of the two, worsening the
  parallelism famine for the smaller one.

**Verdict: do not implement.** If you want the number anyway, §6 E0 gives a
30-minute experiment that bounds the entire prize without writing any solver
code.

### 1.2 The prize is smaller than it looks — measure it first

Before any fusion work, measure the *irreducible* launch floor directly:

> **E0.** In the existing captured Kapla graph, replace the 90 iterate launches
> per substep with a no-op kernel of **identical launch geometry** (same grid,
> same block dim, same argument struct — just `return` at the top). Replay and
> take the GPU time. That number is the exact launch/schedule floor for the
> current dispatch shape.

Interpretation:

- Floor ≥ 50% of the 5.5 µs average → fusion is the dominant lever, go to §1.4.
- Floor ≤ 20% → **fusion is capped at 20% and §1.3 is worth more than everything
  in §1.4.**

My prediction *[derived]*: graph-replayed launches on this class of device cost
~1.5–2.5 µs of end-to-end occupancy each for a 1,504-block grid, so the floor is
in the 25–40% range and both levers matter — but §1.3 is cheaper and lands first.

A second cheap measurement worth taking at the same time: **the per-partition
element histogram** for the settled tower. The unrolled dispatcher
(`dispatch/single_world_mass_splitting_unrolled.py:44`) deliberately disables
tail fusion (`fuse_threshold = wp.int32(-1)`) and issues a full 1,504-block
launch for **every** partition including the smallest. If the tail partitions
hold only a few hundred rows, they are pure launch floor and a size-aware
dispatch (or a balanced re-colouring pass) recovers them for almost no work.

### 1.3 The change I would make first: right-size the grid, then prefetch across items

Two changes, both static kernel-factory knobs, both bitwise exact.

**(a) Grid right-sizing.** Size the persistent grid to expected per-partition
work, not to total capacity:

```python
# solver_phoenx.py:_singleworld_total_threads
expected_partition = ceil(constraint_capacity / max(1, max_colored_partitions + 1))
capacity_blocks = ceil(expected_partition * OVERSUBSCRIBE / block_dim)   # OVERSUBSCRIBE ~ 1.0-2.0
```

Still a compile-time constant chosen at construction, so graph capture is
unaffected. Correctness is unaffected for *any* grid size because the
grid-stride loop covers the colour regardless. For Kapla this takes 1,504 blocks
→ ~170–340 blocks, i.e. 1–2 warps/SM with real work each instead of 8 warps/SM
of which ~1 has work.

On its own this is roughly neutral-to-slightly-positive (fewer blocks to
schedule, slightly worse tail balance). Its real purpose is to make (b)
possible: **with one item per thread there is nothing to prefetch.**

**(b) Cross-item software pipelining.** In
`_make_singleworld_rigid_direct_color_func` / the `for t in range(...)` loop,
restructure to a prologue + steady-state that issues item `t+stride`'s scattered
loads *before* item `t`'s solve and writeback:

```
# prologue
nxt_cid   = element_ids_by_color[start + base]
nxt_hdr   = contact_header(nxt_cid)          # b1, b2, first, count, mu, slot1, slot2
nxt_body  = load_endpoints(nxt_hdr)          # v, w, inv_mass, inv_inertia_sym6  (x2)

while base < count:
    cur_hdr, cur_body = nxt_hdr, nxt_body
    if base + stride < count:                # issue EARLY, before the solve
        nxt_cid  = element_ids_by_color[start + base + stride]
        nxt_hdr  = contact_header(nxt_cid)
        nxt_body = load_endpoints(nxt_hdr)
    solve_and_writeback(cur_hdr, cur_body)   # ~2-4 us of dependent work to hide under
    base += stride
```

Why this is the right target and not the contact rows: **L1 hit rate is 78% and
L2 is 54%.** The per-point streams (`cc_get_normal`, `r0`, `r1`, `eff_*`,
`bias_*`) are contiguous per column and are what is *hitting*. The
`bodies.velocity[b1] / inverse_inertia_world[b1] / copy_state.velocity[slot]`
reads are indexed by arbitrary body ids over ~11k bodies — those are the misses,
and they sit at the **head** of the dependent chain:

```
element_ids_by_color[·] → cid → contact header (b1,b2,first,count,slot1,slot2) → body state → first point
```

That is three to four fully-exposed round trips before any arithmetic can start,
which is exactly the shape of a 5.5 µs kernel doing ~1 item per thread
*[derived]*. Prefetching moves the *next* item's three round trips underneath
the *current* item's solve.

**Register cost.** Live prefetch state ≈ header (8 ints) + two endpoints
(2 × (vec3 + vec3 + float + sym6) = 2 × 13 floats) ≈ 34 registers. 101 → ~135.
Per §0 you have room to ~240 before the 8-warps/SM geometry is even touched, and
with (a) you need fewer warps resident anyway. **This is the one place where
`PERF_NOTES.md`'s "higher occupancy by register caps → spills" law does not
apply, because we are spending registers, not saving them.**

**Expected magnitude** *[derived]*: MLP per warp goes from ~1 outstanding
scattered load to ~2. On a workload where 60% of issue gaps are long-scoreboard
and 91% of scheduler cycles have no eligible warp, the memory-latency component
should compress by close to 2x, giving perhaps 1.3–1.6x on the iterate kernel
and, at 66% of Kapla GPU time, **+15–30% Kapla FPS**. If a 2-deep pipeline
works, `PREFETCH_DEPTH = 3` is a one-line follow-up.

**Why this is general, not Kapla-specific.** It is a static factory axis
(`prefetch_depth`) on the same kernel every single-world scene uses. Any scene
whose per-colour work exceeds the grid gets more items per thread and more
benefit; scenes with one item per thread degrade to the prologue and are
unchanged. No scene detection, no if-elif.

**Interaction with `array_noalias`.** The notes record `array_noalias` as ~+1%
end to end and "do not depend on the experimental branch". Worth noting *why* it
was small: with one item per thread there are no loads to hoist above the
writeback stores. The source-level prefetch above achieves the hoist *without*
the annotation, because the loads are literally written before the stores. If
you later retry `noalias`, retry it **on top of** the pipelined kernel — that is
where it would compound.

### 1.4 If you still want to remove global barriers: resident subdomains, not grid sync

This is the architecture I would propose in place of cooperative launch. It is a
larger change; do it only if E0 says the launch floor is large, and only after
§1.3.

**Observation that determines the design:** within a colour, every body appears
at most once (that is what the colouring guarantees). **There is no body-state
reuse inside a colour — reuse exists only *across* colours and *across*
iterations.** Therefore on-chip body caching is worthless unless several colours
execute inside the same block, which requires every constraint touching a body
to be owned by one block. That forces spatial partitioning, and it also answers
Q2 (§2).

**Design — "resident subdomain block Gauss–Seidel":**

1. **Partition.** Sort bodies by Morton code of centre-of-mass, cut into `P`
   equal runs. Deterministic, reuses `wp.utils.radix_sort_pairs`, no new
   dependency, no graph library. `P ≈ 2–4 × sm_count`.
2. **Classify.** A constraint is *interior* if both endpoints are in the same
   part, else *boundary*. One pass, deterministic.
3. **Interior solve.** One block per part. Block preamble stages that part's body
   state (`velocity`, `angular_velocity`, `inverse_mass`, `inverse_inertia_world`
   as sym6 = 52 B/body) into shared memory — **coalesced, because Morton
   renumbering makes the part's bodies contiguous**. Then run all local colours ×
   all solver iterations with `__syncthreads()` between colours. Write body state
   back once.
4. **Boundary solve.** Boundary constraints go through the existing
   mass-splitting machinery (cross-part endpoints are exactly what copy-state
   slots model) in a small global pass interleaved with the interior sweeps.

**What it buys.** 90 global launches per substep → ~10–20; the other ~80
barriers become `__syncthreads()` (tens of cycles instead of ~2 µs). Body state
is read from global once per part per substep instead of once per incident
constraint per iteration — at degree ~7 and 10 iterations that is ~70x fewer body
round trips *[derived]*, and the remaining in-block accesses are shared-memory
random access, which has no coalescing penalty at all. That is a *latency*
argument, not a bandwidth argument: the measured 15–17% DRAM throughput says you
are not short of bandwidth.

**Capacity check** *[derived]*: 11,340 bricks / 188 parts ≈ 60 bodies/part ≈
3.1 KB of shared memory. Even 512 bodies/part is 27 KB. Shared memory is not the
constraint; **surface-to-volume is**. At 60 bodies/part a large fraction of
contacts are boundary. This is the design's real risk and the thing to measure
first (see E3 in §6): compute the interior/boundary split as a function of `P`
offline before writing a kernel. If interior fraction at `P = 188` is below ~60%,
the design does not pay and should be dropped.

**Honest caveat.** Unlike §1.3 this **changes the Gauss–Seidel ordering** (it is
a block-GS / additive-Schwarz hybrid). Theory says intra-part convergence should
*improve* (more sequential coupling inside a part), but it must be gated on
Kapla drift/settle/contact-count and the stacking tests, not on trajectory
equality.

### 1.5 Determinism

- §1.3(a) grid right-sizing and §1.3(b) prefetch are **bitwise identical**.
  Within a colour the items are independent by construction (disjoint bodies), so
  the result does not depend on which thread runs which item or when the loads
  issue. `parallel_id` in the overflow partition derives from the CSR slot
  (`t_slot / ms_batch_size`), not from `tid`, so the overflow path is grid-size
  independent too. **Gate: exact trajectory equality on the Kapla tower, plus the
  existing capture/replay regressions.**
- §1.4 changes ordering; gate on invariants and quality, per the existing rule.

### 1.6 Blackwell-specific option worth one paragraph

Thread-block **clusters** give a genuine hardware barrier (`cluster.sync()`) and
distributed shared memory across up to 16 blocks. That is the one mechanism that
*is* materially cheaper than a global barrier, and it would let a "part" in §1.4
span 16 blocks instead of one — directly fixing the surface-to-volume problem.
Warp has no cluster API, so this is a tier-2 idea, but it is the only version of
"cheap grid synchronisation" that is not just an atomic spin. If §1.4 measures
well but is limited by part size, this is the follow-up.

---

## 2. Making mutable body / copy-state access coalesced

### 2.1 The honest answer: you cannot coalesce it, and you should stop trying

The per-colour independence property above means each body is touched *once* per
colour by *one* constraint. Within a warp, the 32 lanes touch 64 distinct body
ids. After the existing `(colour, family, body_min, eid)` sort
(`graph_coloring_incremental.py:1018`) `b1` is monotone across lanes, but with
gaps: a colour touches ~2 × count distinct bodies drawn from ~11k, so consecutive
lanes' `b1` differ by ~2–3. With `velocity` as `vec3` (12 B) that is a 24–36 B
stride: 32 lanes span ~0.8–1.2 KB = 24–36 sectors for 384 useful bytes, ~35%
efficiency. `b2` is unordered and is worse.

**This is already close to the best a constraint-parallel schedule can do**, and
it explains the reported 77–78% "excessive sectors" without implying a fixable
bug. The notes' own law applies: *do not infer a bandwidth bottleneck from sector
utilisation alone.* At 15–17% DRAM throughput, halving the sectors would buy
almost nothing. **The lever is round-trip count, not sector efficiency.**

### 2.2 Evaluation of the listed ideas

| Idea | Verdict |
| --- | --- |
| Colour-local endpoint state + deterministic inter-colour transitions | **Reject.** Duplicating state per colour means a gather/scatter reconciliation between every pair of colours — that is the same traffic you removed, plus 9x the storage, plus new barriers. It only pays if the transition is free, and it is not. |
| Alternative copy-state indexing / edge-vertex renumbering | **Partially worth doing, cheap.** Morton renumbering of bodies (§1.4 step 1) is independently useful even without subdomains: it tightens the `b1`/`b2` locality the existing sort exploits and improves the L2 hit rate on the scattered endpoint reads. Deterministic, bitwise-exact if you renumber at build time and keep the mapping stable. Low cost, low-single-digit expected. |
| Persistent on-chip body caches across colours | **This is the only one that can win**, and it requires §1.4's partitioning to be legal. See §1.4. |
| Graph partitioning into resident tiles with boundary reconciliation | Same as above. This is the recommendation. |
| Endpoint-oriented (body-parallel) schedule | **Reject as stated.** A body-parallel sweep needs exclusive access to *both* endpoints of each constraint; the only ways to get that are (i) an owner rule, which makes the sweep Jacobi and changes the fixed point, or (ii) copy-state slots, which is what mass splitting already is. It is not a new mechanism, it is the mechanism you have. |

### 2.3 Dynamic contacts and generality

The subdomain design must survive contacts appearing/disappearing every frame.
It does, because the partition is over **bodies**, not constraints: bodies are
stable, Morton keys are recomputed per frame from positions in one cheap sort,
and constraint classification is one pass over the (already rebuilt) constraint
list. Nothing is cached across frames, so there is no invalidation problem. That
is also why it generalises: it is a property of "one world with many bodies", not
of brick towers. It should help the dragon soft body and any granular scene for
the same reason.

---

## 3. Parallelism inside contact manifolds

### 3.1 The neutral proxy is trustworthy, and the reason is structural

4.095 vs 4.099 µs was not a measurement failure. Two independent reasons say
cooperative manifolds cannot win here:

**(a) It redistributes parallelism, it does not create any.** Giving 8 lanes to
one column means a warp covers 4 columns instead of 32. Total lanes are fixed;
you have converted thread-level parallelism into lane-level parallelism 1:1, then
added shuffle latency and a `__syncwarp`-shaped dependency. In a workload that is
*starved* of independent items (§0), spending items to buy lanes is exactly
backwards.

**(b) It preloaded the data that was already cached.** L1 hit rate is 78%. The
six `vec3` row streams the probe preloaded are contiguous within a column and are
the part that hits. The misses are the scattered `bodies.*[b1|b2]` and
`copy_state.*[slot]` reads — which the probe did not touch, and which a per-column
lane group cannot help with anyway (there are only two of them per column, so
there is nothing to spread across 8 lanes).

So the proxy is **not** missing an effect of the real 101-register kernel. If
anything the real kernel would look worse, because the cooperative version
raises live state per column while cutting the columns in flight.

### 3.2 If you want one more decisive test before closing this out

The cheapest falsification of "the misses are body state, not rows" is a
**source-level ablation, not a new kernel**:

> **E-manifold.** In an isolated harness, run the production iterate over the
> real Kapla colour with (i) production body loads, and (ii) body loads replaced
> by `bodies.velocity[lane]`-style *contiguous* indices (physically wrong, but
> same instruction count and same registers). If (ii) is dramatically faster,
> body scatter is the bottleneck and §1.3/§1.4 are correct. If (ii) is close to
> (i), the row streams matter more than I think and cooperative manifolds deserve
> a second look.

Cost: an afternoon. It discriminates between the two competing hypotheses with
one number, and it is the *only* manifold-related experiment I would fund.

### 3.3 Warp specialisation / async copies / segmented subgroups

- **Warp specialisation** (producer warps issuing loads, consumer warps solving)
  needs producer and consumer co-resident in a block with shared-memory
  handoff. At 8 one-warp blocks/SM there is no block to specialise inside; you
  would have to abandon `_SINGLEWORLD_BLOCK_DIM = 32`, and the note at
  `solver_phoenx.py:500` records that Kapla FPS scales *monotonically* as block
  dim falls 256 → 32 (+97%). Reject.
- **`cp.async` / TMA staged copies** move *contiguous* tiles. The problematic
  access is a gather over 11k bodies. Not applicable without §1.4's renumbering,
  at which point §1.4's shared staging is the better use of the mechanism.
- **A projection/update refactor** (split the loop into "compute all `jv`" then
  "apply all impulses") **changes the physics** — it converts sequential GS over
  the manifold points into block-Jacobi over them. Out of scope under the
  fixed-point constraint.

**Recommendation: close cooperative manifolds as rejected**, record §3.1(a) and
(b) as the reasons in `PERF_NOTES.md`, and redirect the effort to §1.3.

---

## 4. Highest-value remaining reduced-coordinate G1 optimisation

### 4.1 What the code shows

Reading `_advance_and_publish_reduced_articulations_warp_kernel`
(`articulations/reduced.py:3022`, shared body around `:2600–2833`), every one of
the three depth traversals resolves, per lane per depth:

```python
joint       = articulation_depth_joint[index]      # 1
child       = joint_child[joint]                   # 2  (depends on 1)
dof_start   = joint_qd_start[joint]                # 2
dof_end     = joint_qd_start[joint + 1]            # 2
joint_kind  = joint_type[joint]                    # 2
parent_lane = joint_parent_lane[joint]             # 2
child_start[joint], child_start[joint + 1]         # 2
...then joint_s[dof], joint_d_inv[dof, ·], joint_u_matrix[dof]   # 3 (depends on 2)
```

Two observations, and I think the second is the important one:

1. **Two of the three or four dependent levels carry no physics.** Levels 1 and 2
   are pure address resolution. Only level 3 (`joint_s`, `joint_d_inv`,
   `joint_u_matrix`, `body_bias`, `body_q_com`) is state.
2. **Levels 1 and 2 are world-invariant.** For a fleet of identical G1s, the
   depth ordering, parent-lane map, dof counts, joint types and child ranges are
   the *same* for every one of the 8,192 worlds, modulo a per-world base offset.
   They are being re-fetched per world, per depth, per pass, per substep.

This is precisely the "avoidable repeated topology/factor loads" bucket in the
question, and it is on the critical path of the two largest kernels in the G1
profile (fused advance/publish 235 µs at 69.9% L1TEX dependency gaps; packed
contact rows 234.75 µs at 76.1%).

### 4.2 The change: a depth-ordered joint descriptor

Replace the pointer chase with **one coalesced structure load indexed by
`index`** — the depth-ordered slot the loop already iterates on:

```python
@wp.struct
class JointDepthDesc:      # 32 bytes, one per depth-ordered slot
    joint:        wp.int32
    child:        wp.int32
    dof_start:    wp.int32
    dof_count:    wp.int32
    joint_type:   wp.int32
    parent_lane:  wp.int32
    child_first:  wp.int32
    child_count:  wp.int32
```

Built once per topology change (same place `articulation_depth_joint` is built).
Then:

```python
d = joint_depth_desc[index]      # coalesced: lane -> index is unit-stride
# levels 1 and 2 are gone; joint_s[d.dof_start + row] issues immediately
```

Why this should be a real win, not a wash:

- **Collapses 3–4 dependent levels to 2.** The 6–8 scalar loads at level 2 become
  one 32 B structure load.
- **Turns scattered into unit-stride.** `index = depth_start + lane`, so the 32
  lanes read 32 consecutive descriptors — 8 sectors fully used instead of ~6
  scattered scalar gathers. This directly attacks the recorded "4.3 useful bytes
  per 32-byte sector".
- **Deduplicable across the fleet.** If topology signatures match (the common
  homogeneous-fleet case, detected at build time by hashing the depth/parent/dof
  layout), store **one** descriptor table for the whole fleet and index it by
  local slot, adding `world * joints_per_world` where a global joint id is needed.
  Footprint drops from 8,192 × 30 × 32 B ≈ 7.8 MB to ~1 KB — **permanently
  L1-resident for every world**. That converts levels 1–2 from DRAM/L2 traffic to
  ~zero. This is a build-time dedupe of identical data, not a scene-specialised
  kernel, so it stays inside the "general static factory" rule.

**Bitwise exact.** Same values, same order, same arithmetic — only the storage
layout of already-computed integers changes. Gate on exact G1 trajectory
equality plus the serial-oracle 8/16/32-lane tests.

**Expected magnitude** *[derived]*: if L1TEX dependency gaps are 70% and roughly
half of the dependent levels are address resolution, the ceiling is large; I
would predict **8–20%** on fused advance/publish and a similar effect on packed
contact rows if they share the chase (worth checking —
`_build_packed_generalized_row` at `reduced_contact_block.py:1193` looks like it
does). At 13–14% + 9–10% of G1 GPU time, that is a credible **+3–5% end to end**,
which by this codebase's standards is a large single change.

**Counters that validate it:** `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
and the excessive-sector ratio on the topology arrays specifically; `smsp__
average_warps_issue_stalled_long_scoreboard_per_issue_active` before/after;
eligible warps/scheduler (0.22 → should rise). Falsified if sectors drop but the
stall fraction and eligible-warp count do not move.

### 4.3 Distinguishing the four categories the question asks about

| Category | Assessment |
| --- | --- |
| **Irreducible tree dependency latency** | The depth recurrence itself: ~7 levels × (shuffle + small dense solve). Genuinely irreducible without changing dynamics. This is the floor and §4.2 does not touch it. |
| **Avoidable repeated topology/factor loads** | §4.2. **This is where the remaining value is.** Currently paid 3 times per substep (three passes) × 8,192 worlds for data that is one table. |
| **World interleaving / batched traversals** | **Reject.** Two worlds per warp doubles live traversal state; at 101 registers and 22.9% occupancy that halves warps/SM, leaving MLP unchanged. Unlike Kapla (§0), G1 has *no* spare register budget — it is at 14.6 warps/SM against a ~19 ceiling. Different regime, opposite prescription. |
| **Data that can stay resident across rows/substeps** | Topology (§4.2) — yes, permanently. Factor `d_inv`/`u_matrix` — no, they are refreshed per substep and the notes already record "reuse factor/kinematics across substeps" as physics-changing. `joint_parent_lane`/depth structure — yes. |
| **Changes that merely move traffic into staging kernels** | The descriptor build is O(joints) once per topology change, not per substep, so it does not. Contrast with the rejected "build-time patch-reduced rows" and "depth-major factor scratch", which paid per substep. That is the distinction that made those lose and should make this one win. |

### 4.4 Exact inspection targets

- `articulations/reduced.py:2600–2833` — the three depth loops (the chase).
- `articulations/reduced.py:3022` `_advance_and_publish_reduced_articulations_warp_kernel`.
- `articulations/reduced.py:2917` `_factor_and_advance_reduced_articulations_warp_kernel`.
- `articulations/reduced_contact_block.py:1193` `_build_packed_generalized_row` —
  check whether it repeats the same `joint → child → dof_start` chase; if so it
  gets the descriptor for free.
- Data structures: `articulation_depth_joint`, `joint_parent_lane`, `child_start`,
  `child_joint`, `joint_qd_start`, `joint_type`, `joint_child` — all candidates
  to be folded into `JointDepthDesc`.

---

## 5. PhoenXRL learner / rollout

### 5.1 Inspect-first conclusion: the learner is done; stop optimising it

The chronology in `PERF_NOTES.md` is unambiguous. Isolated update went
250 → 231 → 209 → 200 → 197 → 181 → 166 → 112.6 → ~99 ms through a long series of
individually correct wins. Over the same span, end-to-end gains were repeatedly
"about 0.5%", "neutral because rollout hid update", "neutral for rollout-bound G1
but improved Ant". Rollout is ~273 ms against a ~99 ms update under leapfrog
overlap.

Applying the codebase's own predictor: with update fully hidden, the marginal
value of *any* further learner optimisation is **≈ 0 for G1**, bounded above by
whatever fraction of update time exceeds rollout time (currently none).

**Recommendation: freeze learner throughput work for G1.** Keep it for Ant/PBT
(where it demonstrably shows up) and as headroom, and say so explicitly in
`PERF_NOTES.md` so the next person does not re-derive it. Two consequences:

- The remaining "Open ideas" entries (FP32 cuBLAS backward contractions, Muon
  Newton–Schulz through cuBLAS) are correctly scoped as *headroom*, not
  throughput. Land them if they simplify code; do not measure them against G1
  wall-clock.
- The cuBLAS/Warp split is already at the right boundary: large dense
  contractions on cuBLAS, everything shape-irregular or fusable in Warp. I see no
  operation left where the crossover is misplaced. The one pattern that would
  still merit a generated fused Warp kernel is any remaining
  *elementwise-chain-between-two-GEMMs* (bias + activation + mask) that currently
  round-trips a full activation slab to DRAM — worth a single `nsys` pass to
  confirm none remain, then closing the question.

### 5.2 Where the wall-clock actually is: rollout, i.e. §4

Rollout *is* physics. The learner question therefore reduces to Q4. This is the
main structural point: **there is no "learner/rollout pipeline gap" left to
close; there is a rollout cost.**

### 5.3 Further overlap: the limit is algorithmic, not mechanical

Rollout/update already overlap via graph leapfrog with rollout on a
higher-priority stream (measured 1.361 → 1.382 M samples/s; rollout-first
rejected at 1.191 M). Pushing overlap further does not break *determinism* —
seeded replay stays exact as long as the data dependence pattern is fixed — it
increases **policy lag**, which changes the objective by making the data more
off-policy.

So the honest framing: additional overlap is an *algorithmic* trade, and it
should be paid for with the importance-sampling machinery already in the tree
(`_compute_vtrace_returns`, `ppo.py:1396`). If you want a deeper pipeline, gate
it on V-trace correction and measure sample efficiency, not iteration time. Do
not buy overlap with uncorrected staleness — the active-action-only PPO episode
is the cautionary precedent (better short statistics, failed the frozen screen).

### 5.4 How to measure wall-to-quality across seeds properly

The current protocol has a known defect the notes already flag: "Seed 42 was also
the strongest early seed in a 11/29/42/47/73 screen. This is evidence of seed
selection." And "Seed 11 missed at 100 M under both backends."

The statistical point: **samples-to-gate is right-censored.** Seeds that never
reach the gate within budget have no finite value, so means and medians over the
seeds that *did* finish are biased, and dropping the censored seeds is exactly
the selection effect above. The correct tool is survival analysis:

- Fixed seed set, ≥ 8 seeds, frozen before any comparison. Never re-select.
- Primary statistic: **Kaplan–Meier estimate of P(reach gate ≤ N samples)**, with
  non-reaching seeds entered as censored at the budget, not dropped.
- A/B comparison: **log-rank test** on the two seed sets, plus the KM median with
  a bootstrap CI. Report the CI; a point estimate is not a result.
- Secondary: samples-to-gate *conditional on reaching*, reported separately, and
  the reach *rate* itself — those are two different failure modes (slow vs.
  never), and the AMP/motion-prior campaign is aimed squarely at the second.
- Wall-clock conversion at the very end: `wall = KM-median-samples ÷ measured
  samples/s`, so throughput and sample-efficiency stay separable, as the question
  asks.
- Power: with 8 seeds, log-rank detects roughly a 2x hazard ratio. **Anything
  claiming a 10–20% sample-efficiency win from 8 seeds is not measurable** — say
  so up front rather than after the fact.

This costs nothing but discipline, and it is the difference between the
motion-prior campaign producing a decision and producing another anecdote.

---

## 6. Ranked experiment sequence

Three architectural experiments, front-loaded by a cheap measurement that
re-ranks the rest.

### E0 — Bound the launch floor (½ day, do this first)

- **What:** replay the Kapla graph with the 90 iterate launches per substep
  replaced by a geometry-identical no-op kernel. Also dump the per-partition
  element histogram.
- **Upside:** none directly — it *prices* every fusion idea in §1.4/§1.6 before
  anyone builds one, and it is the only thing that can justify or kill them.
- **Accept/reject:** floor ≥ 50% of 5.5 µs → fusion is the main lever; ≤ 20% →
  **permanently close cooperative launch and subdomain fusion**, and spend
  everything on E1.
- **Gates:** none, it is a measurement.
- **Changes next decision:** directly re-ranks E3 vs. E1/E2.

### E1 — Kapla: right-sized grid + 2-deep cross-item prefetch (2–4 days) — **highest expected value**

- **What:** §1.3(a) + §1.3(b). Static factory axis `prefetch_depth ∈ {1, 2}`,
  `OVERSUBSCRIBE` constant in `_singleworld_total_threads`.
- **Upside:** *[derived]* 1.3–1.6x on a kernel that is 51.7% of Kapla GPU time →
  **+15–30% Kapla FPS**. It is the only proposal that attacks the resource the
  profile says is scarce (outstanding requests/thread) using the resource the
  profile says is free (~140 spare registers/thread).
- **Cost:** low. One loop restructure in
  `_make_singleworld_rigid_direct_color_func` plus one sizing constant.
- **Accept/reject:** Kapla FPS ≥ +8% *and* registers/thread ≤ 200 *and* no
  spilling. Reject on any spill.
- **Gates:** **bitwise trajectory equality** on the Kapla tower (this change must
  not move a single bit), plus drift/settle/contact-count, plus the existing
  capture/replay and stacking regressions. Add a register-count guard test in the
  style of the existing factor-split guard.
- **Changes next decision:** if it lands, immediately try `prefetch_depth = 3`
  and re-test `array_noalias` *on top of it* (§1.3). If it is neutral **and** E0
  showed a small launch floor, then Kapla is genuinely at its structural ceiling
  and the honest conclusion is that 11,340 bricks underfills a 188-SM Blackwell
  by ~10x — record that and stop.

### E2 — G1: depth-ordered joint descriptor + fleet-wide dedupe (3–5 days)

- **What:** §4.2.
- **Upside:** *[derived]* 8–20% on fused advance/publish and possibly on packed
  contact rows; **+3–5% end-to-end G1 physics**, which is also the only thing
  that moves RL wall-clock (§5.2).
- **Cost:** moderate. New build-time table, three call sites in `reduced.py`, one
  topology-signature hash for the dedupe.
- **Accept/reject:** ≥ +5% on the fused advance/publish median with excessive
  sectors on topology arrays falling ≥ 50%. If sectors fall but time does not,
  the address chase was not the critical path — record that and stop, it is
  itself a valuable negative.
- **Gates:** **bitwise trajectory equality** on production G1, serial-oracle
  8/16/32-lane tests, wide-tree (>32 joint) rejection path preserved, plus a test
  that the dedupe path and the per-world path produce identical descriptors.
- **Changes next decision:** if the descriptor wins, audit `reduced_loop.py` and
  the patch-row builders for the same chase.

### E3 — Kapla: subdomain feasibility, offline only (1 day, gated on E0)

- **What:** Morton-sort the settled tower's bodies, cut into `P ∈ {94, 188, 376}`
  parts, and **just count** interior vs. boundary constraints and per-part body
  counts. No kernel.
- **Upside:** decides whether §1.4 is worth 2–3 weeks, for one day.
- **Accept/reject:** interior fraction at `P = 188` ≥ 60% → §1.4 is worth
  building. < 60% → drop it (or escalate to clusters, §1.6, only if E0 showed a
  large launch floor).
- **Gates:** none.
- **Changes next decision:** the only path to §1.4. If E0 says the floor is small
  *and* E3 says the boundary fraction is high, close the whole barrier-removal
  line of work and write it into `PERF_NOTES.md` as rejected with reasons.

### E4 — Kapla: `(partitions K, iterations I)` stability surface (1–2 days)

- **What:** 2-D sweep of `MASS_SPLITTING_MAX_COLORED_PARTITIONS × solver_iterations`
  against the drift/settle/contact-count gates. Cost per substep is
  `(K+1) × I` launches; you are at `(8, 10) = 90`.
- **Upside:** `PERF_NOTES.md` records ~10% FPS per partition, dominated by graph
  replay overhead. If `(6, 11) = 77` or `(5, 12) = 72` holds the gates, that is a
  double-digit win with **zero new code**.
- **Cost:** benchmark time only.
- **Accept/reject:** any `(K, I)` with fewer total launches that passes all Kapla
  gates. Reject anything that only passes at the tested settle time — re-run
  longer.
- **Gates:** full Kapla regime gate (drift, speed, finite state, contact count).
- **Note:** "eight partitions is the stability floor" was measured at fixed `I`.
  The trade against `I` is the axis that has not been mapped, and it is nearly
  free to map.

### Not recommended

- **Cooperative launch / `this_grid().sync()`** — §1.1. Same atomic barrier as
  the version already rejected, plus a Warp runtime extension. Expected value ≈ 0.
- **Cooperative manifolds** — §3.1. Redistributes scarce parallelism and
  preloads already-cached data. Close it, with reasons recorded.
- **Further learner throughput work for G1** — §5.1. Fully hidden by rollout.
- **World interleaving in reduced ABA** — §4.3. G1 has no spare register budget;
  this is Kapla's prescription applied to the wrong regime.

### Suggested order

`E0 → E1 → E2` in parallel with `E4` (benchmark-time only), then `E3` **only if**
E0 reports a large launch floor.

---

## 7. Summary of the reframing

| Regime | What the profile actually says | Prescription |
| --- | --- | --- |
| **Kapla** | ~1 item/thread, ~140 spare registers/thread, latency-starved, 9x parallelism division by the colour barrier | **Spend registers on memory-level parallelism.** Right-size the grid, prefetch across items. |
| **G1 reduced** | Plenty of items, *no* spare registers (14.6 of ~19 warps/SM), dependent address chase on the critical path | **Spend nothing; remove chase levels.** Depth-ordered descriptor, fleet-wide dedupe. |
| **RL learner** | Fully hidden behind rollout | **Stop.** Optimise rollout (= G1 physics) or sample efficiency, and measure the latter with censored-data statistics. |

The two solver regimes need *opposite* register policies. Several rejected
experiments in `PERF_NOTES.md` look like they applied one regime's lesson to the
other — most visibly "higher occupancy by register caps: spills or recomputation
worsened latency", which is correct for G1 and inapplicable to Kapla. Recording
the split explicitly would be worth as much as any single optimisation.
