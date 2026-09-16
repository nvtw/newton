# Physical ordinary head, split overflow: local design

No production edits or GPU work. Source audit found consistent current scaling,
not a mismatched effective-mass/scatter count in the inspected rigid paths.

## Current behavior and exact consequence

record_all_interactions_kernel gives every regular color partition0.
Overflow batch j also uses partition j; the first overflow batch shares pid0.
The deduplicated body partition count N is the number of allocated references,
not just overflow-row count. pid0 lookup returns this total N.
slot_cache stamps that same N into contact endpoint caches. Bilateral block
preparation and _ms_load_body_pair likewise both multiply physical mobility by N.
contact_endpoint response and scatter agree. Average uses exactly those N slots.

Thus the current regular head is one part of the split solve. For an isolated
scalar head constraint on a body with N=98 copies, a complete slot0 correction
of velocity error u changes physical mean by only u/98 if other copies do not
change. Residual97u/98 is an algorithmic finite-iteration effect. Setting only
the head mobility to physical while still writing slot0 would NOT fix it:
it would dilute the actual head impulse by98 and break physical impulse accounting.

References:
- mass_splitting/access.py:get_state_index, read_velocity_unified
- mass_splitting/slot_cache.py:_cache_slots_for_partition
- constraints/bilateral_joint.py:prepare_bilateral_joint_blocks
- constraints/constraint_joint.py:_ms_load_body_pair
- constraints/contact_endpoint.py:response and scatter with cached counts
- mass_splitting/kernels.py:_make_average_and_broadcast_kernel
- dispatch/single_world_mass_splitting.py:solve/relax
- solver_phoenx.py:_rebuild_mass_splitting_graph, _singleworld_head_plus_tail_sweep

Graph -> deduplicate/count -> endpoint caches execute in stream order before
preparation. Each normal iteration averages after the complete current sweep;
writeback precedes physical integration. Relax broadcasts after integration,
preserving newly rotated inertia/angular velocity. No inspected count mismatch.

## Coherent alternative matching an unsplit ordinary head

Keep existing graph, including pid0. Change HEAD access and response policy:

1. Every regular row uses slot=-1 and count=1 for each endpoint, reading and
   writing the physical body state. Solve all regular colors in their existing
   internal color order. Within each color its proper coloring remains race-free.
2. Broadcast resulting physical state to ALL body copy slots.
3. Solve overflow batches using original graph slots and N-scaled inverse mass
   and inertia. Batches remain sequential internally and parallel across batches.
4. Average N copies and write back physical body state.

An inert pid0 reference on a body absent from overflow batch0 is safe: it keeps
the post-head baseline. It dilutes tail convergence but does not alter the
physical impulse sum. If pid0 participates in overflow batch0, it is a normal
active overflow copy. Do not add/remove references merely by adjusting divisors.

For each body, after head physical impulse Jh and tail impulses Jt:
v_end = v_start + W*Jh + (1/N)sum(N*W*Jt)
      = v_start + W*(Jh + sum Jt).
The same equation holds for angular velocity at fixed world inertia. Pairing
forces at a shared world point gives total P/L balance (plus explicit static
reactions). Averaging cannot increase the sum of copy kinetic energies with
copy masses/inertias M/N. This does NOT imply a biased motor-driven full step
must dissipate kinetic energy.

## Required implementation touchpoints

- Slot/count cache builder needs an explicit regular-head physical policy:
  regular slot=-1,count1; overflow unchanged. Keep static/invalid endpoints safe.
- Bilateral preparation currently uses count_per_node unconditionally when
  copies exist. It must instead consume each JOINT's selected endpoint counts,
  or receive its head/tail ownership mask, including cooperative preparation.
  Otherwise response/scatter disagree despite corrected contact caches.
- Joint iterate currently _ms_load_body_pair uses parallel_id0 lookup directly.
  It needs an explicit physical-head branch or cached slots/counts. A sentinel
  partition ID not present in graph is possible locally but should not become
  an undocumented production convention.
- Contact preparation, cached warm start, normal/tangent response, and scatter
  must all use matching per-row physical or split cache values. Changing counts
  after preparing effective masses is invalid.
- Current persistent fused sweep can traverse BOTH regular colors and overflow
  inside one launch. Add a boundary so broadcast occurs between phases.
  A change to head launch alone will miss regular colors drained by fused tail.
  CPU/head CUDA/fused CUDA variants must share the ownership policy.
- Do not rescale saved physical lambda just because the endpoint policy changes.
  Warm-start each lambda exactly once with its new matching mobility.

## Warm start, iterations, and relaxation

Refresh geometry/factors without applying impulses where practical; then apply
regular warm impulses to physical state, broadcast, apply overflow warm impulses
to copies, average/writeback. Cached preparation must perform the same impulse
sequence without rebuilding geometry unnecessarily.

For each biased iteration: physical head -> broadcast -> overflow -> average
and writeback. Repeat using the last physical mean. Existing global direct or
reduced projectors must retain their current physical-state synchronization;
first local prototype should use ordinary block joints and owned-row exclusions.

After pose integration refresh all copies as today. For each relaxation sweep,
use physical head -> broadcast -> overflow -> average/writeback, with zero
recovery bias and existing speculative eligibility. Actual new post-integration
inertia must be used. Geometry/factor refresh policy must remain consistent.

Reverse traversal needs a deliberate choice. To preserve original reversed
color order: overflow -> average/writeback -> physical regular head in descending
order -> final broadcast. If the new algorithm always prioritizes physical head,
its head->overflow order changes reversed sweep semantics; document and validate,
rather than claim byte equivalence. Do not overwrite unsynchronized overflow
updates by broadcasting first.

## Minimum regression gates

1. Mixed regular/overflow body with unequal counts1/2/98 and a real bilateral
   plus off-center friction contact: inspect cached response/scatter counts,
   lambda identity, all6momentum and external reaction work after each phase.
2. Head-only scalar closure with inert copies must close physically in one solve,
   not by1/N. Current behavior is fail-before for the intended head policy,
   NOT evidence the existing split solver violates momentum.
3. Tail-only control, first-batch pid0 sharing, head-only body with inertpid0,
   zero overflow, changing overflow membership/counts, forward/reverse traversal.
4. Warm-start nonzero lambdas and cached preparation, with no duplicate impulse;
   anisotropic post-integrate relaxation must preserve the updated state.
5. Actual Colibri contact/joint bounds and drift before any speed acceptance.
   Group256/all-one-copy failure already proves eliminating splitting alone
   cannot guarantee mechanism stability; contact/recovery convergence also matters.
