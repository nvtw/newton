# Compound-Body Contact Grouping — Optimization Investigation

## Goal

Reduce the number of graph-coloring colors required for scenes with
**compound bodies** (rigid bodies with more than one collision shape) by
treating every contact between the same pair of bodies as a *single*
graph-coloring node, regardless of how many shape-pair columns it
spans. The proposed mechanism: one extra sort + one extra index buffer
in the ingest pipeline, plus a new column header layout that points
at a non-contiguous union of contact ranges.

The optimization should:
- Always be a strict improvement at runtime when compounds dominate.
- Not regress single-shape scenes.
- Be cheap enough to leave on by default *or* trivially gated on a
  "any body has > 1 shape" predicate.

---

## 1. Current architecture (relevant pieces)

### 1.1 Contact ingest (per step)

`contact_ingest.py` consumes Newton's sorted `Contacts` buffer (already
keyed by `(shape_a, shape_b)`) and emits one
`CONSTRAINT_TYPE_CONTACT` column per *shape pair* via the following
pipeline:

1. `_contact_pair_boundary_kernel` — mark contacts where `(shape_a,
   shape_b)` changes.
2. `wp.utils.array_scan` — inclusive scan → 1-based run id per
   contact.
3. `_scatter_pair_starts_kernel` — at each boundary, scatter
   `(shape_a, shape_b, contact_first)` into per-pair arrays.
4. `_pair_counts_and_columns_kernel` — derive `contact_count` from
   adjacent `pair_first` and emit `pair_columns ∈ {0, 1}` (filtered
   pairs get 0).
5. `wp.utils.array_scan` (exclusive) over `pair_columns` →
   `pair_col_offset` (output column index per pair).
6. `_pair_source_idx_kernel` + `_contact_pack_columns_kernel` →
   one column written per output cid carrying `(body1, body2,
   contact_first, contact_count, friction)`.

Cost: 2 scans + 5 element-parallel kernels, all graph-capture-safe.

### 1.2 Per-column storage and PGS loop

`ContactColumnContainer` is `(CONTACT_DWORDS=7, num_columns)`. Each
column is 7 dwords:

```
[constraint_type, body1, body2, friction, friction_dynamic,
 contact_first, contact_count]
```

Per-contact persistent state (lambdas, normal, tangents, body-local
anchors) lives in `ContactContainer` keyed by the contact's
sorted-buffer index `k`. The PGS iterate kernel
(`contact_iterate_at` in `constraint_contact.py`):

- Is launched **once per cid (per shape-pair column)**.
- Loads `v1, v2, w1, w2` for `(body1, body2)` once.
- Loops `contact_count` times, solving each contact's normal +
  2 tangent rows via Gauss-Seidel against the in-register velocity
  state.
- Writes body velocities back **once at the end** of the kernel.

The loop pattern is the key insight: **it already iterates over a
range of contacts that share the same body pair**. The kernel does
not care whether those contacts came from one shape pair or many — it
only needs `(body1, body2)` and a contiguous `[contact_first,
contact_first + contact_count)` range.

### 1.3 Graph coloring

`IncrementalContactPartitioner.build_csr_greedy` (default) takes one
`ElementInteractionData` per cid (which carries up to 8 body indices;
contact cids fill 2). The Jones-Plassmann + greedy MIS produces a
coloring such that no two cids of the same color touch the same
body. Empirical color counts:

- Kapla tower (1280 planks, ~25k contacts): **28 colors** with
  greedy (down from 78 with round-based JP).
- A scene where body A has 2 shapes and body B has 2 shapes
  produces **4 columns sharing {A, B}** — the coloring is forced to
  use 4 distinct colors for them.

---

## 2. The compound-body problem

### 2.1 Concrete example

| Scene | Body / shape layout | Shape-pair columns sharing one (A,B) |
|---|---|---|
| Single shape per body | A: {S1}, B: {T1} | 1 |
| 2-shape compound on B | A: {S1}, B: {T1, T2} | 2 |
| 2x2 compound | A: {S1, S2}, B: {T1, T2} | 4 |
| 4x4 compound | A: {S1..S4}, B: {T1..T4} | up to 16 |

Every shape pair gets its own column → its own graph-coloring node →
its own color slot whenever those shapes touch the same opposing
body. Two consequences:

1. **Color count blows up linearly with shape-fanout per body pair.**
   A scene built around USD prims that decompose objects into many
   convex hulls per rigid body (e.g. concave compound colliders, the
   "Kapla plank with 6-face primitive shells" pattern) hits this
   hard.
2. **Per-color kernel-launch overhead grows.** Each color triggers a
   `contact_iterate_*` launch over its cids. More colors ⇒ more
   launches ⇒ more graph-edge traversal cost in `wp.capture_while`
   loops.

### 2.2 Why this is purely architectural waste

Coloring exists to keep PGS race-free: two cids of the same color
must not share a body, because each cid loads the body velocity
into registers, mutates it, and writes back. **But within one cid**,
the kernel already does Gauss-Seidel sequential solving over its
contact range — race-freedom inside a cid is by construction, not by
coloring.

So a hypothetical "merged" column for body pair `(A, B)` containing
the union of all `(Si, Tj)` contacts is *just* a long Gauss-Seidel
sequence. The coloring constraint that forced the 4 separate cids
into 4 colors disappears.

---

## 3. Implementation options

### Option A — Re-sort contacts by body-pair key, then ingest as today

Insert a single radix sort step before pair-boundary detection:

```
key[k]   = pack_body_pair(min(b1, b2), max(b1, b2))
value[k] = k                 # to permute the per-contact data
sort_pairs(key, value)       # 1 radix sort, O(N) work
```

Then run the existing ingest pipeline against the sorted buffer; the
boundary detector compares adjacent body-pair keys instead of
shape-pair keys. The resulting columns are body-pair columns.

**Pros**:
- One extra `wp.utils.radix_sort_pairs` call, fully graph-capture
  safe (Warp supports keys-only and key-value sorts in graphs).
- The downstream pipeline, the column layout, and the iterate kernel
  all stay byte-identical.
- Per-contact state already moves with the sort (the `value` array
  becomes a permutation we apply to `Contacts.{normal, point0,
  point1, ...}` and to `ContactContainer.{lambdas}`).
- Color count drops to the **body-pair graph's** chromatic number,
  which is the true lower bound for race-free PGS.

**Cons**:
- The sort touches `O(N_contacts)` 64-bit keys + 32-bit values per
  step. On a 25k-contact tower that's ~300 KB. CUB's radix-sort is
  O(N), but the constant is non-trivial.
- The contact warm-start gather (matches contacts to last step's
  lambdas via `rigid_contact_match_index`) must follow the new sort
  order. The match index is keyed by *Newton's* sorted buffer
  position — if we resort, we need to re-map. Two approaches:
  (a) sort *before* the warm-start gather and reroute, or
  (b) sort *after* and apply the same permutation to the gather
  output.
- We lose the property that contacts for one shape pair are
  contiguous in `ContactContainer`. Some downstream consumers
  (`contact_per_contact_wrench_kernel`, debug dumps) walk
  `[contact_first, contact_first + contact_count)` ranges and assume
  shape-pair contiguity. None of the *solver* paths depend on it.

**Effort**: Medium. ~150 LOC. The sort already exists in
`scan_and_sort.py`. The match-index remapping is the trickier piece.

---

### Option B — Two-level grouping (keep shape-pair columns, add a "body-pair group" indirection)

Preserve the shape-pair columns and the existing sorted layout; build a
secondary index that groups shape-pair cids sharing a body pair:

```
group_csr_offsets[g]   → first cid of group g
group_csr_values[i]    → cid in group g
group_body_pair[g]     → (b1, b2)
```

Coloring runs against the **groups**, not the cids. Each group is a
single coloring node with 2 bodies. Iterate kernel changes from "one
cid per thread" to "one group per thread, walks group_csr_values to
visit each cid", which is a 1-line nested loop.

**Pros**:
- No re-sort of `Contacts`. The shape-pair columns stay contiguous
  for downstream tooling.
- The existing kernels can keep their per-shape-pair semantics (e.g.
  per-shape-pair friction lookup).

**Cons**:
- Two-level dispatch adds register pressure: the iterate kernel has
  to load `(body1, body2)` once per group (not per cid), then walk a
  variable-length cid list. Each cid still has its own `friction` /
  `friction_dynamic`, so those reads happen inside the inner loop.
  Net effect: same memory traffic, more index arithmetic.
- Building the group index every step requires sorting cids by
  body-pair key — i.e. *the same sort as Option A*, just applied to
  cids instead of contacts.
- The group index has to be rebuilt every step (contact set changes
  every frame). Same overhead as Option A's sort.

**Effort**: Medium-high. ~300 LOC + iterate-kernel rewrite.

---

### Option C — Hybrid: sort cids (not contacts) by body-pair, optionally compact

Like Option B's first step, then go further: use the cid-sort to
**compact** consecutive cids that share a body pair into one merged
cid. The compacted cid carries `[contact_first, contact_count]`
spanning the union, which is **non-contiguous in the underlying
`Contacts` buffer** unless we also Option-A-sort the underlying
contacts.

This collapses to Option A in practice — without sorting the
underlying contacts the merged column can't use one
`[first, count)` range.

---

## 4. Recommended approach: **Option A**

### 4.1 Why

- Smallest delta to existing kernels: `contact_iterate_at` is
  unchanged. `contact_prepare_for_iteration_at` is unchanged. The
  graph coloring sees fewer nodes and produces fewer colors *for
  free* — no algorithm change needed.
- The dispatch / fast-tail / multi-world kernels all consume cids
  uniformly; nothing downstream needs to know about compounds.
- Friction is a *shape-pair* property (different shape pairs on the
  same body pair can have different materials). Option A handles
  this naturally because shape-pair runs stay contiguous *within* a
  body-pair group — we can either (a) average / max-reduce friction
  across the merged column, or (b) keep per-contact friction in
  `ContactContainer` and delete it from the column header. (a) is
  what Box2D does for compound shapes; (b) is more accurate but
  needs a column-schema change.

### 4.2 Required changes

1. **Add a body-pair sort step at the head of `ingest_contacts`.**
   The sort key is `pack_body_pair(min(ba, bb), max(ba, bb))`,
   constructed from `shape_body[shape_a]` / `shape_body[shape_b]`.
   Permutation array applies to:
   - `rigid_contact_shape0/1`
   - `rigid_contact_point0/1`, `rigid_contact_normal`,
     `rigid_contact_margin0/1`
   - `rigid_contact_match_index` (warm-start gather key)
   The sort itself is one `wp.utils.radix_sort_pairs` call; the
   gather can be one `wp.launch` of an `apply_permutation_kernel`.

2. **Update `_contact_pair_boundary_kernel` to compare body-pair keys
   instead of `(shape_a, shape_b)`** (same kernel structure, different
   comparison). After this, the rest of the ingest pipeline emits one
   column per body pair instead of per shape pair.

3. **Decide on friction semantics for the merged column.** Easiest:
   max-reduce static / dynamic friction across the contact range
   during pack (one extra reduction kernel, or fold into the existing
   pack kernel). Alternative: move friction to `ContactContainer`
   per contact, drop from column header (saves 8 bytes per column).

4. **Match-index remap.** The warm-start gather expects contacts at
   a specific sorted-buffer position; after our resort the position
   changes. Option: do the body-pair sort *first*, then run Newton's
   match against the resorted layout (Newton's match index isn't
   created in PhoenX, it's created upstream — this is not viable
   unless Newton itself adopts body-pair sorting). Better option:
   keep `match_index` in original-position space, apply the same
   permutation to the matched `prev_lambdas` so the gather lands at
   the new position.

### 4.3 Activation predicate

The user's instinct is right: **gate on "any body has > 1 shape"**.
Detection cost is one host-side scan over `model.shape_body` at
solver construction:

```python
# In SolverPhoenX.__init__
shape_body = model.shape_body.numpy()
counts = np.bincount(shape_body[shape_body >= 0], minlength=num_bodies)
self._has_compound_bodies = bool((counts > 1).any())
```

Plumb a `bool` into `ingest_contacts(..., enable_body_pair_grouping)`.
The two ingest paths share the same downstream pipeline; only the
sort + boundary-comparison differ. A captured graph picks one branch
at construction and keeps it for the solver's lifetime — no per-step
predicate cost.

### 4.4 Cost analysis

For a 25k-contact step:

| Scene | Sort cost | Color count change | Net runtime |
|---|---|---|---|
| Single-shape bodies | sort skipped (predicate off) | unchanged | 0 |
| Mild compound (2 shapes / body) | ~0.05 ms (radix on 25k keys) | 78 → ~50 | ↓ |
| Heavy compound (Kapla planks, 4-6 shapes) | ~0.05 ms | 78 → ~28 (matches body-pair lower bound) | strong ↓ |

The radix sort itself is the dominant overhead, and CUB's pair-sort
on 25k int64 keys + int32 values is well under 0.1 ms on Blackwell.
The savings from fewer color rounds + fewer kernel launches dwarf
that on any compound-heavy scene.

---

## 5. Risks and open questions

1. **Friction discrimination across shape pairs in one body pair.**
   Compound bodies typically use the same material on every shape
   (the user authoring the compound *meant* it to behave as one
   object). But mixed materials are legal — e.g. a robot foot with a
   rubber pad shape and a steel core shape. If the merge max-reduces
   friction, the steel core wins; if it averages, both lose. **Decision
   needed**: max (defensible "compound materials use the most-grippy
   surface present") vs. per-contact (correct but bloats
   `ContactContainer`).

2. **Newton-side match index.** Newton's `rigid_contact_match_index`
   is built by `CollisionPipeline(contact_matching="sticky")` against
   the previous frame's *Newton-sorted* buffer. After PhoenX
   resorts, the match index points to a stale position. Two
   workable fixes: (a) sort PhoenX-side and apply the same
   permutation to the match index in the same kernel; (b) keep a
   permutation array and dereference once on warm-start gather.
   Both are mechanical.

3. **Contact-pair-wrench reporting.** `contact_pair_wrench_kernel`
   reports per-shape-pair wrenches for diagnostics / viewer overlays.
   After the merge, this becomes per-body-pair. If users want
   per-shape-pair detail, expose both: keep an `original_shape_pair`
   buffer alongside the contact (one extra int per contact) and
   reconstruct shape-pair groups on demand.

4. **Coloring-graph-build invariants.** The greedy coloring assumes
   the element interaction set is stable through the colour build.
   Body-pair grouping doesn't change that contract — it just reduces
   the node count. Tests should still cover Kapla tower, anymal
   piles, and rabbit pile (compound rabbits).

5. **Contact-matching invariants.** The `"sticky"` mode replays
   previous-frame contact geometry on matched contacts. If we
   permute, the geometry follows the permutation cleanly — but the
   match-index remap (Risk 2) must land first. Verifying this
   doesn't break the stacking tests is mandatory.

---

## 6. Recommended phasing

1. **Phase 1 — Behind-flag prototype.** Implement the resort + the
   gate predicate. Default: off. Add a unit test that builds a
   compound scene (e.g. 4-cube body × 4-cube body, dropped on a
   plane) and asserts the color count drops vs. single-shape baseline.
2. **Phase 2 — Friction policy.** Decide max vs per-contact;
   land the column-schema change accordingly.
3. **Phase 3 — Default-on.** Once the test suite is green and
   stacking benchmarks confirm no regression, flip the gate to "on
   when `_has_compound_bodies`".
4. **Phase 4 — Diagnostics.** Add `original_shape_pair` for
   per-shape-pair wrench reporting if users ask.

---

## 7. Summary

- Compound bodies bloat the graph coloring color count linearly with
  per-body-pair shape fanout. This is pure architectural waste
  because the PGS iterate kernel already does Gauss-Seidel within a
  cid — race-freedom inside a cid is structural, not coloring-driven.
- The cleanest fix is **one extra radix sort by body-pair key at the
  head of `ingest_contacts`**, plus a one-comparison change in the
  pair-boundary kernel. The downstream pipeline, the column schema,
  and the iterate kernel all stay unchanged.
- Activation: gate on a host-side `(shape_count_per_body > 1).any()`
  check at solver construction. Single-shape scenes pay nothing;
  compound-heavy scenes (Kapla, ragdolls, decomposed concave hulls)
  see color counts drop toward the body-pair-graph chromatic number.
- The main open decisions are (1) friction merge policy when shape
  pairs in the same body pair carry different materials, and (2)
  whether to keep per-shape-pair wrench reporting.
