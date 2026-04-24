# PhoenX Bug Log

Bugs found in the PhoenX solver pipeline. Each entry carries a repro, root
cause (when known), state, and a pointer to the regression test that locks
the fix in.

**States:**
- **open** — known, not yet fixed
- **fixed** — fix landed; regression test exists
- **wontfix** — documented limitation with a guard that fails loud

---

## #1 — int32 overflow in contact-ingest pair key packing

**State:** open
**Discovered:** 2026-04-24
**Repro:** any scene with `num_shapes >= 46_341` triggers silent
corruption. In practice, `h1_flat` at `num_worlds >= 859` hits it
(900/1024 worlds observed in the wild). `g1_flat` at similar
per-world shape counts will too.

**Root cause (confirmed):**
`_contact_key_kernel` in `contact_ingest.py:182` packs the
``(shape_a, shape_b)`` pair of each contact into a single ``int32``
key:

```python
keys[tid] = sa * num_shapes + sb
```

with `sa`, `sb`, and the `keys` array all typed as `int32`. For
`num_shapes ≥ ceil(sqrt(2^31)) ≈ 46_341`, the product
`sa * num_shapes` silently wraps. The corrupted `keys` feed
`runlength_encode_variable_length`, whose `run_values` get unpacked by
`_pair_metadata_kernel` back into `(sa, sb)` via `key // num_shapes`
— which then points at the **wrong** shape pair. Every downstream
consumer (body lookup, per-world routing, graph coloring) inherits
the misattribution.

**Observed downstream symptom (h1_flat @ 900 worlds, frame 1):**

- `model.collide()` produces 12 600 legitimate ground contacts (all
  shape-vs-`shape_world=-1`, zero cross-world). The collision
  pipeline is fine.
- After the PhoenX solver's ingest, the element table has
  `b0 = 3264 (world 163)` repeated across every single contact
  constraint, with `b1` scattered over bodies 9…17 998 spanning all
  900 worlds. Zero constraints reference the static ground.
- The graph colouring, asked to process ~7 200 constraints all
  routing through body 3264, can only fit one per colour →
  `num_colors = 7 202`, which in turn trips Bug #2.

**Scaling threshold (h1_flat, 54 shapes/world + 1 ground):**

| num_worlds | num_shapes | `num_shapes²` |     int32 safe? | observed n_colors |
| ---------- | ---------: | ------------: | --------------: | ----------------: |
| 800        |     43 201 |       1.87e9 |               ✓ |                 8 |
| 858        |     46 333 |       2.15e9 | at the boundary |                 — |
| 900        |     48 601 |       2.36e9 |          **✗**  |            7 202  |
| 1024       |     55 297 |       3.06e9 |          **✗**  |            8 193  |

**Fix plan:** Drop the packed-int32 key entirely. Contacts are
already sorted by `(shape0, shape1)` before ingest, so we can
compute pair boundaries by a simple "did the pair change from the
previous entry?" mark, inclusive-scan into run ids, and scatter
`pair_shape_a / pair_shape_b / pair_first` directly at the mark==1
positions. No key packing, no RLE helper, scales cleanly to int32
shape counts (≥2 billion shapes).

**Regression test:** _pending._

---

## #2 — `MAX_COLORS` overflow: silent buffer overrun in IncrementalContactPartitioner

**State:** fixed (2026-04-24, commit _pending-tw-tbd_)
**Discovered:** 2026-04-24 (as a downstream effect of Bug #1, but is its own
memory-safety bug on any path that produces >1024 colours).
**Repro:** Bug #1 triggers it in practice. A synthetic repro: any
constraint graph whose Jones-Plassmann colouring yields > `MAX_COLORS`
(= 1024) partitions.

**Symptom:** `IncrementalContactPartitioner._color_starts` is allocated
with shape `(MAX_COLORS + 1,) = (1025,)`. When the coloring loop
produces more colours than that, the kernel writes
`color_starts[c]` for `c > 1024`, out-of-bounds into the adjacent Warp
mempool allocation. Downstream kernels then read corrupted values. The
comment at the `MAX_COLORS` definition even states _"no realistic
physics constraint graph can exceed it"_ — but nothing enforces the
invariant.

**Observed signature:**
- Sometimes a silent CUDA `illegal memory access` surfaced only on
  module unload.
- Sometimes the coloring gradually feeds on its own corruption and
  colour counts blow up to 7000–8000.
- Rerun-to-rerun non-determinism at 1024 worlds (timing varied from
  3 ms to 10 s in the profiler).

**Root cause:** missing bounds check against `MAX_COLORS` inside
`build_csr`; no host-side post-build assertion either.

**Fix landed:**

- `incremental_tile_compact_csr_and_advance_kernel` now takes a
  `max_colors: int` parameter and an `overflow_flag: wp.array[int]`
  output. Before reading `color_starts[cc]` or writing
  `color_starts[cc + 1]`, it checks `cc >= max_colors`; if so, lane 0
  sets the flag, zeros `num_remaining` (which forces the outer
  `capture_while` to terminate on its next iteration), and every lane
  early-returns before any buffer access.
- `IncrementalContactPartitioner` allocates `_overflow_flag` at
  construction, zeros it at the top of `build_csr`, and reads it back
  with `.numpy()` after the capture_while exits. If it's set, raises
  `RuntimeError` with a descriptive message. When the device is
  already inside a user-level graph capture (e.g. a step captured into
  a CUDA graph), the readback is skipped — the kernel-side early exit
  still protects the buffer and the user sees the raise on the next
  uncaptured step.

**Regression test:**
`newton/_src/solvers/phoenx/tests/test_graph_coloring.py::TestMaxColorsOverflowGuard`.
- `test_hub_graph_exceeding_max_colors_raises`: clique of
  `MAX_COLORS + 100` elements all touching body 0 forces one colour
  per element. Asserts `build_csr` raises with "MAX_COLORS" in the
  message and `num_colors[0] <= MAX_COLORS` after.
- `test_budget_graph_below_max_colors_does_not_raise`: same shape
  with `MAX_COLORS − 50` elements; asserts the build completes.
