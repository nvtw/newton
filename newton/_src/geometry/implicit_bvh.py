# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit (pointer-free) AABB tree, a GPU alternative to :class:`warp.Bvh`.

The tree is a complete binary heap: node ``1`` is the root, node ``i`` has
children ``2 i`` and ``2 i + 1``, and the ``leaf_base`` leaves are the nodes
``[leaf_base, 2 * leaf_base)`` (they span the two deepest levels). Topology is
implied by the node index, so a build is a sort plus a few launches, a refit is
a single launch, and traversal needs no child pointers.

Leaves are ordered along a Hilbert curve of the item centers. Each subtree is
split where its run of curve keys changes in the highest bit, i.e. at a curve
cell boundary as in an LBVH, as far as the subtree's leaf capacity allows; a
few spare leaf slots give room for such splits. Spare leaves have inverted
bounds and ``last_leaf == -1``, as do subtrees holding only them.

Queries from kernels take :attr:`ImplicitBvh.data` and return item indices:

* :func:`implicit_bvh_query_aabb` / :func:`implicit_bvh_query_aabb_next`:
  iterate the items whose boxes overlap a box.
* :func:`implicit_bvh_query_ray` / :func:`implicit_bvh_query_point`: the
  closest box along a ray / nearest box to a point.
* Factories that fuse a user function into the traversal, for items that are
  not plain boxes (triangles, shapes, ...). The user function is inlined, so
  the threads of a warp stay in step; ``context`` is any value (typically a
  :func:`warp.struct`) passed through to it:

  * closest-item queries, :func:`implicit_bvh_make_query_ray` and
    :func:`implicit_bvh_make_query_point`: the user function
    ``(item, query..., bound, context) -> distance`` measures one item, and
    the query returns ``(item, distance)`` of the closest one;
  * visiting queries, :func:`implicit_bvh_make_query_ray_foreach` and
    :func:`implicit_bvh_make_query_pairs`: the user function
    ``(item, value, state, context) -> state`` is called for every hit and
    threads a ``state`` value through, which the query returns.
"""

from __future__ import annotations

import math
from typing import Any

import warp as wp

wp.set_module_options({"enable_backward": False})

_REDUCE_THREADS = 4096
_WARP_LEVELS = 5  # log2 of the warp size

# ---------------------------------------------------------------------------
# CUDA intrinsics
# ---------------------------------------------------------------------------


@wp.func_native("""
#if defined(__CUDA_ARCH__)
const float4 v = __ldg(reinterpret_cast<const float4*>(ptr) + index);
return wp::vec4(v.x, v.y, v.z, v.w);
#else
const float* f = reinterpret_cast<const float*>(ptr) + 4 * index;
return wp::vec4(f[0], f[1], f[2], f[3]);
#endif
""")
def _load_float4(ptr: wp.uint64, index: wp.int32) -> wp.vec4:
    """One 16-byte read-only load (a node is two of them)."""
    ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
const float4 v = __ldcg(reinterpret_cast<const float4*>(ptr) + index);
return wp::vec4(v.x, v.y, v.z, v.w);
#else
const float* f = reinterpret_cast<const float*>(ptr) + 4 * index;
return wp::vec4(f[0], f[1], f[2], f[3]);
#endif
""")
def _load_float4_coherent(ptr: wp.uint64, index: wp.int32) -> wp.vec4:
    """16-byte load past L1, to read nodes other blocks wrote in the same launch."""
    ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__threadfence();
#endif
""")
def _thread_fence():
    """Make this thread's global writes visible before its later writes and atomics."""
    ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return __float_as_int(value);
#else
int bits;
memcpy(&bits, &value, sizeof(bits));
return bits;
#endif
""")
def _float_bits(value: wp.float32) -> wp.int32:
    """Reinterpret the bits of a float as an int."""
    ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return __clz(value);
#else
int n = 0;
while (n < 32 && !(value & (0x80000000u >> n))) ++n;
return n;
#endif
""")
def _clz(value: wp.uint32) -> wp.int32:
    """Count leading zero bits."""
    ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return __shfl_xor_sync(0xffffffffu, value, lane_mask);
#else
return value;
#endif
""")
def _shfl_xor_f(value: wp.float32, lane_mask: wp.int32) -> wp.float32:
    """Exchange a float with lane ``lane ^ lane_mask`` of the warp."""
    ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return __shfl_xor_sync(0xffffffffu, value, lane_mask);
#else
return value;
#endif
""")
def _shfl_xor_i(value: wp.int32, lane_mask: wp.int32) -> wp.int32:
    """Exchange an int with lane ``lane ^ lane_mask`` of the warp."""
    ...


@wp.func_native("""
#if defined(__CUDA_ARCH__)
return __all_sync(0xffffffffu, predicate);
#else
return predicate;
#endif
""")
def _warp_all(predicate: bool) -> bool:
    """Whether ``predicate`` holds on every lane of the warp."""
    ...


# ---------------------------------------------------------------------------
# Tree data
# ---------------------------------------------------------------------------


@wp.struct
class ImplicitBvhNode:
    """One 32-byte node record, so visiting a node touches one memory sector.

    ``max_span`` is the largest leaf AABB edge in the subtree and ``last_leaf``
    the largest leaf slot in the subtree (``-1`` for empty subtrees); both let
    self-overlap queries report each pair exactly once.
    """

    lower: wp.vec3
    max_span: wp.float32
    upper: wp.vec3
    last_leaf: wp.int32


@wp.struct
class ImplicitBvhData:
    """Kernel-side view of an :class:`ImplicitBvh`.

    .. experimental::
    """

    nodes: wp.array[ImplicitBvhNode]
    nodes_ptr: wp.uint64  # address of ``nodes`` for vectorized read-only loads
    leaf_item: wp.array[wp.int32]
    leaf_base: wp.int32
    item_count: wp.int32


@wp.func
def _node_load(tree: ImplicitBvhData, node: wp.int32) -> tuple[wp.vec3, wp.vec3, wp.float32, wp.int32]:
    """Return ``(lower, upper, max_span, last_leaf)`` of ``node``."""
    a = _load_float4(tree.nodes_ptr, node * wp.int32(2))
    b = _load_float4(tree.nodes_ptr, node * wp.int32(2) + wp.int32(1))
    return wp.vec3(a[0], a[1], a[2]), wp.vec3(b[0], b[1], b[2]), a[3], _float_bits(b[3])


@wp.func
def implicit_bvh_next_subtree(node: wp.int32) -> wp.int32:
    """Return the depth-first successor of ``node``'s subtree (``0`` when done)."""
    while node > wp.int32(1):
        if (node & wp.int32(1)) == wp.int32(0):
            return node + wp.int32(1)
        node = node >> wp.int32(1)
    return wp.int32(0)


@wp.func
def aabb_disjoint(a_lower: wp.vec3, a_upper: wp.vec3, b_lower: wp.vec3, b_upper: wp.vec3) -> bool:
    """Whether two axis-aligned boxes are separated."""
    return (
        a_lower[0] > b_upper[0]
        or b_lower[0] > a_upper[0]
        or a_lower[1] > b_upper[1]
        or b_lower[1] > a_upper[1]
        or a_lower[2] > b_upper[2]
        or b_lower[2] > a_upper[2]
    )


@wp.func
def _ray_box_entry(origin: wp.vec3, inv_dir: wp.vec3, lower: wp.vec3, upper: wp.vec3, max_t: wp.float32) -> wp.float32:
    """Return the slab entry parameter in ``[0, max_t]``, or ``-1`` on a miss."""
    t1 = wp.cw_mul(lower - origin, inv_dir)
    t2 = wp.cw_mul(upper - origin, inv_dir)
    t_near = wp.min(t1, t2)
    t_far = wp.max(t1, t2)
    t_enter = wp.max(wp.max(t_near[0], t_near[1]), wp.max(t_near[2], wp.float32(0.0)))
    t_exit = wp.min(wp.min(t_far[0], t_far[1]), wp.min(t_far[2], max_t))
    if t_enter <= t_exit:
        return t_enter
    return wp.float32(-1.0)


@wp.func
def _point_box_dist_sq(p: wp.vec3, lower: wp.vec3, upper: wp.vec3) -> wp.float32:
    d = wp.max(wp.max(lower - p, p - upper), wp.vec3(0.0))
    return wp.dot(d, d)


# ---------------------------------------------------------------------------
# Build kernels
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _reset_bounds(bounds: wp.array[wp.vec3]):
    bounds[0] = wp.vec3(1.0e30, 1.0e30, 1.0e30)
    bounds[1] = wp.vec3(-1.0e30, -1.0e30, -1.0e30)


@wp.kernel(enable_backward=False)
def _center_bounds(lowers: wp.array[wp.vec3], uppers: wp.array[wp.vec3], count: int, bounds: wp.array[wp.vec3]):
    i = wp.tid()
    low = wp.vec3(1.0e30, 1.0e30, 1.0e30)
    high = wp.vec3(-1.0e30, -1.0e30, -1.0e30)
    while i < count:
        c = (lowers[i] + uppers[i]) * 0.5
        low = wp.min(low, c)
        high = wp.max(high, c)
        i += _REDUCE_THREADS
    if low[0] <= high[0]:
        wp.atomic_min(bounds, 0, low)
        wp.atomic_max(bounds, 1, high)


@wp.func
def _spread_10(v: wp.uint32) -> wp.uint32:
    v = (v * wp.uint32(0x00010001)) & wp.uint32(0xFF0000FF)
    v = (v * wp.uint32(0x00000101)) & wp.uint32(0x0F00F00F)
    v = (v * wp.uint32(0x00000011)) & wp.uint32(0xC30C30C3)
    v = (v * wp.uint32(0x00000005)) & wp.uint32(0x49249249)
    return v


@wp.func
def _hilbert_30(x: wp.uint32, y: wp.uint32, z: wp.uint32) -> wp.uint32:
    """30-bit 3D Hilbert index of 10-bit coordinates (Skilling's transpose algorithm).

    Consecutive Hilbert cells are face neighbors, so every run of consecutive
    leaves stays spatially compact; a Morton run that crosses a high-level cell
    boundary covers two distant regions instead.
    """
    q = wp.uint32(1) << wp.uint32(9)
    while q > wp.uint32(1):
        p = q - wp.uint32(1)
        if (x & q) != wp.uint32(0):
            x = x ^ p
        if (y & q) != wp.uint32(0):
            x = x ^ p
        else:
            t = (x ^ y) & p
            x = x ^ t
            y = y ^ t
        if (z & q) != wp.uint32(0):
            x = x ^ p
        else:
            t = (x ^ z) & p
            x = x ^ t
            z = z ^ t
        q = q >> wp.uint32(1)
    y = y ^ x
    z = z ^ y
    t = wp.uint32(0)
    q = wp.uint32(1) << wp.uint32(9)
    while q > wp.uint32(1):
        if (z & q) != wp.uint32(0):
            t = t ^ (q - wp.uint32(1))
        q = q >> wp.uint32(1)
    return (_spread_10(x ^ t) << wp.uint32(2)) | (_spread_10(y ^ t) << wp.uint32(1)) | _spread_10(z ^ t)


@wp.kernel(enable_backward=False)
def _hilbert_keys(
    lowers: wp.array[wp.vec3],
    uppers: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
    keys: wp.array[wp.int32],
    values: wp.array[wp.int32],
):
    i = wp.tid()
    lo = bounds[0]
    extent = bounds[1] - lo
    # One scale for all axes: normalizing each axis separately would stretch
    # flat scenes (e.g. terrain) and let their thin axis dominate the curve.
    scale = 1023.0 / wp.max(wp.max(extent[0], extent[1]), wp.max(extent[2], 1.0e-30))
    c = (lowers[i] + uppers[i]) * 0.5 - lo
    x = wp.uint32(wp.clamp(c[0] * scale, 0.0, 1023.0))
    y = wp.uint32(wp.clamp(c[1] * scale, 0.0, 1023.0))
    z = wp.uint32(wp.clamp(c[2] * scale, 0.0, 1023.0))
    keys[i] = wp.int32(_hilbert_30(x, y, z))
    values[i] = i


@wp.kernel(enable_backward=False)
def _identity_order(values: wp.array[wp.int32]):
    i = wp.tid()
    values[i] = i


@wp.func
def _overlap(lo: int, hi: int, range_lo: int, range_hi: int) -> int:
    return wp.max(0, wp.min(hi, range_hi) - wp.max(lo, range_lo))


@wp.func
def _subtree_capacity(node: int, leaf_base: int, deep_start: int) -> int:
    """Number of leaves under ``node``.

    Leaves ``[deep_start, 2 leaf_base)`` are on the deepest level and
    ``[leaf_base, deep_start)`` one level up, where ``deep_start`` is the power
    of two ``>= leaf_base``.
    """
    if node >= leaf_base:
        return 1
    shift = _clz(wp.uint32(node)) - _clz(wp.uint32(deep_start))
    return _overlap(node << shift, (node + 1) << shift, deep_start, 2 * leaf_base) + _overlap(
        node << (shift - 1), (node + 1) << (shift - 1), leaf_base, deep_start
    )


@wp.kernel(enable_backward=False)
def _split_level(
    keys: wp.array[wp.int32],
    count: int,
    level_start: int,
    leaf_base: int,
    deep_start: int,
    node_start: wp.array[wp.int32],
):
    """Split the sorted item runs of one tree level.

    Internal node ``n`` holds the items ``[node_start[n], end)``, where ``end``
    is the next node's start on the same level (``count`` for the last). The run
    is split where the highest bit differing between its first and last key
    changes, i.e. at a curve cell boundary, clamped so that both children fit.
    """
    n = level_start + wp.tid()
    a = node_start[n]
    b = count
    if n + 1 < 2 * level_start:
        b = node_start[n + 1]
    m = (a + b) // 2
    if b - a >= 2:
        ka = keys[a]
        kb = keys[b - 1]
        if ka != kb:
            h = 31 - _clz(wp.uint32(ka ^ kb))
            value = (kb >> h) << h
            lo = a
            hi = b - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if keys[mid] < value:
                    lo = mid + 1
                else:
                    hi = mid
            m = lo
    left = _subtree_capacity(2 * n, leaf_base, deep_start)
    right = _subtree_capacity(2 * n + 1, leaf_base, deep_start)
    m = wp.clamp(wp.clamp(m, b - right, a + left), a, b)
    node_start[2 * n] = a
    node_start[2 * n + 1] = m


@wp.func
def _leaf_node(position: int, leaf_base: int, deep_start: int) -> int:
    """Heap index of the leaf at left-to-right ``position`` (deepest level first)."""
    deep_count = 2 * leaf_base - deep_start
    if position < deep_count:
        return deep_start + position
    return leaf_base + position - deep_count


@wp.kernel(enable_backward=False)
def _assign_leaves(
    order: wp.array[wp.int32],
    node_start: wp.array[wp.int32],
    count: int,
    split: int,
    leaf_base: int,
    deep_start: int,
    leaf_item: wp.array[wp.int32],
):
    """Place the sorted items on the leaves: by the split runs, or by rank."""
    position = wp.tid()
    node = _leaf_node(position, leaf_base, deep_start)
    item = wp.int32(-1)
    if split != 0:
        a = node_start[node]
        b = count
        if position + 1 < leaf_base:
            b = node_start[_leaf_node(position + 1, leaf_base, deep_start)]
        if b > a:
            item = order[a]
    elif position < count:
        item = order[position]
    leaf_item[node - leaf_base] = item


@wp.kernel(enable_backward=False)
def _clear_internal(node_ints: wp.array[wp.int32]):
    node_ints[wp.tid() * 8 + 7] = wp.int32(-1)


_ARRIVED = wp.constant(wp.int32(-(1 << 31)))
"""Mark the first child to arrive leaves in its parent's ``last_leaf`` during an update."""


@wp.func
def _climb(nodes: wp.array[ImplicitBvhNode], nodes_ptr: wp.uint64, node_ints: wp.array[wp.int32], node: int):
    """Propagate bounds from a written ``node`` towards the root.

    Of a node's two children, the first to arrive marks the parent (an atomic
    exchange of its ``last_leaf``) and stops; the second computes the parent,
    whose new record replaces the mark, and continues. Children written by
    other blocks are read past L1, and every write is fenced before the next
    arrival.
    """
    _thread_fence()
    node = node >> 1
    while node >= 1:
        if wp.atomic_exch(node_ints, node * 8 + 7, _ARRIVED) != _ARRIVED:
            return
        l0 = _load_float4_coherent(nodes_ptr, node * 4)
        l1 = _load_float4_coherent(nodes_ptr, node * 4 + 1)
        r0 = _load_float4_coherent(nodes_ptr, node * 4 + 2)
        r1 = _load_float4_coherent(nodes_ptr, node * 4 + 3)
        record = ImplicitBvhNode()
        record.lower = wp.min(wp.vec3(l0[0], l0[1], l0[2]), wp.vec3(r0[0], r0[1], r0[2]))
        record.max_span = wp.max(l0[3], r0[3])
        record.upper = wp.max(wp.vec3(l1[0], l1[1], l1[2]), wp.vec3(r1[0], r1[1], r1[2]))
        record.last_leaf = wp.max(_float_bits(l1[3]), _float_bits(r1[3]))
        nodes[node] = record
        _thread_fence()
        node = node >> 1


@wp.func
def _make_leaf(lowers: wp.array[wp.vec3], uppers: wp.array[wp.vec3], item: wp.int32, slot: wp.int32) -> ImplicitBvhNode:
    record = ImplicitBvhNode()
    record.lower = wp.vec3(1.0e30, 1.0e30, 1.0e30)
    record.upper = wp.vec3(-1.0e30, -1.0e30, -1.0e30)
    record.max_span = wp.float32(-1.0)
    record.last_leaf = -1
    if item >= wp.int32(0):
        record.lower = lowers[item]
        record.upper = uppers[item]
        span = record.upper - record.lower
        record.max_span = wp.max(span[0], wp.max(span[1], span[2]))
        record.last_leaf = slot
    return record


@wp.kernel(enable_backward=False)
def _refit(
    lowers: wp.array[wp.vec3],
    uppers: wp.array[wp.vec3],
    leaf_item: wp.array[wp.int32],
    leaf_base: int,
    deep_start: int,
    nodes: wp.array[ImplicitBvhNode],
    nodes_ptr: wp.uint64,
    node_ints: wp.array[wp.int32],
):
    """Write the leaves and update all internal nodes in one launch.

    Threads cover the deepest leaves ``[deep_start, 2 leaf_base)`` and then the
    heap range ``[leaf_base & ~31, deep_start)`` holding the leaves one level
    up, both starting at a multiple of 32. A warp of 32 leaves thus owns one
    complete subtree: it reduces the subtree's five lowest levels with register
    shuffles and only its first lane climbs on. Other warps climb per lane.
    """
    t = wp.tid()
    deep_count = 2 * leaf_base - deep_start
    deep_threads = (deep_count + 31) // 32 * 32
    node = deep_start + t
    active = t < deep_count
    if t >= deep_threads:
        node = (leaf_base // 32) * 32 + t - deep_threads
        active = node >= leaf_base and node < deep_start
    record = _make_leaf(lowers, uppers, wp.int32(-1), 0)
    if active:
        record = _make_leaf(lowers, uppers, leaf_item[node - leaf_base], node - leaf_base)
        nodes[node] = record
    if _warp_all(active and deep_start >= 32):
        lane = t & 31
        for k in range(_WARP_LEVELS):
            m = 1 << k
            other_lower = wp.vec3(
                _shfl_xor_f(record.lower[0], m), _shfl_xor_f(record.lower[1], m), _shfl_xor_f(record.lower[2], m)
            )
            other_upper = wp.vec3(
                _shfl_xor_f(record.upper[0], m), _shfl_xor_f(record.upper[1], m), _shfl_xor_f(record.upper[2], m)
            )
            record.lower = wp.min(record.lower, other_lower)
            record.upper = wp.max(record.upper, other_upper)
            record.max_span = wp.max(record.max_span, _shfl_xor_f(record.max_span, m))
            record.last_leaf = wp.max(record.last_leaf, _shfl_xor_i(record.last_leaf, m))
            node = node >> 1
            if (lane & ((m << 1) - 1)) == 0:
                nodes[node] = record
        if lane == 0:
            _climb(nodes, nodes_ptr, node_ints, node)
    elif active:
        _climb(nodes, nodes_ptr, node_ints, node)


@wp.kernel(enable_backward=False)
def _update_internal(
    leaf_base: int, nodes: wp.array[ImplicitBvhNode], nodes_ptr: wp.uint64, node_ints: wp.array[wp.int32]
):
    _climb(nodes, nodes_ptr, node_ints, leaf_base + wp.tid())


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------


def _default_slack(item_capacity: int) -> float:
    """Leaf slots per item used when :class:`ImplicitBvh` is given no ``slack``.

    Spare slots let subtrees split at curve cell boundaries, which matters most
    for small and medium trees (1.75 up to 2^17 items); large trees gain little
    and taper to 1.15 at 2^20 items, about the memory of a ``wp.Bvh``.
    """
    lo, hi = 1 << 17, 1 << 20
    f = min(max(math.log(max(item_capacity, 1) / lo) / math.log(hi / lo), 0.0), 1.0)
    return 1.75 + f * (1.15 - 1.75)


class ImplicitBvh:
    """Bounding volume hierarchy over the boxes ``[lowers[i], uppers[i]]``, built on the GPU.

    :meth:`build` and :meth:`refit` only launch kernels on preallocated
    buffers, so they can be recorded in a CUDA graph. Rebuilding every step is
    cheap; :meth:`refit` keeps the leaf layout of the last build and only
    updates bounds. Pass :attr:`data` to kernels calling the
    ``implicit_bvh_query_*`` functions; items are box indices.

    .. experimental::

    Args:
        item_capacity: Largest item count passed to :meth:`build`.
        device: Device to allocate on.
        slack: Leaf slots per item; ``None`` chooses a value that favors query
            speed for small trees and memory for large ones.
        power_of_two: Round the leaf slots up to a power of two, which keeps all
            leaves on one level (leaf ``slot`` is then left-to-right position ``slot``).
    """

    def __init__(
        self,
        item_capacity: int,
        *,
        device: wp.DeviceLike = None,
        slack: float | None = None,
        power_of_two: bool = False,
    ) -> None:
        if item_capacity < 1:
            raise ValueError("item_capacity must be positive")
        slack = _default_slack(item_capacity) if slack is None else slack
        if slack < 1.0:
            raise ValueError("slack must be at least 1")
        self.device = wp.get_device(device)
        self.item_capacity = int(item_capacity)
        slots = max(2, math.ceil(self.item_capacity * slack))
        if power_of_two:
            slots = 1 << (slots - 1).bit_length()
        self.leaf_base = slots
        """Number of leaf slots; leaf ``slot`` is node ``leaf_base + slot``."""
        self.deep_start = 1 << (self.leaf_base - 1).bit_length()
        """First node of the deepest level; leaves ``[leaf_base, deep_start)`` are one level up."""
        self._depth = self.deep_start.bit_length() - 1
        self.item_count = 0
        """Item count of the last :meth:`build`."""
        self.nodes = wp.empty(2 * self.leaf_base, dtype=ImplicitBvhNode, device=self.device)
        self.leaf_item = wp.empty(self.leaf_base, dtype=wp.int32, device=self.device)
        self.bounds = wp.empty(2, dtype=wp.vec3, device=self.device)
        # The build scratch (sort keys and order, twice the item count each, and
        # the item run start of every node) occupies at most the first
        # ``3 leaf_base / 4`` internal records, which are only written after the
        # scratch is consumed; the leaf records start at ``leaf_base``.
        self._node_ints = wp.array(ptr=self.nodes.ptr, dtype=wp.int32, shape=(16 * self.leaf_base,), device=self.device)
        self._keys = self._node_ints[: 2 * self.item_capacity]
        self._order = self._node_ints[2 * self.item_capacity : 4 * self.item_capacity]
        self._node_start = self._node_ints[4 * self.item_capacity : 4 * self.item_capacity + 2 * self.leaf_base]
        self.data = ImplicitBvhData()
        """Kernel-side view to pass to kernels using the tree's query functions."""
        self.data.nodes = self.nodes
        self.data.nodes_ptr = self.nodes.ptr
        self.data.leaf_item = self.leaf_item
        self.data.leaf_base = self.leaf_base
        self.data.item_count = 0
        self._clear_marks()

    @classmethod
    def from_boxes(
        cls, lowers: wp.array[wp.vec3], uppers: wp.array[wp.vec3], *, device: wp.DeviceLike = None
    ) -> ImplicitBvh:
        """Build a tree over the boxes ``[lowers[i], uppers[i]]`` with default settings."""
        tree = cls(max(len(lowers), 1), device=device if device is not None else lowers.device)
        tree.build(lowers, uppers)
        return tree

    @property
    def slot_count(self) -> int:
        """Number of leaf slots; the launch size of queries made with :func:`implicit_bvh_make_query_pairs`."""
        return self.leaf_base

    @property
    def memory_bytes(self) -> int:
        """Bytes of device memory held by the tree, including build scratch."""
        return self.nodes.capacity + self.leaf_item.capacity + self.bounds.capacity

    def build(
        self,
        lowers: wp.array[wp.vec3],
        uppers: wp.array[wp.vec3],
        count: int | None = None,
        sort: bool = True,
    ) -> None:
        """Rebuild the tree over the first ``count`` boxes.

        Args:
            lowers: Lower box corners.
            uppers: Upper box corners.
            count: Number of items, at most ``item_capacity``; defaults to ``len(lowers)``.
            sort: Order the leaves along a Hilbert curve of the box centers. Without
                sorting the leaves keep the input order and are split by count,
                which only suits input that is already spatially coherent.
        """
        count = len(lowers) if count is None else int(count)
        if count > self.item_capacity:
            raise ValueError(f"count {count} exceeds item_capacity {self.item_capacity}")
        self.item_count = count
        self.data.item_count = count
        launch = {"device": self.device, "record_tape": False}
        sort = sort and count > 0
        if sort:
            wp.launch(_reset_bounds, dim=1, inputs=[self.bounds], **launch)
            wp.launch(
                _center_bounds, dim=min(count, _REDUCE_THREADS), inputs=[lowers, uppers, count, self.bounds], **launch
            )
            wp.launch(_hilbert_keys, dim=count, inputs=[lowers, uppers, self.bounds, self._keys, self._order], **launch)
            wp.utils.radix_sort_pairs(self._keys, self._order, count)
        elif count > 0:
            wp.launch(_identity_order, dim=count, inputs=[self._order], **launch)
        if sort:
            self._node_start[0:2].zero_()
            for level in range(self._depth):
                internal = min(2 << level, self.leaf_base) - (1 << level)
                if internal > 0:
                    wp.launch(
                        _split_level,
                        dim=internal,
                        inputs=[self._keys, count, 1 << level, self.leaf_base, self.deep_start, self._node_start],
                        **launch,
                    )
        wp.launch(
            _assign_leaves,
            dim=self.leaf_base,
            inputs=[self._order, self._node_start, count, int(sort), self.leaf_base, self.deep_start, self.leaf_item],
            **launch,
        )
        self._clear_marks()
        self.refit(lowers, uppers)

    def refit(self, lowers: wp.array[wp.vec3], uppers: wp.array[wp.vec3]) -> None:
        """Recompute all node bounds from moved boxes, keeping the leaf layout of the last build."""
        deep_threads = (2 * self.leaf_base - self.deep_start + 31) // 32 * 32
        shallow_threads = (self.deep_start - self.leaf_base // 32 * 32 + 31) // 32 * 32
        wp.launch(
            _refit,
            dim=deep_threads + shallow_threads,
            inputs=[
                lowers,
                uppers,
                self.leaf_item,
                self.leaf_base,
                self.deep_start,
                self.nodes,
                self.nodes.ptr,
                self._node_ints,
            ],
            device=self.device,
            record_tape=False,
        )

    def update_internal_nodes(self) -> None:
        """Recompute the internal nodes after the leaf records were written directly."""
        wp.launch(
            _update_internal,
            dim=self.leaf_base,
            inputs=[self.leaf_base, self.nodes, self.nodes.ptr, self._node_ints],
            device=self.device,
            record_tape=False,
        )

    def _clear_marks(self) -> None:
        """Reset the internal records' ``last_leaf``, which updates use as arrival marks."""
        wp.launch(_clear_internal, dim=self.leaf_base, inputs=[self._node_ints], device=self.device, record_tape=False)


# ---------------------------------------------------------------------------
# AABB overlap iterator
# ---------------------------------------------------------------------------


@wp.struct
class ImplicitBvhAabbQuery:
    """State of an AABB-overlap query; ``item`` is the current hit or ``-1``.

    .. experimental::
    """

    lower: wp.vec3
    upper: wp.vec3
    node: wp.int32
    item: wp.int32


@wp.func
def implicit_bvh_query_aabb_next(tree: ImplicitBvhData, query: ImplicitBvhAabbQuery) -> ImplicitBvhAabbQuery:
    """Advance ``query`` to its next overlapping item (``item == -1`` when exhausted).

    .. experimental::
    """
    node = query.node
    item = wp.int32(-1)
    while node != wp.int32(0) and item < wp.int32(0):
        lower, upper, _span, _last = _node_load(tree, node)
        if aabb_disjoint(lower, upper, query.lower, query.upper):
            node = implicit_bvh_next_subtree(node)
        elif node < tree.leaf_base:
            node = node * wp.int32(2)
        else:
            item = tree.leaf_item[node - tree.leaf_base]
            node = implicit_bvh_next_subtree(node)
    result = query
    result.node = node
    result.item = item
    return result


@wp.func
def implicit_bvh_query_aabb(tree: ImplicitBvhData, lower: wp.vec3, upper: wp.vec3) -> ImplicitBvhAabbQuery:
    """Start an AABB-overlap query positioned on the first hit (``item == -1`` if none).

    Iterate with::

        q = implicit_bvh_query_aabb(tree, lower, upper)
        while q.item >= 0:
            ...  # q.item overlaps [lower, upper]
            q = implicit_bvh_query_aabb_next(tree, q)

    .. experimental::
    """
    query = ImplicitBvhAabbQuery()
    query.lower = lower
    query.upper = upper
    query.node = wp.where(tree.item_count > wp.int32(0), wp.int32(1), wp.int32(0))
    query.item = wp.int32(-1)
    return implicit_bvh_query_aabb_next(tree, query)


# ---------------------------------------------------------------------------
# Fused nearest-first queries
# ---------------------------------------------------------------------------


@wp.func
def _ray_node_entry(
    tree: ImplicitBvhData, node: wp.int32, origin: wp.vec3, inv_dir: wp.vec3, max_t: wp.float32
) -> wp.float32:
    """Ray entry parameter into ``node``'s box within ``max_t``, or ``-1``."""
    lower, upper, _span, last_leaf = _node_load(tree, node)
    if last_leaf < wp.int32(0):
        return wp.float32(-1.0)
    return _ray_box_entry(origin, inv_dir, lower, upper, max_t)


def _make_ray_traversal(leaf_func: Any, closest: bool) -> Any:
    """Build a front-to-back ray traversal calling ``leaf_func`` at every leaf the ray enters.

    Of two children the one the ray enters first is visited first; a trail with
    one bit per level records whether the other is pending, so the walk needs no
    stack. With ``closest``, ``leaf_func`` returns a hit distance that shrinks
    ``max_t``; a pending sibling is then tested again, otherwise it is entered
    directly (a pending leaf with the entry it was tested with).
    """

    @wp.func
    def traverse(
        tree: ImplicitBvhData, origin: wp.vec3, direction: wp.vec3, max_t: wp.float32, state: Any, context: Any
    ):
        inv_dir = wp.vec3(1.0 / direction[0], 1.0 / direction[1], 1.0 / direction[2])
        start_max_t = max_t
        best = wp.int32(-1)
        node = wp.int32(1)
        level = wp.int32(-1)
        trail = wp.uint32(0)
        expand = bool(False)
        t = wp.float32(-1.0)
        pending_t = wp.float32(-1.0)
        if tree.item_count > wp.int32(0):
            t = _ray_node_entry(tree, wp.int32(1), origin, inv_dir, max_t)
            if t >= wp.float32(0.0):
                level = wp.int32(0)
                expand = True
        while level >= wp.int32(0):
            if expand:
                if node >= tree.leaf_base:
                    if t <= max_t:
                        item = tree.leaf_item[node - tree.leaf_base]
                        if wp.static(closest):
                            t_hit = t
                            if wp.static(leaf_func is not None):
                                t_hit = leaf_func(item, origin, direction, max_t, context)
                            if t_hit >= wp.float32(0.0) and t_hit < max_t:
                                max_t = t_hit
                                best = item
                        else:
                            state = leaf_func(item, t, state, context)
                    expand = False
                else:
                    left = node * wp.int32(2)
                    t_left = _ray_node_entry(tree, left, origin, inv_dir, max_t)
                    t_right = _ray_node_entry(tree, left + wp.int32(1), origin, inv_dir, max_t)
                    if t_left >= wp.float32(0.0) or t_right >= wp.float32(0.0):
                        level += wp.int32(1)
                        bit = wp.uint32(1) << wp.uint32(level)
                        trail = trail | bit
                        if t_left >= wp.float32(0.0) and t_right >= wp.float32(0.0):
                            trail = trail & ~bit
                        if t_left >= wp.float32(0.0) and (t_right < wp.float32(0.0) or t_left <= t_right):
                            node = left
                            t = t_left
                            pending_t = t_right
                        else:
                            node = left + wp.int32(1)
                            t = t_right
                            pending_t = t_left
                    else:
                        expand = False
            elif level == wp.int32(0):
                level = wp.int32(-1)
            else:
                bit = wp.uint32(1) << wp.uint32(level)
                if (trail & bit) == wp.uint32(0):
                    trail = trail | bit
                    node = node ^ wp.int32(1)
                    if max_t < start_max_t:
                        t = _ray_node_entry(tree, node, origin, inv_dir, max_t)
                    elif node >= tree.leaf_base:
                        t = pending_t
                    else:
                        t = wp.float32(0.0)
                    expand = t >= wp.float32(0.0)
                else:
                    node = node >> wp.int32(1)
                    level -= wp.int32(1)
        return best, max_t, state

    return traverse


def implicit_bvh_make_query_ray(hit_func: Any) -> Any:
    """Create a fused closest-hit ray query.

    ``hit_func(item: int, origin: vec3, direction: vec3, max_t: float, context: Any) -> float``
    intersects the ray with item ``item`` and returns the hit distance, or a
    negative value on a miss. The returned function
    ``query(tree, origin, direction, max_t, context) -> (item, t)`` visits items
    front to back, shrinks ``max_t`` with every hit, and returns the closest
    item (``-1`` if none) and its distance.

    .. experimental::
    """
    traverse = _make_ray_traversal(hit_func, True)

    @wp.func
    def ray_closest(tree: ImplicitBvhData, origin: wp.vec3, direction: wp.vec3, max_t: wp.float32, context: Any):
        item, t, _state = traverse(tree, origin, direction, max_t, wp.int32(0), context)
        return item, t

    return ray_closest


_ray_closest_box = _make_ray_traversal(None, True)


@wp.func
def implicit_bvh_query_ray(
    tree: ImplicitBvhData, origin: wp.vec3, direction: wp.vec3, max_t: wp.float32
) -> tuple[wp.int32, wp.float32]:
    """Return ``(item, t)`` of the first box a ray enters within ``max_t`` (``item == -1`` if none).

    ``t`` is the entry distance in units of ``direction``; a ray starting inside a box enters it at ``0``.

    .. experimental::
    """
    item, t, _state = _ray_closest_box(tree, origin, direction, max_t, wp.int32(0), wp.int32(0))
    return item, t


def implicit_bvh_make_query_ray_foreach(visit_func: Any) -> Any:
    """Create a fused query visiting every item whose box a ray enters.

    ``visit_func(item: int, t: float, state: Any, context: Any) -> Any`` is
    called with the box entry distance ``t`` of each item within ``max_t`` and
    returns the updated ``state``. The returned function
    ``query(tree, origin, direction, max_t, state, context)`` returns the final state.

    .. experimental::
    """
    traverse = _make_ray_traversal(visit_func, False)

    @wp.func
    def ray_foreach(
        tree: ImplicitBvhData, origin: wp.vec3, direction: wp.vec3, max_t: wp.float32, state: Any, context: Any
    ):
        _item, _t, result = traverse(tree, origin, direction, max_t, state, context)
        return result

    return ray_foreach


@wp.func
def _node_dist_sq(tree: ImplicitBvhData, node: wp.int32, point: wp.vec3) -> wp.float32:
    """Squared distance from ``point`` to ``node``'s box, or ``-1`` for an empty subtree."""
    lower, upper, _span, last_leaf = _node_load(tree, node)
    if last_leaf < wp.int32(0):
        return wp.float32(-1.0)
    return _point_box_dist_sq(point, lower, upper)


def implicit_bvh_make_query_point(dist_func: Any) -> Any:
    """Create a fused nearest-item query.

    ``dist_func(item: int, point: vec3, max_dist: float, context: Any) -> float``
    returns the distance from ``point`` to item ``item`` (any value ``>= max_dist``
    when it is not closer). The returned function
    ``query(tree, point, max_dist, context) -> (item, dist)`` visits subtrees
    nearest first, prunes those farther than the best distance so far, and
    returns the nearest item (``-1`` if none lies within ``max_dist``) and its
    distance.

    .. experimental::
    """

    @wp.func
    def point_closest(tree: ImplicitBvhData, point: wp.vec3, max_dist: wp.float32, context: Any):
        best = wp.int32(-1)
        r2 = max_dist * max_dist
        node = wp.int32(1)
        level = wp.int32(-1)
        trail = wp.uint32(0)
        expand = bool(False)
        if tree.item_count > wp.int32(0):
            d = _node_dist_sq(tree, wp.int32(1), point)
            if d >= wp.float32(0.0) and d <= r2:
                level = wp.int32(0)
                expand = True
        while level >= wp.int32(0):
            if expand:
                if node >= tree.leaf_base:
                    item = tree.leaf_item[node - tree.leaf_base]
                    if wp.static(dist_func is None):
                        dist = wp.sqrt(_node_dist_sq(tree, node, point))
                    else:
                        dist = dist_func(item, point, max_dist, context)
                    if dist < max_dist:
                        max_dist = dist
                        r2 = dist * dist
                        best = item
                    expand = False
                else:
                    left = node * wp.int32(2)
                    d_left = _node_dist_sq(tree, left, point)
                    d_right = _node_dist_sq(tree, left + wp.int32(1), point)
                    in_left = d_left >= wp.float32(0.0) and d_left <= r2
                    in_right = d_right >= wp.float32(0.0) and d_right <= r2
                    if in_left or in_right:
                        level += wp.int32(1)
                        bit = wp.uint32(1) << wp.uint32(level)
                        trail = trail | bit
                        if in_left and in_right:
                            trail = trail & ~bit
                        if in_left and (not in_right or d_left <= d_right):
                            node = left
                        else:
                            node = left + wp.int32(1)
                    else:
                        expand = False
            elif level == wp.int32(0):
                level = wp.int32(-1)
            else:
                bit = wp.uint32(1) << wp.uint32(level)
                if (trail & bit) == wp.uint32(0):
                    # The pending sibling may be out of reach after the radius shrank.
                    trail = trail | bit
                    node = node ^ wp.int32(1)
                    d = _node_dist_sq(tree, node, point)
                    expand = d >= wp.float32(0.0) and d <= r2
                else:
                    node = node >> wp.int32(1)
                    level -= wp.int32(1)
        return best, max_dist

    return point_closest


_point_closest_box = implicit_bvh_make_query_point(None)


@wp.func
def implicit_bvh_query_point(
    tree: ImplicitBvhData, point: wp.vec3, max_dist: wp.float32
) -> tuple[wp.int32, wp.float32]:
    """Return ``(item, dist)`` of the box nearest to ``point`` within ``max_dist`` (``item == -1`` if none).

    ``dist`` is ``0`` for a point inside a box.

    .. experimental::
    """
    return _point_closest_box(tree, point, max_dist, wp.int32(0))


def implicit_bvh_make_query_pairs(visit_func: Any) -> Any:
    """Create a fused query visiting every pair of overlapping items once.

    ``visit_func(item: int, other: int, state: Any, context: Any) -> Any`` is
    called for each item whose box overlaps that of the query slot's item and
    returns the updated ``state``. The returned function
    ``query(tree, slot, state, context)`` returns the final state. Launched
    over :attr:`ImplicitBvh.slot_count` threads (larger launches are fine,
    surplus slots visit nothing), the queries together visit every
    overlapping pair exactly once: a pair belongs to the item with the smaller
    box span (ties: smaller slot). Each slot walks up to the root and searches
    only the sibling subtrees on its way, as its ancestors contain it.

    .. experimental::
    """

    @wp.func
    def self_foreach(tree: ImplicitBvhData, slot: wp.int32, state: Any, context: Any):
        if slot >= tree.leaf_base:
            return state
        leaf = tree.leaf_base + slot
        lower, upper, span, last_leaf = _node_load(tree, leaf)
        item = tree.leaf_item[slot]
        ancestor = wp.where(last_leaf >= wp.int32(0), leaf, wp.int32(1))
        node = ancestor ^ wp.int32(1)
        stop = implicit_bvh_next_subtree(node)
        while ancestor > wp.int32(1):
            if node == stop:
                ancestor = ancestor >> wp.int32(1)
                node = ancestor ^ wp.int32(1)
                stop = implicit_bvh_next_subtree(node)
            else:
                n_lower, n_upper, n_span, n_last = _node_load(tree, node)
                if (
                    n_span < span
                    or (n_span == span and n_last <= slot)
                    or aabb_disjoint(n_lower, n_upper, lower, upper)
                ):
                    node = implicit_bvh_next_subtree(node)
                elif node < tree.leaf_base:
                    node = node * wp.int32(2)
                else:
                    state = visit_func(item, tree.leaf_item[node - tree.leaf_base], state, context)
                    node = implicit_bvh_next_subtree(node)
        return state

    return self_foreach
