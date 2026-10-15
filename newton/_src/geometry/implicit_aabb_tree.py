# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit (pointer-free) AABB tree with stackless traversal.

The tree is a *complete* binary tree stored in heap order: node ``1`` is the
root, node ``i`` has children ``2 i`` and ``2 i + 1``, and the ``leaf_base``
(next power of two of the leaf count) leaves occupy nodes
``[leaf_base, 2 * leaf_base)``. Leaves are ordered along a Morton curve of the
item centers, so spatially close items share subtrees. Unused padding leaves
carry inverted bounds and ``last_leaf == -1``; every internal node whose
subtree contains only padding inherits both, so queries reject it at once.

Because the topology is implied by the node index, building requires only a
radix sort plus ``log2(leaf_base)`` bottom-up reduction launches, a refit
re-reads the bounds in the stored leaf order, and traversal needs no stack:
:func:`implicit_tree_next_subtree` advances a depth-first walk past a node's
subtree by index arithmetic alone. All operations are CUDA-graph capturable.

Use from kernels through :class:`ImplicitAABBTreeData` (``tree.data``)::

    q = implicit_tree_query_aabb(tree, lower, upper)
    while q.item >= 0:
        ...  # q.item overlaps [lower, upper]
        q = implicit_tree_query_aabb_next(tree, q)
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})

_REDUCE_THREADS = 4096


@wp.struct
class ImplicitTreeNode:
    """One 32-byte node record, so visiting a node touches one memory sector.

    ``max_span`` is the largest leaf AABB edge in the subtree and ``last_leaf``
    the largest leaf slot in the subtree (``-1`` for padding-only subtrees).
    Both let self-overlap queries report each pair exactly once.
    """

    lower: wp.vec3
    max_span: wp.float32
    upper: wp.vec3
    last_leaf: wp.int32


@wp.struct
class ImplicitAABBTreeData:
    """Kernel-side view of an :class:`ImplicitAABBTree`."""

    nodes: wp.array[ImplicitTreeNode]
    leaf_item: wp.array[wp.int32]
    leaf_base: wp.int32
    leaf_count: wp.int32


@wp.struct
class ImplicitTreeAabbQuery:
    """State of an AABB-overlap query; ``item`` is the current hit or ``-1``."""

    lower: wp.vec3
    upper: wp.vec3
    node: wp.int32
    item: wp.int32


@wp.struct
class ImplicitTreeRayQuery:
    """State of a ray query; lower ``max_t`` between steps to prune (closest hit)."""

    origin: wp.vec3
    inv_dir: wp.vec3
    max_t: wp.float32
    t: wp.float32
    node: wp.int32
    item: wp.int32


@wp.struct
class ImplicitTreePointQuery:
    """State of a point-distance query; lower ``max_dist`` between steps to prune.

    Reports every item whose AABB lies within ``max_dist`` of ``point``.
    Traversal is unordered, so shrink ``max_dist`` as closer items are found.
    """

    point: wp.vec3
    max_dist: wp.float32
    node: wp.int32
    item: wp.int32


@wp.struct
class ImplicitTreeSelfQuery:
    """State of a self-overlap query from one leaf slot.

    Iterates the leaves whose AABB overlaps the query leaf such that every
    overlapping pair is reported by exactly one of its two leaves.
    """

    lower: wp.vec3
    upper: wp.vec3
    span: wp.float32
    slot: wp.int32
    ancestor: wp.int32
    stop: wp.int32
    node: wp.int32
    item: wp.int32


@wp.func
def implicit_tree_next_subtree(node: wp.int32) -> wp.int32:
    """Return the depth-first successor of ``node``'s subtree (``0`` when done)."""
    while node > wp.int32(1):
        if (node & wp.int32(1)) == wp.int32(0):
            return node + wp.int32(1)
        node = node >> wp.int32(1)
    return wp.int32(0)


@wp.func
def _spread_10(v: wp.uint32) -> wp.uint32:
    v = (v * wp.uint32(0x00010001)) & wp.uint32(0xFF0000FF)
    v = (v * wp.uint32(0x00000101)) & wp.uint32(0x0F00F00F)
    v = (v * wp.uint32(0x00000011)) & wp.uint32(0xC30C30C3)
    v = (v * wp.uint32(0x00000005)) & wp.uint32(0x49249249)
    return v


@wp.func
def _aabb_disjoint(a_lower: wp.vec3, a_upper: wp.vec3, b_lower: wp.vec3, b_upper: wp.vec3) -> bool:
    return (
        a_lower[0] > b_upper[0]
        or b_lower[0] > a_upper[0]
        or a_lower[1] > b_upper[1]
        or b_lower[1] > a_upper[1]
        or a_lower[2] > b_upper[2]
        or b_lower[2] > a_upper[2]
    )


@wp.kernel(enable_backward=False)
def _reset_bounds(bounds: wp.array[wp.vec3]):
    bounds[0] = wp.vec3(1.0e30, 1.0e30, 1.0e30)
    bounds[1] = wp.vec3(-1.0e30, -1.0e30, -1.0e30)


@wp.kernel(enable_backward=False)
def _center_bounds(
    lowers: wp.array[wp.vec3],
    uppers: wp.array[wp.vec3],
    count: int,
    bounds: wp.array[wp.vec3],
):
    tid = wp.tid()
    stride = wp.int32(_REDUCE_THREADS)
    low = wp.vec3(1.0e30, 1.0e30, 1.0e30)
    high = wp.vec3(-1.0e30, -1.0e30, -1.0e30)
    i = tid
    while i < count:
        c = (lowers[i] + uppers[i]) * 0.5
        low = wp.min(low, c)
        high = wp.max(high, c)
        i += stride
    if low[0] <= high[0]:
        wp.atomic_min(bounds, 0, low)
        wp.atomic_max(bounds, 1, high)


@wp.kernel(enable_backward=False)
def _morton_keys(
    lowers: wp.array[wp.vec3],
    uppers: wp.array[wp.vec3],
    bounds: wp.array[wp.vec3],
    keys: wp.array[wp.int32],
    values: wp.array[wp.int32],
):
    i = wp.tid()
    lo = bounds[0]
    extent = bounds[1] - lo
    c = (lowers[i] + uppers[i]) * 0.5 - lo
    x = wp.uint32(wp.clamp(c[0] * 1023.0 / wp.max(extent[0], 1.0e-30), 0.0, 1023.0))
    y = wp.uint32(wp.clamp(c[1] * 1023.0 / wp.max(extent[1], 1.0e-30), 0.0, 1023.0))
    z = wp.uint32(wp.clamp(c[2] * 1023.0 / wp.max(extent[2], 1.0e-30), 0.0, 1023.0))
    code = (_spread_10(x) << wp.uint32(2)) | (_spread_10(y) << wp.uint32(1)) | _spread_10(z)
    keys[i] = wp.int32(code)
    values[i] = i


@wp.kernel(enable_backward=False)
def _identity_order(values: wp.array[wp.int32]):
    i = wp.tid()
    values[i] = i


@wp.kernel(enable_backward=False)
def _write_leaves(
    lowers: wp.array[wp.vec3],
    uppers: wp.array[wp.vec3],
    order: wp.array[wp.int32],
    count: int,
    leaf_base: int,
    nodes: wp.array[ImplicitTreeNode],
    leaf_item: wp.array[wp.int32],
):
    slot = wp.tid()
    record = ImplicitTreeNode()
    record.lower = wp.vec3(1.0e30, 1.0e30, 1.0e30)
    record.upper = wp.vec3(-1.0e30, -1.0e30, -1.0e30)
    record.max_span = wp.float32(-1.0)
    record.last_leaf = -1
    item = wp.int32(-1)
    if slot < count:
        item = order[slot]
        record.lower = lowers[item]
        record.upper = uppers[item]
        span = record.upper - record.lower
        record.max_span = wp.max(span[0], wp.max(span[1], span[2]))
        record.last_leaf = slot
    leaf_item[slot] = item
    nodes[leaf_base + slot] = record


@wp.kernel(enable_backward=False)
def _build_level(nodes: wp.array[ImplicitTreeNode], start: int):
    node = start + wp.tid()
    left = nodes[node * 2]
    right = nodes[node * 2 + 1]
    record = ImplicitTreeNode()
    record.lower = wp.min(left.lower, right.lower)
    record.upper = wp.max(left.upper, right.upper)
    record.last_leaf = wp.max(left.last_leaf, right.last_leaf)
    record.max_span = wp.max(left.max_span, right.max_span)
    nodes[node] = record


@wp.func
def implicit_tree_query_aabb_next(tree: ImplicitAABBTreeData, query: ImplicitTreeAabbQuery) -> ImplicitTreeAabbQuery:
    """Advance ``query`` to its next overlapping item (``item == -1`` when exhausted)."""
    # A single flat loop without early returns compiles to much tighter code
    # than nested loops or returns from inside the loop.
    node = query.node
    item = wp.int32(-1)
    while node != wp.int32(0) and item < wp.int32(0):
        record = tree.nodes[node]
        if _aabb_disjoint(record.lower, record.upper, query.lower, query.upper):
            node = implicit_tree_next_subtree(node)
        elif node < tree.leaf_base:
            node = node * wp.int32(2)
        else:
            item = tree.leaf_item[node - tree.leaf_base]
            node = implicit_tree_next_subtree(node)
    result = query
    result.node = node
    result.item = item
    return result


@wp.func
def implicit_tree_query_aabb(tree: ImplicitAABBTreeData, lower: wp.vec3, upper: wp.vec3) -> ImplicitTreeAabbQuery:
    """Start an AABB-overlap query positioned on the first hit (``item == -1`` if none)."""
    query = ImplicitTreeAabbQuery()
    query.lower = lower
    query.upper = upper
    query.node = wp.where(tree.leaf_count > wp.int32(0), wp.int32(1), wp.int32(0))
    query.item = wp.int32(-1)
    return implicit_tree_query_aabb_next(tree, query)


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
def implicit_tree_query_ray_next(tree: ImplicitAABBTreeData, query: ImplicitTreeRayQuery) -> ImplicitTreeRayQuery:
    """Advance ``query`` to the next item whose AABB the ray enters within ``max_t``."""
    node = query.node
    item = wp.int32(-1)
    t = wp.float32(0.0)
    while node != wp.int32(0) and item < wp.int32(0):
        record = tree.nodes[node]
        t = wp.float32(-1.0)
        if record.last_leaf >= wp.int32(0):
            t = _ray_box_entry(query.origin, query.inv_dir, record.lower, record.upper, query.max_t)
        if t < wp.float32(0.0):
            node = implicit_tree_next_subtree(node)
        elif node < tree.leaf_base:
            node = node * wp.int32(2)
        else:
            item = tree.leaf_item[node - tree.leaf_base]
            node = implicit_tree_next_subtree(node)
    result = query
    result.node = node
    result.item = item
    result.t = t
    return result


@wp.func
def implicit_tree_query_ray(
    tree: ImplicitAABBTreeData, origin: wp.vec3, direction: wp.vec3, max_t: wp.float32
) -> ImplicitTreeRayQuery:
    """Start a ray query positioned on the first hit; ``query.t`` is the AABB entry parameter."""
    query = ImplicitTreeRayQuery()
    query.origin = origin
    query.inv_dir = wp.vec3(1.0 / direction[0], 1.0 / direction[1], 1.0 / direction[2])
    query.max_t = max_t
    query.t = wp.float32(0.0)
    query.node = wp.where(tree.leaf_count > wp.int32(0), wp.int32(1), wp.int32(0))
    query.item = wp.int32(-1)
    return implicit_tree_query_ray_next(tree, query)


@wp.func
def _point_box_dist_sq(p: wp.vec3, lower: wp.vec3, upper: wp.vec3) -> wp.float32:
    d = wp.max(wp.max(lower - p, p - upper), wp.vec3(0.0))
    return wp.dot(d, d)


@wp.func
def implicit_tree_query_point_next(tree: ImplicitAABBTreeData, query: ImplicitTreePointQuery) -> ImplicitTreePointQuery:
    """Advance ``query`` to the next item whose AABB lies within ``max_dist``."""
    node = query.node
    item = wp.int32(-1)
    r2 = query.max_dist * query.max_dist
    while node != wp.int32(0) and item < wp.int32(0):
        record = tree.nodes[node]
        if record.last_leaf < wp.int32(0) or _point_box_dist_sq(query.point, record.lower, record.upper) > r2:
            node = implicit_tree_next_subtree(node)
        elif node < tree.leaf_base:
            node = node * wp.int32(2)
        else:
            item = tree.leaf_item[node - tree.leaf_base]
            node = implicit_tree_next_subtree(node)
    result = query
    result.node = node
    result.item = item
    return result


@wp.func
def implicit_tree_query_point(
    tree: ImplicitAABBTreeData, point: wp.vec3, max_dist: wp.float32
) -> ImplicitTreePointQuery:
    """Start a point-distance query positioned on the first item within ``max_dist``."""
    query = ImplicitTreePointQuery()
    query.point = point
    query.max_dist = max_dist
    query.node = wp.where(tree.leaf_count > wp.int32(0), wp.int32(1), wp.int32(0))
    query.item = wp.int32(-1)
    return implicit_tree_query_point_next(tree, query)


@wp.func
def implicit_tree_query_self_next(tree: ImplicitAABBTreeData, query: ImplicitTreeSelfQuery) -> ImplicitTreeSelfQuery:
    """Advance a self-overlap query; see :class:`ImplicitTreeSelfQuery`."""
    ancestor = query.ancestor
    stop = query.stop
    node = query.node
    item = wp.int32(-1)
    while ancestor > wp.int32(1) and item < wp.int32(0):
        if node == stop:
            # Sibling subtree exhausted: continue with the parent's sibling.
            ancestor = ancestor >> wp.int32(1)
            node = ancestor ^ wp.int32(1)
            stop = implicit_tree_next_subtree(node)
            continue
        record = tree.nodes[node]
        max_span = record.max_span
        # The leaf with the smaller span (ties: smaller slot) reports a pair.
        if (
            max_span < query.span
            or (max_span == query.span and record.last_leaf <= query.slot)
            or _aabb_disjoint(record.lower, record.upper, query.lower, query.upper)
        ):
            node = implicit_tree_next_subtree(node)
        elif node < tree.leaf_base:
            node = node * wp.int32(2)
        else:
            item = tree.leaf_item[node - tree.leaf_base]
            node = implicit_tree_next_subtree(node)
    result = query
    result.ancestor = ancestor
    result.stop = stop
    result.node = node
    result.item = item
    return result


@wp.func
def implicit_tree_query_self(tree: ImplicitAABBTreeData, slot: wp.int32) -> ImplicitTreeSelfQuery:
    """Start a self-overlap query for leaf ``slot`` (item ``tree.leaf_item[slot]``).

    Launch one thread per slot in ``[0, leaf_count)``; together the threads
    report every overlapping leaf pair exactly once. Only the sibling subtrees
    along the leaf-to-root path are visited since ancestors always contain the
    query leaf.
    """
    query = ImplicitTreeSelfQuery()
    leaf_node = tree.leaf_base + slot
    record = tree.nodes[leaf_node]
    query.lower = record.lower
    query.upper = record.upper
    query.span = record.max_span
    query.slot = slot
    query.ancestor = leaf_node
    query.node = leaf_node ^ wp.int32(1)
    query.stop = implicit_tree_next_subtree(query.node)
    query.item = wp.int32(-1)
    return implicit_tree_query_self_next(tree, query)


class ImplicitAABBTree:
    """Graph-capturable implicit AABB tree, an alternative to :class:`warp.Bvh`.

    The tree owns all its buffers; :meth:`build` and :meth:`refit` only launch
    kernels and may therefore be recorded in a CUDA graph. Item ``i`` is the
    box ``[lowers[i], uppers[i]]``.

    Args:
        max_leaf_count: Capacity, i.e. the largest item count passed to :meth:`build`.
        device: Device to allocate on.
    """

    def __init__(self, max_leaf_count: int, device: wp.DeviceLike = None) -> None:
        if max_leaf_count < 1:
            raise ValueError("max_leaf_count must be positive")
        self.device = wp.get_device(device)
        self.max_leaf_count = int(max_leaf_count)
        self.leaf_base = 1 << max(1, (self.max_leaf_count - 1).bit_length())
        """Number of leaf slots (power of two); leaf ``slot`` is node ``leaf_base + slot``."""
        self.depth = self.leaf_base.bit_length() - 1
        self.leaf_count = 0
        """Item count of the last :meth:`build`."""
        self.nodes = wp.empty(2 * self.leaf_base, dtype=ImplicitTreeNode, device=self.device)
        self.leaf_item = wp.empty(self.leaf_base, dtype=wp.int32, device=self.device)
        # Radix sort scratch needs twice the sorted length.
        self.keys = wp.empty(2 * self.max_leaf_count, dtype=wp.int32, device=self.device)
        self.order = wp.empty(2 * self.max_leaf_count, dtype=wp.int32, device=self.device)
        self.bounds = wp.empty(2, dtype=wp.vec3, device=self.device)
        self.data = ImplicitAABBTreeData()
        """Kernel-side view to pass to kernels using the ``implicit_tree_query_*`` functions."""
        self.data.nodes = self.nodes
        self.data.leaf_item = self.leaf_item
        self.data.leaf_base = self.leaf_base
        self.data.leaf_count = 0

    @property
    def memory_bytes(self) -> int:
        """Bytes of device memory held by the tree, including build scratch."""
        return sum(a.capacity for a in (self.nodes, self.leaf_item, self.keys, self.order, self.bounds))

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
            count: Number of items, at most ``max_leaf_count``; defaults to ``len(lowers)``.
            sort: Order leaves along a Morton curve of the box centers. Disable it
                when the input is already spatially coherent (e.g. mesh triangles).
        """
        count = len(lowers) if count is None else int(count)
        if count > self.max_leaf_count:
            raise ValueError(f"count {count} exceeds max_leaf_count {self.max_leaf_count}")
        self.leaf_count = count
        self.data.leaf_count = count
        if count == 0:
            wp.launch(
                _write_leaves,
                dim=self.leaf_base,
                inputs=[lowers, uppers, self.order, 0, self.leaf_base, self.nodes, self.leaf_item],
                device=self.device,
                record_tape=False,
            )
            self.update_internal_nodes()
            return
        if sort:
            wp.launch(_reset_bounds, dim=1, inputs=[self.bounds], device=self.device, record_tape=False)
            wp.launch(
                _center_bounds,
                dim=min(count, _REDUCE_THREADS),
                inputs=[lowers, uppers, count, self.bounds],
                device=self.device,
                record_tape=False,
            )
            wp.launch(
                _morton_keys,
                dim=count,
                inputs=[lowers, uppers, self.bounds, self.keys, self.order],
                device=self.device,
                record_tape=False,
            )
            wp.utils.radix_sort_pairs(self.keys, self.order, count)
        else:
            wp.launch(_identity_order, dim=count, inputs=[self.order], device=self.device, record_tape=False)
        self.refit(lowers, uppers)

    def refit(self, lowers: wp.array[wp.vec3], uppers: wp.array[wp.vec3]) -> None:
        """Recompute all node bounds from moved boxes, keeping the leaf order of the last build."""
        wp.launch(
            _write_leaves,
            dim=self.leaf_base,
            inputs=[lowers, uppers, self.order, self.leaf_count, self.leaf_base, self.nodes, self.leaf_item],
            device=self.device,
            record_tape=False,
        )
        self.update_internal_nodes()

    def update_internal_nodes(self) -> None:
        """Recompute internal nodes bottom-up after the leaf records were written."""
        for level in range(self.depth - 1, -1, -1):
            start = 1 << level
            wp.launch(_build_level, dim=start, inputs=[self.nodes, start], device=self.device, record_tape=False)
