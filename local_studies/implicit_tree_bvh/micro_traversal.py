import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
import numpy as np
import warp as wp
from bench_bvh import (
    k_aabb_implicit,
    k_aabb_implicit_inline,
    k_ray_all_bvh,
    k_ray_all_implicit,
    scene_boxes,
    time_graph,
)

from newton._src.geometry.implicit_aabb_tree import (
    ImplicitAABBTree,
    ImplicitAABBTreeData,
    ImplicitTreeAabbQuery,
    _aabb_disjoint,
    _ray_box_entry,
    implicit_tree_next_subtree,
    implicit_tree_query_aabb_next,
    implicit_tree_query_ray,
    implicit_tree_query_ray_next,
)

wp.config.quiet = True


@wp.kernel
def k_aabb_inline_nohoist(
    tree: ImplicitAABBTreeData, qlo: wp.array[wp.vec3], qhi: wp.array[wp.vec3], counts: wp.array[wp.int32]
):
    i = wp.tid()
    lo = qlo[i]
    hi = qhi[i]
    n = int(0)
    node = wp.int32(1)
    while node != 0:
        r = tree.nodes[node]
        if _aabb_disjoint(r.lower, r.upper, lo, hi):
            node = implicit_tree_next_subtree(node)
        elif node < tree.leaf_base:
            node = node * 2
        else:
            n += 1
            node = implicit_tree_next_subtree(node)
    counts[i] = n


@wp.kernel
def k_aabb_inline_arrays(
    nodes: wp.array[wp.vec4], leaf_base: int, qlo: wp.array[wp.vec3], qhi: wp.array[wp.vec3], counts: wp.array[wp.int32]
):
    i = wp.tid()
    lo = qlo[i]
    hi = qhi[i]
    n = int(0)
    node = wp.int32(1)
    while node != 0:
        a = nodes[2 * node]
        b = nodes[2 * node + 1]
        rl = wp.vec3(a[0], a[1], a[2])
        ru = wp.vec3(b[0], b[1], b[2])
        if _aabb_disjoint(rl, ru, lo, hi):
            node = implicit_tree_next_subtree(node)
        elif node < leaf_base:
            node = node * 2
        else:
            n += 1
            node = implicit_tree_next_subtree(node)
    counts[i] = n


@wp.kernel
def k_ray_inline(tree: ImplicitAABBTreeData, ro: wp.array[wp.vec3], rd: wp.array[wp.vec3], counts: wp.array[wp.int32]):
    i = wp.tid()
    o = ro[i]
    d = rd[i]
    inv = wp.vec3(1.0 / d[0], 1.0 / d[1], 1.0 / d[2])
    nodes = tree.nodes
    leaf_base = tree.leaf_base
    n = int(0)
    node = wp.int32(1)
    while node != 0:
        r = nodes[node]
        t = float(-1.0)
        if r.last_leaf >= 0:
            t = _ray_box_entry(o, inv, r.lower, r.upper, 1.0e30)
        if t < 0.0:
            node = implicit_tree_next_subtree(node)
        elif node < leaf_base:
            node = node * 2
        else:
            n += 1
            node = implicit_tree_next_subtree(node)
    counts[i] = n


@wp.kernel
def k_ray_inline_nodes_visited(
    tree: ImplicitAABBTreeData, ro: wp.array[wp.vec3], rd: wp.array[wp.vec3], counts: wp.array[wp.int32]
):
    i = wp.tid()
    o = ro[i]
    d = rd[i]
    inv = wp.vec3(1.0 / d[0], 1.0 / d[1], 1.0 / d[2])
    nodes = tree.nodes
    leaf_base = tree.leaf_base
    n = int(0)
    node = wp.int32(1)
    while node != 0:
        r = nodes[node]
        n += 1
        t = float(-1.0)
        if r.last_leaf >= 0:
            t = _ray_box_entry(o, inv, r.lower, r.upper, 1.0e30)
        if t < 0.0:
            node = implicit_tree_next_subtree(node)
        elif node < leaf_base:
            node = node * 2
        else:
            node = implicit_tree_next_subtree(node)
    counts[i] = n


@wp.kernel
def k_aabb_single_site(
    tree: ImplicitAABBTreeData, qlo: wp.array[wp.vec3], qhi: wp.array[wp.vec3], counts: wp.array[wp.int32]
):
    i = wp.tid()
    n = int(0)
    q = ImplicitTreeAabbQuery()
    q.lower = qlo[i]
    q.upper = qhi[i]
    q.node = 1
    q.item = -1
    while q.node != 0:
        q = implicit_tree_query_aabb_next(tree, q)
        if q.item >= 0:
            n += 1
    counts[i] = n


@wp.kernel
def k_aabb_while_true(
    tree: ImplicitAABBTreeData, qlo: wp.array[wp.vec3], qhi: wp.array[wp.vec3], counts: wp.array[wp.int32]
):
    i = wp.tid()
    n = int(0)
    q = ImplicitTreeAabbQuery()
    q.lower = qlo[i]
    q.upper = qhi[i]
    q.node = 1
    q.item = -1
    while True:
        q = implicit_tree_query_aabb_next(tree, q)
        if q.item < 0:
            break
        n += 1
    counts[i] = n


@wp.func
def next_break(tree: ImplicitAABBTreeData, query: ImplicitTreeAabbQuery) -> ImplicitTreeAabbQuery:
    node = query.node
    item = wp.int32(-1)
    while node != wp.int32(0):
        record = tree.nodes[node]
        if _aabb_disjoint(record.lower, record.upper, query.lower, query.upper):
            node = implicit_tree_next_subtree(node)
        elif node < tree.leaf_base:
            node = node * wp.int32(2)
        else:
            item = tree.leaf_item[node - tree.leaf_base]
            node = implicit_tree_next_subtree(node)
            break
    result = query
    result.node = node
    result.item = item
    return result


@wp.func
def next_tuple(tree: ImplicitAABBTreeData, lower: wp.vec3, upper: wp.vec3, node: wp.int32):
    item = wp.int32(-1)
    while node != wp.int32(0):
        record = tree.nodes[node]
        if _aabb_disjoint(record.lower, record.upper, lower, upper):
            node = implicit_tree_next_subtree(node)
        elif node < tree.leaf_base:
            node = node * wp.int32(2)
        else:
            item = tree.leaf_item[node - tree.leaf_base]
            node = implicit_tree_next_subtree(node)
            break
    return node, item


@wp.kernel
def k_aabb_break(
    tree: ImplicitAABBTreeData, qlo: wp.array[wp.vec3], qhi: wp.array[wp.vec3], counts: wp.array[wp.int32]
):
    i = wp.tid()
    n = int(0)
    q = ImplicitTreeAabbQuery()
    q.lower = qlo[i]
    q.upper = qhi[i]
    q.node = 1
    q.item = -1
    q = next_break(tree, q)
    while q.item >= 0:
        n += 1
        q = next_break(tree, q)
    counts[i] = n


@wp.kernel
def k_aabb_tuple(
    tree: ImplicitAABBTreeData, qlo: wp.array[wp.vec3], qhi: wp.array[wp.vec3], counts: wp.array[wp.int32]
):
    i = wp.tid()
    n = int(0)
    lo = qlo[i]
    hi = qhi[i]
    node, item = next_tuple(tree, lo, hi, 1)
    while item >= 0:
        n += 1
        node, item = next_tuple(tree, lo, hi, node)
    counts[i] = n


@wp.kernel
def k_ray_len_impl(
    tree: ImplicitAABBTreeData, ro: wp.array[wp.vec3], rd: wp.array[wp.vec3], max_t: float, counts: wp.array[wp.int32]
):
    i = wp.tid()
    n = int(0)
    q = implicit_tree_query_ray(tree, ro[i], rd[i], max_t)
    while q.item >= 0:
        n += 1
        q = implicit_tree_query_ray_next(tree, q)
    counts[i] = n


@wp.kernel
def k_ray_len_bvh(
    bvh: wp.uint64, ro: wp.array[wp.vec3], rd: wp.array[wp.vec3], max_t: float, counts: wp.array[wp.int32]
):
    i = wp.tid()
    n = int(0)
    item = int(0)
    q = wp.bvh_query_ray(bvh, ro[i], rd[i])
    while wp.bvh_query_next(q, item, max_t):
        n += 1
    counts[i] = n


@wp.kernel
def k_ray_visits(
    tree: ImplicitAABBTreeData, ro: wp.array[wp.vec3], rd: wp.array[wp.vec3], max_t: float, counts: wp.array[wp.int32]
):
    i = wp.tid()
    o = ro[i]
    d = rd[i]
    inv = wp.vec3(1.0 / d[0], 1.0 / d[1], 1.0 / d[2])
    n = int(0)
    node = wp.int32(1)
    while node != 0:
        r = tree.nodes[node]
        n += 1
        t = float(-1.0)
        if r.last_leaf >= 0:
            t = _ray_box_entry(o, inv, r.lower, r.upper, max_t)
        if t < 0.0:
            node = implicit_tree_next_subtree(node)
        elif node < tree.leaf_base:
            node = node * 2
        else:
            node = implicit_tree_next_subtree(node)
    counts[i] = n


@wp.kernel
def k_aabb_visits(
    tree: ImplicitAABBTreeData, qlo: wp.array[wp.vec3], qhi: wp.array[wp.vec3], counts: wp.array[wp.int32]
):
    i = wp.tid()
    lo = qlo[i]
    hi = qhi[i]
    n = int(0)
    node = wp.int32(1)
    while node != 0:
        r = tree.nodes[node]
        n += 1
        if _aabb_disjoint(r.lower, r.upper, lo, hi):
            node = implicit_tree_next_subtree(node)
        elif node < tree.leaf_base:
            node = node * 2
        else:
            node = implicit_tree_next_subtree(node)
    counts[i] = n


def main(n):
    rng = np.random.default_rng(1)
    lo_np, hi_np, side = scene_boxes(n, rng)
    lo = wp.array(lo_np, dtype=wp.vec3)
    hi = wp.array(hi_np, dtype=wp.vec3)
    tree = ImplicitAABBTree(n)
    tree.build(lo, hi)
    bvh = wp.Bvh(lo, hi, constructor="lbvh")
    q = 1 << 18
    c = rng.uniform(0, side, (q, 3)).astype(np.float32)
    qlo = wp.array(c - 1.0, dtype=wp.vec3)
    qhi = wp.array(c + 1.0, dtype=wp.vec3)
    d = rng.standard_normal((q, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ro = wp.array(c, dtype=wp.vec3)
    rd = wp.array(d, dtype=wp.vec3)
    counts = wp.zeros(q, dtype=wp.int32)
    nodes4 = wp.array(tree.nodes.numpy().view(np.float32).reshape(-1, 4), dtype=wp.vec4)

    def run(name, k, inputs):
        def fn():
            wp.launch(k, dim=q, inputs=[*inputs, counts])

        fn()
        ms, _ = time_graph(fn)
        print(f"n={n} {name:28s} {ms:.4f} ms  sum={int(counts.numpy().sum())}")

    run("aabb visits", k_aabb_visits, [tree.data, qlo, qhi])
    for L in (0.5, 8.0):
        run(f"ray visits L={L}", k_ray_visits, [tree.data, ro, rd, L])
    for L in (0.5, 2.0, 8.0, 1e30):
        run(f"ray impl L={L}", k_ray_len_impl, [tree.data, ro, rd, L])
        run(f"ray bvh  L={L}", k_ray_len_bvh, [bvh.id, ro, rd, L])
    return
    run("aabb iter", k_aabb_implicit, [tree.data, qlo, qhi])
    run("aabb single site", k_aabb_single_site, [tree.data, qlo, qhi])
    run("aabb while true", k_aabb_while_true, [tree.data, qlo, qhi])
    run("aabb break", k_aabb_break, [tree.data, qlo, qhi])
    run("aabb tuple", k_aabb_tuple, [tree.data, qlo, qhi])
    run("aabb inline", k_aabb_implicit_inline, [tree.data, qlo, qhi])
    run("aabb inline nohoist", k_aabb_inline_nohoist, [tree.data, qlo, qhi])
    run("aabb inline vec4", k_aabb_inline_arrays, [nodes4, tree.leaf_base, qlo, qhi])
    return
    run("ray iter", k_ray_all_implicit, [tree.data, ro, rd])
    run("ray inline", k_ray_inline, [tree.data, ro, rd])
    run("ray inline nodes visited", k_ray_inline_nodes_visited, [tree.data, ro, rd])
    run("ray bvh", k_ray_all_bvh, [bvh.id, ro, rd])


for n in [int(a) for a in sys.argv[1:]]:
    main(n)
