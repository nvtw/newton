# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU-parallel lock-free Union-Find (disjoint set) built on Warp.

Implements the algorithm from `wjakob/dset <https://github.com/wjakob/dset>`_
adapted for GPU execution.  Each element packs ``(rank, parent)`` into a single
``uint64`` entry and uses 64-bit atomic compare-and-swap for lock-free
:func:`uf_find` (with path compression) and :func:`uf_unite` (union-by-rank).

All kernel launches use a fixed ``dim`` equal to the pre-allocated capacity and
read the actual element count from a device-side ``int32`` array, making the
entire init / unite / find sequence graph-capturable via
:class:`wp.ScopedCapture`.
"""

from __future__ import annotations

import warp as wp

# ---------------------------------------------------------------------------
# Native bit-packing helpers
# ---------------------------------------------------------------------------
# Entry layout:  [63..32] = rank (31 bits + sign bit unused)
#                [31.. 0] = parent


@wp.func_native("""
return (int32_t)((uint32_t)(entry & 0xFFFFFFFFull));
""")
def uf_parent(entry: wp.uint64) -> int:
    """Extract the parent index (low 32 bits) from a packed entry."""
    ...


@wp.func_native("""
return (int32_t)((entry >> 32) & 0x7FFFFFFFull);
""")
def uf_rank(entry: wp.uint64) -> int:
    """Extract the rank (high 31 bits) from a packed entry."""
    ...


@wp.func_native("""
return (((uint64_t)(uint32_t)rank) << 32) | ((uint64_t)(uint32_t)parent);
""")
def uf_make_entry(rank: int, parent: int) -> wp.uint64:
    """Pack *rank* and *parent* into a single ``uint64`` entry."""
    ...


# ---------------------------------------------------------------------------
# Device functions
# ---------------------------------------------------------------------------


@wp.func
def uf_find(entries: wp.array(dtype=wp.uint64), id: int) -> int:
    """Lock-free find with path compression.

    Walks toward the root while attempting to shorten the path via atomic CAS.
    Safe under concurrent modifications -- failed CAS attempts are harmless.
    """
    cur = id
    while True:
        entry = entries[cur]
        p = uf_parent(entry)
        if p == cur:
            return cur

        gp_entry = entries[p]
        gp = uf_parent(gp_entry)
        if gp == -1:
            return -1

        r = uf_rank(entry)
        new_entry = uf_make_entry(r, gp)
        if entry != new_entry:
            wp.atomic_cas(entries, cur, entry, new_entry)
        cur = gp
    return -1


@wp.func
def uf_unite(entries: wp.array(dtype=wp.uint64), id1: int, id2: int):
    """Lock-free union-by-rank with atomic CAS.

    Repeatedly finds roots of *id1* and *id2*, then attempts to make the
    lower-rank root point to the higher-rank root.  On a rank tie the winner's
    rank is bumped via a second CAS (best-effort -- failure is safe).
    """
    a = id1
    b = id2
    while True:
        a = uf_find(entries, a)
        b = uf_find(entries, b)

        if a < 0 or b < 0:
            return
        if a == b:
            return

        ra = uf_rank(entries[a])
        rb = uf_rank(entries[b])

        if ra > rb or (ra == rb and a < b):
            tmp_id = a
            a = b
            b = tmp_id
            tmp_r = ra
            ra = rb
            rb = tmp_r

        old_entry = uf_make_entry(ra, a)
        new_entry = uf_make_entry(ra, b)

        if wp.atomic_cas(entries, a, old_entry, new_entry) != old_entry:
            continue

        if ra == rb:
            old_b = uf_make_entry(rb, b)
            new_b = uf_make_entry(rb + 1, b)
            wp.atomic_cas(entries, b, old_b, new_b)

        return


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@wp.kernel
def uf_init_kernel(
    entries: wp.array(dtype=wp.uint64),
    count: wp.array(dtype=wp.int32),
):
    """Initialise every element as its own singleton set."""
    tid = wp.tid()
    if tid < count[0]:
        entries[tid] = wp.uint64(tid)


@wp.kernel
def uf_unite_pairs_kernel(
    entries: wp.array(dtype=wp.uint64),
    pairs: wp.array(dtype=wp.vec2i),
    pair_count: wp.array(dtype=wp.int32),
):
    """Unite elements for each ``(i, j)`` pair."""
    tid = wp.tid()
    if tid < pair_count[0]:
        p = pairs[tid]
        uf_unite(entries, p[0], p[1])


@wp.kernel
def uf_find_roots_kernel(
    entries: wp.array(dtype=wp.uint64),
    roots: wp.array(dtype=wp.int32),
    count: wp.array(dtype=wp.int32),
):
    """Write the root representative of each element into *roots*."""
    tid = wp.tid()
    if tid < count[0]:
        roots[tid] = uf_find(entries, tid)


# ---------------------------------------------------------------------------
# Host-side driver
# ---------------------------------------------------------------------------


class UnionFind:
    """Host-side driver for the GPU Union-Find data structure.

    Allocates device arrays once at construction and exposes :meth:`init`,
    :meth:`unite_pairs`, and :meth:`find_roots` as thin wrappers around the
    corresponding Warp kernels.  All launches use a fixed ``dim`` equal to
    *capacity* so they are safe to record inside a CUDA graph.

    Args:
        capacity: maximum number of elements.
        device: Warp device string or object (``None`` for the default device).
    """

    def __init__(self, capacity: int, device: wp.context.Device | str | None = None):
        self.capacity = capacity
        self.device = wp.get_device(device)
        self.entries = wp.zeros(capacity, dtype=wp.uint64, device=self.device)
        self.roots = wp.zeros(capacity, dtype=wp.int32, device=self.device)

    def init(self, count: wp.array(dtype=wp.int32)):
        """Reset elements ``[0, count)`` to singleton sets.

        Args:
            count: device array of shape ``(1,)`` holding the active element
                count.  Read on-device so no host sync is needed.
        """
        wp.launch(
            uf_init_kernel,
            dim=self.capacity,
            inputs=[self.entries, count],
            device=self.device,
        )

    def unite_pairs(self, pairs: wp.array(dtype=wp.vec2i), pair_count: wp.array(dtype=wp.int32)):
        """Unite elements for each ``(i, j)`` pair in *pairs*.

        Args:
            pairs: device array of ``vec2i`` with shape ``(max_pairs,)``.
            pair_count: device array of shape ``(1,)`` holding the number of
                active pairs.
        """
        wp.launch(
            uf_unite_pairs_kernel,
            dim=pairs.shape[0],
            inputs=[self.entries, pairs, pair_count],
            device=self.device,
        )

    def find_roots(self, count: wp.array(dtype=wp.int32)):
        """Write root of each element into :attr:`roots`.

        Args:
            count: device array of shape ``(1,)`` holding the active element
                count.
        """
        wp.launch(
            uf_find_roots_kernel,
            dim=self.capacity,
            inputs=[self.entries, self.roots, count],
            device=self.device,
        )
