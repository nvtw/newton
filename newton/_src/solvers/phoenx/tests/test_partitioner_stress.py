# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Stress / regression tests for :class:`IncrementalContactPartitioner`.

This is the partitioner :class:`PhoenXWorld` calls every step from
``solver_phoenx.py:2056`` to color the constraint graph. The
``example_soft_body_drop`` scene was observed to hang for many minutes
in :meth:`build_csr_greedy_with_jp_fallback` (specifically inside
``wp.capture_while(self._num_remaining, self._capture_speculative_step)``
when both ``capture_while_greedy_coloring=True`` and
``speculative_coloring=True``). Each test below targets a graph
topology the production scenes hit (dense soft-body voxel grid, dense
cloth-triangle mesh, joint chain, star, isolated singletons) and
asserts the partitioner returns within a bounded wall-clock so future
regressions to the MIS / speculative-coloring kernels surface as a
clean test failure rather than a silent hang.

Each test runs both ``build_csr`` (plain JP) and
``build_csr_greedy_with_jp_fallback`` (the production path) with every
combination of the two perf flags so we cover the matrix exhaustively.
The TIMEOUT_SECONDS budget is generous (45 s) — even slow paths should
finish in seconds; anything longer is a hang.
"""

from __future__ import annotations

import itertools
import signal
import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.graph_coloring.graph_coloring_common import (
    ElementInteractionData,
)
from newton._src.solvers.phoenx.graph_coloring.graph_coloring_incremental import (
    IncrementalContactPartitioner,
)

# Wall-clock budget for one build_csr call. Production scenes finish in
# < 200 ms on this host; 45 s is a safety net for slow CI nodes and
# trip-wires the hang we hit in soft_body_drop.
TIMEOUT_SECONDS: int = 45


class _BuildTimeout(Exception):
    """Raised when ``build_csr*`` exceeds :data:`TIMEOUT_SECONDS`."""


def _build_with_timeout(partitioner: IncrementalContactPartitioner, method: str) -> None:
    """Call ``partitioner.<method>()`` under a SIGALRM watchdog.

    ``signal.SIGALRM`` is delivered to the main thread so unittest sees
    the failure as an exception instead of the worker hanging until
    pytest / CTest timeout (which can take minutes on a multi-test
    run). Restores the previous handler before returning.
    """

    def _handler(signum, frame):
        raise _BuildTimeout(
            f"partitioner.{method}() exceeded {TIMEOUT_SECONDS}s wall budget — likely hung in capture_while"
        )

    prev = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.alarm(TIMEOUT_SECONDS)
        getattr(partitioner, method)()
        wp.synchronize()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


_ELEMENT_DTYPE = np.dtype({"names": ["bodies"], "formats": ["8i4"], "offsets": [0], "itemsize": 32})


def _elements_from_python(body_lists: list[list[int]], device) -> tuple[wp.array, wp.array]:
    """Pack ``body_lists`` into the (elements, num_elements) Warp arrays
    the partitioner consumes.

    Each list is the body indices an element touches (1..8 ids). Inactive
    slots are -1; the partitioner reads slots until it hits one. The
    on-device dtype matches :class:`ElementInteractionData` (a packed
    ``vec8i`` of body slots).
    """
    n = len(body_lists)
    struct_arr = np.zeros(n, dtype=_ELEMENT_DTYPE)
    bodies_view = struct_arr["bodies"]
    bodies_view[:] = -1
    for i, bodies in enumerate(body_lists):
        if len(bodies) > 8:
            raise ValueError(f"element {i} has {len(bodies)} bodies; max is 8")
        bodies_view[i, : len(bodies)] = bodies
    elements = wp.array(struct_arr, dtype=ElementInteractionData, device=device)
    num_elements = wp.array([n], dtype=wp.int32, device=device)
    return elements, num_elements


def _soft_body_voxel_graph(
    *, grid_x: int, grid_y: int, num_cubes: int, cube_resolution: int
) -> tuple[list[list[int]], int]:
    """Synthesise the element graph for a soft-body-drop-style scene.

    Each voxel decomposes into 5 tets sharing 8 corner particles
    (``newton.ModelBuilder.add_soft_grid`` convention). Particles are
    shared across neighbouring voxels (so the graph is dense even at
    small ``cube_resolution``). Returns ``(elements, num_nodes)``.
    """
    elements: list[list[int]] = []
    voxel_id_to_corners: dict[tuple[int, int, int], list[int]] = {}
    next_node = 0

    def corner(ix: int, iy: int, iz: int) -> int:
        nonlocal next_node
        key = (ix, iy, iz)
        if key not in voxel_id_to_corners:
            voxel_id_to_corners[key] = [next_node]
            next_node += 1
        return voxel_id_to_corners[key][0]

    # 5-tet split per voxel — corners labelled the same way as
    # ``MeshBuilder.GenerateTetrahedronBlock`` (Jitter2). Topology is
    # what the partitioner sees; exact tet indices don't matter beyond
    # touching the right corner ids.
    tet_local_corners = [
        (0, 1, 3, 4),
        (1, 2, 3, 6),
        (4, 5, 1, 6),
        (4, 7, 3, 6),
        (1, 3, 4, 6),
    ]
    for px in range(grid_x):
        for py in range(grid_y):
            for cube in range(num_cubes):
                for vx in range(cube_resolution):
                    for vy in range(cube_resolution):
                        for vz in range(cube_resolution):
                            cx = px * (cube_resolution + 1) + vx
                            cy = py * (cube_resolution + 1) + vy
                            cz = cube * (cube_resolution + 1) + vz
                            voxel_corners = [
                                corner(cx, cy, cz),
                                corner(cx + 1, cy, cz),
                                corner(cx + 1, cy + 1, cz),
                                corner(cx, cy + 1, cz),
                                corner(cx, cy, cz + 1),
                                corner(cx + 1, cy, cz + 1),
                                corner(cx + 1, cy + 1, cz + 1),
                                corner(cx, cy + 1, cz + 1),
                            ]
                            for local in tet_local_corners:
                                elements.append([voxel_corners[i] for i in local])
    return elements, next_node


def _cloth_triangle_graph(*, rows: int, cols: int) -> tuple[list[list[int]], int]:
    """Synthesise the element graph for a regular cloth-grid mesh.

    A grid of ``rows`` x ``cols`` vertices is meshed into two
    triangles per quad, producing ``2 * (rows - 1) * (cols - 1)`` cloth
    triangles. Adjacent triangles share two vertices, so the resulting
    element graph is dense (each interior vertex touches 6 triangles).
    """

    def vid(r: int, c: int) -> int:
        return r * cols + c

    elements: list[list[int]] = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            elements.append([vid(r, c), vid(r + 1, c), vid(r, c + 1)])
            elements.append([vid(r + 1, c), vid(r + 1, c + 1), vid(r, c + 1)])
    return elements, rows * cols


def _validate_coloring(partitioner: IncrementalContactPartitioner, num_elements: int) -> None:
    """Cheap O(N * deg) host-side coloring legality check.

    Two elements that share any body must not share a colour, except
    in the overflow bucket (``colour == max_colored_partitions``) when
    mass splitting is configured. ``IncrementalContactPartitioner``'s
    own internals don't expose a single "is the colouring valid" hook,
    so we just walk the elements and assert no neighbour collision.
    """
    elements_np = partitioner._elements.numpy()
    colors_np = partitioner._interaction_id_to_partition.numpy()
    overflow = partitioner._max_colored_partitions_kernel_arg

    body_to_color = {}
    for eid in range(num_elements):
        color = int(colors_np[eid])
        if overflow >= 0 and color == overflow:
            # overflow bucket: mass splitting resolves conflicts here
            continue
        bodies = [int(b) for b in elements_np[eid]["bodies"] if int(b) >= 0]
        for b in bodies:
            if (b, color) in body_to_color and body_to_color[(b, color)] != eid:
                raise AssertionError(
                    f"colour collision: element {eid} and element {body_to_color[(b, color)]} "
                    f"both share body {b} at colour {color}"
                )
            body_to_color[(b, color)] = eid


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class _PartitionerStressBase(unittest.TestCase):
    """Shared harness — sub-classes override :meth:`make_graph` to pick a
    topology, then every combination of the two performance flags
    (``capture_while`` x ``speculative``) is exercised."""

    device: str = "cuda:0"

    def make_graph(self) -> tuple[list[list[int]], int]:
        raise NotImplementedError

    def _run(
        self,
        *,
        capture_while: bool,
        speculative: bool,
        method: str,
        max_colored_partitions: int | None,
        enable_warm_start: bool,
    ) -> None:
        elements_py, num_nodes = self.make_graph()
        n_elements = len(elements_py)
        if n_elements == 0:
            self.skipTest("graph has zero elements")

        elements, num_elements = _elements_from_python(elements_py, self.device)
        partitioner = IncrementalContactPartitioner(
            max_num_interactions=n_elements,
            max_num_nodes=num_nodes,
            device=self.device,
            seed=0,
            max_colored_partitions=max_colored_partitions,
            enable_warm_start=enable_warm_start,
        )
        partitioner.set_capture_while_greedy(capture_while)
        partitioner.set_speculative_coloring(speculative)
        partitioner.reset(elements, num_elements)

        _build_with_timeout(partitioner, method)
        _validate_coloring(partitioner, n_elements)

    def _exercise_matrix(
        self, *, methods: tuple[str, ...] = ("build_csr", "build_csr_greedy_with_jp_fallback")
    ) -> None:
        # Production scene flag combos:
        #   max_colored_partitions=None  → JP-only path (rigid scenes)
        #   max_colored_partitions=0     → all-overflow (soft-body / dense Tonge mass splitting)
        #   max_colored_partitions=12    → bounded-colour Tonge (default for cloth / Kapla)
        # enable_warm_start: True is the production default; False guards the cold path.
        partition_caps = (None, 0, 12)
        for capture_while, speculative, method, cap, warm in itertools.product(
            [False, True], [False, True], methods, partition_caps, [False, True]
        ):
            with self.subTest(
                capture_while=capture_while,
                speculative=speculative,
                method=method,
                max_colored_partitions=cap,
                enable_warm_start=warm,
            ):
                self._run(
                    capture_while=capture_while,
                    speculative=speculative,
                    method=method,
                    max_colored_partitions=cap,
                    enable_warm_start=warm,
                )


class TestPartitionerSoftBodyVoxel(_PartitionerStressBase):
    """Reproduces the soft-body-drop topology: 4x4 grid of 2-cube piles
    at cube_resolution=1 = 160 dense-graph tets sharing ~256 particles."""

    def make_graph(self) -> tuple[list[list[int]], int]:
        return _soft_body_voxel_graph(grid_x=4, grid_y=4, num_cubes=2, cube_resolution=1)

    def test_build_csr_matrix(self):
        self._exercise_matrix()


class TestPartitionerSoftBodyDense(_PartitionerStressBase):
    """Larger soft-body voxel grid (1 280 tets) — exercises the MIS
    convergence on a denser graph than the default soft_body_drop
    scene."""

    def make_graph(self) -> tuple[list[list[int]], int]:
        return _soft_body_voxel_graph(grid_x=4, grid_y=4, num_cubes=4, cube_resolution=2)

    def test_build_csr_matrix(self):
        self._exercise_matrix()


class TestPartitionerClothGrid(_PartitionerStressBase):
    """Regular 32x32 cloth grid = 1 922 triangles. Each interior vertex
    touches 6 triangles → high-degree graph that the cloth-hanging
    example exercises."""

    def make_graph(self) -> tuple[list[list[int]], int]:
        return _cloth_triangle_graph(rows=32, cols=32)

    def test_build_csr_matrix(self):
        self._exercise_matrix()


class TestPartitionerSingletons(_PartitionerStressBase):
    """All-disjoint elements: every element touches its own private
    body, so the chromatic number is 1. Catches off-by-one in
    'no remaining' detection."""

    def make_graph(self) -> tuple[list[list[int]], int]:
        return [[i] for i in range(64)], 64

    def test_build_csr_matrix(self):
        self._exercise_matrix()


class TestPartitionerStar(_PartitionerStressBase):
    """Star topology: N elements all sharing one central body — forces
    N distinct colours (or the overflow bucket). Hits the worst case
    for greedy and speculative coloring with capture_while."""

    def make_graph(self) -> tuple[list[list[int]], int]:
        n = 32
        return [[0, 1 + i] for i in range(n)], n + 1

    def test_build_csr_matrix(self):
        self._exercise_matrix()


class TestPartitionerChain(_PartitionerStressBase):
    """Joint-chain topology: each element shares one body with the
    next. 2-colourable but the speculative-coloring path is sensitive
    to the priority ordering — covers a different MIS regime than
    the dense soft-body / cloth graphs."""

    def make_graph(self) -> tuple[list[list[int]], int]:
        n = 256
        return [[i, i + 1] for i in range(n)], n + 1

    def test_build_csr_matrix(self):
        self._exercise_matrix()


if __name__ == "__main__":
    unittest.main()
