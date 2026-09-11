# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check scalar symbolic initialization and packed response arithmetic."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.factorize.scalar_pattern import _ScalarFactorPattern
from newton._src.solvers.kamino._src.solvers.dvi.kernels import (
    _solve_bilateral_unilateral_response_symbolic,
    make_solve_bilateral_unilateral_response_compact_kernel,
)


def _left_looking_fill(n, edges, inverse):
    structural = np.eye(n, dtype=bool)
    for a, b in edges:
        row, col = sorted((int(inverse[a]), int(inverse[b])), reverse=True)
        structural[row, col] = True
    lower = np.eye(n, dtype=bool)
    for row in range(n):
        for col in range(row):
            lower[row, col] = structural[row, col] or np.any(lower[row, :col] & lower[col, :col])
    return lower, int(lower.sum() - structural.sum())


class TestScalarPattern(unittest.TestCase):
    def test_conditional_initialization_rebuild(self):
        """Rebuild changing topology and permutations inside an actual CUDA conditional graph."""
        device = wp.get_device()
        if not device.is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("Conditional symbolic initialization requires CUDA conditional graphs")
        ns = np.array([0, 1, 3, 17, 33, 65], dtype=np.int32)
        vio = np.cumsum(np.r_[0, ns[:-1]]).astype(np.int32)
        owner = _ScalarFactorPattern(ns, device)
        rng = np.random.default_rng(900813)

        def ints(values):
            return wp.array(np.asarray(values, dtype=np.int32), dtype=wp.int32, device=device)

        records = [(w, r, c) for w, n in enumerate(ns) for r in range(n) for c in range(r)]
        pairs = [ints(column) for column in zip(*records, strict=True)]
        inverse, offsets = ints(np.concatenate([np.arange(n) for n in ns])), ints(vio)
        enabled = ints([1])

        def initialize():
            owner.initialize(*pairs, inverse, offsets)

        initialize()
        with wp.ScopedCapture(device=device) as capture:
            wp.capture_if(enabled, on_true=initialize)
        for kind in ("dense", "star", "path", "empty", "dense"):
            new_records, expected_starts, expected_columns, orders = [], [], [], []
            fill_entries = 0
            for world, n in enumerate(ns):
                order = rng.permutation(n)
                if kind == "star" and n:
                    position = int(np.flatnonzero(order == 0)[0])
                    order[0], order[position] = order[position], order[0]
                inv = np.argsort(order)
                orders.extend(inv)
                if kind == "dense":
                    edges = [(r, c) for r in range(n) for c in range(r)]
                elif kind == "star":
                    edges = [(r, 0) for r in range(1, n)]
                elif kind == "path":
                    edges = [(r, r - 1) for r in range(1, n)]
                else:
                    edges = []
                padded = edges + [(0, 0)] * (int(n * (n - 1) // 2) - len(edges))
                new_records.extend((world, r, c) for r, c in padded)
                lower, fill = _left_looking_fill(int(n), edges, inv)
                fill_entries += fill
                for row in range(n):
                    expected_starts.append(len(expected_columns))
                    expected_columns.extend(np.flatnonzero(lower[row]))
                expected_starts.append(len(expected_columns))
            for array, values in zip(pairs, zip(*new_records, strict=True), strict=True):
                array.assign(np.asarray(values, dtype=np.int32))
            inverse.assign(np.asarray(orders, dtype=np.int32))
            before_starts, before_columns = owner.starts.numpy(), owner.columns.numpy()
            enabled.fill_(0)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(owner.starts.numpy(), before_starts)
            np.testing.assert_array_equal(owner.columns.numpy(), before_columns)
            enabled.fill_(1)
            for _ in range(2):
                wp.capture_launch(capture.graph)
                with self.subTest(topology=kind):
                    np.testing.assert_array_equal(owner.starts.numpy(), expected_starts)
                    np.testing.assert_array_equal(owner.columns.numpy()[: len(expected_columns)], expected_columns)
            if kind == "star":
                self.assertGreater(fill_entries, 0)

    def test_response_capacity_and_nonfinite_retry(self):
        """Preserve compact response accuracy and nonfinite propagation without writing skipped worlds."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Captured symbolic response requires CUDA")
        rng = np.random.default_rng(92911)
        ns = np.array([0, 1, 17, 33], dtype=np.int32)
        tiles = (ns + 31) // 32
        slots = np.cumsum(np.r_[0, (tiles * (tiles + 1) // 2)[:-1]]).astype(np.int64)
        vio = np.cumsum(np.r_[0, ns[:-1]]).astype(np.int32)
        capacity, guard = 33, 11
        sizes = ns * capacity
        rio = (guard + np.cumsum(np.r_[0, (sizes + guard)[:-1]])).astype(np.int32)
        total = int(np.sum(sizes + guard) + guard)
        owner = _ScalarFactorPattern(ns, device)
        orders, inverses, records, lowers, panels = [], [], [], [], []
        for world, (n, count) in enumerate(zip(ns, tiles, strict=True)):
            order = rng.permutation(n)
            orders.append(order)
            inverses.extend(np.argsort(order))
            pattern = np.tril(np.arange(n)[:, None] % 3 == np.arange(n)[None, :] % 3)
            lower = rng.normal(0, 0.025, (n, n)).astype(np.float32)
            lower[~pattern] = 0.0
            np.fill_diagonal(lower, 1.0)
            # Retain currently cancelled entries in the structural superset.
            for row in range(n):
                for col in range(row):
                    if pattern[row, col]:
                        records.append((world, order[row], order[col]))
                        if (row + col) % 2 == 0:
                            lower[row, col] = 0.0
            lowers.append(lower.astype(np.float64))
            padded = np.zeros((count * 32, count * 32), dtype=np.float32)
            padded[:n, :n] = lower
            panels.extend(
                padded[i * 32 : (i + 1) * 32, j * 32 : (j + 1) * 32].ravel() for i in range(count) for j in range(i + 1)
            )

        def array(values, dtype=wp.int32):
            return wp.array(np.asarray(values), dtype=dtype, device=device)

        njc, dim, offsets, vector_offsets, stride = [array(x) for x in (ns, ns, rio, vio, np.full(len(ns), capacity))]
        factor_offsets = array(slots, wp.int64)
        factor = array(np.asarray(panels).ravel(), wp.float32)
        permutation, inverse = array(np.concatenate(orders)), array(inverses)
        owner.initialize(*[array(x) for x in zip(*records, strict=True)], inverse, vector_offsets)
        scale = wp.ones(int(ns.sum()), dtype=wp.float32, device=device)
        prefix = wp.zeros_like(permutation)
        coupling = wp.zeros(total, dtype=wp.float32, device=device)
        outputs = [wp.zeros_like(coupling) for _ in range(2)]

        def launch():
            for index, kernel in enumerate(
                (
                    make_solve_bilateral_unilateral_response_compact_kernel(True),
                    _solve_bilateral_unilateral_response_symbolic,
                )
            ):
                inputs = [
                    dim,
                    njc,
                    factor_offsets,
                    vector_offsets,
                    scale,
                    factor,
                    permutation,
                    offsets,
                    stride,
                    coupling,
                    outputs[index],
                    prefix,
                ]
                wp.launch(
                    kernel,
                    dim=(len(ns), capacity),
                    block_dim=128,
                    inputs=inputs + (owner.inputs if index else []),
                    device=device,
                )

        launch()
        with wp.ScopedCapture(device=device) as capture:
            launch()
        sentinel = np.float32(-381.25)
        for mode in ("finite", "signed_zero", "infinity", "nan"):
            for nu in (0, 1, 5, 6, 17, 24, 33, 1):
                counts = np.full(len(ns), nu, dtype=np.int32)
                counts[0] = 0
                dim.assign(ns + counts)
                rhs = np.full(total, sentinel, dtype=np.float32)
                local_rhs = []
                for world, (n, count) in enumerate(zip(ns, counts, strict=True)):
                    values = rng.normal(0, 0.2, (n, count)).astype(np.float32)
                    if mode == "signed_zero":
                        values.fill(-0.0)
                    elif mode in ("infinity", "nan") and n and count:
                        values[orders[world][0], 0] = np.inf if mode == "infinity" else np.nan
                    rhs[rio[world] : rio[world] + n * capacity].reshape(n, capacity)[:, :count] = values
                    local_rhs.append(values[orders[world]].astype(np.float64))
                coupling.assign(rhs)
                for output in outputs:
                    output.fill_(sentinel)
                wp.capture_launch(capture.graph)
                baseline, actual = [output.numpy() for output in outputs]
                inactive = np.ones(total, dtype=bool)
                for world, (n, count) in enumerate(zip(ns, counts, strict=True)):
                    if count * count > n * capacity:
                        continue
                    segment = slice(rio[world], rio[world] + n * count)
                    inactive[segment] = False
                    with self.subTest(mode=mode, dimension=int(n), rhs=int(count)):
                        if mode != "signed_zero":
                            np.testing.assert_array_equal(
                                actual[segment].view(np.uint32), baseline[segment].view(np.uint32)
                            )
                        else:
                            np.testing.assert_array_equal(actual[segment], baseline[segment])
                        if mode in ("finite", "signed_zero") and n:
                            expected = np.linalg.solve(lowers[world], local_rhs[world])
                            np.testing.assert_allclose(
                                actual[segment].reshape(n, count), expected, atol=2e-6, rtol=2e-6
                            )
                np.testing.assert_array_equal(actual[inactive], sentinel)
                np.testing.assert_array_equal(baseline[inactive], sentinel)


if __name__ == "__main__":
    unittest.main()
