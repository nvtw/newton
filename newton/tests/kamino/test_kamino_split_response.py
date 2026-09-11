# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for split compact response and cooperative fallback."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.dvi.kernels import (
    _solve_bilateral_unilateral_response_compact,
    _solve_bilateral_unilateral_response_cooperative,
)
from newton._src.solvers.kamino._src.solvers.dvi.sparse_kernels import (
    _assemble_compact_unilateral_schur_tiled,
)


def _assert_bits_equal(a, b):
    np.testing.assert_array_equal(a.copy().view(np.uint32), b.copy().view(np.uint32))


class TestSplitResponse(unittest.TestCase):
    def test_captured_capacity_boundaries(self):
        """Preserve response accuracy and fallback bits across changing compact capacity."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Captured split response requires CUDA")
        rng = np.random.default_rng(812)
        ns = np.array([84, 32, 0], dtype=np.int32)
        capacities = np.array([128, 128, 8], dtype=np.int32)
        guard = 11
        sizes = ns * capacities
        offsets = np.concatenate(([guard], guard + np.cumsum(sizes + guard)[:-1])).astype(np.int32)
        mio = np.concatenate(([0], np.cumsum(ns * ns)[:-1])).astype(np.int32)
        vio = np.concatenate(([0], np.cumsum(ns)[:-1])).astype(np.int32)
        qio = np.concatenate(([0], np.cumsum(ns + capacities)[:-1])).astype(np.int32)
        factors, scales, orders, prefixes, jacobians = [], [], [], [], []
        local_factors, local_scales, local_orders = [], [], []
        for n in ns:
            bodies = max(2, int(n) // 4 + 2)
            J = np.zeros((n, bodies * 6), dtype=np.float64)
            for row in range(n):
                body = row % bodies
                J[row, body * 6 : body * 6 + 6] = rng.normal(size=6)
                other = (body + 1) % bodies
                J[row, other * 6 : other * 6 + 6] = rng.normal(size=6)
            D = J @ J.T
            scale = np.sqrt(1.0 / (np.diag(D) + 1.0)).astype(np.float32)
            order = rng.permutation(n).astype(np.int32)
            A = scale[:, None] * D * scale[None, :] + np.eye(n)
            L = np.linalg.cholesky(A[np.ix_(order, order)]).astype(np.float32)
            prefix = [int(np.flatnonzero(L[row])[0]) // 16 * 16 for row in range(n)]
            factors.extend(L.ravel())
            scales.extend(scale)
            orders.extend(order)
            prefixes.extend(prefix)
            jacobians.append(J)
            local_factors.append(L)
            local_scales.append(scale)
            local_orders.append(order)

        def ints(x):
            return wp.array(np.asarray(x, dtype=np.int32), dtype=wp.int32, device=device)

        def floats(x):
            return wp.array(np.asarray(x, dtype=np.float32), dtype=wp.float32, device=device)

        dim = ints(ns)
        njc, bmio, bvio, rio, stride, qoffset = map(ints, (ns, mio, vio, offsets, capacities, qio))
        L, scale, order, prefix = floats(factors), floats(scales), ints(orders), ints(prefixes)
        total = int(np.sum(sizes + guard) + guard)
        coupling = wp.zeros(total, dtype=wp.float32, device=device)
        outputs = [wp.zeros_like(coupling) for _ in range(2)]
        scratch = [wp.zeros_like(coupling) for _ in range(2)]
        q = [wp.zeros(int(np.sum(ns + capacities)), dtype=wp.float32, device=device) for _ in range(2)]

        def launch(index):
            if index:
                wp.launch(
                    _solve_bilateral_unilateral_response_compact,
                    dim=(len(ns), 128),
                    block_dim=128,
                    inputs=[dim, njc, bmio, bvio, scale, L, order, rio, stride, coupling, outputs[index], prefix],
                    device=device,
                )
            wp.launch(
                _solve_bilateral_unilateral_response_cooperative,
                dim=len(ns) * 64 * 32,
                block_dim=256,
                inputs=[
                    dim,
                    njc,
                    bmio,
                    bvio,
                    scale,
                    L,
                    order,
                    True,
                    rio,
                    stride,
                    coupling,
                    scratch[index],
                    outputs[index],
                    0,
                    64,
                    True,
                    prefix,
                    bool(index),
                ],
                device=device,
            )
            wp.launch(
                _assemble_compact_unilateral_schur_tiled,
                dim=(len(ns), 16, 128),
                block_dim=128,
                inputs=[dim, njc, qoffset, rio, stride, outputs[index], scratch[index], q[index], 16],
                device=device,
            )

        graphs = []
        for index in range(2):
            launch(index)
            with wp.ScopedCapture(device=device) as capture:
                launch(index)
            graphs.append(capture.graph)
        for nus in ([103, 64, 0], [104, 65, 1], [0, 0, 0], [125, 64, 0], [103, 65, 0]):
            dim.assign(ns + np.asarray(nus, dtype=np.int32))
            rhs = np.full(total, -999.0, dtype=np.float32)
            references = []
            for world, (n, nu, capacity, off) in enumerate(zip(ns, nus, capacities, offsets, strict=True)):
                J = jacobians[world]
                W = np.zeros((J.shape[1], nu), dtype=np.float64)
                for col in range(max(0, nu - 1)):
                    body = int(rng.integers(J.shape[1] // 6))
                    W[body * 6 : body * 6 + 6, col] = rng.normal(0, 0.1, 6)
                C = (J @ W).astype(np.float32)
                view = rhs[off : off + n * capacity].reshape(n, capacity)
                view[:, :nu] = C
                b = (local_scales[world][:, None] * C)[local_orders[world]]
                white = np.linalg.solve(local_factors[world].astype(np.float64), b.astype(np.float64))
                references.append(white)
            coupling.assign(rhs)
            for index in range(2):
                outputs[index].fill_(-456.0)
                scratch[index].fill_(-123.0)
                q[index].fill_(-789.0)
                wp.capture_launch(graphs[index])
            a, b = [x.numpy() for x in outputs]
            sa, sb = [x.numpy() for x in scratch]
            expected_q = np.full(q[0].size, -789.0, dtype=np.float32)
            gap_mask = np.ones(total, dtype=bool)
            for world, (n, nu, capacity, off) in enumerate(zip(ns, nus, capacities, offsets, strict=True)):
                size = n * capacity
                gap_mask[off : off + size] = False
                compact = nu * nu <= size
                if compact:
                    white = references[world]
                    np.testing.assert_allclose(b[off : off + n * nu].reshape(n, nu), white, atol=2.0e-6, rtol=2.0e-6)
                    np.testing.assert_allclose(
                        sb[off : off + nu * nu].reshape(nu, nu), white.T @ white, atol=5.0e-6, rtol=5.0e-6
                    )
                    np.testing.assert_array_equal(b[off + n * nu : off + size], -456.0)
                    np.testing.assert_array_equal(sb[off + nu * nu : off + size], -123.0)
                    expected_q[qio[world] + n : qio[world] + n + nu] = 0.0
                    if nu:
                        np.testing.assert_array_equal(b[off : off + n * nu].reshape(n, nu)[:, -1], 0.0)
                        np.testing.assert_array_equal(sb[off : off + nu * nu].reshape(nu, nu)[-1], 0.0)
                else:
                    _assert_bits_equal(b[off : off + size], a[off : off + size])
                    _assert_bits_equal(sb[off : off + size], sa[off : off + size])
                    inactive = b[off : off + size].reshape(n, capacity)[:, nu:]
                    np.testing.assert_array_equal(inactive, -456.0)
            np.testing.assert_array_equal(b[gap_mask], -456.0)
            np.testing.assert_array_equal(sb[gap_mask], -123.0)
            np.testing.assert_array_equal(q[1].numpy(), expected_q)
            np.testing.assert_array_equal(q[0].numpy(), expected_q)


if __name__ == "__main__":
    unittest.main()
