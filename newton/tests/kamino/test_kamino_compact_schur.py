# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check compact Schur assembly and pipelined contact sweeps at their dispatch boundaries."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.dvi.sparse_kernels import (
    _assemble_compact_unilateral_schur_blocked,
    _solve_dvi_compact_schur_pgs_cooperative,
    _solve_dvi_sparse_inequalities_pgs_cooperative,
)
from newton._src.solvers.kamino._src.solvers.dvi.types import DVIConfigStruct, DVIStatus


class TestKaminoCompactSchur(unittest.TestCase):
    def setUp(self):
        if not wp.get_cuda_device_count():
            self.skipTest("Compact tile and warp kernels require CUDA")
        self.device = wp.get_cuda_devices()[0]

    def ints(self, values):
        return wp.array(values, dtype=wp.int32, device=self.device)

    def floats(self, values):
        return wp.array(np.asarray(values, dtype=np.float32).ravel(), dtype=wp.float32, device=self.device)

    def test_blocked_gram_boundaries(self):
        """Match a dense Gram oracle and preserve guards at empty, partial, and fallback sizes."""
        rng = np.random.default_rng(28931)
        cases = ((0, 0), (32, 0), (1, 1), (33, 3), (65, 31), (129, 65), (129, 127), (129, 128), (129, 129))
        for n, nu in cases:
            with self.subTest(n=n, nu=nu):
                stride = max(nu, 1)
                offset = 7
                capacity = n * stride
                white = rng.normal(size=(n, nu)).astype(np.float32)
                response = self.floats(np.pad(white.ravel(), (offset, capacity - n * nu + 7)))
                result = wp.full(offset + capacity + 7, -123.0, device=self.device)
                q = wp.full(n + nu + 7, -77.0, device=self.device)
                wp.launch(
                    _assemble_compact_unilateral_schur_blocked,
                    dim=(1, 256),
                    inputs=[
                        self.ints([n + nu]),
                        self.ints([n]),
                        self.ints([0]),
                        self.ints([offset]),
                        self.ints([stride]),
                        response,
                        result,
                        q,
                    ],
                    device=self.device,
                    block_dim=256,
                )
                expected = np.full(offset + capacity + 7, -123.0, dtype=np.float32)
                expected_q = np.full(n + nu + 7, -77.0, dtype=np.float32)
                if nu <= 128 and nu * nu <= capacity:
                    expected[offset : offset + nu * nu] = (white.astype(np.float64).T @ white).ravel()
                    expected_q[n : n + nu] = 0.0
                np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4, atol=5e-5)
                np.testing.assert_array_equal(q.numpy(), expected_q)

    def test_pipelined_sweeps_match_general_kernel(self):
        """Preserve mixed-constraint updates and reversed schedules through 128 compact rows."""
        rng = np.random.default_rng(28932)
        for nb, nl, nc in ((0, 0, 1), (2, 2, 3), (31, 1, 32), (0, 0, 42), (7, 0, 2)):
            with self.subTest(bounded=nb, limits=nl, contacts=nc):
                nu = nb + nl + 3 * nc
                n = max(32, nu)
                slots = nb + nl + nc
                dense = rng.normal(0, 0.1, (nu, nu)).astype(np.float32)
                operator = dense @ dense.T + np.eye(nu, dtype=np.float32)
                q0 = rng.normal(size=nu).astype(np.float32)
                initial = rng.uniform(0.0, 0.1, n + nu).astype(np.float32)
                order = rng.permutation(slots).astype(np.int32)
                # Reverse colors and group-local rows independently, preserving group order.
                group_bounds = np.unique(np.linspace(0, slots, min(4, slots) + 1).astype(np.int32))
                groups = len(group_bounds) - 1
                color_bounds = np.unique(np.array([0, groups // 2, groups], dtype=np.int32))
                config = DVIConfigStruct()
                # A non-fused iteration selects the general implementation without
                # its dispatch to the pipeline. One outer iteration makes the two
                # schedules equivalent, including odd tangent sweeps.
                config.max_alternating_iterations = 1
                config.inequality_sweeps_per_iteration = 8
                config.regularization = 0.001
                config.omega = 0.8
                config.tolerance = 1e-5
                data = {
                    "problem_nbc": self.ints([nb]),
                    "problem_nl": self.ints([nl]),
                    "problem_nc": self.ints([nc]),
                    "problem_njc": self.ints([n]),
                    "problem_bcio": self.ints([0]),
                    "problem_lio": self.ints([0]),
                    "problem_cio": self.ints([0]),
                    "problem_uio": self.ints([0]),
                    "problem_bcgo": self.ints([n]),
                    "problem_lcgo": self.ints([n + nb]),
                    "problem_ccgo": self.ints([n + nb + nl]),
                    "problem_vio": self.ints([0]),
                    "bilateral_vio": self.ints([0]),
                    "response_mio": self.ints([0]),
                    "response_stride": self.ints([nu]),
                    "limit_indices": self.ints(list(range(nl))),
                    "contact_indices": self.ints(list(range(nc))),
                    "problem_mu": self.floats(np.full(nc, 0.6)),
                    "problem_bound_lower": self.floats(np.full(nb, -0.3)),
                    "problem_bound_upper": self.floats(np.full(nb, 0.7)),
                    "problem_P": self.floats(np.ones(n + nu)),
                    "problem_v_b": self.floats(np.zeros(n + nu)),
                    "problem_diag": self.floats(np.concatenate([np.ones(n), operator.diagonal()])),
                    "projected_diag": self.floats(np.concatenate([np.ones(n), operator.diagonal()])),
                    "compact_schur": self.floats(-operator.T),
                    "inequality_num_colors": self.ints([len(color_bounds) - 1]),
                    "inequality_ids_by_color": self.ints(order),
                    "inequality_color_starts": self.ints(color_bounds),
                    "inequality_group_starts": self.ints(group_bounds),
                    "solver_config": wp.array([config], dtype=DVIConfigStruct, device=self.device),
                    "enable_compact_schur": True,
                    "block_iteration": 0,
                }
                results = []
                for kernel in (
                    _solve_dvi_sparse_inequalities_pgs_cooperative,
                    _solve_dvi_compact_schur_pgs_cooperative,
                ):
                    data["compact_q"] = self.floats(np.concatenate([np.zeros(n), q0]))
                    data["solution_lambdas"] = self.floats(initial)
                    data["solver_status"] = wp.zeros(1, dtype=DVIStatus, device=self.device)
                    inputs = []
                    for arg in kernel.adj.args:
                        if arg.label in data:
                            inputs.append(data[arg.label])
                        else:
                            # Full compact sweeps never read sparse body-space inputs.
                            shape = (1, 2) if arg.type.ndim == 2 else 1
                            inputs.append(wp.zeros(shape, dtype=arg.type.dtype, device=self.device))
                    wp.launch(kernel, dim=32, inputs=inputs, device=self.device, block_dim=32)
                    results.append((data["solution_lambdas"].numpy(), data["compact_q"].numpy()))
                np.testing.assert_array_equal(results[1][0][:n], initial[:n])
                np.testing.assert_allclose(results[0][0], results[1][0], atol=3e-6, rtol=3e-6)
                np.testing.assert_allclose(results[0][1], results[1][1], atol=3e-6, rtol=3e-6)


if __name__ == "__main__":
    unittest.main()
