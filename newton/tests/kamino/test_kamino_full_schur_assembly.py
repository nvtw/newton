# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check parallel full-Schur assembly against a dense oracle."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.solvers.dvi.sparse_kernels import (
    _prepare_full_sparse_unilateral_schur as kernel,
)
from newton._src.solvers.kamino._src.solvers.dvi.types import DVIConfigStruct


class TestKaminoFullSchurAssembly(unittest.TestCase):
    def test_dense_oracle(self):
        """Cover mixed row types, capacity boundaries, and inactive worlds."""
        rng = np.random.default_rng(123)
        cases = [
            (33, 3, 2, 2, 16),
            (256, 31, 1, 32, 256),
            (256, 32, 1, 32, 256),
            (600, 512, 0, 0, 600),
            (600, 513, 0, 0, 600),
            (2, 1, 1, 2, 2),
            (33, 0, 0, 0, 8),
            (33, 3, 2, 2, 16),
        ]
        data = {
            name: []
            for name in (
                "bsm_num_nzb",
                "bsm_nzb_start",
                "bsm_nzb_coords",
                "bsm_nzb_values",
                "jacobian_nzb_values",
                "bsm_row_start",
                "bsm_col_start",
                "bounded_nzb_offsets",
                "limit_nzb_offsets",
                "contact_nzb_offsets",
                "limit_indices",
                "contact_indices",
                "problem_nbc",
                "problem_nl",
                "problem_nc",
                "problem_bcio",
                "problem_lio",
                "problem_cio",
                "problem_vio",
                "problem_P",
                "problem_v_f",
                "eta",
                "problem_njc",
                "response_mio",
                "response_stride",
                "compact_schur",
                "compact_q",
                "solver_config",
                "body_space",
                "solution_lambdas",
            )
        }
        expected_s, expected_q = [], []

        for wid, (n, nb, nl, nc, stride) in enumerate(cases):
            nu = nb + nl + 3 * nc
            weighted = np.zeros((nu, 12))
            jacobian = np.zeros((nu, 12))
            start = len(data["bsm_nzb_values"])
            data["bsm_nzb_start"].append(start)
            data["bsm_row_start"].append(len(data["eta"]))
            data["bsm_col_start"].append(len(data["body_space"]))
            data["problem_bcio"].append(len(data["bounded_nzb_offsets"]))
            data["problem_lio"].append(len(data["limit_indices"]))
            data["problem_cio"].append(len(data["contact_indices"]))
            data["problem_vio"].append(len(data["problem_P"]))
            data["response_mio"].append(len(data["compact_schur"]))
            for name, value in (
                ("problem_njc", n),
                ("problem_nbc", nb),
                ("problem_nl", nl),
                ("problem_nc", nc),
                ("response_stride", stride),
            ):
                data[name].append(value)

            def block(row, body, active=True, n=n, weighted=weighted, jacobian=jacobian):
                index = len(data["bsm_nzb_values"])
                w, j = rng.normal(size=(2, 6)).astype(np.float32)
                data["bsm_nzb_coords"].append((n + row, body * 6))
                data["bsm_nzb_values"].append(w)
                data["jacobian_nzb_values"].append(j)
                if active:
                    weighted[row, body * 6 : body * 6 + 6] = w
                    jacobian[row, body * 6 : body * 6 + 6] = j
                return index

            for row in range(nb):
                first = block(row, 0)
                second = block(row, 1) if row % 2 else -1
                data["bounded_nzb_offsets"].append((first, second))
            for limit in range(nl):
                active = limit % 2 == 0
                data["limit_indices"].append(len(data["limit_nzb_offsets"]) if active else -1)
                data["limit_nzb_offsets"].append(block(nb + limit, 0, active))
                block(nb + limit, 1, active)
            for contact in range(nc):
                data["contact_indices"].append(len(data["contact_nzb_offsets"]))
                data["contact_nzb_offsets"].append(len(data["bsm_nzb_values"]))
                for body in range(1 + contact % 2):
                    for component in range(3):
                        block(nb + nl + contact * 3 + component, body)
            data["bsm_num_nzb"].append(len(data["bsm_nzb_values"]) - start)
            p, vf, eta, lambdas, q = rng.normal(size=(5, n + nu)).astype(np.float32)
            body = rng.normal(size=12).astype(np.float32)
            schur = rng.normal(size=n * stride).astype(np.float32)
            expected_matrix, expected_velocity = schur.copy(), q.copy()
            if wid != len(cases) - 1 and 0 < nu <= 512 and nu * nu <= n * stride:
                operator = (weighted @ jacobian.T) * p[None, n:] + np.diag(eta[n:])
                expected_matrix[: nu * nu] -= operator.T.ravel()
                expected_velocity[n:] = weighted @ body + eta[n:] * lambdas[n:] + vf[n:]
            expected_s.extend(expected_matrix)
            expected_q.extend(expected_velocity)
            for name, values in (
                ("problem_P", p),
                ("problem_v_f", vf),
                ("eta", eta),
                ("solution_lambdas", lambdas),
                ("compact_q", q),
                ("body_space", body),
                ("compact_schur", schur),
            ):
                data[name].extend(values)
            config = DVIConfigStruct()
            config.max_alternating_iterations = 0 if wid == len(cases) - 1 else 8
            data["solver_config"].append(config)

        for device in wp.get_devices():
            arrays = {}
            for arg in kernel.adj.args:
                name = arg.label
                if name == "enable_compact_schur":
                    arrays[name] = True
                elif name == "workers_per_world":
                    arrays[name] = 128
                elif name == "block_iteration":
                    arrays[name] = 0
                else:
                    arrays[name] = wp.array(data[name], dtype=arg.type.dtype, device=device)
            reference_s = None
            reference_q = None
            for workers in (32, 128, 4096):
                arrays["workers_per_world"] = workers
                arrays["compact_schur"].assign(np.asarray(data["compact_schur"], dtype=np.float32))
                arrays["compact_q"].assign(np.asarray(data["compact_q"], dtype=np.float32))
                wp.launch(
                    kernel,
                    dim=len(cases) * workers,
                    inputs=[arrays[arg.label] for arg in kernel.adj.args],
                    block_dim=128,
                    device=device,
                )
                actual_s = arrays["compact_schur"].numpy()
                actual_q = arrays["compact_q"].numpy()
                np.testing.assert_allclose(actual_s, expected_s, atol=1e-5, rtol=1e-5)
                np.testing.assert_allclose(actual_q, expected_q, atol=1e-5, rtol=1e-5)
                if reference_s is not None:
                    np.testing.assert_array_equal(actual_s, reference_s)
                    np.testing.assert_array_equal(actual_q, reference_q)
                reference_s, reference_q = actual_s, actual_q


if __name__ == "__main__":
    unittest.main()
