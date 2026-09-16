# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Prepared FP32 mass-metric hard-row groups with original impulse bookkeeping.

Local diagnostic only: preserve physical responses and hard targets under an
invertible row transformation. Drives stay in original coordinates.
"""

import importlib.abc
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

from . import colored_d6_alternative, colored_d6_physx_order, colored_d6_scalar, colored_d6_static_last

NAME = "newton._src.solvers.phoenx.constraints.bilateral_joint_data"


class DataFinder(importlib.abc.MetaPathFinder):
    """Add isolated FP32 scratch fields; leave canonical data definitions intact."""

    def __init__(self):
        path = Path(__file__).resolve().parents[2] / "newton/_src/solvers/phoenx/constraints/bilateral_joint_data.py"
        source = path.read_text()
        source = source.replace(
            "@wp.struct", "Mat66f = wp.types.matrix(shape=(6, 6), dtype=wp.float32)\n\n\n@wp.struct", 1
        )
        source += """    transformed0: wp.array2d[wp.spatial_vector]
    transformed1: wp.array2d[wp.spatial_vector]
    transformed_bias: wp.array2d[wp.float32]
    transform: wp.array[Mat66f]
"""
        self.path = Path(tempfile.mkdtemp(prefix="colibri_d6_preconditioned_")) / "bilateral_joint_data.py"
        self.path.write_text(source)

    def find_spec(self, fullname, path=None, target=None):
        if fullname == NAME:
            return importlib.util.spec_from_file_location(fullname, self.path)
        return None


TRANSFORM = """    transform = wp.identity(n=6, dtype=wp.float32)
    for i in range(count):
        row = data.row_indices[cid, i]
        local = data.row_local[row]
        data.transformed0[cid, i] = data.wrench0[structural, local]
        data.transformed1[cid, i] = data.wrench1[structural, local]
        data.transformed_bias[cid, i] = data.bias[structural, local]
    for i in range(count):
        row_i = data.row_indices[cid, i]
        local_i = data.row_local[row_i]
        if not data.row_dynamic[row_i]:
            for j in range(i):
                row_j = data.row_indices[cid, j]
                local_j = data.row_local[row_j]
                same_group = (local_i < wp.int32(3)) == (local_j < wp.int32(3))
                if not data.row_dynamic[row_j] and same_group:
                    ji0 = data.transformed0[cid, i]
                    ji1 = data.transformed1[cid, i]
                    jj0 = data.transformed0[cid, j]
                    jj1 = data.transformed1[cid, j]
                    rj0 = data.response0[cid, j]
                    rj1 = data.response1[cid, j]
                    denom = wp.dot(jj0, rj0) + wp.dot(jj1, rj1)
                    wp.expect_eq(denom > wp.float32(0.0), True)
                    coefficient = (wp.dot(ji0, rj0) + wp.dot(ji1, rj1)) / denom
                    data.transformed0[cid, i] = ji0 - coefficient * jj0
                    data.transformed1[cid, i] = ji1 - coefficient * jj1
                    data.response0[cid, i] -= coefficient * rj0
                    data.response1[cid, i] -= coefficient * rj1
                    data.transformed_bias[cid, i] -= coefficient * data.transformed_bias[cid, j]
                    for k in range(6):
                        transform[i, k] -= coefficient * transform[j, k]
    data.transform[cid] = transform
"""


def allocate(solver, output):
    """Allocate prepared scratch before capture and audit the final transform."""
    import numpy as np
    import warp as wp

    from newton._src.solvers.phoenx.constraints.bilateral_joint_data import Mat66f

    data = solver.world.constraints.bilateral
    count = data.row_count.shape[0]
    device = data.row_count.device
    data.transformed0 = wp.zeros((count, 6), dtype=wp.spatial_vector, device=device)
    data.transformed1 = wp.zeros((count, 6), dtype=wp.spatial_vector, device=device)
    data.transformed_bias = wp.zeros((count, 6), dtype=wp.float32, device=device)
    data.transform = wp.zeros(count, dtype=Mat66f, device=device)

    def audit():
        rows = data.row_indices.numpy()[0, :6]
        local = data.row_local.numpy()[rows]
        structural = int(data.structural_index.numpy()[0])
        transform = data.transform.numpy()[0].astype(float)
        j0 = data.wrench0.numpy()[structural, local].astype(float)
        j1 = data.wrench1.numpy()[structural, local].astype(float)
        actual0 = data.transformed0.numpy()[0].astype(float)
        actual1 = data.transformed1.numpy()[0].astype(float)
        bias = data.bias.numpy()[structural, local].astype(float)
        actual_bias = data.transformed_bias.numpy()[0].astype(float)
        np.testing.assert_allclose(actual0, transform @ j0, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(actual1, transform @ j1, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(actual_bias, transform @ bias, atol=2e-7, rtol=2e-6)
        assert abs(np.linalg.det(transform) - 1.0) < 1e-6
        dynamic = data.row_dynamic.numpy()[rows]
        np.testing.assert_array_equal(transform[dynamic], np.eye(6)[dynamic])
        result = {
            "transform": transform.tolist(),
            "determinant": float(np.linalg.det(transform)),
            "jacobian_error": float(max(np.max(abs(actual0 - transform @ j0)), np.max(abs(actual1 - transform @ j1)))),
            "target_error": float(np.max(abs(actual_bias - transform @ bias))),
            "data_module": sys.modules[NAME].__file__,
            "scope": "FP32 separate angular/linear hard-row groups; original-equation audits remain authoritative",
        }
        Path(output).with_suffix(".preconditioned.json").write_text(json.dumps(result, indent=2))

    return audit


def main():
    """Run prepared hard rows with PhysX row and static-contact ordering."""
    assert NAME not in sys.modules
    sys.meta_path.insert(0, DataFinder())
    marker = "    data.valid[cid] = wp.int32(1)"
    assert colored_d6_scalar.PREPARE.count(marker) == 1
    colored_d6_scalar.PREPARE = colored_d6_scalar.PREPARE.replace(marker, TRANSFORM + marker)
    source = colored_d6_physx_order.ordered_iteration(colored_d6_scalar.ITERATE)
    source = source.replace("j0 = data.wrench0[structural, local]", "j0 = data.transformed0[cid, i]")
    source = source.replace("j1 = data.wrench1[structural, local]", "j1 = data.transformed1[cid, i]")
    source = source.replace("residual += data.bias[structural, local]", "residual += data.transformed_bias[cid, i]")
    source = source.replace(
        "                    data.accumulated[row] += impulse",
        """                    transform = data.transform[cid]
                    for k in range(data.row_count[cid]):
                        original_row = data.row_indices[cid, k]
                        data.accumulated[original_row] += transform[i, k] * impulse""",
    )
    colored_d6_scalar.ITERATE = source
    original = colored_d6_alternative.transformed_source

    def transformed_source():
        path, source = original()
        marker = '        print("COLORED_BLOCK_PGS_OWNERSHIP_GATE_PASS", flush=True)'
        assert source.count(marker) == 1
        source = source.replace(
            marker,
            """        from local_studies.colibri.colored_d6_preconditioned import allocate
        phase_audits.append(allocate(solver, output))
"""
            + marker,
        )
        return path, source

    colored_d6_alternative.transformed_source = transformed_source
    colored_d6_static_last.main()


if __name__ == "__main__":
    main()
