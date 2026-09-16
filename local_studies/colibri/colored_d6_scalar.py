# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Local FP32 D6-style ordered endpoint rows, preserving native drive equations.

Architecture experiment, not a literal complete PhysX TGS implementation:
source joint coordinates/bias and native implicit drive coefficients remain.
No joint row orthogonalization or solver-offset slop is introduced.
"""

import hashlib
import importlib.abc
import importlib.util
import sys
from pathlib import Path

PREPARE = """@wp.func
def _prepare_scalar_rows(constraints: ConstraintContainer, bodies: BodyContainer, cid: wp.int32):
    data = constraints.bilateral
    count = data.row_count[cid]
    data.valid[cid] = wp.int32(0)
    if count == 0:
        return
    structural = data.structural_index[cid]
    body0 = constraint_get_body1(constraints, cid)
    body1 = constraint_get_body2(constraints, cid)
    for i in range(count):
        row = data.row_indices[cid, i]
        local = data.row_local[row]
        data.response0[cid, i] = _response(bodies, body0, data.wrench0[structural, local])
        data.response1[cid, i] = _response(bodies, body1, data.wrench1[structural, local])
    data.valid[cid] = wp.int32(1)


@wp.kernel(enable_backward=False)
def prepare_bilateral_joint_blocks(constraints: ConstraintContainer, bodies: BodyContainer, copy_state: CopyStateContainer):
    _prepare_scalar_rows(constraints, bodies, wp.tid())


@wp.kernel(enable_backward=False)
def _prepare_bilateral_joint_blocks_cooperative(constraints: ConstraintContainer, bodies: BodyContainer, copy_state: CopyStateContainer):
    tid = wp.tid()
    if tid % 8 == 0:
        _prepare_scalar_rows(constraints, bodies, tid // 8)


"""

ITERATE = """def get_iterate_bilateral_joint_block(cooperative: bool = False):
    @wp.func
    def iterate(
        constraints: ConstraintContainer, cid: wp.int32,
        bodies: BodyContainer, particles: ParticleContainer,
        copy_state: CopyStateContainer, num_bodies: wp.int32,
        parallel_id: wp.int32, use_bias: wp.bool, tile_lane: wp.int32,
    ):
        if wp.static(cooperative):
            if tile_lane != 0:
                return
        data = constraints.bilateral
        if data.enabled == 0 or data.row_count[cid] == 0:
            return
        structural = data.structural_index[cid]
        body0 = constraint_get_body1(constraints, cid)
        body1 = constraint_get_body2(constraints, cid)
        v0, v1, w0, w1, inv_m0, inv_m1, inv_i0, inv_i1, slot0, slot1 = _ms_load_body_pair(
            bodies, particles, copy_state, body0, body1, parallel_id, num_bodies)
        for i in range(data.row_count[cid]):
            row = data.row_indices[cid, i]
            local = data.row_local[row]
            j0 = data.wrench0[structural, local]
            j1 = data.wrench1[structural, local]
            response0 = data.response0[cid, i]
            response1 = data.response1[cid, i]
            residual = wp.dot(j0, wp.spatial_vector(v0, w0)) + wp.dot(j1, wp.spatial_vector(v1, w1))
            denominator = wp.dot(j0, response0) + wp.dot(j1, response1)
            if data.row_dynamic[row]:
                diagonal = wp.float32(1.0) / data.dynamic_mass[row]
                residual += diagonal * data.accumulated[row] - data.reference[row]
                denominator += diagonal
            elif use_bias:
                residual += data.bias[structural, local]
            if denominator > wp.float32(0.0):
                impulse = -residual / denominator
                data.accumulated[row] += impulse
                v0 += impulse * wp.spatial_top(response0)
                w0 += impulse * wp.spatial_bottom(response0)
                v1 += impulse * wp.spatial_top(response1)
                w1 += impulse * wp.spatial_bottom(response1)
        _ms_store_body_pair(bodies, particles, copy_state, body0, body1, slot0, slot1, num_bodies, v0, w0, v1, w1)
    return iterate


"""


class ScalarFinder(importlib.abc.MetaPathFinder):
    """Override only the joint module in this isolated process."""

    def __init__(self, path):
        self.path = path

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "newton._src.solvers.phoenx.constraints.bilateral_joint":
            return importlib.util.spec_from_file_location(fullname, self.path)
        return None


def install():
    root = Path(__file__).resolve().parents[2]
    source = (root / "newton/_src/solvers/phoenx/constraints/bilateral_joint.py").read_text()
    first = source.index("@wp.kernel(enable_backward=False)\ndef prepare_bilateral_joint_blocks(")
    last = source.index("def get_iterate_bilateral_joint_block(")
    tail = source.index("_iterate_bilateral_scalar =", last)
    candidate = source[:first] + PREPARE + ITERATE + source[tail:]
    compile(candidate, "<scalar_d6>", "exec")
    digest = hashlib.sha256(candidate.encode()).hexdigest()
    directory = Path("/tmp/colibri_d6_scalar_" + digest[:12])
    directory.mkdir(exist_ok=True)
    path = directory / "bilateral_joint.py"
    path.write_text(candidate)
    assert "newton._src.solvers.phoenx.constraints.bilateral_joint" not in sys.modules
    sys.meta_path.insert(0, ScalarFinder(path))
    print("SCALAR_D6_SOURCE_SHA256", digest, flush=True)
    return path


def main():
    install()
    from .colored_d6_alternative import transformed_source  # noqa: PLC0415

    path, source = transformed_source()
    source = source.replace('"path": "colored_block_pgs"', '"path": "colored_d6_scalar"')
    exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})


if __name__ == "__main__":
    main()
