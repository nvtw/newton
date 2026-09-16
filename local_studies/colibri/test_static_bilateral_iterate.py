# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent CPU arithmetic-order comparison for constant-index joint sweeps."""

import ast
import copy
import inspect
from types import SimpleNamespace

import numpy as np

from local_studies.colibri.static_bilateral_iterate import module, source


def compile_python(text):
    """Interpret the actual original/transformed callback with NumPy containers."""
    tree = ast.parse(text)
    node = tree.body[0]
    node.decorator_list = []
    for arg in node.args.args:
        arg.annotation = None
    node.returns = None
    ast.fix_missing_locations(tree)

    class Matrix:
        def __init__(self, value):
            self.value = value

        def __mul__(self, vector):
            return self.value @ vector

    def load(bodies, particles, copies, a, b, parallel, count):
        return (
            bodies.v[a].copy(),
            bodies.v[b].copy(),
            bodies.w[a].copy(),
            bodies.w[b].copy(),
            bodies.im[a],
            bodies.im[b],
            Matrix(bodies.ii[a]),
            Matrix(bodies.ii[b]),
            -1,
            -1,
        )

    def store(bodies, particles, copies, a, b, sa, sb, count, va, wa, vb, wb):
        bodies.v[a], bodies.w[a], bodies.v[b], bodies.w[b] = va, wa, vb, wb

    def spatial(*values):
        return np.concatenate(values) if values else np.zeros(6, np.float32)

    def dot(a, b):
        value = np.float64(0)
        for k in range(6):
            value += np.float64(a[k]) * np.float64(b[k])
        return value

    namespace = {
        "wp": SimpleNamespace(
            float64=np.float64,
            float32=np.float32,
            spatial_vector=spatial,
            spatial_top=lambda x: x[:3],
            spatial_bottom=lambda x: x[3:],
        ),
        "Vec6d": lambda: np.zeros(6, np.float64),
        "_dot_double": dot,
        "_ms_load_body_pair": load,
        "_ms_store_body_pair": store,
        "constraint_get_body1": lambda c, cid: 0,
        "constraint_get_body2": lambda c, cid: 1,
    }
    exec(compile(tree, "<callback-arithmetic>", "exec"), namespace)
    return namespace[node.name]


def main():
    """Check all row counts, dynamic rows, bias phases and wide diagonal scales."""
    original = compile_python(inspect.getsource(module.iterate_bilateral_joint_block.func))
    expanded = compile_python(source())
    rng = np.random.default_rng(932)
    checks = 0
    for count in range(7):
        for use_bias in (False, True):
            for scale in (1e-6, 1.0, 1e6):
                lower = np.tril(rng.normal(size=(6, 6))) * 0.2 + np.eye(6)
                data = SimpleNamespace(
                    enabled=1,
                    row_count=np.array([count]),
                    valid=np.ones(1),
                    structural_index=np.zeros(1, int),
                    row_indices=np.arange(6)[None],
                    row_local=np.arange(6),
                    row_dynamic=np.array([0, 1, 0, 1, 0, 1]),
                    wrench0=rng.normal(size=(1, 6, 6)).astype(np.float32),
                    wrench1=rng.normal(size=(1, 6, 6)).astype(np.float32),
                    accumulated=rng.normal(size=6).astype(np.float32),
                    dynamic_mass=np.ones(6, np.float32) * 1.3,
                    reference=rng.normal(size=6).astype(np.float32),
                    bias=rng.normal(size=(1, 6)).astype(np.float32),
                    lower=lower[None],
                    diagonal=np.array([[scale, 1, 2, 3, 4, scale]]),
                )
                bodies = SimpleNamespace(
                    v=rng.normal(size=(2, 3)).astype(np.float32),
                    w=rng.normal(size=(2, 3)).astype(np.float32),
                    im=np.array([0.7, 1.3], np.float32),
                    ii=np.array([np.eye(3), 2 * np.eye(3)], np.float32),
                )
                outputs = []
                for func in (original, expanded):
                    c, b = SimpleNamespace(bilateral=copy.deepcopy(data)), copy.deepcopy(bodies)
                    func(c, 0, b, None, None, 2, 0, use_bias)
                    outputs.append((b.v, b.w, c.bilateral.accumulated))
                for a, b in zip(*outputs, strict=True):
                    assert a.tobytes() == b.tobytes()
                checks += 1
    print("CPU arithmetic exact cases:", checks)


if __name__ == "__main__":
    main()
