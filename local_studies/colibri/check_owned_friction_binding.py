"""Fail closed on actual owned callback source and executed CPU break semantics."""

import hashlib
import inspect

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_friction_metric_with_break


@wp.kernel
def probe(out: wp.array[wp.vec3f]):
    k = wp.tid()
    rhs = wp.float32(0.0)
    if k == 1:
        rhs = wp.float32(-2.0)
    out[k] = contact_project_friction_metric_with_break(1.0, 0.0, 1.0, rhs, 0.0, 0.99, 0.0, 1.0, 1.0)


def check(module):
    source = inspect.getsource(module.iterate_maximal_contact_runs_kernel.func)
    assert "contact_project_friction_metric_with_break(" in source
    assert "contacts.lambdas[CC_FRICTION_BROKEN, contact] = tangents[2]" in source
    assert "normal_load = normal_lambda" in source
    assert "normal_load +=" not in source, "Old diagonal recovery subtraction still active"
    out = wp.zeros(2, dtype=wp.vec3f, device="cpu")
    wp.launch(probe, dim=2, inputs=[out], device="cpu")
    values = out.numpy().copy()
    np.testing.assert_allclose(values[0], [0.99, 0, 0], rtol=0, atol=2e-8)
    np.testing.assert_allclose(values[1], [1, 0, 1], rtol=0, atol=2e-6)
    return {
        "actual_kernel_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "cpu_stick_99_percent_and_actual_sliding": values.tolist(),
        "actual_owned_total_normal_load_source": True,
    }
