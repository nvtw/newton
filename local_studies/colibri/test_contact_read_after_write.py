import unittest

import numpy as np
import warp as wp

from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_get_normal_lambda,
    contact_container_zeros,
)
from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_normal_velocity_update


@wp.kernel(enable_backward=False)
def _probe(cc: ContactContainer, out: wp.array[wp.float32]):
    old = cc_get_normal_lambda(cc, 0)
    impulse = contact_project_normal_velocity_update(
        cc,
        0,
        wp.vec3(1.0, 0.0, 0.0),
        wp.float32(0.0),
        wp.float32(0.1),
        wp.float32(-0.1),
        wp.float32(1.0),
        wp.float32(0.0),
        wp.float32(1.0),
        wp.float32(0.001),
        wp.float32(10.0),
        wp.float32(0.2),
    )
    out[0] = old
    out[1] = cc_get_normal_lambda(cc, 0)
    out[2] = cc.impulses[0, 0]
    out[3] = impulse[0]


class TestReadAfterWrite(unittest.TestCase):
    def test_probe(self):
        cc = contact_container_zeros(1, device="cuda:0")
        out = wp.zeros(4, dtype=wp.float32, device="cuda:0")
        wp.launch(_probe, dim=1, inputs=[cc, out], device="cuda:0")
        print("probe", out.numpy())
        np.testing.assert_allclose(out.numpy(), [0.0, 0.0002, 0.0002, 0.0002], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
