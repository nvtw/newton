"""CPU gates for the combined history-only matching experiment."""

# ruff: noqa: E402 -- install source overlays before importing Newton.

from local_studies.colibri.combined_gap_matching import install

original, modified, directory, constructor, installed = install()

import numpy as np
import warp as wp

import newton
from local_studies.colibri.check_history_only_matching import run
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_contact import ContactViews
from newton._src.solvers.phoenx.constraints.contact_container import contact_container_zeros
from newton._src.solvers.phoenx.constraints.contact_ingest import _contact_warmstart_gather_kernel

candidate_constructor = newton.CollisionPipeline.__init__
newton.CollisionPipeline.__init__ = constructor
baseline = run(False)
newton.CollisionPipeline.__init__ = candidate_constructor
candidate = run(False)
assert len(installed) == 1
assert baseline[1][0] < 0 and candidate[1][0] == 0 and candidate[2][0] < 0
for b, c in zip(baseline, candidate, strict=True):
    for name in b[1]:
        np.testing.assert_array_equal(b[1][name], c[1][name])


def array(values, dtype=wp.int32):
    return wp.array(values, dtype=dtype, device="cpu")


bodies = body_container_zeros(2, device="cpu")
bodies.orientation.assign(np.array([[0, 0, 0, 1], [0, 0, 0, 1]], dtype=np.float32))
contacts = ContactViews()
contacts.rigid_contact_count = array([1])
contacts.rigid_contact_point0 = array([[0, 0, 0]], wp.vec3f)
contacts.rigid_contact_point1 = array([[0, 0, 5.0e-6]], wp.vec3f)
contacts.rigid_contact_normal = array([[0, 0, 1]], wp.vec3f)
contacts.rigid_contact_shape0 = array([0])
contacts.rigid_contact_shape1 = array([1])
contacts.rigid_contact_match_index = array([0])
contacts.rigid_contact_margin0 = array([0], wp.float32)
contacts.rigid_contact_margin1 = array([0], wp.float32)
contacts.shape_body = array([0, 1])
contacts.shape_type = array([int(newton.GeoType.SPHERE)] * 2)
cc = contact_container_zeros(1, device="cpu")
history = np.zeros(cc.prev_lambdas.shape, dtype=np.float32)
history[2, 0] = 1
history[3, 0] = 1
history[6:12, 0] = np.arange(6, dtype=np.float32) * 0.001
cc.prev_lambdas.assign(history)
impulses = np.zeros(cc.prev_impulses.shape, dtype=np.float32)
impulses[:3, 0] = [1, 0.1, 0.2]
cc.prev_impulses.assign(impulses)
zero = array([0])
for gap in (5.0e-6, -5.0e-6):
    contacts.rigid_contact_point1.assign(np.array([[0, 0, gap]], dtype=np.float32))
    wp.launch(
        _contact_warmstart_gather_kernel,
        dim=1,
        inputs=[
            zero,
            zero,
            zero,
            1,
            contacts.rigid_contact_match_index,
            zero,
            zero,
            1,
            bodies,
            contacts,
            0,
            1,
            zero,
            cc,
        ],
        device="cpu",
    )
    np.testing.assert_array_equal(cc.lambdas.numpy()[6:12], history[6:12])
    if gap > 0:
        np.testing.assert_array_equal(cc.impulses.numpy()[:3], 0)
    else:
        assert float(cc.impulses.numpy()[0, 0]) == 1.0
        np.testing.assert_allclose(np.linalg.norm(cc.impulses.numpy()[1:3, 0]), np.sqrt(0.05), rtol=1e-6)
print(
    "PASS: combined constructor installs once; fresh geometry exact; positive-gap impulses cleared, anchors retained; negative-gap warmstart retained"
)
