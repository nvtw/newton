"""CPU buffer regression for the tangent-only recovery trial overlay."""

import numpy as np
import warp as wp

from local_studies.colibri.tangent_recovery_trial import shift_tangent_target
from newton._src.solvers.phoenx.constraints.constraint_contact import (
    ContactColumnContainer,
    contact_column_container_zeros,
    contact_set_body1,
    contact_set_body2,
    contact_set_contact_count,
    contact_set_contact_first,
)
from newton._src.solvers.phoenx.constraints.contact_container import (
    ContactContainer,
    cc_set_bias_t1,
    cc_set_bias_t2,
    cc_set_normal,
    cc_set_r0,
    cc_set_r1,
    cc_set_tangent1,
    contact_container_zeros,
)
from newton._src.solvers.phoenx.constraints.contact_projection import contact_project_friction_metric


@wp.kernel
def setup(columns: ContactColumnContainer, contacts: ContactContainer):
    """Set one off-center contact and a sentinel inactive column."""
    contact_set_body1(columns, 0, 0)
    contact_set_body2(columns, 0, 1)
    contact_set_contact_first(columns, 0, 0)
    contact_set_contact_count(columns, 0, 1)
    cc_set_normal(contacts, 0, wp.vec3f(0.0, 0.0, 1.0))
    cc_set_tangent1(contacts, 0, wp.vec3f(1.0, 0.0, 0.0))
    cc_set_r0(contacts, 0, wp.vec3f(0.0))
    cc_set_r1(contacts, 0, wp.vec3f(0.0, 0.1, 0.0))
    cc_set_bias_t1(contacts, 0, wp.float32(2e-6))
    cc_set_bias_t2(contacts, 0, wp.float32(-3e-6))


@wp.kernel
def zero_friction(out: wp.array[wp.vec2f]):
    """Even huge recovery cannot produce friction when both radii are zero."""
    out[0] = contact_project_friction_metric(0.5, 0.1, 1.0, 1e5, -1e5, 0.0, 0.0, 0.0, 0.0)


def main():
    """Check target shift, unaffected normal rows, inactive tail, exact restoration."""
    columns = contact_column_container_zeros(2, device="cpu")
    contacts = contact_container_zeros(2, device="cpu")
    wp.launch(setup, dim=1, inputs=[columns, contacts], device="cpu")
    before = contacts.derived.numpy().copy()
    recovery = wp.array(
        np.array([[0] * 6, [1e-4, 2e-4, 3e-4, 0.0, 0.0, 1e-3]], dtype=np.float32), dtype=wp.spatial_vector, device="cpu"
    )
    saved = wp.zeros(2, dtype=wp.vec2f, device="cpu")
    scheduled = wp.array([0, 123456], dtype=wp.int32, device="cpu")
    ends = wp.array([1], dtype=wp.int32, device="cpu")
    inputs = [columns, contacts, scheduled, ends, recovery, saved]
    wp.launch(shift_tangent_target, dim=2, inputs=[*inputs, False], device="cpu")
    after = contacts.derived.numpy()
    np.testing.assert_allclose(after[4:6, 0], np.array([2e-6, 197e-6]), rtol=0, atol=2e-11)
    untouched = np.ones(before.shape, dtype=bool)
    untouched[4:6, 0] = False
    assert before[untouched].tobytes() == after[untouched].tobytes()
    wp.launch(shift_tangent_target, dim=2, inputs=[*inputs, True], device="cpu")
    assert before.tobytes() == contacts.derived.numpy().tobytes()
    out = wp.zeros(1, dtype=wp.vec2f, device="cpu")
    wp.launch(zero_friction, dim=1, inputs=[out], device="cpu")
    np.testing.assert_array_equal(out.numpy(), np.zeros((1, 2), dtype=np.float32))
    print("PASS: target arithmetic, normal/tail bytes, exact restore, zero-friction metric")


if __name__ == "__main__":
    main()
