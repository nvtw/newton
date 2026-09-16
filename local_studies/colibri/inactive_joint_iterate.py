"""Local exact-zero axial iteration shortcut after bilateral block solving."""

import warp as wp

from newton._src.solvers.phoenx.constraints import constraint_joint as joint
from newton._src.solvers.phoenx.constraints.joint_inequality import joint_constraint_iterate_inequality as original


@wp.func
def iterate(
    constraints: joint.ConstraintContainer,
    cid: wp.int32,
    bodies: joint.BodyContainer,
    particles: joint.ParticleContainer,
    copy_state: joint.CopyStateContainer,
    num_bodies: wp.int32,
    parallel_id: wp.int32,
    idt: wp.float32,
    sor_boost: wp.float32,
    use_bias: wp.bool,
):
    mode = joint.read_int(constraints, joint._OFF_JOINT_MODE, cid)
    if (
        constraints.bilateral.enabled != 0
        and bodies.has_position_level_writers[0] == 0
        and (mode == joint.JOINT_MODE_REVOLUTE or mode == joint.JOINT_MODE_PRISMATIC)
        and joint.read_int(constraints, joint._OFF_CLAMP, cid) == joint._CLAMP_NONE
        and joint.read_float(constraints, joint._OFF_FRICTION_COEFFICIENT, cid) <= 0.0
    ):
        body0 = joint.read_int(constraints, joint._OFF_BODY1, cid)
        body1 = joint.read_int(constraints, joint._OFF_BODY2, cid)
        v0 = wp.vec3f()
        v1 = wp.vec3f()
        w0 = wp.vec3f()
        w1 = wp.vec3f()
        if copy_state.highest_index_in_use[0] == 0:
            v0, w0 = joint.body_load_vw(bodies, body0)
            v1, w1 = joint.body_load_vw(bodies, body1)
        else:
            v0, factor0, slot0 = joint.read_velocity_unified(
                bodies, particles, copy_state, body0, parallel_id, num_bodies
            )
            v1, factor1, slot1 = joint.read_velocity_unified(
                bodies, particles, copy_state, body1, parallel_id, num_bodies
            )
            w0, angular_factor0, angular_slot0 = joint.read_angular_velocity_unified(
                bodies, copy_state, body0, parallel_id, num_bodies
            )
            w1, angular_factor1, angular_slot1 = joint.read_angular_velocity_unified(
                bodies, copy_state, body1, parallel_id, num_bodies
            )
        # The original zero-wrench arithmetic can normalize signed zeros.
        # Preserve those cases exactly through the original callback.
        nonzero = True
        for component in range(3):
            if v0[component] == 0.0 or v1[component] == 0.0 or w0[component] == 0.0 or w1[component] == 0.0:
                nonzero = False
        if nonzero:
            joint.constraint_write_multiplier(constraints, joint._MUL_ACC_FRICTION, cid, 0.0)
            return
    original(constraints, cid, bodies, particles, copy_state, num_bodies, parallel_id, idt, sor_boost, use_bias)


def install():
    """Replace only the local dispatcher inequality callback."""
    from newton._src.solvers.phoenx import solver_phoenx_kernels

    solver_phoenx_kernels.joint_constraint_iterate_inequality = iterate
