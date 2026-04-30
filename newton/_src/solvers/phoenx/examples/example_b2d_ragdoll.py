# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Box2D port: ``Joints / Ragdoll``
#
# Articulated humanoid skeleton: torso + head + 2 upper arms + 2 forearms
# + 2 thighs + 2 shins, joined by revolute hinges. Drops to the ground.
# The 2D Box2D ragdoll's hinges become 3D revolute joints; we orient all
# hinges about world +y so the figure swings in the X-Z plane.
#
# Run: python -m newton._src.solvers.phoenx.examples.example_b2d_ragdoll
###########################################################################

from __future__ import annotations

import warp as wp

import newton
from newton._src.solvers.phoenx.examples._ported_example_base import (
    PortedExample,
    default_capsule_half_extents,
    default_sphere_half_extents,
    run_ported_example,
)

#: D-only velocity drive on every revolute, playing the role of Box2D's
#: ``m_jointFrictionTorque`` (motor with ``motorSpeed=0``, small max
#: torque). Without it limbs are completely floppy on impact.
JOINT_DAMPING = 0.05


class Example(PortedExample):
    sim_substeps = 12
    solver_iterations = 16

    def build_scene(self, builder: newton.ModelBuilder):
        builder.add_ground_plane()

        # Capsules along +z; "length" arg is half_height (cylindrical mid).
        # Joint axes about world +y so all hinges bend forward / backward.
        spawn_z = 4.0
        torso_h = 0.4  # half-height of torso cylinder
        torso_r = 0.15
        head_r = 0.18
        upper_h = 0.2
        lower_h = 0.22
        thigh_h = 0.25
        shin_h = 0.28
        limb_r = 0.07

        joints: list[int] = []
        extents: list = []

        # Torso (root -- floating link via free joint? we use world->torso
        # FREE joint for simplicity)
        torso = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, spawn_z), q=wp.quat_identity()),
        )
        builder.add_shape_capsule(torso, radius=torso_r, half_height=torso_h)
        extents.append(default_capsule_half_extents(torso_r, torso_h))
        free = builder.add_joint_free(child=torso)
        joints.append(free)

        # Head above torso.
        head = builder.add_link(
            xform=wp.transform(p=wp.vec3(0.0, 0.0, spawn_z + torso_h + head_r + 0.03), q=wp.quat_identity()),
        )
        builder.add_shape_sphere(head, radius=head_r)
        extents.append(default_sphere_half_extents(head_r))
        joints.append(
            builder.add_joint_revolute(
                parent=torso,
                child=head,
                parent_xform=wp.transform(p=wp.vec3(0.0, 0.0, torso_h + 0.02), q=wp.quat_identity()),
                child_xform=wp.transform(p=wp.vec3(0.0, 0.0, -head_r - 0.01), q=wp.quat_identity()),
                axis=(0.0, 1.0, 0.0),
                limit_lower=-0.5,
                limit_upper=0.5,
                target_vel=0.0,
                target_kd=JOINT_DAMPING,
                actuator_mode=newton.JointTargetMode.VELOCITY,
            )
        )

        def add_limb(parent, parent_xform, half_h, child_xform_offset_z, joint_lo, joint_hi):
            child = builder.add_link(xform=wp.transform_identity())
            builder.add_shape_capsule(child, radius=limb_r, half_height=half_h)
            extents.append(default_capsule_half_extents(limb_r, half_h))
            joints.append(
                builder.add_joint_revolute(
                    parent=parent,
                    child=child,
                    parent_xform=parent_xform,
                    child_xform=wp.transform(p=wp.vec3(0.0, 0.0, child_xform_offset_z), q=wp.quat_identity()),
                    axis=(0.0, 1.0, 0.0),
                    limit_lower=joint_lo,
                    limit_upper=joint_hi,
                    target_vel=0.0,
                    target_kd=JOINT_DAMPING,
                    actuator_mode=newton.JointTargetMode.VELOCITY,
                )
            )
            return child

        # Arms.
        for sign in (+1.0, -1.0):
            shoulder_xform = wp.transform(p=wp.vec3(sign * (torso_r + 0.02), 0.0, torso_h - 0.02), q=wp.quat_identity())
            upper = add_limb(torso, shoulder_xform, upper_h, upper_h + 0.01, -1.5, 1.5)
            elbow_xform = wp.transform(p=wp.vec3(0.0, 0.0, -upper_h - 0.01), q=wp.quat_identity())
            add_limb(upper, elbow_xform, lower_h, lower_h + 0.01, -2.2, 0.0)
        # Legs.
        for sign in (+1.0, -1.0):
            hip_xform = wp.transform(p=wp.vec3(sign * (torso_r * 0.6), 0.0, -torso_h - 0.02), q=wp.quat_identity())
            thigh = add_limb(torso, hip_xform, thigh_h, thigh_h + 0.01, -1.5, 1.5)
            knee_xform = wp.transform(p=wp.vec3(0.0, 0.0, -thigh_h - 0.01), q=wp.quat_identity())
            add_limb(thigh, knee_xform, shin_h, shin_h + 0.01, 0.0, 2.2)

        builder.add_articulation(joints)
        return extents

    def configure_camera(self, viewer):
        viewer.set_camera(pos=wp.vec3(4.0, -3.0, 2.5), pitch=-10.0, yaw=140.0)


if __name__ == "__main__":
    run_ported_example(Example)
