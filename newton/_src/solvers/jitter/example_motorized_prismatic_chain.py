# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Motorized Prismatic Chain
#
# A vertical column of ``NUM_BODIES`` cubes joined head-to-tail by
# prismatic (slider) joints. Each slider's free DoF is translation along
# world ``+y``, so the stack behaves like a set of concentric telescopic
# sections. The top joint anchors cube 0 to the static world; every
# other joint connects cube ``k-1`` to cube ``k``.
#
# The joint type is a :class:`JointMode.PRISMATIC` unified joint (see
# :mod:`constraint_actuated_double_ball_socket` for the math):
#
#   * The positional lock kills 5 DoF -- 3 rotational (the cubes must
#     keep their relative orientation) and 2 translational
#     perpendicular to the slide axis (no lateral drift). The
#     formulation is *pure point-matching*: two tangent-plane ties at
#     each of two user-supplied anchors plus one tangent-plane tie at
#     an auto-derived third anchor (perpendicular to the slide axis at
#     rest length).
#   * The free DoF is ``translation along +y``. In this example it's
#     driven by a soft *position* drive towards :data:`TARGET_POSITION`
#     [m] with a finite peak force cap -- the chain extends or
#     retracts along +y as the user changes ``TARGET_POSITION``.
#
# Run: ``python -m newton._src.solvers.jitter.example_motorized_prismatic_chain``
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.jitter.picking import JitterPicking, register_with_viewer_gl
from newton._src.solvers.jitter.solver_jitter import pack_body_xforms_kernel
from newton._src.solvers.jitter.world_builder import (
    DriveMode,
    JointMode,
    WorldBuilder,
)

NUM_CUBES = 6
HALF_EXTENT = 0.5
NUM_BODIES = NUM_CUBES + 1  # +1 for the static world anchor body
NUM_JOINTS = NUM_CUBES

_INV_INERTIA = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

# Slide axis: world +y. The second anchor is one metre away from the
# first (rest_length = 1 m) so the unit-axis convention recommended by
# :class:`JointDescriptor` applies: any value of ``target`` is then
# measured as "metres of displacement along the slide axis".
_SLIDE_AXIS = (0.0, 1.0, 0.0)

# Spacing between consecutive cubes at rest [m]. 2*HALF_EXTENT stacks
# them edge-to-edge -- tight enough to visualise, loose enough that
# the prismatic limits (when enabled) are nowhere near the rest pose.
_REST_SPACING = 2.0 * HALF_EXTENT

# Peak linear force the drive may apply per substep [N]. Generous so
# the chain converges to ``TARGET_POSITION`` under gravity without
# visibly oscillating.
_DRIVE_MAX_FORCE = 200.0

# Per-joint target displacement along +y relative to the joint's
# anchor1 (i.e. body1's end of the slider) [m]. Zero = rest pose; >0
# = extend; <0 = retract. The chain's total extension is
# ``NUM_JOINTS * TARGET_POSITION``.
TARGET_POSITION = 0.0


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        # ---- Build the Jitter World via WorldBuilder ------------------
        b = WorldBuilder()
        world_body = b.world_body  # body 0, auto-created static anchor

        # Cubes stacked below the world anchor along -y. Identity
        # orientation; no diamond-rotate is needed because the slide
        # axis is world-aligned.
        cube_ids: list[int] = []
        for j in range(NUM_CUBES):
            cube_ids.append(
                b.add_dynamic_body(
                    position=(0.0, -(j + 1) * _REST_SPACING, 0.0),
                    inverse_mass=1.0,
                    inverse_inertia=_INV_INERTIA,
                )
            )

        # One prismatic joint per junction. Joint k connects body_a
        # (world anchor for k=0, else cube k-1) to cube k. The first
        # anchor sits at body_a's "bottom face" -- midway between
        # body_a and body_b at rest -- and the second anchor is one
        # metre further along +y, yielding ``rest_length = 1 m``.
        self.joint_handles = []
        for k in range(NUM_JOINTS):
            body_a = world_body if k == 0 else cube_ids[k - 1]
            body_b = cube_ids[k]
            # Midpoint between body_a's COM and body_b's COM at rest.
            y_mid = -(k + 0.5) * _REST_SPACING if k > 0 else -0.5 * _REST_SPACING
            anchor1 = (0.0, y_mid, 0.0)
            anchor2 = (
                anchor1[0] + _SLIDE_AXIS[0],
                anchor1[1] + _SLIDE_AXIS[1],
                anchor1[2] + _SLIDE_AXIS[2],
            )
            self.joint_handles.append(
                b.add_joint(
                    body1=body_a,
                    body2=body_b,
                    anchor1=anchor1,
                    anchor2=anchor2,
                    mode=JointMode.PRISMATIC,
                    drive_mode=DriveMode.POSITION,
                    target=TARGET_POSITION,
                    max_force_drive=_DRIVE_MAX_FORCE,
                    hertz_drive=4.0,
                    damping_ratio_drive=1.0,
                )
            )

        self.world = b.finalize(
            substeps=1,
            solver_iterations=8,
            device=self.device,
        )

        # ---- Rendering scratch ---------------------------------------
        self._xforms = wp.zeros(NUM_BODIES, dtype=wp.transform, device=self.device)

        # ---- Picking --------------------------------------------------
        half_extents_np = np.zeros((NUM_BODIES, 3), dtype=np.float32)
        half_extents_np[1:] = HALF_EXTENT
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = JitterPicking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.picking.apply_force()
            self.world.step(self.sim_dt)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        wp.launch(
            pack_body_xforms_kernel,
            dim=NUM_BODIES,
            inputs=[self.world.bodies, self._xforms],
            device=self.device,
        )

        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_shapes(
            "/world/cubes",
            newton.GeoType.BOX,
            (HALF_EXTENT, HALF_EXTENT, HALF_EXTENT),
            self._xforms[1:],
        )
        self.viewer.end_frame()

    def test_final(self):
        # Sanity: no cube flew off to infinity and no NaNs. With
        # ``TARGET_POSITION = 0`` the chain should stay within the
        # initial bounding box ``[0, -NUM_CUBES * _REST_SPACING]``
        # plus a small drift margin.
        positions = self.world.bodies.position.numpy()
        floor = -(NUM_CUBES + 2) * _REST_SPACING
        for i in range(1, NUM_BODIES):
            assert np.isfinite(positions[i]).all(), (
                f"body {i} produced non-finite position"
            )
            assert positions[i, 1] > floor, (
                f"body {i} fell below prismatic-chain floor "
                f"({positions[i, 1]} < {floor})"
            )
            assert abs(positions[i, 0]) < 1.0, (
                f"body {i} drifted laterally along x ({positions[i, 0]})"
            )
            assert abs(positions[i, 2]) < 1.0, (
                f"body {i} drifted laterally along z ({positions[i, 2]})"
            )


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
