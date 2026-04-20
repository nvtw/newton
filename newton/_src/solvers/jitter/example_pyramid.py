# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Pyramid
#
# A box pyramid of configurable height sitting on a ground plane, left
# to settle under gravity. Drives Newton's :class:`CollisionPipeline`
# to generate persistent, matched contacts and feeds them to the
# Jitter solver -- the first self-contained contact-stability test
# for this solver.
#
# Pattern (each level ``L`` has ``layers - L`` cubes in a row):
#     layer 0: ########  (width `layers`)
#     layer 1:  ######
#     layer 2:   ####
#     ...
#
# Runs two gravity-settle stages: starting slightly above the ground
# so contacts warm-start cleanly, then lets the tower come to rest.
# :meth:`Example.test_final` asserts every cube is at rest within
# 10 cm of its nominal grid position.
#
# The scene is built via ``newton.ModelBuilder`` + Newton's
# ``CollisionPipeline`` (which produces the sorted, matched contact
# buffer the Jitter solver consumes) and a thin state-sync adapter
# keeps ``state.body_q`` / ``state.body_qd`` in lockstep with Jitter's
# own :class:`BodyContainer`. This is the integration pattern we'll
# generalise once Jitter becomes an official Newton solver.
#
# Run:  python -m newton._src.solvers.jitter.example_pyramid
# Options: --layers N (default 10)
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.jitter.body import MOTION_DYNAMIC, MOTION_STATIC
from newton._src.solvers.jitter.solver_jitter import pack_body_xforms_kernel
from newton._src.solvers.jitter.world_builder import (
    RigidBodyDescriptor,
    WorldBuilder,
)

DEFAULT_LAYERS = 10
BOX_HALF = 0.5
# Tight spacing produces a well-aligned pyramid; keep a small
# horizontal gap so the broad phase doesn't produce spurious inter-
# cube contacts on the very first frame.
BOX_SPACING = 2.01 * BOX_HALF


# ---------------------------------------------------------------------------
# Newton <-> Jitter state sync kernels
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _newton_to_jitter_kernel(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    # out
    position: wp.array[wp.vec3],
    orientation: wp.array[wp.quat],
    velocity: wp.array[wp.vec3],
    angular_velocity: wp.array[wp.vec3],
):
    """Push Newton's body state into Jitter's :class:`BodyContainer`.

    Newton stores the body-origin transform in ``body_q`` and the
    COM as a body-local offset in ``body_com``. Jitter's
    ``BodyContainer.position`` is the COM in world space, so we
    rotate ``body_com`` into world and add it to ``body_q``'s
    translation. Orientation is shared verbatim.

    ``body_qd`` is ``spatial_vector(linear_COM_velocity,
    angular_velocity)`` in world frame so the copy is direct.
    """
    i = wp.tid()
    q = body_q[i]
    pos_body = wp.transform_get_translation(q)
    rot = wp.transform_get_rotation(q)
    position[i] = pos_body + wp.quat_rotate(rot, body_com[i])
    orientation[i] = rot
    qd = body_qd[i]
    velocity[i] = wp.vec3f(qd[0], qd[1], qd[2])
    angular_velocity[i] = wp.vec3f(qd[3], qd[4], qd[5])


@wp.kernel(enable_backward=False)
def _jitter_to_newton_kernel(
    position: wp.array[wp.vec3],
    orientation: wp.array[wp.quat],
    velocity: wp.array[wp.vec3],
    angular_velocity: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    # out
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Write Jitter's body state back into Newton's ``body_q`` / ``body_qd``.

    Reverse of :func:`_newton_to_jitter_kernel`: Jitter's ``position``
    is the COM in world space, Newton's ``body_q`` is the body-origin
    transform, so we subtract the rotated COM offset.
    """
    i = wp.tid()
    rot = orientation[i]
    com_world = wp.quat_rotate(rot, body_com[i])
    pos_body = position[i] - com_world
    body_q[i] = wp.transform(pos_body, rot)
    body_qd[i] = wp.spatial_vector(velocity[i], angular_velocity[i])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quat_rotate_np(q, v) -> np.ndarray:
    """NumPy equivalent of ``wp.quat_rotate`` with xyzw layout.

    Duplicated here (rather than pulled from a shared utils module)
    because the finalize path runs purely on the host and we want to
    keep this example standalone.
    """
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array(
        [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ],
        dtype=np.float32,
    )


def _build_jitter_world_from_model(model: newton.Model):
    """Mirror Newton's body set into a :class:`WorldBuilder`.

    Returns a populated builder and the Newton-index -> Jitter-index
    map (Jitter body 0 is the auto-created static world anchor, so
    Newton body ``i`` ends up at Jitter body ``i + 1``).
    """
    body_q_np = model.body_q.numpy()
    body_inv_mass_np = model.body_inv_mass.numpy() if model.body_inv_mass is not None else None
    body_inv_inertia_np = (
        model.body_inv_inertia.numpy() if model.body_inv_inertia is not None else None
    )
    body_com_np = model.body_com.numpy()

    builder = WorldBuilder()
    newton_to_jitter: dict[int, int] = {}

    for i in range(model.body_count):
        inv_m = float(body_inv_mass_np[i]) if body_inv_mass_np is not None else 0.0
        inv_I = (
            body_inv_inertia_np[i]
            if body_inv_inertia_np is not None
            else np.zeros((3, 3), dtype=np.float32)
        )
        # A non-positive inverse mass means a static body in Newton.
        is_static = inv_m <= 0.0
        pose = body_q_np[i]
        origin_pos = pose[:3]
        rot = pose[3:7]  # xyzw
        com_local = body_com_np[i]
        com_world = origin_pos + _quat_rotate_np(rot, com_local)

        if is_static:
            motion = int(MOTION_STATIC)
            zero = np.zeros((3, 3), dtype=np.float32)
            desc = RigidBodyDescriptor(
                position=tuple(com_world.tolist()),
                orientation=tuple(rot.tolist()),
                motion_type=motion,
                inverse_mass=0.0,
                inverse_inertia=tuple(tuple(float(x) for x in r) for r in zero),
                affected_by_gravity=False,
            )
        else:
            desc = RigidBodyDescriptor(
                position=tuple(com_world.tolist()),
                orientation=tuple(rot.tolist()),
                motion_type=int(MOTION_DYNAMIC),
                inverse_mass=inv_m,
                inverse_inertia=tuple(tuple(float(x) for x in r) for r in inv_I),
                affected_by_gravity=True,
            )
        newton_to_jitter[i] = builder.add_body(desc)

    return builder, newton_to_jitter


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.solver_iterations = 20

        self.viewer = viewer
        self.device = wp.get_device()
        self.layers = int(getattr(args, "layers", DEFAULT_LAYERS))

        # ---- Scene: ground plane + pyramid of unit cubes ------------
        # Newton's default up-axis (and plane normal) is +Z, so the
        # pyramid stacks along +Z. ``add_shape_plane`` with no args
        # drops an infinite XY ground at Z=0.
        mb = newton.ModelBuilder()
        mb.add_shape_plane(-1, wp.transform_identity(), width=0.0, length=0.0)

        self._box_bodies: list[int] = []
        self._nominal_positions: list[tuple[float, float, float]] = []
        for level in range(self.layers):
            num_in_row = self.layers - level
            row_width = (num_in_row - 1) * BOX_SPACING
            for i in range(num_in_row):
                x = -row_width * 0.5 + i * BOX_SPACING
                y = 0.0
                # Start each cube a hair above its resting height so
                # the first-frame contacts warm-start cleanly (rather
                # than being generated from initial deep penetration).
                z = level * BOX_SPACING + BOX_HALF + 1.0e-3
                body = mb.add_body(
                    xform=wp.transform(p=wp.vec3(x, y, z), q=wp.quat_identity()),
                )
                mb.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)
                self._box_bodies.append(body)
                # Nominal resting z: level * BOX_SPACING + BOX_HALF.
                self._nominal_positions.append((x, y, level * BOX_SPACING + BOX_HALF))

        self.model = mb.finalize()
        print(
            f"pyramid: layers={self.layers} boxes={len(self._box_bodies)} "
            f"shapes={self.model.shape_count}"
        )

        # ---- Collision pipeline ------------------------------------
        # ``contact_matching=True`` is required for the Jitter solver
        # (the warm-start gather kernel reads ``rigid_contact_match_index``).
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching=True,
        )
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        # ---- Build the Jitter world mirroring Newton's body set -----
        builder, newton_to_jitter = _build_jitter_world_from_model(self.model)
        # Upper bound on contact columns: every box/plane pair yields
        # up to 4 contacts (= 1 column), and box/box pairs in a stack
        # add roughly one column per horizontal+vertical contact. A
        # safe envelope is "one column per contact", which is the
        # coarsest bound the ingest pipeline needs.
        max_contact_columns = max(16, rigid_contact_max)
        num_shapes = int(self.model.shape_count)
        self.world = builder.finalize(
            substeps=1,
            solver_iterations=self.solver_iterations,
            gravity=(0.0, 0.0, -9.81),
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=num_shapes,
            device=self.device,
        )
        self._newton_to_jitter = newton_to_jitter

        # Shape-to-Jitter-body map. Newton's ``shape_body`` holds
        # Newton body indices (or -1 for the world). We shift by +1
        # to match Jitter's body layout (the static anchor is Jitter
        # body 0) and remap -1 to that anchor.
        shape_body_np = self.model.shape_body.numpy()
        shape_body_jitter = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(shape_body_jitter, dtype=wp.int32, device=self.device)

        # ---- Viewer setup ------------------------------------------
        self.viewer.set_model(self.model)
        span = self.layers * BOX_SPACING
        self.viewer.set_camera(
            pos=wp.vec3(span * 2.5, span * 2.0, span * 1.2),
            pitch=-18.0,
            yaw=-55.0,
        )
        self._xforms = wp.zeros(self.world.num_bodies, dtype=wp.transform, device=self.device)

    # ----------------------------------------------------------------
    # Simulation
    # ----------------------------------------------------------------

    def simulate(self) -> None:
        """One render frame: refresh contacts, sub-step Jitter, sync state."""
        # Push current Newton state into Jitter bodies. Done once per
        # frame; during sub-stepping Jitter integrates internally.
        self._sync_newton_to_jitter()

        # Newton's ``body_q`` already reflects the previous frame's
        # Jitter result (we synced back at the end of the previous
        # frame), so the collision pipeline sees the up-to-date
        # transforms.
        self.contacts = self.model.collide(
            self.state, collision_pipeline=self.collision_pipeline
        )

        for _ in range(self.sim_substeps):
            self.world.step(
                dt=self.sim_dt,
                contacts=self.contacts,
                shape_body=self._shape_body,
            )

        self._sync_jitter_to_newton()

    def _sync_newton_to_jitter(self) -> None:
        # Target slots [1, 1+body_count) inside the Jitter body
        # container (body 0 is the static anchor we don't touch).
        n = self.model.body_count
        wp.launch(
            _newton_to_jitter_kernel,
            dim=n,
            inputs=[
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
            ],
            outputs=[
                self.world.bodies.position[1 : 1 + n],
                self.world.bodies.orientation[1 : 1 + n],
                self.world.bodies.velocity[1 : 1 + n],
                self.world.bodies.angular_velocity[1 : 1 + n],
            ],
            device=self.device,
        )

    def _sync_jitter_to_newton(self) -> None:
        n = self.model.body_count
        wp.launch(
            _jitter_to_newton_kernel,
            dim=n,
            inputs=[
                self.world.bodies.position[1 : 1 + n],
                self.world.bodies.orientation[1 : 1 + n],
                self.world.bodies.velocity[1 : 1 + n],
                self.world.bodies.angular_velocity[1 : 1 + n],
                self.model.body_com,
            ],
            outputs=[self.state.body_q, self.state.body_qd],
            device=self.device,
        )

    def step(self) -> None:
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self) -> None:
        wp.launch(
            pack_body_xforms_kernel,
            dim=self.world.num_bodies,
            inputs=[self.world.bodies, self._xforms],
            device=self.device,
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.end_frame()

    # ----------------------------------------------------------------
    # Test
    # ----------------------------------------------------------------

    def test_final(self) -> None:
        """Assert every cube sits at rest near its nominal pyramid slot.

        After the settle run each cube's COM should be within
        ``pos_tol`` of its nominal (x, y, z) and its speed should be
        below ``vel_tol``. Any failure means the solver couldn't
        stabilise the stack -- usually a sign of the contact
        warm-start or speculative-separation logic regressing.
        """
        pos_tol = 0.10  # [m] -- roughly 10% of a cube edge
        vel_tol = 0.5  # [m/s]

        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        for nbody, nominal in zip(self._box_bodies, self._nominal_positions, strict=True):
            jbody = self._newton_to_jitter[nbody]
            pos = positions[jbody]
            vel = velocities[jbody]
            assert np.isfinite(pos).all(), f"body {nbody} position non-finite ({pos})"
            disp = float(np.linalg.norm(pos - np.asarray(nominal, dtype=np.float32)))
            speed = float(np.linalg.norm(vel))
            assert disp < pos_tol, (
                f"body {nbody}: displaced {disp:.3f} m from nominal {nominal} "
                f"(pos {tuple(float(x) for x in pos)})"
            )
            assert speed < vel_tol, f"body {nbody}: still moving at {speed:.3f} m/s"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--layers",
            type=int,
            default=DEFAULT_LAYERS,
            help="Number of pyramid layers (bottom row width).",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
