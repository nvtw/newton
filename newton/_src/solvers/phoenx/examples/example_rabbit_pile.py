# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Rabbit Pile
#
# Drops a stack of SDF-backed bunnies onto a ground plane. Every
# bunny has its own signed-distance field (``Mesh.build_sdf``), so
# every bunny-vs-bunny collision is a dense SDF manifold -- this is
# the most contact-heavy PhoenX scene in the suite and is the
# intended stress test for:
#
#   * The 6-slot-per-column contact-packing in
#     :mod:`constraint_contact` (single bunny-vs-bunny pair can
#     spill across multiple columns).
#   * The per-frame warm-start match-index pipeline.
#   * The Jones-Plassmann graph colouring on a non-trivial body-pair
#     topology (cubes-on-a-plane is almost fully ordered).
#   * The hard-contact PGS convergence under many simultaneous
#     body-body constraints (the standard "pile of junk" stability
#     regression).
#
# Requires CUDA (SDF narrow phase is CUDA-only).
#
# Run:  python -m newton._src.solvers.phoenx.examples.example_rabbit_pile
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.constraints.constraint_contact import CONTACT_DWORDS
from newton._src.solvers.phoenx.constraints.constraint_container import (
    constraint_container_zeros,
)
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel,
    newton_to_phoenx_kernel,
    phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.picking import (
    Picking,
    register_with_viewer_gl,
)
from newton._src.solvers.phoenx.solver_phoenx import (
    DEFAULT_SHAPE_GAP,
    PhoenXWorld,
    pack_body_xforms_kernel,
)

#: Per-bunny uniform scale. The bunny USD ships in unit-metre units,
#: which is too big to pile neatly; 0.2 makes each bunny ~20 cm tall.
BUNNY_SCALE = 0.2

#: Narrow-band SDF range [m] in mesh coords. The bunny vertices are
#: O(1 m) so 5 mm is a few voxels wide at max_resolution=256.
BUNNY_SDF_NARROW_BAND = (-0.005, 0.005)

#: Max sparse-SDF grid dimension. 256 is a good tradeoff for the
#: bunny: retains ear / foot features without exploding memory.
BUNNY_SDF_MAX_RESOLUTION = 256

#: Contact friction coefficient. Medium -- enough that bunnies hold
#: their stacked poses without instantly splaying outward.
BUNNY_FRICTION = 0.5


def _load_bunny_mesh(build_sdf: bool) -> newton.Mesh:
    """Load the bunny triangle mesh with an optional baked SDF.

    Tries Newton's USD loader (canonical asset); falls back to a
    12-vertex icosahedron when the optional ``pxr`` dep is missing
    so the example still runs on bare-CUDA machines.
    """
    try:
        from pxr import Usd

        import newton.usd as newton_usd  # noqa: PLC0415

        stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        mesh = newton_usd.get_mesh(stage.GetPrimAtPath("/root/bunny"))
    except ModuleNotFoundError:
        # Icosahedron fallback. Same contact-pipeline path as the
        # bunny, just a simpler mesh.
        t = 0.5 * (1.0 + 5.0**0.5)
        verts = np.array(
            [
                [-1, t, 0],
                [1, t, 0],
                [-1, -t, 0],
                [1, -t, 0],
                [0, -1, t],
                [0, 1, t],
                [0, -1, -t],
                [0, 1, -t],
                [t, 0, -1],
                [t, 0, 1],
                [-t, 0, -1],
                [-t, 0, 1],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                0,
                11,
                5,
                0,
                5,
                1,
                0,
                1,
                7,
                0,
                7,
                10,
                0,
                10,
                11,
                1,
                5,
                9,
                5,
                11,
                4,
                11,
                10,
                2,
                10,
                7,
                6,
                7,
                1,
                8,
                3,
                9,
                4,
                3,
                4,
                2,
                3,
                2,
                6,
                3,
                6,
                8,
                3,
                8,
                9,
                4,
                9,
                5,
                2,
                4,
                11,
                6,
                2,
                10,
                8,
                6,
                7,
                9,
                8,
                1,
            ],
            dtype=np.int32,
        )
        mesh = newton.Mesh(verts, faces)

    if build_sdf and mesh.sdf is None:
        mesh.build_sdf(
            max_resolution=BUNNY_SDF_MAX_RESOLUTION,
            narrow_band_range=BUNNY_SDF_NARROW_BAND,
            margin=DEFAULT_SHAPE_GAP,
        )
    return mesh


class Example:
    """SDF-backed bunny pile for :class:`PhoenXWorld`.

    Lays out ``num_bunnies`` bunnies on a small grid above the
    ground plane with random yaws, then lets gravity pile them up.
    Every bunny collides with every other bunny via SDF narrow-phase;
    typical peak contact counts are a few hundred points per frame,
    which is an order of magnitude more than any other example and
    is the primary stress test for PhoenX's dense-manifold code
    path.
    """

    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 4
        self.solver_iterations = 16
        self._frame: int = 0

        self.num_bunnies = int(getattr(args, "num_bunnies", 24))

        self.viewer = viewer
        self.device = wp.get_device()
        if not self.device.is_cuda:
            raise RuntimeError("example_rabbit_pile requires CUDA (SDF narrow phase is CUDA-only).")

        bunny_mesh = _load_bunny_mesh(build_sdf=True)

        mb = newton.ModelBuilder()
        mb.default_shape_cfg.gap = DEFAULT_SHAPE_GAP
        mb.add_shape_plane(
            -1,
            wp.transform_identity(),
            width=0.0,
            length=0.0,
        )

        # Scatter bunnies on a 3D grid with random yaw + small tilt so
        # every bunny-vs-bunny pair has a non-degenerate SDF contact
        # frame. Grid side is ``ceil(num_bunnies ** (1/3))`` so the
        # cube comfortably holds every bunny; the top layer is partial
        # when ``num_bunnies`` isn't a perfect cube.
        #
        # The bunny mesh's scaled bounding box is roughly 0.32 m x
        # 0.32 m at :data:`BUNNY_SCALE` = 0.2, with the mesh origin
        # ~12 cm from the geometric centre; a random tilt of up to
        # 0.4 rad sweeps a vertex up to ~0.36 m from the transform
        # origin. Pick spacings above those bounds so no two bunnies
        # spawn intersecting and the bottom layer clears the plane.
        horizontal_spacing = 0.75  # > 2 * max_radius
        vertical_spacing = 0.8  # > 2 * max_radius
        base_z = 0.45  # > max_radius so lowest tilt doesn't dip below z = 0
        rng = np.random.default_rng(seed=42)
        side = max(1, int(np.ceil(self.num_bunnies ** (1.0 / 3.0))))
        self._grid_side = side
        bunny_ids: list[int] = []
        for i in range(self.num_bunnies):
            layer = i // (side * side)
            row = (i // side) % side
            col = i % side
            # Alternate-row half-step stagger so adjacent layers
            # interlock instead of stacking columns -- bunnies slot
            # into each other's gaps as they settle, which produces a
            # tighter pile and lets the SDF narrow phase see more
            # body-pair combinations.
            stagger = 0.5 * horizontal_spacing if (layer % 2) else 0.0
            x = (col - 0.5 * (side - 1)) * horizontal_spacing + stagger
            y = (row - 0.5 * (side - 1)) * horizontal_spacing + stagger
            z = base_z + layer * vertical_spacing
            yaw = float(rng.uniform(-np.pi, np.pi))
            tilt_axis = wp.vec3(*rng.normal(size=3).astype(np.float32))
            tilt_axis = tilt_axis / (float(np.linalg.norm([*tilt_axis])) + 1e-9)
            tilt = wp.quat_from_axis_angle(tilt_axis, float(rng.uniform(-0.4, 0.4)))
            yaw_q = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
            q = yaw_q * tilt
            body = mb.add_body(
                xform=wp.transform(wp.vec3(x, y, z), q),
                label=f"bunny_{i}",
            )
            mb.add_shape_mesh(
                body,
                mesh=bunny_mesh,
                scale=(BUNNY_SCALE, BUNNY_SCALE, BUNNY_SCALE),
                cfg=newton.ModelBuilder.ShapeConfig(
                    density=1000.0,
                    mu=BUNNY_FRICTION,
                    gap=DEFAULT_SHAPE_GAP,
                ),
            )
            bunny_ids.append(body)

        self._bunny_ids = bunny_ids
        self.model = mb.finalize()

        # ---- Collision pipeline ---------------------------------------
        self.collision_pipeline = newton.CollisionPipeline(self.model, contact_matching="sticky")
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.model.body_q.assign(self.state.body_q)

        # ---- PhoenX body container (slot 0 = static world anchor) ----
        num_phoenx_bodies = int(self.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        bodies.orientation.assign(np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32))
        wp.launch(
            init_phoenx_bodies_kernel,
            dim=self.model.body_count,
            inputs=[
                self.model.body_q,
                self.state.body_qd,
                self.model.body_com,
                self.model.body_inv_mass,
                self.model.body_inv_inertia,
            ],
            outputs=[
                bodies.position,
                bodies.orientation,
                bodies.velocity,
                bodies.angular_velocity,
                bodies.inverse_mass,
                bodies.inverse_inertia,
                bodies.inverse_inertia_world,
                bodies.motion_type,
                bodies.body_com,
            ],
            device=self.device,
        )
        self.bodies = bodies

        # ---- Contact-only constraint container ------------------------
        # The bunny pile is the largest contact pool we ship; reserve
        # room for every narrow-phase point to land in its own column
        # in the worst case so the ingest never stalls on a full
        # column-table.
        max_contact_columns = max(64, (rigid_contact_max + 5) // 6)
        self.constraints = constraint_container_zeros(
            num_constraints=max_contact_columns,
            num_dwords=CONTACT_DWORDS,
            device=self.device,
        )

        shape_body_np = self.model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=self.device)

        # ---- Solver ---------------------------------------------------
        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self.constraints,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=1,
            gravity=(0.0, 0.0, -9.81),
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            default_friction=BUNNY_FRICTION,
            device=self.device,
        )

        # ---- Viewer ---------------------------------------------------
        self._xforms = wp.zeros(num_phoenx_bodies, dtype=wp.transform, device=self.device)
        self.viewer.set_model(self.model)
        # Pull the camera back in proportion to the grid so the whole
        # pile stays in frame regardless of ``--num-bunnies``.
        cam_dist = max(1.2, 0.8 * side)
        cam_height = max(0.8, 0.6 * side)
        self.viewer.set_camera(
            pos=wp.vec3(cam_dist, -cam_dist, cam_height),
            pitch=-20.0,
            yaw=135.0,
        )

        # ---- Picking --------------------------------------------------
        half_extents_np = np.full((num_phoenx_bodies, 3), BUNNY_SCALE * 0.5, dtype=np.float32)
        half_extents_np[0] = 0.0  # world anchor
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        self.graph = None
        self.capture()

    def capture(self) -> None:
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self) -> None:
        self._sync_newton_to_phoenx()
        self.model.collide(
            self.state,
            contacts=self.contacts,
            collision_pipeline=self.collision_pipeline,
        )
        self.picking.apply_force()
        self.world.step(
            dt=self.frame_dt,
            contacts=self.contacts,
            shape_body=self._shape_body,
        )
        self._sync_phoenx_to_newton()

    def _sync_newton_to_phoenx(self) -> None:
        n = self.model.body_count
        wp.launch(
            newton_to_phoenx_kernel,
            dim=n,
            inputs=[self.state.body_q, self.state.body_qd, self.model.body_com],
            outputs=[
                self.bodies.position[1 : 1 + n],
                self.bodies.orientation[1 : 1 + n],
                self.bodies.velocity[1 : 1 + n],
                self.bodies.angular_velocity[1 : 1 + n],
            ],
            device=self.device,
        )

    def _sync_phoenx_to_newton(self) -> None:
        n = self.model.body_count
        wp.launch(
            phoenx_to_newton_kernel,
            dim=n,
            inputs=[
                self.bodies.position[1 : 1 + n],
                self.bodies.orientation[1 : 1 + n],
                self.bodies.velocity[1 : 1 + n],
                self.bodies.angular_velocity[1 : 1 + n],
                self.model.body_com,
            ],
            outputs=[self.state.body_q, self.state.body_qd],
            device=self.device,
        )

    def step(self) -> None:
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._frame += 1

    def render(self) -> None:
        wp.launch(
            pack_body_xforms_kernel,
            dim=self.world.num_bodies,
            inputs=[self.bodies, self._xforms],
            device=self.device,
        )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.end_frame()

    def test_final(self) -> None:
        """Every bunny has finite state and stayed roughly over the
        pile footprint. Tolerance scales with the grid side -- the
        spawn extent is ``(side - 1) * 0.75 m`` across plus a
        half-step stagger and some settling spread."""
        spawn_half_extent = 0.5 * (self._grid_side - 1) * 0.75 + 0.5 * 0.75
        xy_tol = max(2.0, spawn_half_extent + 1.5)
        positions = self.bodies.position.numpy()
        velocities = self.bodies.velocity.numpy()
        for i, body in enumerate(self._bunny_ids):
            slot = body + 1
            pos = positions[slot]
            vel = velocities[slot]
            assert np.isfinite(pos).all(), f"bunny {i} pos non-finite ({pos})"
            assert np.isfinite(vel).all(), f"bunny {i} vel non-finite ({vel})"
            xy = float(np.linalg.norm(pos[:2]))
            assert xy < xy_tol, f"bunny {i} flew off the pile: xy={xy:.3f} m, tol={xy_tol:.3f}, pos={pos}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--num-bunnies",
            type=int,
            default=100,
            help=(
                "Number of bunnies in the pile. Each bunny has its "
                "own SDF; more bunnies = more contacts. 24 takes "
                "~25 FPS on an RTX A6000; scale up for stress testing."
            ),
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
