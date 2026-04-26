# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Nut & Bolt
#
# PhoenX variant of :mod:`example_nut_bolt`. Same scene (an SDF nut
# dropped onto a fixed SDF bolt) and the same ``CollisionPipeline``
# driven contact ingest, but drives the simulation through
# :class:`PhoenXWorld` instead of the jitter ``World``. The two
# examples produce visually similar threading behaviour, so this is
# also a side-by-side validation of the PhoenX contact path on a
# dense-manifold (multi-column) scene.
#
# Requires CUDA (SDF narrow phase is CUDA-only), ``trimesh``, and
# the IsaacGymEnvs asset cache (auto-downloaded on first run).
#
# Run:  python -m newton._src.solvers.phoenx.examples.example_nut_bolt
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros

# Contact matching mode. The shared jitter/phoenx default is
# ``"sticky"`` -- it pins each matched contact's body-frame anchors
# and world-frame normal to their first-frame values, which kills
# stack jitter but also freezes the contact normal. On the nut-bolt
# scene, the first frame's contacts land on the bolt head's flat
# top (vertical normal), so sticky mode keeps that vertical normal
# even once the nut should be threading -- the normal-row impulse
# ends up pure-vertical and produces no torque about +Z.
# ``"latest"`` keeps matching (so warm-start still works) but uses
# each frame's fresh narrow-phase normal, which tilts along the
# helical SDF surface and naturally rotates the nut. XPBD and
# MuJoCo's solver don't go through this matching layer at all, so
# this is a PhoenX-specific sticky-vs-fresh tradeoff.
_CONTACT_MATCHING = "latest"
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
    PhoenXWorld,
    pack_body_xforms_kernel,
)

# Assembly + asset source -- identical to the jitter / XPBD / MuJoCo
# nut-bolt examples so the same mesh pair is compared across solvers.
ASSEMBLY_STR = "m20_loose"
ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"

# Shape config. Matches the XPBD / MuJoCo nut-bolt knobs verbatim
# so the three solvers can be compared side-by-side on the same
# scene. The ``ke`` / ``kd`` stiffness terms are no-ops for jitter
# / PhoenX (they're a MuJoCo/SemiImplicit penalty-solver knob).
#
# ``mu = 0.01`` is *deliberate*: for a rigid nut on a rigid helical
# bolt SDF, the contact normals are purely radial + axial (they
# have no component in the tangent-around-axis direction), so the
# *normal impulse alone* produces zero torque about +Z. The nut
# rotates because Coulomb friction opposes the axial slip along
# the helix surface -- and the helix's tangent direction has a
# rotational component, so the friction force has one too. Set
# ``mu = 0`` and the nut sinks straight down the bolt without
# threading; this is physically correct.
SHAPE_CFG = newton.ModelBuilder.ShapeConfig(
    margin=0.0,
    mu=0.01,
    ke=1e7,
    kd=1e4,
    gap=0.005,
    density=8000.0,
    mu_torsional=0.0,
    mu_rolling=0.0,
    is_hydroelastic=False,
)

MESH_SDF_MAX_RESOLUTION = 512
MESH_SDF_NARROW_BAND_RANGE = (-0.005, 0.005)


def _load_mesh_with_sdf(mesh_file: str, gap: float) -> tuple[newton.Mesh, wp.vec3]:
    """Load a trimesh, center its bounding box at the origin, and bake
    an SDF inside the configured narrow band.

    The SDF is baked from unscaled mesh vertices; any ``scene_scale``
    the caller applies ends up as a runtime shape scale in
    :meth:`add_shape_mesh`, and the runtime narrow-phase rescales SDF
    distances / gradients by ``min_scale`` on each query. Keeping the
    SDF in mesh units preserves the asset's authored resolution
    regardless of ``scene_scale``.

    Returns the :class:`newton.Mesh` and the pre-centering world
    offset so the caller can compensate by shifting the shape origin
    (the SDF is generated around the mesh's local origin, not its
    original AABB centre). ``center_vec`` is in mesh (unscaled) units.
    """
    import trimesh

    mesh_data = trimesh.load(mesh_file, force="mesh")
    vertices = np.array(mesh_data.vertices, dtype=np.float32)
    indices = np.array(mesh_data.faces.flatten(), dtype=np.int32)
    min_extent = vertices.min(axis=0)
    max_extent = vertices.max(axis=0)
    center = (min_extent + max_extent) / 2
    vertices = vertices - center
    center_vec = wp.vec3(float(center[0]), float(center[1]), float(center[2]))

    mesh = newton.Mesh(vertices, indices)
    mesh.build_sdf(
        max_resolution=MESH_SDF_MAX_RESOLUTION,
        narrow_band_range=MESH_SDF_NARROW_BAND_RANGE,
        margin=gap,
    )
    return mesh, center_vec


class Example:
    """Drop-an-SDF-nut-onto-an-SDF-bolt scene for :class:`PhoenXWorld`.

    Pipeline per frame mirrors the tower example -- sync state,
    run Newton's :class:`CollisionPipeline`, apply picking force,
    call :meth:`PhoenXWorld.step`, sync back -- so the same
    CUDA-graph capture pattern works.
    """

    def __init__(self, viewer, args):
        # Timing + iteration budget matches the XPBD nut-bolt scene in
        # :mod:`newton.examples.contacts.example_nut_bolt_sdf` so the
        # two solvers can be compared head-to-head: fps=120, 5
        # substeps, 10 solver iterations per substep. XPBD also runs
        # with ``angular_damping=0`` (no rotational drag); PhoenX
        # keeps damping multipliers at 1.0 by default
        # (``body_container_zeros``), so neither solver introduces
        # any rotational damping on this scene.
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 5
        self.solver_iterations = 10
        self._frame: int = 0

        # Scene-wide geometric scale. ``1.0`` matches the M20 bolt
        # assembly from the IsaacGymEnvs asset pack verbatim. Setting
        # this to e.g. ``10.0`` produces an oversized nut/bolt pair in
        # the same physical layout -- useful for exercising the solver
        # at non-default length scales (contact thicknesses, SDF
        # narrow band, picking AABB, camera distance all track).
        # Matches the ``scene_scale`` knob in
        # :mod:`newton.examples.contacts.example_nut_bolt_sdf`.
        self.scene_scale = float(getattr(args, "scene_scale", 1.0))

        self.viewer = viewer
        self.device = wp.get_device()
        if not self.device.is_cuda:
            raise RuntimeError("example_nut_bolt requires CUDA (SDF narrow phase is CUDA-only).")

        # ---- Fetch the nut/bolt meshes --------------------------------
        print("Downloading nut/bolt assets...")
        asset_path = newton.examples.download_external_git_folder(ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER)
        print(f"Assets downloaded to: {asset_path}")

        bolt_file = str(asset_path / f"factory_bolt_{ASSEMBLY_STR}.obj")
        nut_file = str(asset_path / f"factory_nut_{ASSEMBLY_STR}_subdiv_3x.obj")
        # SDF / gap stay in mesh (unscaled) units; the shape scale on
        # ``add_shape_mesh`` applies uniformly at contact-query time.
        bolt_mesh, bolt_center = _load_mesh_with_sdf(bolt_file, gap=SHAPE_CFG.gap)
        nut_mesh, nut_center = _load_mesh_with_sdf(nut_file, gap=SHAPE_CFG.gap)
        shape_cfg = SHAPE_CFG

        # ---- Build the Newton scene: bolt (static) + nut (dynamic) ----
        mb = newton.ModelBuilder()
        mb.default_shape_cfg.gap = shape_cfg.gap

        # Static bolt. The mesh is kept in mesh-local (unscaled)
        # units; the shape scale scales it up at runtime. The shape
        # xform places the scaled mesh back where the asset author
        # had it, which means the body-frame offset is
        # ``scale * bolt_center``.
        bolt_body = mb.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            label="bolt",
        )
        scaled_bolt_center = wp.vec3(
            bolt_center[0] * self.scene_scale,
            bolt_center[1] * self.scene_scale,
            bolt_center[2] * self.scene_scale,
        )
        mb.add_shape_mesh(
            bolt_body,
            mesh=bolt_mesh,
            xform=wp.transform(scaled_bolt_center, wp.quat_identity()),
            scale=(self.scene_scale, self.scene_scale, self.scene_scale),
            cfg=shape_cfg,
        )

        # Dynamic nut, 4 cm above the bolt head with a pi/8 rotation
        # about +Z so its threads are phase-offset from the bolt's
        # (without this the two SDFs line up perfectly and the nut
        # slides straight down a single groove).
        nut_xform = wp.transform(
            wp.vec3(0.0, 0.0, 0.041 * self.scene_scale),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 8),
        )
        nut_body = mb.add_body(xform=nut_xform, label="nut")
        scaled_nut_center = wp.vec3(
            nut_center[0] * self.scene_scale,
            nut_center[1] * self.scene_scale,
            nut_center[2] * self.scene_scale,
        )
        mb.add_shape_mesh(
            nut_body,
            mesh=nut_mesh,
            xform=wp.transform(scaled_nut_center, wp.quat_identity()),
            scale=(self.scene_scale, self.scene_scale, self.scene_scale),
            cfg=shape_cfg,
        )
        self._bolt_body = bolt_body
        self._nut_body = nut_body

        self.model = mb.finalize()

        # Pin the bolt: zero its inverse mass + inertia so
        # :func:`init_phoenx_bodies_kernel` marks it
        # :data:`MOTION_STATIC`. The nut's density-derived mass is
        # unaffected because the mesh inertia computation already ran
        # during ``finalize``.
        body_inv_mass_np = self.model.body_inv_mass.numpy()
        body_inv_inertia_np = self.model.body_inv_inertia.numpy()
        body_inv_mass_np[bolt_body] = 0.0
        body_inv_inertia_np[bolt_body] = np.zeros((3, 3), dtype=np.float32)
        self.model.body_inv_mass.assign(wp.array(body_inv_mass_np, dtype=wp.float32))
        self.model.body_inv_inertia.assign(wp.array(body_inv_inertia_np, dtype=wp.mat33))

        print(f"[PhoenX Nut-Bolt] bodies={self.model.body_count} shapes={self.model.shape_count}")

        # ---- Collision pipeline ---------------------------------------
        self.collision_pipeline = newton.CollisionPipeline(self.model, contact_matching=_CONTACT_MATCHING)
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.model.body_q.assign(self.state.body_q)

        # ---- PhoenX body container (slot 0 = static world anchor) ----
        num_phoenx_bodies = int(self.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        # Seed every slot's orientation to identity so the rotation-
        # to-matrix call in :func:`_phoenx_update_inertia_and_clear_forces_kernel`
        # doesn't blow up on the zero-quaternion default.
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

        # ---- Joint-only constraint container ---------------------------
        # Contact column storage lives in :class:`ContactColumnContainer`;
        # the joint-side constraint container only needs a 1-row
        # placeholder in this contact-only scene.
        self.constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
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
            rigid_contact_max=rigid_contact_max,
            default_friction=SHAPE_CFG.mu,
            device=self.device,
        )

        # ---- Viewer ---------------------------------------------------
        self._xforms = wp.zeros(num_phoenx_bodies, dtype=wp.transform, device=self.device)
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(
                0.25 * self.scene_scale,
                -0.25 * self.scene_scale,
                0.15 * self.scene_scale,
            ),
            pitch=-25.0,
            yaw=135.0,
        )

        # ---- Picking --------------------------------------------------
        # Half-extents for the ray cast; slot 0 (world anchor) stays
        # at zero so rays ignore it. The bolt and nut use loose AABB
        # half-extents so the picking ray latches onto either body.
        # The AABBs scale with the geometry so the ray still hits at
        # non-default scene scales.
        half_extents_np = np.zeros((num_phoenx_bodies, 3), dtype=np.float32)
        half_extents_np[bolt_body + 1] = (
            0.02 * self.scene_scale,
            0.02 * self.scene_scale,
            0.04 * self.scene_scale,
        )
        half_extents_np[nut_body + 1] = (
            0.025 * self.scene_scale,
            0.025 * self.scene_scale,
            0.015 * self.scene_scale,
        )
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        # Initial nut XY for :meth:`test_final`.
        self._nut_initial_xy = (0.0, 0.0)

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
        # Picking force is accumulated into bodies.force once per
        # frame -- :class:`PhoenXWorld`'s per-substep force kernel
        # consumes it and the post-step ``_clear_forces`` zeros it
        # for the next frame.
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
        """After the settle run the nut must still be near the bolt
        axis (no off-axis fly-off) and both bodies must have finite
        state. The nut may have rotated significantly (threading) so
        we don't assert on orientation.
        """
        positions = self.bodies.position.numpy()
        velocities = self.bodies.velocity.numpy()
        nut_slot = self._nut_body + 1
        nut_pos = positions[nut_slot]
        nut_vel = velocities[nut_slot]
        assert np.isfinite(nut_pos).all(), f"nut position non-finite ({nut_pos})"
        assert np.isfinite(nut_vel).all(), f"nut velocity non-finite ({nut_vel})"
        nut_xy_dist = float(
            np.linalg.norm(
                np.asarray(nut_pos[:2], dtype=np.float32) - np.asarray(self._nut_initial_xy, dtype=np.float32)
            )
        )
        max_drift = 0.1 * self.scene_scale
        assert nut_xy_dist < max_drift, (
            f"nut flew off-axis: xy_dist={nut_xy_dist:.4f} m "
            f"(max={max_drift:.4f} m at scene_scale={self.scene_scale}; "
            f"pos={tuple(float(x) for x in nut_pos)})"
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--scene-scale",
            type=float,
            default=1.0,
            help=(
                "Uniform geometric scale applied to the nut/bolt assembly "
                "(positions, mesh scale, SDF narrow band, contact gap, "
                "picking AABB, and camera). 1.0 matches the original "
                "M20 asset size; larger / smaller values produce a "
                "geometrically similar scene at a different length scale."
            ),
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
