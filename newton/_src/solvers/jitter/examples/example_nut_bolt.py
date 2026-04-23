# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Jitter Nut & Bolt
#
# A single SDF nut dropped onto a single SDF bolt, simulated with the
# Jitter solver. This is the canonical "one pair, many contact points"
# stress case for a rigid-body contact solver: the subdivided nut/bolt
# mesh pair produces dozens of manifold contacts per frame, which the
# Jitter contact ingest splits into multiple 6-slot columns.
#
# Mirrors the MuJoCo/XPBD ``newton.examples.contacts.example_nut_bolt_sdf``
# but routes contacts through the Jitter solver via the same
# ``CollisionPipeline`` + ``build_jitter_world_from_model`` plumbing
# the pyramid / stack demos use.
#
# Requires CUDA (SDF narrow phase is CUDA-only), the ``trimesh``
# package (for loading the mesh files), and the IsaacGymEnvs asset
# cache (auto-downloaded on first run via
# ``newton.examples.download_external_git_folder``).
#
# Run:  python -m newton._src.solvers.jitter.examples.example_nut_bolt
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.jitter.constraints.contact_matching_config import JITTER_CONTACT_MATCHING
from newton._src.solvers.jitter.examples.example_jitter_common import (
    build_jitter_world_from_model,
    jitter_to_newton_kernel,
    newton_to_jitter_kernel,
)
from newton._src.solvers.jitter.picking import JitterPicking, register_with_viewer_gl
from newton._src.solvers.jitter.solver_jitter import pack_body_xforms_kernel

# Assembly + asset source (matches newton.examples.contacts.example_nut_bolt_sdf).
ASSEMBLY_STR = "m20_loose"
ISAACGYM_ENVS_REPO_URL = "https://github.com/isaac-sim/IsaacGymEnvs.git"
ISAACGYM_NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"

# Shape config. Matches the per-shape tuning used by the upstream
# nut/bolt example so Jitter sees the same contact / friction settings
# the other solvers do; the ke/kd stiffness terms are no-ops for
# Jitter (they're a MuJoCo/SemiImplicit penalty-solver knob).
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
    an SDF inside the configured narrow band. Returns the
    :class:`newton.Mesh` and the pre-centering world offset so the
    caller can compensate by shifting the body origin.
    """
    import trimesh  # noqa: PLC0415  -- deferred: heavy dep, only needed on call.

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
    def __init__(self, viewer, args):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 5
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.solver_iterations = 16
        self._frame: int = 0

        self.viewer = viewer
        self.device = wp.get_device()
        if not self.device.is_cuda:
            raise RuntimeError(
                "example_nut_bolt requires CUDA (SDF narrow phase is CUDA-only)."
            )

        # ---- Fetch the nut/bolt meshes --------------------------------
        print("Downloading nut/bolt assets...")
        asset_path = newton.examples.download_external_git_folder(
            ISAACGYM_ENVS_REPO_URL, ISAACGYM_NUT_BOLT_FOLDER
        )
        print(f"Assets downloaded to: {asset_path}")

        bolt_file = str(asset_path / f"factory_bolt_{ASSEMBLY_STR}.obj")
        nut_file = str(asset_path / f"factory_nut_{ASSEMBLY_STR}_subdiv_3x.obj")
        bolt_mesh, bolt_center = _load_mesh_with_sdf(bolt_file, gap=SHAPE_CFG.gap)
        nut_mesh, nut_center = _load_mesh_with_sdf(nut_file, gap=SHAPE_CFG.gap)

        # ---- Build the Newton scene: bolt (static) + nut (dynamic) ----
        mb = newton.ModelBuilder()
        mb.default_shape_cfg.gap = SHAPE_CFG.gap

        # Static bolt. We set zero mass via a dummy inertia here and
        # Newton's finalize infers a static body from the resulting
        # non-positive inverse mass. Shape origin is shifted by
        # ``bolt_center`` so the SDF (centered at its local origin)
        # aligns with the world origin.
        bolt_body = mb.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            label="bolt",
        )
        mb.add_shape_mesh(
            bolt_body,
            mesh=bolt_mesh,
            xform=wp.transform(bolt_center, wp.quat_identity()),
            scale=(1.0, 1.0, 1.0),
            cfg=SHAPE_CFG,
        )

        # Dynamic nut spawned 4 cm above the bolt head. Small initial
        # rotation (pi/8 around Z) so the nut's threads land with an
        # offset relative to the bolt -- otherwise the two SDFs would
        # line up perfectly and slide down a single groove.
        nut_xform = wp.transform(
            wp.vec3(0.0, 0.0, 0.041),
            wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 8),
        )
        nut_body = mb.add_body(xform=nut_xform, label="nut")
        mb.add_shape_mesh(
            nut_body,
            mesh=nut_mesh,
            xform=wp.transform(nut_center, wp.quat_identity()),
            scale=(1.0, 1.0, 1.0),
            cfg=SHAPE_CFG,
        )
        self._bolt_body = bolt_body
        self._nut_body = nut_body

        self.model = mb.finalize()

        # Pin the bolt by zeroing its inverse mass + inertia. Newton's
        # finalize picks the body up as dynamic by default (non-zero
        # mass from the mesh + density), but the upstream example
        # pattern "bolt static, nut dynamic" is the physical setup
        # users expect -- otherwise the bolt free-falls alongside the
        # nut. We do this in-place after finalize so the mesh density
        # still contributes to the nut's inertia computation.
        body_inv_mass_np = self.model.body_inv_mass.numpy()
        body_inv_inertia_np = self.model.body_inv_inertia.numpy()
        body_inv_mass_np[bolt_body] = 0.0
        body_inv_inertia_np[bolt_body] = np.zeros((3, 3), dtype=np.float32)
        self.model.body_inv_mass.assign(wp.array(body_inv_mass_np, dtype=wp.float32))
        self.model.body_inv_inertia.assign(
            wp.array(body_inv_inertia_np, dtype=wp.mat33)
        )

        print(f"nut_bolt: bodies={self.model.body_count} shapes={self.model.shape_count}")

        # ---- Collision pipeline ---------------------------------------
        self.collision_pipeline = newton.CollisionPipeline(
            self.model, contact_matching=JITTER_CONTACT_MATCHING
        )
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.model.body_q.assign(self.state.body_q)

        # ---- Build the Jitter world mirroring Newton's body set -------
        builder, newton_to_jitter = build_jitter_world_from_model(self.model)
        # The nut/bolt manifold can generate tens of contacts per
        # frame; 64 columns is well above the observed peak and avoids
        # truncation when the SDF narrow phase reports a dense frame.
        max_contact_columns = max(64, (rigid_contact_max + 5) // 6)
        self.world = builder.finalize(
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            gravity=(0.0, 0.0, -9.81),
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=int(self.model.shape_count),
            default_friction=SHAPE_CFG.mu,
            device=self.device,
        )
        self._newton_to_jitter = newton_to_jitter

        shape_body_np = self.model.shape_body.numpy()
        shape_body_jitter = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(
            shape_body_jitter, dtype=wp.int32, device=self.device
        )

        self._sync_newton_to_jitter()

        # ---- Rendering scratch + viewer -------------------------------
        self._xforms = wp.zeros(
            self.world.num_bodies, dtype=wp.transform, device=self.device
        )
        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(0.25, -0.25, 0.15),
            pitch=-25.0,
            yaw=135.0,
        )

        # ---- Picking -------------------------------------------------
        # Picking bounds: rough mesh AABB half-extents so the viewer's
        # ray-cast can grab either body. Index 0 is the static Jitter
        # world anchor (zero extents so rays ignore it).
        half_extents_np = np.zeros((self.world.num_bodies, 3), dtype=np.float32)
        bolt_j = newton_to_jitter[bolt_body]
        nut_j = newton_to_jitter[nut_body]
        half_extents_np[bolt_j] = (0.02, 0.02, 0.04)
        half_extents_np[nut_j] = (0.025, 0.025, 0.015)
        self._half_extents = wp.array(
            half_extents_np, dtype=wp.vec3f, device=self.device
        )
        self.picking = JitterPicking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        # Initial nut position for the ``test_final`` bounding check.
        self._nut_initial_xy = (0.0, 0.0)

        self.graph = None
        self.capture()

    def capture(self) -> None:
        """Record a CUDA graph for the entire per-frame simulate path."""
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self) -> None:
        """One rendered frame: sync state, collide, sub-step, sync back."""
        self._sync_newton_to_jitter()
        self.model.collide(
            self.state,
            contacts=self.contacts,
            collision_pipeline=self.collision_pipeline,
        )
        self.world.step(
            dt=self.frame_dt,
            contacts=self.contacts,
            shape_body=self._shape_body,
            picking=self.picking,
        )
        self._sync_jitter_to_newton()

    def _sync_newton_to_jitter(self) -> None:
        n = self.model.body_count
        wp.launch(
            newton_to_jitter_kernel,
            dim=n,
            inputs=[self.state.body_q, self.state.body_qd, self.model.body_com],
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
            jitter_to_newton_kernel,
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
            inputs=[self.world.bodies, self._xforms],
            device=self.device,
        )
        self.viewer.begin_frame(self.sim_time)
        # Draw through the Newton state so the viewer can render the
        # actual mesh geometry, not a primitive stand-in.
        self.viewer.log_state(self.state)
        self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.end_frame()

    # ----------------------------------------------------------------
    # Test
    # ----------------------------------------------------------------

    def test_final(self) -> None:
        """After the settle run the nut must still be near the bolt
        axis (no off-axis fly-off) and both bodies must have finite
        state. Rotation is expected -- the nut threads onto the bolt
        -- so we don't assert on orientation change.
        """
        positions = self.world.bodies.position.numpy()
        velocities = self.world.bodies.velocity.numpy()
        nut_j = self._newton_to_jitter[self._nut_body]
        nut_pos = positions[nut_j]
        nut_vel = velocities[nut_j]
        assert np.isfinite(nut_pos).all(), f"nut position non-finite ({nut_pos})"
        assert np.isfinite(nut_vel).all(), f"nut velocity non-finite ({nut_vel})"
        nut_xy = np.asarray(nut_pos[:2], dtype=np.float32)
        nut_xy_dist = float(np.linalg.norm(nut_xy - np.asarray(self._nut_initial_xy)))
        # Loose bound: the nut can precess slightly around the bolt
        # axis while settling, but a catastrophic friction / manifold
        # failure sends it flying on the order of tens of cm.
        assert nut_xy_dist < 0.1, (
            f"nut flew off-axis: xy_dist={nut_xy_dist:.4f} m "
            f"(pos={tuple(float(x) for x in nut_pos)})"
        )

    @staticmethod
    def create_parser():
        return newton.examples.create_parser()


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
