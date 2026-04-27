# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Kapla Tower
#
# Port of PhoenX's ``Demo15: Kapla Tower`` (see
# ``PhoenX/src/Viewer/Demos/Demo15.cs``). The C# demo loads
# ``KaplaTower2.usda`` -- a USD ``PointInstancer`` describing
# ~11k Kapla-style wooden bricks arranged into a cylindrical tower --
# and lets PhoenX settle them onto the ground.
#
# We don't depend on USD here: the brick prototype's full extents and
# every instance's position / orientation have been extracted into
# :mod:`kapla_tower_data` (regenerate with
# ``python _extract_kapla_usda.py`` from the repo root).
#
# The USDA is +Z-up (``upAxis = "Z"``); PhoenX is +Y-up so the C#
# demo rotates the whole instancer by ``-pi/2`` about +X. Newton is
# also +Z-up, so we keep positions and orientations as-is. The
# overall scene is uniformly scaled by :data:`GLOBAL_SCALING` to
# match the C# demo's metric size.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_kapla_tower
###########################################################################

from __future__ import annotations

import os
import pathlib

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import MOTION_KINEMATIC, body_container_zeros
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel as _init_phoenx_bodies_kernel,
)
from newton._src.solvers.phoenx.examples.example_common import (
    newton_to_phoenx_kernel as _newton_to_phoenx_kernel,
)
from newton._src.solvers.phoenx.examples.example_common import (
    phoenx_to_newton_kernel as _phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.examples.kapla_tower_data import (
    BRICK_FULL_EXTENTS,
    NUM_BRICKS,
    ORIENTATIONS,
    POSITIONS,
)
from newton._src.solvers.phoenx.picking import (
    Picking,
    register_with_viewer_gl,
)
from newton._src.solvers.phoenx.solver_config import (
    PHOENX_CONTACT_MATCHING,
)
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld


# Patch ``state.body_q`` for the camera collider so Newton's
# CollisionPipeline (which broad/narrow-phases against ``body_q``)
# tracks the live camera, not the spawn position.
@wp.kernel(enable_backward=False)
def _write_camera_body_q_kernel(
    body_id: wp.int32,
    pos: wp.array[wp.vec3f],
    orient: wp.array[wp.quatf],
    body_q: wp.array[wp.transform],
):
    body_q[body_id] = wp.transform(pos[0], orient[0])


# Mirrors C# ``Demo15.cs``: globalScaling 0.01 there, 0.1 here for a
# ~70 cm tabletop tower; ground sits at 0.35 * scale.
GLOBAL_SCALING: float = 0.1
BRICK_DENSITY: float = 1000.0
GROUND_HEIGHT: float = 0.35 * GLOBAL_SCALING

# Single-world layout wins on the dense ~11k-body contact pool.
USE_BIG_WORLD_MODE: bool = True
STEP_LAYOUT: str = "single_world" if USE_BIG_WORLD_MODE else "multi_world"

# Tile the single ``KaplaTower2.usda`` instancer into a 2D grid centred
# on the origin. ``(1, 1)`` reproduces the original scene; bigger
# grids scale brick count, SAP candidates and contact pool linearly,
# which the collision-pipeline budgets below absorb via ``Nx * Ny``.
TOWER_GRID_DIMS: tuple[int, int] = (2, 2)
# Centre-to-centre spacing [m]. Tower footprint ~7.1 x 5.0 m at
# scale 0.1; 9 m leaves ~2 m clearance so neighbours don't leak into
# each other's SAP lists during settling.
TOWER_GRID_SPACING: tuple[float, float] = (9.0, 9.0)

# Invisible kinematic sphere parented to the camera. PhoenX's
# inferred-velocity path on :data:`MOTION_KINEMATIC` bodies turns
# fly-throughs into impulses on the bricks. Sphere (rotation-
# invariant) so we don't need to track camera orientation.
CAMERA_COLLIDER_ENABLED: bool = True
CAMERA_COLLIDER_RADIUS: float = 0.4
# Newton-side mass density only; PhoenX-side inverse mass is zeroed
# right after init so the value just needs to be finite.
CAMERA_COLLIDER_DENSITY: float = 1000.0


class Example:
    """PhoenX Kapla Tower -- port of ``Demo15`` from PhoenXDemo.

    Per-frame pipeline (captured into a CUDA graph): sync Newton
    state -> PhoenX, run :class:`CollisionPipeline`, step
    :class:`PhoenXWorld`, sync back.
    """

    # Frames 0..WARMUP_FRAMES-1 pin global damping to 1.0 so the
    # USD-extracted brick poses (which carry mm-scale overlaps where
    # planks meet) don't kick the tower into divergence on frame 1.
    WARMUP_FRAMES: int = 20

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.frame_index: int = 0
        self.sim_substeps = 4
        self.solver_iterations = 5
        self.velocity_iterations = 1

        self._build_scene()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        builder = newton.ModelBuilder()
        builder.add_ground_plane(height=GROUND_HEIGHT)

        hx = 0.5 * GLOBAL_SCALING * BRICK_FULL_EXTENTS[0]
        hy = 0.5 * GLOBAL_SCALING * BRICK_FULL_EXTENTS[1]
        hz = 0.5 * GLOBAL_SCALING * BRICK_FULL_EXTENTS[2]

        # USDA stores quats as half-precision; renormalise so Newton
        # gets unit quaternions in ``body_q``.
        positions = (POSITIONS * GLOBAL_SCALING).astype(np.float32)
        quats = ORIENTATIONS.astype(np.float32)
        norms = np.linalg.norm(quats, axis=1, keepdims=True)
        quats = quats / np.maximum(norms, 1e-12)

        # 1 cm AABB margin. Newton default (5 cm) is ~4x bigger than
        # the bricks and would explode the SAP candidate count.
        cfg = newton.ModelBuilder.ShapeConfig(density=BRICK_DENSITY, gap=0.01)

        nx, ny = (int(d) for d in TOWER_GRID_DIMS)
        if nx < 1 or ny < 1:
            raise ValueError(f"TOWER_GRID_DIMS components must be >= 1 (got {TOWER_GRID_DIMS})")
        spacing_x, spacing_y = (float(s) for s in TOWER_GRID_SPACING)
        self._tower_grid_dims: tuple[int, int] = (nx, ny)
        self._tower_grid_spacing: tuple[float, float] = (spacing_x, spacing_y)

        cell_centres_xy: list[tuple[float, float]] = []
        for j in range(ny):
            for i in range(nx):
                cx = (i - 0.5 * (nx - 1)) * spacing_x
                cy = (j - 0.5 * (ny - 1)) * spacing_y
                cell_centres_xy.append((cx, cy))
        self._cell_centres_xy = cell_centres_xy

        # Per-cell newton-id list (cell_index = j * nx + i, row-major).
        self._brick_newton_ids: list[list[int]] = [[] for _ in range(nx * ny)]
        for cell_index, (cx, cy) in enumerate(cell_centres_xy):
            for i in range(NUM_BRICKS):
                qx, qy, qz, qw = quats[i]
                px, py, pz = positions[i]
                body = builder.add_body(
                    xform=wp.transform(
                        p=wp.vec3(float(px) + cx, float(py) + cy, float(pz)),
                        q=wp.quat(float(qx), float(qy), float(qz), float(qw)),
                    ),
                )
                builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg)
                self._brick_newton_ids[cell_index].append(body)

        # Camera collider added as dynamic so finalize() accepts it,
        # then flipped to :data:`MOTION_KINEMATIC` and inverse-mass-
        # zeroed below.
        self._camera_body_newton_id: int | None = None
        # Must match the ``viewer.set_camera`` pos below so the first
        # frame doesn't slam the collider into the tower.
        self._camera_collider_initial_pos: tuple[float, float, float] = (1.2, 0.0, 0.4)
        if CAMERA_COLLIDER_ENABLED:
            cam_cfg = newton.ModelBuilder.ShapeConfig(
                density=CAMERA_COLLIDER_DENSITY,
                gap=0.01,
                is_visible=False,
            )
            cam_body = builder.add_body(
                xform=wp.transform(
                    p=wp.vec3(*(float(c) for c in self._camera_collider_initial_pos)),
                    q=wp.quat_identity(),
                ),
            )
            builder.add_shape_sphere(cam_body, radius=float(CAMERA_COLLIDER_RADIUS), cfg=cam_cfg)
            self._camera_body_newton_id = cam_body

        # SAP doesn't need the explicit O(N^2) pair list, and skipping
        # avoids minutes of Python work + tens of GB of speculative
        # pair allocation.
        self.model = builder.finalize(skip_shape_contact_pairs=True)
        total_tower_bricks = sum(len(ids) for ids in self._brick_newton_ids)
        print(
            f"[PhoenX KaplaTower] bodies={self.model.body_count} "
            f"shapes={self.model.shape_count} "
            f"tower_grid={nx}x{ny} "
            f"tower_bricks={total_tower_bricks} "
            f"camera_collider={'yes' if self._camera_body_newton_id is not None else 'no'} "
            f"brick_full_extents={BRICK_FULL_EXTENTS} scale={GLOBAL_SCALING}"
        )

        # SAP broad phase + warm-start contact_matching for stable
        # settling. ``shape_pairs_max`` / ``rigid_contact_max``
        # budgets sized from the observed single-tower numbers
        # (~800k SAP candidates, ~750k contacts) and scaled linearly
        # by the cell count -- replicated cells are spatially
        # disjoint so SAP drops cross-cell pairs.
        num_cells = nx * ny
        shape_pairs_max = 1_500_000 * num_cells
        rigid_contact_max_pipeline = 900_000 * num_cells
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching=PHOENX_CONTACT_MATCHING,
            broad_phase="sap",
            shape_pairs_max=shape_pairs_max,
            rigid_contact_max=rigid_contact_max_pipeline,
        )
        self.contacts = self.collision_pipeline.contacts()
        rigid_contact_max = int(self.contacts.rigid_contact_point0.shape[0])

        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.model.body_q.assign(self.state.body_q)

        # PhoenX body container -- slot 0 stays static world anchor.
        num_phoenx_bodies = int(self.model.body_count) + 1
        bodies = body_container_zeros(num_phoenx_bodies, device=self.device)
        wp.copy(
            bodies.orientation,
            wp.array(
                np.tile([0.0, 0.0, 0.0, 1.0], (num_phoenx_bodies, 1)).astype(np.float32),
                dtype=wp.quatf,
                device=self.device,
            ),
        )
        wp.launch(
            _init_phoenx_bodies_kernel,
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

        # Flip the camera collider to MOTION_KINEMATIC and zero its
        # inverse mass / inertia so the solver treats it as an
        # immovable but velocity-aware rail.
        if self._camera_body_newton_id is not None:
            slot = self._camera_body_newton_id + 1
            self._camera_phoenx_slot: int = slot
            mt_np = bodies.motion_type.numpy()
            mt_np[slot] = int(MOTION_KINEMATIC)
            bodies.motion_type.assign(mt_np)
            inv_m_np = bodies.inverse_mass.numpy()
            inv_m_np[slot] = 0.0
            bodies.inverse_mass.assign(inv_m_np)
            inv_I_np = bodies.inverse_inertia.numpy()
            inv_I_np[slot] = np.zeros((3, 3), dtype=np.float32)
            bodies.inverse_inertia.assign(inv_I_np)
            inv_Iw_np = bodies.inverse_inertia_world.numpy()
            inv_Iw_np[slot] = np.zeros((3, 3), dtype=np.float32)
            bodies.inverse_inertia_world.assign(inv_Iw_np)
        else:
            self._camera_phoenx_slot: int | None = None  # type: ignore[no-redef]

        self.bodies = bodies

        # Joint-side container only needs a placeholder row -- contact
        # columns live in PhoenXWorld's own ContactColumnContainer.
        self.constraints = PhoenXWorld.make_constraint_container(
            num_joints=0,
            device=self.device,
        )

        shape_body_np = self.model.shape_body.numpy()
        shape_body_phoenx = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        self._shape_body = wp.array(shape_body_phoenx, dtype=wp.int32, device=self.device)

        self.world = PhoenXWorld(
            bodies=self.bodies,
            constraints=self.constraints,
            substeps=self.sim_substeps,
            solver_iterations=self.solver_iterations,
            velocity_iterations=self.velocity_iterations,
            gravity=(0.0, 0.0, -9.81),
            rigid_contact_max=rigid_contact_max,
            step_layout=STEP_LAYOUT,
            max_thread_blocks=256,
            device=self.device,
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(
            pos=wp.vec3(1.2, 0.0, 0.4),
            pitch=-12.0,
            yaw=180.0,
        )

        # Picking AABBs: brick = box half-extents, camera collider =
        # sphere radius cubed. Slot 0 is the static world anchor.
        half_extents_np = np.zeros((self.world.num_bodies, 3), dtype=np.float32)
        for cell_ids in self._brick_newton_ids:
            for newton_idx in cell_ids:
                half_extents_np[newton_idx + 1] = (hx, hy, hz)
        if self._camera_body_newton_id is not None:
            r = float(CAMERA_COLLIDER_RADIUS)
            half_extents_np[self._camera_body_newton_id + 1] = (r, r, r)
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        # Camera-collider buffers, capture-safe by construction:
        # ``_camera_pos_host`` is pinned (so wp.copy is a fixed-pointer
        # DMA the captured graph can replay), ``_camera_pos_host_np``
        # aliases its bytes for in-place Python writes between
        # replays, and ``_camera_pos_arr`` is the device mirror.
        # Orientation is identity for the sphere collider.
        if self._camera_phoenx_slot is not None:
            self._camera_body_id_arr = wp.array([int(self._camera_phoenx_slot)], dtype=wp.int32, device=self.device)
            self._camera_pos_arr = wp.array(
                [self._camera_collider_initial_pos],
                dtype=wp.vec3f,
                device=self.device,
            )
            self._camera_orient_arr = wp.array([(0.0, 0.0, 0.0, 1.0)], dtype=wp.quatf, device=self.device)
            self._camera_pos_host = wp.array(
                [self._camera_collider_initial_pos],
                dtype=wp.vec3f,
                device="cpu",
                pinned=True,
            )
            # ``numpy()`` on a pinned CPU array returns an aliased
            # view, so in-place writes to this handle mutate the
            # underlying wp.array without allocations.
            self._camera_pos_host_np = self._camera_pos_host.numpy()

        # Warm-up damping (relaxed in :meth:`step` after WARMUP_FRAMES).
        # Read from a device slot at replay time, so toggling later
        # doesn't require a recapture.
        self.world.set_global_linear_damping(1.0)
        self.world.set_global_angular_damping(1.0)

        # Capture the per-frame pipeline into a single CUDA graph.
        # The PGS sweep uses ``wp.capture_while`` internally, which
        # only takes its conditional-graph fast path when there's an
        # outer capture active.
        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # ------------------------------------------------------------------
    # Simulation + rendering
    # ------------------------------------------------------------------

    def simulate(self) -> None:
        # Stage the camera position on the device, hand it to PhoenX as
        # the kinematic target, AND patch the same slot of
        # ``state.body_q`` so Newton's CollisionPipeline (which
        # broad/narrow-phases against ``body_q``) sees the collider at
        # the live camera location.
        if self._camera_phoenx_slot is not None:
            wp.copy(self._camera_pos_arr, self._camera_pos_host)
            self.world.set_kinematic_poses_batch(
                body_ids=self._camera_body_id_arr,
                positions=self._camera_pos_arr,
                orientations=self._camera_orient_arr,
            )
            wp.launch(
                _write_camera_body_q_kernel,
                dim=1,
                inputs=[
                    wp.int32(self._camera_body_newton_id),
                    self._camera_pos_arr,
                    self._camera_orient_arr,
                    self.state.body_q,
                ],
                device=self.device,
            )
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
        # Camera collider is always the last body and is driven by the
        # kinematic target, not the dynamic-body sync. Cap the range
        # at n-1 to leave its PhoenX slot alone.
        n = self.model.body_count
        if self._camera_body_newton_id is not None:
            n -= 1
        if n <= 0:
            return
        wp.launch(
            _newton_to_phoenx_kernel,
            dim=n,
            inputs=[
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
            ],
            outputs=[
                self.bodies.position[1 : 1 + n],
                self.bodies.orientation[1 : 1 + n],
                self.bodies.velocity[1 : 1 + n],
                self.bodies.angular_velocity[1 : 1 + n],
            ],
            device=self.device,
        )

    def _sync_phoenx_to_newton(self) -> None:
        # Symmetric to :meth:`_sync_newton_to_phoenx` -- skip the camera
        # slot. ``state.body_q[camera]`` is owned by the host-side
        # update in :meth:`simulate`.
        n = self.model.body_count
        if self._camera_body_newton_id is not None:
            n -= 1
        if n <= 0:
            return
        wp.launch(
            _phoenx_to_newton_kernel,
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

    def _update_camera_collider(self) -> None:
        """Stage the camera position into the pinned host buffer.

        The captured graph re-reads the bytes via ``wp.copy`` on each
        replay, so the in-place numpy write is all that's needed
        between frames. Headless viewers (no ``camera.pos``) fall
        back to the initial spawn position.
        """
        if self._camera_phoenx_slot is None:
            return
        cam_pos = getattr(getattr(self.viewer, "camera", None), "pos", None)
        if cam_pos is None:
            cam_xyz = self._camera_collider_initial_pos
        else:
            cam_xyz = (float(cam_pos.x), float(cam_pos.y), float(cam_pos.z))
        self._camera_pos_host_np[0] = cam_xyz

    def step(self) -> None:
        # Release the warm-up damping pin once we're past the settle.
        # Toggling the device-side slot is capture-safe; no recapture.
        if self.frame_index == self.WARMUP_FRAMES:
            self.world.set_global_linear_damping(0.0)
            self.world.set_global_angular_damping(0.0)
        self._update_camera_collider()
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self._print_step_report()
        # Optional snapshot for the standalone coloring bench.
        dump_frame_env = os.environ.get("PHOENX_DUMP_COLORING_GRAPH")
        if dump_frame_env is not None and int(dump_frame_env) == self.frame_index:
            self._dump_coloring_graph()
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def _dump_coloring_graph(self) -> None:
        """Snapshot the active constraint graph to ``kapla_graph.npz``
        for the standalone coloring benchmark to replay."""
        partitioner = self.world._partitioner
        n_active = int(self.world._num_active_constraints.numpy()[0])
        elements_struct = partitioner._elements.numpy()[:n_active]
        bodies = elements_struct["bodies"].astype(np.int32, copy=False)
        cost = partitioner._cost_values.numpy()[:n_active].astype(np.int32, copy=False)
        jitter = partitioner._random_values.numpy()[:n_active].astype(np.int32, copy=False)
        out_path = pathlib.Path("kapla_graph.npz").resolve()
        np.savez(
            out_path,
            bodies=bodies,
            cost_values=cost,
            random_values=jitter,
            num_bodies=np.int32(self.world.num_bodies),
            frame_index=np.int32(self.frame_index),
        )
        print(f"[PhoenX KaplaTower] dumped coloring graph to {out_path} (frame={self.frame_index}, n={n_active})")

    def _print_step_report(self) -> None:
        report = self.world.step_report()
        # ``max_body_degree`` is the chromatic lower bound; the ratio
        # measures how close the colourer is to it.
        slack = f"{report.num_colors / report.max_body_degree:.2f}x" if report.max_body_degree > 0 else "n/a"
        fields = [
            f"step={self.frame_index}",
            f"contacts={report.num_contact_columns}",
            f"active_constraints={report.num_active_constraints}",
            f"colors={report.num_colors}",
            f"max_body_degree={report.max_body_degree}",
            f"colors/lower_bound={slack}",
            f"color_sizes={report.color_sizes}",
        ]
        if report.per_world_num_colors is not None:
            fields.append(f"per_world_num_colors={report.per_world_num_colors}")
            fields.append(f"per_world_color_sizes={report.per_world_color_sizes}")
        print("[PhoenX KaplaTower] " + " ".join(fields))

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.end_frame()

    # ------------------------------------------------------------------
    # Post-settle validation
    # ------------------------------------------------------------------

    def test_final(self) -> None:
        """After settling, no brick may have escaped its cell's
        envelope. Catches NaNs / ejected bodies without
        over-constraining the exact settled geometry.
        """
        # 4x the unscaled tower radius is generous post-scale.
        tower_radius = float(np.linalg.norm(POSITIONS[:, :2], axis=1).max())
        tower_tolerance = 4.0 * tower_radius * GLOBAL_SCALING
        positions = self.bodies.position.numpy()
        for cell_index, cell_ids in enumerate(self._brick_newton_ids):
            cx, cy = self._cell_centres_xy[cell_index]
            for newton_idx in cell_ids:
                pos = positions[newton_idx + 1]
                assert np.isfinite(pos).all(), f"body {newton_idx} non-finite position: {pos}"
                r_xy = float(np.hypot(pos[0] - cx, pos[1] - cy))
                assert r_xy < tower_tolerance, (
                    f"brick {newton_idx} (cell {cell_index}) flew outside the tower envelope "
                    f"(r_xy={r_xy:.3f}, tol={tower_tolerance:.3f})"
                )

        for newton_idx in self._wrecking_ball_newton_ids:
            pos = positions[newton_idx + 1]
            assert np.isfinite(pos).all(), f"wrecking ball {newton_idx} non-finite position: {pos}"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    # Start paused so the initial (potentially inter-penetrating)
    # brick layout is visible before the solver begins resolving.
    # Press SPACE or toggle the viewer's "Pause" checkbox to run.
    # viewer._paused = True
    newton.examples.run(example, args)
