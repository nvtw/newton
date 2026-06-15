# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example PhoenX Kapla Square Tower
#
# A procedurally generated Kapla-style tower matching the standard
# alternating square-layer arrangement: 40 layers, 4 identical planks per
# layer, with every second layer rotated 45 degrees about +Z.
#
# The simulation loop mirrors the PhoenX Kapla tower example: Newton handles
# collision detection, PhoenX solves the contact constraints, and the body
# state is synced back for rendering.
#
# Run:
#   python -m newton._src.solvers.phoenx.examples.example_kapla_square_tower
#   python -m newton._src.solvers.phoenx.examples.example_kapla_square_tower --grid-side 4
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.body import body_container_zeros
from newton._src.solvers.phoenx.examples.example_common import (
    init_phoenx_bodies_kernel as _init_phoenx_bodies_kernel,
)
from newton._src.solvers.phoenx.examples.example_common import (
    newton_to_phoenx_kernel as _newton_to_phoenx_kernel,
)
from newton._src.solvers.phoenx.examples.example_common import (
    phoenx_to_newton_kernel as _phoenx_to_newton_kernel,
)
from newton._src.solvers.phoenx.picking import (
    Picking,
    register_with_viewer_gl,
)
from newton._src.solvers.phoenx.solver_config import PHOENX_CONTACT_MATCHING
from newton._src.solvers.phoenx.solver_phoenx import PhoenXWorld

# Step/solver settings taken from ``example_kapla_tower.py``.
USE_BIG_WORLD_MODE: bool = True
STEP_LAYOUT: str = "single_world" if USE_BIG_WORLD_MODE else "multi_world"
ENABLE_MASS_SPLITTING: bool = False
MASS_SPLITTING_MAX_COLORED_PARTITIONS: int = 12

# Tower geometry. Dimensions keep the Kapla plank proportions while placing
# the second-largest dimension upright, as in the reference screenshot.
TOWER_HEIGHT_LAYERS: int = 40
PLANKS_PER_LAYER: int = 4
PLANK_LENGTH: float = 0.30
PLANK_UPRIGHT_HEIGHT: float = 0.06
PLANK_THICKNESS: float = 0.02
PLANK_HX: float = 0.5 * PLANK_LENGTH
PLANK_HY: float = 0.5 * PLANK_THICKNESS
PLANK_HZ: float = 0.5 * PLANK_UPRIGHT_HEIGHT

# Pinwheel square: each plank's end touches the side of the next plank.
LAYER_OUTER_SIDE: float = PLANK_LENGTH + PLANK_THICKNESS
TOWER_RADIUS: float = 0.5 * math.sqrt(2.0) * LAYER_OUTER_SIDE
# Centre-to-centre spacing between towers in grid mode.
TOWER_GRID_SPACING: float = 2.0 * TOWER_RADIUS + 0.35
LAYER_ROTATION_STEP: float = math.pi / 4.0
PLANK_DENSITY: float = 1000.0

PLANK_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.96, 0.70, 0.16),
    (0.18, 0.72, 0.22),
    (0.08, 0.72, 0.78),
    (0.93, 0.25, 0.72),
)

# Sleeping bodies render at this fraction of their active brightness.
SLEEP_COLOR_GAIN: float = 0.55

# Sleeping pipeline tuning. ``frames_required`` counts step() calls, so at
# fps=120 a value of 60 means ~0.5 s of sustained low motion before a body
# is allowed to sleep. The island score is ``|v| + 0.5 * diag * |omega|``,
# taken as the maximum over all bodies in a contact island -- a whole
# Kapla tower is a single island, so the threshold has to cover the
# noisiest plank, not the average. 0.15 m/s linear-equivalent lets a
# settled tower sleep within a few seconds without freezing planks that
# are still drifting visibly.
SLEEPING_VELOCITY_THRESHOLD: float = 0.15
SLEEPING_FRAMES_REQUIRED: int = 10


@wp.kernel
def _darken_colors_kernel(
    active: wp.array[wp.vec3],
    gain: float,
    out: wp.array[wp.vec3],
):
    sid = wp.tid()
    out[sid] = gain * active[sid]


@wp.kernel
def _select_shape_color_by_sleep_kernel(
    active_colors: wp.array[wp.vec3],
    sleeping_colors: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    body_island_root: wp.array[wp.int32],
    shape_color_out: wp.array[wp.vec3],
):
    sid = wp.tid()
    body = shape_body[sid]
    sleeping = int(0)
    if body >= 0:
        # ``island_root >= 0`` marks a sleeping body (the value is the
        # lowest body id in its island at sleep time).
        if body_island_root[body + 1] >= 0:
            sleeping = 1
    if sleeping != 0:
        shape_color_out[sid] = sleeping_colors[sid]
    else:
        shape_color_out[sid] = active_colors[sid]


class Example:
    """PhoenX Kapla square tower with alternating 45-degree layers."""

    WARMUP_FRAMES: int = 20

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        self.device = wp.get_device()

        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.frame_index: int = 0
        self.sim_substeps = 6
        self.solver_iterations = 8
        self.velocity_iterations = 1

        self._build_scene()

    # ------------------------------------------------------------------
    # Scene construction
    # ------------------------------------------------------------------

    def _build_scene(self) -> None:
        builder = newton.ModelBuilder()
        builder.add_ground_plane()

        cfg = newton.ModelBuilder.ShapeConfig(density=PLANK_DENSITY, gap=0.002)
        self._plank_newton_ids: list[int] = []
        grid_side = int(getattr(self.args, "grid_side", 1) or 1)
        if grid_side < 1:
            raise ValueError(f"--grid-side must be >= 1 (got {grid_side})")
        self._grid_side: int = grid_side
        self._tower_grid_spacing: float = TOWER_GRID_SPACING
        self._tower_centres_xy: list[tuple[float, float]] = []
        self._tower_plank_newton_ids: list[list[int]] = []

        half_grid = 0.5 * (grid_side - 1) * TOWER_GRID_SPACING
        for grid_x in range(grid_side):
            for grid_y in range(grid_side):
                centre_x = grid_x * TOWER_GRID_SPACING - half_grid
                centre_y = grid_y * TOWER_GRID_SPACING - half_grid
                self._tower_centres_xy.append((centre_x, centre_y))
                tower_ids: list[int] = []

                for layer in range(TOWER_HEIGHT_LAYERS):
                    layer_yaw = LAYER_ROTATION_STEP if layer % 2 else 0.0
                    layer_z = PLANK_HZ + layer * PLANK_UPRIGHT_HEIGHT

                    # Four identical planks form a no-gap pinwheel square:
                    # end 0 -> side 1, end 1 -> side 2, and so on.
                    half_side = 0.5 * LAYER_OUTER_SIDE
                    local_specs = (
                        ((0.5 * PLANK_LENGTH, 0.5 * PLANK_THICKNESS), 0.0),
                        ((PLANK_LENGTH + 0.5 * PLANK_THICKNESS, 0.5 * PLANK_LENGTH), math.pi / 2.0),
                        (
                            (0.5 * PLANK_LENGTH + PLANK_THICKNESS, PLANK_LENGTH + 0.5 * PLANK_THICKNESS),
                            math.pi,
                        ),
                        ((0.5 * PLANK_THICKNESS, 0.5 * PLANK_LENGTH + PLANK_THICKNESS), -math.pi / 2.0),
                    )
                    cos_l = math.cos(layer_yaw)
                    sin_l = math.sin(layer_yaw)

                    for side, ((local_x, local_y), local_yaw) in enumerate(local_specs):
                        centered_x = local_x - half_side
                        centered_y = local_y - half_side
                        world_x = cos_l * centered_x - sin_l * centered_y + centre_x
                        world_y = sin_l * centered_x + cos_l * centered_y + centre_y
                        yaw = layer_yaw + local_yaw
                        body = builder.add_body(
                            xform=wp.transform(
                                p=wp.vec3(float(world_x), float(world_y), float(layer_z)),
                                q=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw),
                            ),
                        )
                        builder.add_shape_box(
                            body,
                            hx=PLANK_HX,
                            hy=PLANK_HY,
                            hz=PLANK_HZ,
                            cfg=cfg,
                            color=PLANK_COLORS[(layer + side) % len(PLANK_COLORS)],
                        )
                        self._plank_newton_ids.append(body)
                        tower_ids.append(body)

                self._tower_plank_newton_ids.append(tower_ids)

        self.model = builder.finalize(skip_shape_contact_pairs=True)
        print(
            f"[PhoenX KaplaSquareTower] bodies={self.model.body_count} "
            f"shapes={self.model.shape_count} "
            f"grid={grid_side}x{grid_side} "
            f"layers={TOWER_HEIGHT_LAYERS} planks_per_layer={PLANKS_PER_LAYER}"
        )

        # SAP broad phase + warm-start contact matching, matching the full
        # Kapla example's contact pipeline style with budgets sized for this
        # smaller procedural tower.
        # Wire the PhoenX sleeping-aware broad-phase filter so the SAP
        # pass drops rigid-rigid pairs where both sides are frozen
        # (sleeping or static / kinematic). Without this, a sleeping
        # plank still generates a contact against the ground every
        # frame; the contact's bias term then re-injects energy into
        # the sleeping island and the tower explodes on settle.
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching=PHOENX_CONTACT_MATCHING,
            broad_phase="sap",
            shape_pairs_max=20_000 * grid_side * grid_side,
            rigid_contact_max=20_000 * grid_side * grid_side,
            broad_phase_filter=PhoenXWorld.broad_phase_filter(),
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
        self.bodies = bodies

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
            mass_splitting=ENABLE_MASS_SPLITTING,
            max_colored_partitions=MASS_SPLITTING_MAX_COLORED_PARTITIONS,
            mass_splitting_unrolled=True,
            mass_splitting_batch_size=8,
            sor_boost=1.0,
            sleeping_velocity_threshold=SLEEPING_VELOCITY_THRESHOLD,
            sleeping_frames_required=SLEEPING_FRAMES_REQUIRED,
            device=self.device,
        )

        # Bind the broad-phase filter data + cache per-shape AABB
        # arrays. After this call the per-frame loop doesn't need to
        # pass ``shape_aabb_*`` to ``world.step()``.
        self.world.attach_collision_pipeline(
            self.collision_pipeline,
            num_rigid_shapes=int(self.model.shape_count),
            shape_body=self.model.shape_body,
        )

        # Two parallel per-shape color buffers (active + dim-for-sleep), plus
        # a kernel that picks between them based on each body's
        # ``island_root`` flag and writes the result into ``model.shape_color``.
        # The viewer's existing ``_sync_shape_colors_from_model`` repacks that
        # array into the OpenGL instance-color VBO every frame, so this is
        # the cheapest way to drive sleep-aware colors without modifying the
        # viewer or any shaders.
        self._shape_color_active = wp.clone(self.model.shape_color)
        self._shape_color_sleeping = wp.empty_like(self.model.shape_color)
        wp.launch(
            _darken_colors_kernel,
            dim=self.model.shape_count,
            inputs=[self._shape_color_active, SLEEP_COLOR_GAIN],
            outputs=[self._shape_color_sleeping],
            device=self.device,
        )

        self.viewer.set_model(self.model)
        grid_extent = max(1.0, TOWER_GRID_SPACING * grid_side)
        self.viewer.set_camera(
            pos=wp.vec3(0.6 * grid_extent, -1.1 * grid_extent, 1.5),
            pitch=-18.0,
            yaw=145.0,
        )

        half_extents_np = np.zeros((self.world.num_bodies, 3), dtype=np.float32)
        for newton_idx in self._plank_newton_ids:
            half_extents_np[newton_idx + 1] = (PLANK_HX, PLANK_HY, PLANK_HZ)
        self._half_extents = wp.array(half_extents_np, dtype=wp.vec3f, device=self.device)
        self.picking = Picking(self.world, self._half_extents)
        register_with_viewer_gl(self.viewer, self.picking)

        # Warm-up damping, matching the full Kapla example's settle path.
        self.world.set_global_linear_damping(1.0)
        self.world.set_global_angular_damping(1.0)

        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    # ------------------------------------------------------------------
    # Simulation + rendering
    # ------------------------------------------------------------------

    def simulate(self) -> None:
        self._sync_newton_to_phoenx()
        # Apply picking force *before* collide() so the wake-on-input
        # pass below sees the user wrench and clears ``island_root`` on
        # the picked plank's whole island. Without this ordering the
        # broad-phase sleeping filter (which runs in collide) drops
        # plank-vs-plank and plank-vs-ground pairs on the wake frame
        # and the tower has no contacts to lean on while the picked
        # plank is being dragged.
        self.picking.apply_force()
        self.world.wake_on_external_input()
        self.model.collide(
            self.state,
            contacts=self.contacts,
            collision_pipeline=self.collision_pipeline,
        )
        # ``attach_collision_pipeline`` cached the per-shape AABB
        # arrays + installed the shape_body map, so no extra args
        # are needed here.
        self.world.step(
            dt=self.frame_dt,
            contacts=self.contacts,
            shape_body=self._shape_body,
        )
        self._sync_phoenx_to_newton()

    def _sync_newton_to_phoenx(self) -> None:
        n = self.model.body_count
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
        n = self.model.body_count
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

    def step(self) -> None:
        if self.frame_index == self.WARMUP_FRAMES:
            self.world.set_global_linear_damping(0.0)
            self.world.set_global_angular_damping(0.0)
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def _refresh_sleep_colors(self) -> None:
        """Write sleep-aware colors into ``model.shape_color`` on the GPU.

        The viewer picks up the new values via its own
        ``_sync_shape_colors_from_model`` pass during ``log_state``.
        """
        wp.launch(
            _select_shape_color_by_sleep_kernel,
            dim=self.model.shape_count,
            inputs=[
                self._shape_color_active,
                self._shape_color_sleeping,
                self.model.shape_body,
                self.bodies.island_root,
            ],
            outputs=[self.model.shape_color],
            device=self.device,
        )

    def render(self) -> None:
        self._refresh_sleep_colors()
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        force_contacts = bool(getattr(self.args, "show_contacts", False))
        if self._grid_side <= 1 or force_contacts:
            self.viewer.log_contacts(self.contacts, self.state)
        self.viewer.end_frame()

    # ------------------------------------------------------------------
    # Post-settle validation
    # ------------------------------------------------------------------

    def test_final(self) -> None:
        """After settling, every plank must remain finite and near the tower."""
        tolerance = 4.0 * TOWER_RADIUS
        positions = self.bodies.position.numpy()
        for tower_index, tower_ids in enumerate(self._tower_plank_newton_ids):
            centre_x, centre_y = self._tower_centres_xy[tower_index]
            for newton_idx in tower_ids:
                pos = positions[newton_idx + 1]
                assert np.isfinite(pos).all(), f"body {newton_idx} non-finite position: {pos}"
                r_xy = float(math.hypot(pos[0] - centre_x, pos[1] - centre_y))
                assert r_xy < tolerance, (
                    f"plank {newton_idx} flew outside tower {tower_index} envelope "
                    f"(r_xy={r_xy:.3f}, tol={tolerance:.3f})"
                )


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--grid-side",
        type=int,
        default=20,
        help="Tile the tower into a grid-side x grid-side grid in one shared world.",
    )
    parser.add_argument(
        "--show-contacts",
        action="store_true",
        help="Render contact-normal arrows in grid mode. Off by default for better viewer performance.",
    )
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
