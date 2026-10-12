# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop 32,000 six-box ragdolls in one PhoenX world.

Command: python -m newton.examples phoenx_ragdoll_pile --viewer optix
Quick test: python -m newton.examples phoenx_ragdoll_pile --grid 4 4 4

Inspired by https://www.youtube.com/watch?v=REPDnj8WZtA. The default grid
matches its 192,001 bodies and 160,000 joints. Geometry and material values
are reconstructed, not exported from the original scene. Contact counts are
simulation outcomes; the reference reports averages of 242,026 manifolds,
600,051 points and 1,326,130 contact rows. All five pivots per figure are D6
joints configured as spherical joints, with no linear freedoms.
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples._render_state import RenderStateBuffer
from newton.examples.phoenx._rigid_solver import make_colored_solver
from newton.examples.phoenx.ragdoll_pile_scene import (
    BODIES,
    BODY_DENSITY,
    DEFAULT_GRID,
    JOINTS,
    PALETTE,
    ragdoll_positions,
    scene_bounds,
)
from newton.geometry import BroadPhaseImplicitTree


def build_scene(*, grid=DEFAULT_GRID, seed=42):
    """Build independent maximal-coordinate figures in one shared world."""
    scene_bounds(grid)
    builder = newton.ModelBuilder()
    ground = builder.add_link(label="ground", is_kinematic=True)
    # Keep the floor unbounded so expanding piles do not lose bodies and
    # contact work over a platform edge. It has one upward contact surface.
    builder.add_shape_plane(
        body=ground,
        width=0.0,
        length=0.0,
        cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.6),
        color=(0.48, 0.55, 0.58),
    )
    cfg = newton.ModelBuilder.ShapeConfig(density=BODY_DENSITY, mu=0.6, restitution=0.0, gap=0.01)
    rng = np.random.default_rng(seed)
    angular_axes = [newton.ModelBuilder.JointDofConfig.create_unlimited(axis) for axis in newton.Axis]
    for index, offset in enumerate(ragdoll_positions(grid)):
        color = PALETTE[int(rng.integers(len(PALETTE)))]
        cfg.collision_group = -(index + 1)
        body_ids = []
        for body in BODIES:
            position = np.asarray(offset) + body.center
            body_id = builder.add_link(
                xform=wp.transform(position, wp.quat_identity()), label=f"ragdoll_{index}/{body.name}"
            )
            body_ids.append(body_id)
            builder.add_shape_box(
                body_id,
                hx=body.half_extents[0],
                hy=body.half_extents[1],
                hz=body.half_extents[2],
                cfg=cfg,
                color=color,
            )
        for child, anchor in JOINTS:
            builder.add_joint_d6(
                parent=body_ids[0],
                child=body_ids[child],
                parent_xform=wp.transform(np.asarray(anchor) - BODIES[0].center, wp.quat_identity()),
                child_xform=wp.transform(np.asarray(anchor) - BODIES[child].center, wp.quat_identity()),
                angular_axes=angular_axes,
                collision_filter_parent=False,
                label=f"ragdoll_{index}/{BODIES[child].name}_pivot",
            )
    return builder


class Example:
    """A deterministic large-world joint and contact stress scene."""

    overlap_simulation_render = True

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.grid = tuple(args.grid)
        self.ragdoll_count = math.prod(self.grid)
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0
        self.frame_index = 0
        self.joint_solver = getattr(args, "joint_solver", "exact")
        print(f"[PhoenX Ragdoll Pile] building {self.ragdoll_count:,} ragdolls...", flush=True)
        builder = build_scene(grid=self.grid, seed=args.seed)
        # Maximal-coordinate joints form a constraint graph without free root
        # joints or reduced-coordinate articulations. Avoid enumerating every
        # possible shape pair for the large shared world.
        self.model = builder.finalize(skip_validation_joints=True, skip_shape_contact_pairs=True)
        self.state = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            contact_matching="sticky",
            shape_pairs_max=max(4096, 128 * self.ragdoll_count),
            rigid_contact_max=max(4096, 96 * self.ragdoll_count),
            # Predict closure over the outer frame without enlarging the
            # physical 10 mm gap. Bound searches by the largest body diameter.
            speculative_contact_gap_max=2.0 * max(float(np.linalg.norm(body.half_extents)) for body in BODIES),
        )
        self.contacts = self.collision_pipeline.contacts()
        self.collision_pipeline.broad_phase = BroadPhaseImplicitTree(
            self.model.shape_world, shape_flags=self.model.shape_flags, device=self.model.device
        )
        self.solver = make_colored_solver(
            self.model,
            self.collision_pipeline,
            self.model.device,
            substeps=args.substeps,
            solver_iterations=args.iterations,
            # Exact mechanism projections converge faster than solving each
            # joint independently. Contacts retain the shared colored path.
            direct_joint_mask=np.ones(self.model.joint_count, dtype=bool) if self.joint_solver == "exact" else None,
        )
        viewer.set_model(self.model)
        if hasattr(viewer, "set_default_color_palette"):
            viewer.set_default_color_palette(PALETTE)
        lower, upper = scene_bounds(self.grid)
        center = 0.5 * (lower + upper)
        distance = max(float(np.linalg.norm(upper - lower)), 5.0)
        viewer.set_camera(
            pos=wp.vec3(float(center[0] + 0.72 * distance), float(center[1] - distance), float(0.65 * distance)),
            pitch=-22.0,
            yaw=126.0,
        )
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        self.render_state = RenderStateBuffer(
            self.model, enabled=bool(getattr(viewer, "supports_simulation_render_overlap", False))
        )
        print(
            f"[PhoenX Ragdoll Pile] grid={self.grid} ragdolls={self.ragdoll_count} "
            f"bodies={self.model.body_count} joints={self.model.joint_count} "
            f"joint_solver={self.joint_solver} substeps={args.substeps} "
            f"iterations={args.iterations} velocity_iterations=1 sleeping=off",
            flush=True,
        )

    def simulate(self):
        self.state.clear_forces()
        self.viewer.apply_forces(self.state)
        self.collision_pipeline.collide(self.state, self.contacts, dt=self.frame_dt)
        self.solver.step(self.state, self.state, self.control, self.contacts, self.frame_dt)

    def step(self):
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def render(self):
        state, time = self.render_state.get(self.state, self.sim_time)
        self.viewer.begin_frame(time)
        self.viewer.log_state(state)
        self.viewer.end_frame()
        self.render_state.release()

    def prepare_render_state(self):
        self.render_state.prepare(self.state, self.sim_time)

    def test_final(self):
        poses = self.state.body_q.numpy()
        assert np.isfinite(poses).all(), "ragdoll poses are non-finite"
        assert np.isfinite(self.state.body_qd.numpy()).all(), "ragdoll velocities are non-finite"
        np.testing.assert_allclose(poses[0], self.model.body_q.numpy()[0], atol=1.0e-6)
        print(f"[PhoenX Ragdoll Pile] body_height_range={poses[1:, 2].min():.3f}..{poses[1:, 2].max():.3f}", flush=True)
        assert poses[1:, 2].min() > -0.5, "ragdolls fell through the floor"
        parents = self.model.joint_parent.numpy()
        children = self.model.joint_child.numpy()
        parent_frames = self.model.joint_X_p.numpy()
        child_frames = self.model.joint_X_c.numpy()

        def anchors(body_ids, frames):
            q = poses[body_ids, 3:6]
            p = frames[:, :3]
            rotated = p + 2.0 * np.cross(q, np.cross(q, p) + poses[body_ids, 6:7] * p)
            return poses[body_ids, :3] + rotated

        errors = np.linalg.norm(anchors(parents, parent_frames) - anchors(children, child_frames), axis=1)
        max_error = float(errors.max(initial=0.0))
        assert max_error < 0.005, f"ragdoll joint attachment error is {max_error:.4f} m"
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        assert count < self.contacts.rigid_contact_max, "contact capacity was exhausted"
        pair_count = int(self.collision_pipeline.broad_phase_pair_count.numpy()[0])
        assert pair_count < self.collision_pipeline.shape_pairs_max, "broad-phase capacity was exhausted"
        print(f"[PhoenX Ragdoll Pile] max_joint_error={max_error:.6f} m contact_points={count}", flush=True)

    def test_post_step(self):
        """Reject capacity loss at any point during a collapse."""
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        pair_count = int(self.collision_pipeline.broad_phase_pair_count.numpy()[0])
        assert count < self.contacts.rigid_contact_max, "contact capacity was exhausted"
        assert pair_count < self.collision_pipeline.shape_pairs_max, "broad-phase capacity was exhausted"


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--grid", type=int, nargs=3, default=DEFAULT_GRID, metavar=("X", "Y", "Z"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--joint-solver", choices=("exact", "colored"), default="exact")
    grid_args, _ = parser.parse_known_args()
    parser.set_defaults(optix_max_instances=6 * math.prod(grid_args.grid) + 1)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
