# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One Kapla tower and ten jointed Hoberman spheres in a single PhoenX world.

Run with ``uv run -m newton._src.solvers.phoenx.examples.example_kapla_hoberman``.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.geometry.broad_phase_grid import BroadPhaseGrid
from newton._src.solvers.phoenx.examples.example_hoberman_sphere import make_hoberman_builder
from newton._src.solvers.phoenx.examples.example_kapla_tower import GROUND_HEIGHT, add_kapla_bricks
from newton._src.solvers.phoenx.solver_config import PHOENX_CONTACT_MATCHING

HOBERMAN_COUNT = 10
WARMUP_FRAMES = 10
TOWER_CENTRE_Y = -2.5
SPHERE_RING_RADIUS = 6.6
SPHERE_START_HEIGHT = 4.8
PICK_STIFFNESS = 300.0
PICK_DAMPING = 30.0
PICK_MAX_ACCELERATION = 20.0
HOBERMAN_MID_SPRING_STIFFNESS = 50.0
HOBERMAN_MID_SPRING_DAMPING = 5.0


class Example:
    """Exercise looped joints and dense rigid contacts in one world."""

    overlap_simulation_render = True

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.device = wp.get_device()
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.frame_index = 0

        builder = newton.ModelBuilder()
        builder.add_ground_plane(height=GROUND_HEIGHT)
        tower_ids, _ = add_kapla_bricks(builder, [(0.0, 0.0)])
        self.tower_body_count = len(tower_ids[0])
        self.tower_body_ids = np.asarray(tower_ids[0], dtype=np.int32)

        # Import and tile the articulated asset once, then clone it into
        # the same world. Negative groups filter contacts within each
        # sphere while allowing sphere-ground and sphere-tower contacts.
        sphere = make_hoberman_builder(
            collidable=True,
            gravity=(0.0, 0.0, -9.81),
            mid_spring_stiffness=HOBERMAN_MID_SPRING_STIFFNESS,
            mid_spring_damping=HOBERMAN_MID_SPRING_DAMPING,
        )
        sphere_body_count = sphere.body_count
        sphere_joint_count = len(sphere.joint_parent)
        self.sphere_first_shape = len(builder.shape_body)
        for index in range(HOBERMAN_COUNT):
            angle = 2.0 * math.pi * index / HOBERMAN_COUNT
            offset = wp.transform(
                p=wp.vec3(
                    SPHERE_RING_RADIUS * math.cos(angle),
                    TOWER_CENTRE_Y + SPHERE_RING_RADIUS * math.sin(angle),
                    SPHERE_START_HEIGHT,
                ),
                q=wp.quat_identity(),
            )
            first_shape = len(builder.shape_body)
            builder.add_builder(sphere, xform=offset, label_prefix=f"hoberman_{index:02d}")
            builder.shape_collision_group[first_shape:] = [-(index + 2)] * (len(builder.shape_body) - first_shape)

        builder.color()
        self.model = builder.finalize(skip_validation_joints=True, skip_shape_contact_pairs=True)
        self.tower_start_positions = self.model.body_q.numpy()[self.tower_body_ids, :3].copy()
        self.state = self.model.state()
        self.state.body_q.assign(self.model.body_q)
        self.control = self.model.control()

        pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching=PHOENX_CONTACT_MATCHING,
            broad_phase="sap",
            shape_pairs_max=2_000_000,
            rigid_contact_max=750_000,
        )
        pipeline.broad_phase = BroadPhaseGrid(
            self.model.shape_world,
            shape_flags=self.model.shape_flags,
            cell_width=0.6,
            pair_mode="warp_deterministic",
            device=self.device,
        )
        self.collision_pipeline = pipeline
        self.contacts = pipeline.contacts()
        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            collision_pipeline=pipeline,
            substeps=4,
            solver_iterations=10,
            velocity_iterations=1,
            step_layout="single_world",
            mass_splitting=True,
            mass_splitting_overflow_only=True,
            max_colored_partitions=9,
            mass_splitting_batch_size=1,
            mass_splitting_unrolled=True,
            partitioner_algorithm="endpoint_owner",
            colored_contact_headers=True,
            colored_contact_rows=True,
            joint_mode="maximal_pgs",
            max_thread_blocks=8 * self.device.sm_count,
        )
        self.solver.world.set_global_linear_damping(1.0)
        self.solver.world.set_global_angular_damping(1.0)
        print(
            f"[PhoenX Kapla+Hoberman] tower_bricks={self.tower_body_count} "
            f"spheres={HOBERMAN_COUNT} sphere_bodies={sphere_body_count * HOBERMAN_COUNT} "
            f"sphere_joints={sphere_joint_count * HOBERMAN_COUNT} "
            f"total_bodies={self.model.body_count}"
        )

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(8.0, -12.0, 4.5), pitch=-15.0, yaw=130.0)
        picking = getattr(self.viewer, "picking", None)
        if picking is not None:
            pick_state = picking.pick_state.numpy()
            pick_state[0]["pick_stiffness"] = PICK_STIFFNESS
            pick_state[0]["pick_damping"] = PICK_DAMPING
            pick_state[0]["pick_max_acceleration"] = PICK_MAX_ACCELERATION
            picking.pick_state.assign(pick_state)

        self.graph = None
        if self.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        overlap = self.overlap_simulation_render and self.viewer.supports_simulation_render_overlap
        self._render_states = (self.model.state(), self.model.state()) if overlap else None
        self._render_state_done = tuple(wp.Event(self.device) if overlap else None for _ in range(2))
        self._render_state_index = 0
        self._render_state_prepared = False
        self._render_time = self.sim_time

    def prepare_render_state(self) -> None:
        """Snapshot poses before physics advances on the other CUDA stream."""
        assert self._render_states is not None
        self._render_state_index = 1 - self._render_state_index
        done = self._render_state_done[self._render_state_index]
        if done is not None:
            wp.wait_event(done)
        wp.copy(self._render_states[self._render_state_index].body_q, self.state.body_q)
        self._render_time = self.sim_time
        self._render_state_prepared = True

    def simulate(self) -> None:
        """Advance two 120 Hz physics ticks for one rendered frame."""
        for _ in range(2):
            self.state.clear_forces()
            self.viewer.apply_forces(self.state)
            self.collision_pipeline.collide(self.state, self.contacts)
            self.solver.step(self.state, self.state, self.control, self.contacts, self.frame_dt / 2.0)

    def step(self) -> None:
        if self.frame_index == WARMUP_FRAMES:
            self.solver.world.set_global_linear_damping(0.0)
            self.solver.world.set_global_angular_damping(0.0)
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt
        self.frame_index += 1

    def render(self) -> None:
        state = self._render_states[self._render_state_index] if self._render_state_prepared else self.state
        self.viewer.begin_frame(self._render_time if self._render_state_prepared else self.sim_time)
        self.viewer.log_state(state)
        if self._render_state_prepared:
            done = self._render_state_done[self._render_state_index]
            if done is not None:
                wp.record_event(done)
        self.viewer.end_frame()
        self._render_state_prepared = False

    def test_final(self) -> None:
        """Catch invalid coupled states in the example test runner."""
        poses = self.state.body_q.numpy()
        assert np.isfinite(poses).all(), "mixed scene produced non-finite poses"
        assert np.isfinite(self.state.body_qd.numpy()).all(), "mixed scene produced non-finite velocities"
        joint_parent = self.model.joint_parent.numpy()
        joint_child = self.model.joint_child.numpy()
        joint_type = self.model.joint_type.numpy()
        joint_axis = self.model.joint_axis.numpy()
        joint_qd_start = self.model.joint_qd_start.numpy()
        parent_frame = self.model.joint_X_p.numpy()
        child_frame = self.model.joint_X_c.numpy()
        worst_attachment = 0.0
        worst_hinge_angle = 0.0
        for joint in range(self.model.joint_count):
            parent = int(joint_parent[joint])
            child = int(joint_child[joint])
            if parent < 0 or child < 0:
                continue
            parent_pose = wp.transform(wp.vec3(*poses[parent, :3]), wp.quat(*poses[parent, 3:]))
            child_pose = wp.transform(wp.vec3(*poses[child, :3]), wp.quat(*poses[child, 3:]))
            joint_p = wp.transform(wp.vec3(*parent_frame[joint, :3]), wp.quat(*parent_frame[joint, 3:]))
            joint_c = wp.transform(wp.vec3(*child_frame[joint, :3]), wp.quat(*child_frame[joint, 3:]))
            frame_p = parent_pose * joint_p
            frame_c = child_pose * joint_c
            point_p = wp.transform_get_translation(frame_p)
            point_c = wp.transform_get_translation(frame_c)
            worst_attachment = max(worst_attachment, float(wp.length(point_p - point_c)))
            if int(joint_type[joint]) == int(newton.JointType.REVOLUTE):
                axis = wp.vec3(*joint_axis[int(joint_qd_start[joint])])
                direction_p = wp.transform_vector(frame_p, axis)
                direction_c = wp.transform_vector(frame_c, axis)
                angle = math.atan2(
                    float(wp.length(wp.cross(direction_p, direction_c))), float(wp.dot(direction_p, direction_c))
                )
                worst_hinge_angle = max(worst_hinge_angle, angle)
        print(f"[PhoenX Kapla+Hoberman] maximum_joint_attachment_error={worst_attachment:.6g} m")
        print(f"[PhoenX Kapla+Hoberman] maximum_hinge_axis_error={math.degrees(worst_hinge_angle):.6g} deg")
        tower_positions = poses[self.tower_body_ids, :3]
        tower_delta = tower_positions - self.tower_start_positions
        maximum_tower_drop = float(np.max(-tower_delta[:, 2]))
        maximum_tower_slide = float(np.linalg.norm(tower_delta[:, :2], axis=1).max())
        fallen_bricks = int(np.count_nonzero(-tower_delta[:, 2] > 0.5))
        print(f"[PhoenX Kapla+Hoberman] tower_bricks_fallen_50cm={fallen_bricks}")
        print(f"[PhoenX Kapla+Hoberman] tower_maximum_drop={maximum_tower_drop:.6g} m")
        print(f"[PhoenX Kapla+Hoberman] tower_maximum_slide={maximum_tower_slide:.6g} m")
        assert fallen_bricks == 0, f"Kapla tower lost {fallen_bricks} bricks by more than 50 cm"
        assert worst_attachment < 0.02, f"Hoberman joint attachment drifted {worst_attachment:.6g} m"
        assert worst_hinge_angle < math.radians(10.0), (
            f"Hoberman hinge axis drifted {math.degrees(worst_hinge_angle):.6g} deg"
        )
        joint_cids = self.solver._joint_constraints.joint_idx_to_cid.numpy()
        joint_cids = joint_cids[joint_cids >= 0]
        colors = self.solver.world._partitioner.interaction_id_to_partition.numpy()[joint_cids]
        print(f"[PhoenX Kapla+Hoberman] joint_colors={np.bincount(colors, minlength=10)[:10].tolist()}")
        if self.sim_time >= 1.0:
            count = int(self.contacts.rigid_contact_count.numpy()[0])
            shape0 = self.contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = self.contacts.rigid_contact_shape1.numpy()[:count]
            sphere_contacts = int(
                np.count_nonzero((shape0 >= self.sphere_first_shape) | (shape1 >= self.sphere_first_shape))
            )
            print(f"[PhoenX Kapla+Hoberman] final_contacts={count} sphere_contacts={sphere_contacts}")
            assert sphere_contacts > 0, "Hoberman spheres never reached contact by the final frame"


if __name__ == "__main__":
    viewer, args = newton.examples.init()
    example = Example(viewer, args)
    newton.examples.run(example, args)
