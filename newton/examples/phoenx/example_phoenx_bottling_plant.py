# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simulate the full Berlin bottling plant with PhoenX.

Read the downloaded plant's composed USD with an example-specific builder. The
factory keeps its source geometry, rigid bodies, mass properties, colliders,
joint frames, and drive settings. Meshes are shared between repeated parts.
The source assets remain outside the repository and require NVIDIA Storage
access; see ``newton/examples/assets/bottling_plant/README.md``.

Success criteria:
- Body poses and velocities remain finite.
- Bottles stay off the factory floor, and sealed cases retain all 12 bottles.
- Joint attachments remain within their physical travel.
- Inlet, case-line, sealer, and palletizer controllers remain free of faults.
- Runs of at least 22 simulated seconds fill a case at each of the eight packers.
- Runs of at least 130 simulated seconds deliver a case through every palletizer.

Runs:
- `python -m newton.examples phoenx_bottling_plant`
- `python -m newton.examples phoenx_bottling_plant --viewer null --test --num-frames 600`
- `python -m newton.examples phoenx_bottling_plant --viewer null --timing-frames 600`
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.phoenx._bottling_control import PlantController
from newton.examples.phoenx._bottling_scene import load_plant
from newton.solvers import SolverPhoenX


@wp.struct
class _PlantFilter:
    product: wp.array[wp.int32]
    collision_membership: wp.array[wp.uint64]
    collision_filtered: wp.array[wp.uint64]
    articulation: wp.array[wp.int32]
    shape_body: wp.array[wp.int32]
    held_by: wp.array[wp.int32]


@wp.func
def _filter_pair(pair: wp.vec2i, data: _PlantFilter) -> wp.int32:
    a, b = pair[0], pair[1]
    if (data.collision_membership[a] & data.collision_filtered[b]) != wp.uint64(0) or (
        data.collision_membership[b] & data.collision_filtered[a]
    ) != wp.uint64(0):
        return 0
    # The source excludes bottle bases against every other bottle hull.
    pa, pb = data.product[a], data.product[b]
    if (pa == 2 and pb > 0) or (pb == 2 and pa > 0):
        return 0
    group = data.articulation[a]
    if group > 0 and group == data.articulation[b]:
        return 0
    ba, bb = data.shape_body[a], data.shape_body[b]
    if ba >= 0 and bb >= 0:
        if data.held_by[ba] == bb + 1 or data.held_by[bb] == ba + 1:
            return 0
    return 1


class Example:
    """One factory world, with CUDA graph capture for collision and stepping."""

    overlap_simulation_render = True

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        if args.substeps < 1:
            raise ValueError("--substeps must be positive")
        if args.controllers and not args.motor_off and args.joint_mode != "maximal_pgs":
            raise ValueError("Physical runtime grasp joints require --joint-mode maximal_pgs")
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0
        self.scene = load_plant(
            Path(args.asset_root).expanduser(),
            cutaway=args.cutaway,
            visuals=args.visuals if args.visuals is not None else args.viewer != "null",
            motor_off=args.motor_off,
        )
        if args.controllers and not args.motor_off:
            self.scene.builder.reserve_dynamic_capacity(joints=320)
        self.scene.builder.validate_inertia_detailed = args.diagnostics
        # Several source actuator tensors are positive definite but violate
        # the triangle inequality. Keep those authored approximations unless
        # the caller explicitly asks Newton to balance them.
        self.model = self.scene.builder.finalize(skip_validation_inertia=not args.correct_inertia)
        initial_state = self.model.state()
        newton.eval_ik(self.model, initial_state, self.model.joint_q, self.model.joint_qd)
        coordinates = self.model.joint_q.numpy()
        starts = self.model.joint_q_start.numpy()
        for joint, kind in enumerate(self.model.joint_type.numpy()):
            if kind == int(newton.JointType.REVOLUTE):
                index = starts[joint]
                coordinates[index] = (coordinates[index] + np.pi) % (2.0 * np.pi) - np.pi
        self.model.joint_q.assign(coordinates)
        for mesh, velocity, _ in self.scene.belts:
            mesh.mesh.velocities.assign(velocity)
        product = np.zeros(self.model.shape_count, dtype=np.int32)
        groups = np.zeros(self.model.shape_count, dtype=np.int32)
        scopes = sorted(set(self.scene.self_collision_scopes), key=len)
        for shape, label in enumerate(self.model.shape_label):
            if label.startswith("/Plant/Sim/Product/Bottles/"):
                product[shape] = 2 if "/Collision/Hull_00/" in label else 1
            for group, scope in enumerate(scopes, 1):
                if label.startswith(scope + "/"):
                    groups[shape] = group
        self.filter_data = _PlantFilter()
        membership, filtered = self.scene.collision_masks(self.model.shape_label)
        self.filter_data.collision_membership = wp.array(membership, dtype=wp.uint64, device=self.model.device)
        self.filter_data.collision_filtered = wp.array(filtered, dtype=wp.uint64, device=self.model.device)
        self.filter_data.product = wp.array(product, dtype=wp.int32, device=self.model.device)
        self.filter_data.articulation = wp.array(groups, dtype=wp.int32, device=self.model.device)
        self.filter_data.shape_body = self.model.shape_body
        self.filter_data.held_by = wp.zeros(self.model.body_count, dtype=int, device=self.model.device)
        self.pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching="sticky",
            rigid_contact_max=250_000,
            broad_phase="nxn" if not self.model.device.is_cuda else "sap",
            speculative_contact_gap_max=0.005,
            broad_phase_filter=(_filter_pair, _PlantFilter),
            max_triangle_pairs=1_000_000,
            shape_pairs_max=max(
                65536,
                32 * int(np.count_nonzero(self.model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES))),
            ),
            include_static_kinematic_pairs=False,
        )
        self.pipeline.set_broad_phase_filter_data(self.filter_data)
        if args.joint_mode == "maximal_pgs":
            # Exact solves are faster for these driven scalar-joint mechanisms.
            # D6 closures and runtime grasp/tape joints remain local blocks.
            exact = np.asarray(
                [
                    label.startswith("/Plant/") and kind != int(newton.JointType.D6)
                    for label, kind in zip(self.model.joint_label, self.model.joint_type.numpy(), strict=True)
                ],
                dtype=bool,
            )
            self.solver = SolverPhoenX(
                self.model,
                collision_pipeline=self.pipeline,
                substeps=1,
                solver_iterations=args.iterations,
                contact_chunk_size=64,
                step_layout="single_world",
                joint_mode="maximal_pgs",
                direct_joint_mask=exact,
                direct_contact_mask=(
                    exact
                    & np.asarray(
                        [label.startswith("/Plant/Logistics/Fleet/") for label in self.model.joint_label], dtype=bool
                    )
                    if np.any(exact)
                    else None
                ),
                velocity_iterations=1,
                enable_body_pair_grouping=True,
                parallel_contact_prepare=self.model.device.is_cuda and not args.mass_splitting,
                mass_splitting=args.mass_splitting,
                mass_splitting_overflow_only=args.mass_splitting,
                max_colored_partitions=9,
                mass_splitting_batch_size=1,
                mass_splitting_unrolled=args.mass_splitting,
                partitioner_algorithm="endpoint_owner" if args.mass_splitting else "greedy",
                sleeping_velocity_threshold=0.0,
            )
        else:
            self.solver = SolverPhoenX(
                self.model,
                collision_pipeline=self.pipeline,
                step_layout="single_world",
                substeps=1,
                solver_iterations=args.iterations,
                joint_mode=args.joint_mode,
                reduced_articulations=self.scene.articulations if args.joint_mode == "reduced" else None,
                velocity_iterations=1,
                sleeping_velocity_threshold=0.0,
            )
        # Runtime joints install their default data during solver setup. This
        # filter also handles grasp pairs, alongside the source's hull groups.
        self.pipeline.set_broad_phase_filter_data(self.filter_data)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        self.controller = None
        if args.controllers and not args.motor_off:
            self.controller = PlantController(
                self.scene,
                self.model,
                self.solver,
                self.state_0,
                self.control,
                Path(args.asset_root).expanduser(),
                self.filter_data.held_by,
            )
        self.joint_axes = self.model.joint_axis.numpy()
        self.joint_dofs = self.model.joint_qd_start.numpy()
        self.initial_poses = self.state_0.body_q.numpy()
        self.bottle_bodies = np.asarray(self.scene.bottle_bodies, dtype=np.int32)
        viewer.set_model(self.model)
        viewer.set_camera(wp.vec3(49.0, 48.0, 3.4), pitch=-12.0, yaw=10.0)
        self.graph = None
        if self.model.device.is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        overlap = self.overlap_simulation_render and viewer.supports_simulation_render_overlap
        self._render_states = (self.model.state(), self.model.state()) if overlap else None
        self._render_state_done = tuple(
            wp.Event(self.model.device) if self.model.device.is_cuda else None for _ in range(2)
        )
        self._render_state_index = 0
        self._render_state_prepared = False
        self._render_time = self.sim_time

    def prepare_render_state(self):
        """Snapshot poses before physics runs on the viewer's simulation stream."""
        self._render_state_index = 1 - self._render_state_index
        done = self._render_state_done[self._render_state_index]
        if done is not None:
            wp.wait_event(done)
        wp.copy(self._render_states[self._render_state_index].body_q, self.state_0.body_q)
        self._render_time = self.sim_time
        self._render_state_prepared = True

    def simulate(self):
        self.state_0.clear_forces()
        if self.controller is not None:
            self.controller.simulate()
        self.viewer.apply_forces(self.state_0)
        dt = self.frame_dt / self.args.substeps
        for _ in range(self.args.substeps):
            # Refresh contacts at the physics rate: sealed cases have thin
            # floors that must keep supporting bottles while being lifted.
            self.pipeline.collide(self.state_0, self.contacts, dt=dt)
            self.solver.step(self.state_0, self.state_0, self.control, self.contacts, dt)

    def step(self):
        if self.controller is not None:
            self.controller.before_step()
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        state = self._render_states[self._render_state_index] if self._render_state_prepared else self.state_0
        self.viewer.begin_frame(self._render_time if self._render_state_prepared else self.sim_time)
        self.viewer.log_state(state)
        if self._render_state_prepared:
            done = self._render_state_done[self._render_state_index]
            if done is not None:
                wp.record_event(done)
        self.viewer.end_frame()
        self._render_state_prepared = False

    def _check_attachments(self, poses):
        """Check source joint anchors, allowing each slide its axial travel."""
        maximum = 0.0
        for (bodies, frames, kind), joint in zip(self.scene.joint_frames, self.scene.joints.values(), strict=True):
            if kind == "PhysicsJoint":
                continue
            world = [
                frame if body < 0 else wp.transform_multiply(wp.transform(poses[body, :3], poses[body, 3:]), frame)
                for body, frame in zip(bodies, frames, strict=True)
            ]
            error = np.asarray(world[1].p) - np.asarray(world[0].p)
            if kind == "PhysicsPrismaticJoint":
                axis = self.joint_axes[self.joint_dofs[joint]]
                axis = np.asarray(wp.quat_rotate(world[0].q, wp.vec3(axis)))
                error -= axis * np.dot(axis, error)
            maximum = max(maximum, float(np.linalg.norm(error)))
        assert maximum < 0.025, f"source joint anchor drifted {maximum:.4f} m"
        return maximum

    def _check_products(self, poses):
        bottles = poses[self.bottle_bodies]
        # Even floor-level pallets support the case contents above 0.1 m.
        assert float(bottles[:, 2].min()) > 0.1, "a bottle escaped the line or its case"
        if self.controller is None:
            return

        def rotate(rotation, vectors):
            cross = 2.0 * np.cross(rotation[..., :3], vectors)
            return vectors + rotation[..., 3, None] * cross + np.cross(rotation[..., :3], cross)

        centers = bottles[:, :3] + rotate(bottles[:, 3:], self.model.body_com.numpy()[self.bottle_bodies])
        design = self.controller.case_modules.design.CASE
        half_width = np.array([design["L"], design["W"]]) * 0.5
        expected = int(np.prod(design["cells"]))
        sealed = self.controller.sealed.numpy()
        for index, bodies in enumerate(self.controller.case_body_groups.values()):
            if not sealed[index]:
                continue
            case = poses[bodies[0]]
            inverse = case[3:] * np.array([-1.0, -1.0, -1.0, 1.0])
            local = rotate(inverse, centers - case[:3])
            inside = np.all(np.abs(local[:, :2]) < half_width, axis=1) & (local[:, 2] > 0.0)
            count = np.count_nonzero(inside & (local[:, 2] < design["H"]))
            label = self.model.body_label[bodies[0]]
            assert count == expected, f"sealed case {label} contains {count} bottles, expected {expected}"

    def test_post_step(self):
        """Check the complete body state once per simulated second."""
        if round(self.sim_time / self.frame_dt) % 60:
            return
        poses = self.state_0.body_q.numpy()
        assert np.isfinite(poses).all(), "nonfinite body pose"
        assert np.isfinite(self.state_0.body_qd.numpy()).all(), "nonfinite body velocity"
        self._check_products(poses)

    def test_final(self):
        """Check finite states and report bottle motion and contact workload."""
        poses = self.state_0.body_q.numpy()
        assert np.isfinite(poses).all(), "nonfinite body pose"
        assert np.isfinite(self.state_0.body_qd.numpy()).all(), "nonfinite body velocity"
        drift = self._check_attachments(poses)
        bottles = poses[self.bottle_bodies]
        self._check_products(poses)
        travel = np.linalg.norm(bottles[:, :2] - self.initial_poses[self.bottle_bodies, :2], axis=1)
        print(
            f"PhoenX bottling plant: {len(bottles)} bottles, max travel {travel.max():.3f} m, "
            f"minimum base height {bottles[:, 2].min():.3f} m, "
            f"{self.contacts.rigid_contact_count.numpy()[0]} contacts, max anchor error {drift:.4f} m at {self.sim_time:.2f} s"
        )
        if self.controller is not None:
            print(json.dumps(self.controller.report(), indent=2))
            assert all(plc.fault is None for plc in self.controller.plcs), "an inlet PLC faulted"
            assert all(line.station != "FAULT" for line in self.controller.case_lines), "a case station faulted"
            assert all(sealer.state != "FAULT" for sealer in self.controller.sealers), "a sealer faulted"
            assert all(p.fault is None for p in self.controller.transport.palletizers.values()), "a palletizer faulted"
            if self.sim_time >= 22.0:
                assert (self.controller.placed.numpy() >= 12).all(), "each packer must fill at least one case"
            if self.sim_time + 0.5 * self.frame_dt >= 130.0:
                for name, palletizer in self.controller.transport.palletizers.items():
                    assert palletizer.placed, f"{name} must place at least one case within 130 seconds"
            assert int(self.solver.dynamics.error_count.numpy()[0]) == 0, "a grasp constraint could not be created"

    def run_timing(self):
        """Measure steady-state physics separately from setup and compilation."""
        for _ in range(self.args.timing_warmup):
            self.step()
        wp.synchronize_device(self.model.device)
        started = time.perf_counter()
        for _ in range(self.args.timing_frames):
            self.step()
        wp.synchronize_device(self.model.device)
        elapsed = time.perf_counter() - started
        result = {
            "bodies": self.model.body_count,
            "bottles": len(self.bottle_bodies),
            "shapes": self.model.shape_count,
            "joints": self.model.joint_count,
            "frames": self.args.timing_frames,
            "physics_ms": elapsed * 1000.0 / self.args.timing_frames,
            "physics_fps": self.args.timing_frames / elapsed,
            "substeps": self.args.substeps,
            "iterations": self.args.iterations,
        }
        self.test_final()
        print(json.dumps(result, indent=2))
        if self.args.timing_output:
            Path(self.args.timing_output).write_text(json.dumps(result, indent=2) + "\n")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(viewer="optix", optix_max_instances=262144)
        parser.add_argument(
            "--asset-root",
            default=newton.examples.get_local_asset_directory("bottling_plant"),
            help="Local bottling_plant directory containing assets/, plant/, scenarios/, and sim/.",
        )
        parser.add_argument("--visuals", action=argparse.BooleanOptionalAction, default=None)
        parser.add_argument("--cutaway", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--motor-off", action="store_true", help="Leave conveyor surfaces stationary.")
        parser.add_argument("--controllers", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--mass-splitting", action=argparse.BooleanOptionalAction, default=False)
        parser.add_argument("--diagnostics", action="store_true")
        parser.add_argument(
            "--correct-inertia", action="store_true", help="Balance source actuator inertia approximations."
        )
        parser.add_argument("--substeps", type=int, default=4, help="Collision and solver steps per 60 Hz frame.")
        parser.add_argument("--iterations", type=int, default=4)
        parser.add_argument("--joint-mode", choices=("reduced", "maximal_pgs", "maximal_direct"), default="maximal_pgs")
        parser.add_argument("--timing-frames", type=int, default=0)
        parser.add_argument("--timing-warmup", type=int, default=120)
        parser.add_argument("--timing-output", default=None)
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    if args.timing_frames:
        example.run_timing()
    else:
        newton.examples.run(example, args)
