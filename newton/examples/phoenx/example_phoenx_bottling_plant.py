# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Simulate the full Berlin bottling plant with PhoenX.

Read the downloaded plant's composed USD with an example-specific builder. The
factory keeps its source geometry, rigid bodies, mass properties, colliders,
joint frames, and drive settings. Meshes are shared between repeated parts.
With ``--filling-room-physics``, rigid bottles follow the wet-end belt and
carousel route through the filling room and retain their source mass. The
line starts in production at its rated 425 bottles/min: a packed bed fills the
pasteuriser from its infeed table through the tunnel to its outfeed table, and
a packed field waits on the accumulation table before the IFM.
The source assets remain outside the repository and require NVIDIA Storage
access; see ``newton/examples/assets/bottling_plant/README.md``.

Success criteria:
- Body poses and velocities remain finite.
- Bottles stay off the factory floor, and sealed cases retain all 12 bottles.
- When enabled, the driven wet end moves bottles toward the packer infeed.
- With filling-room physics, a 30-second run delivers an upright bottle to the IFM,
  and no bottle tips on the accumulation table or at the IFM handoff.
- Joint attachments remain within their physical travel.
- Inlet, case-line, sealer, and palletizer controllers remain free of faults.
- Runs of at least 22 simulated seconds fill a case at each of the eight packers.
- Runs of at least 130 simulated seconds deliver a case through every palletizer.
- Runs of at least 130 simulated seconds fill at least three cases at every packer.

Runs:
- `python -m newton.examples phoenx_bottling_plant`
- `python -m newton.examples phoenx_bottling_plant --filling-room-physics`
- `python -m newton.examples phoenx_bottling_plant --viewer null --test --num-frames 600`
- `python -m newton.examples phoenx_bottling_plant --viewer null --timing-frames 600`
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.phoenx.sleeping_filter import (
    PhoenXSleepingFilterData,
    build_phoenx_sleeping_filter_data,
    phoenx_passive_shape,
    phoenx_sleeping_filter,
)
from newton.examples.phoenx._bottling_control import PlantController
from newton.examples.phoenx._bottling_infeed import BOTTLES_PER_MINUTE, BottleSupply
from newton.examples.phoenx._bottling_scene import load_plant
from newton.geometry import BroadPhaseImplicitTree
from newton.solvers import SolverPhoenX

_PRODUCTION_SECONDS = 330.0  # s, the bottling benchmark's default run length


# OptiX allocates an instance for every shape, hidden colliders included.
# With visuals the plant builds 276,775 shapes, and 592,302 with the filling
# room's packed beds (measured 2026-11-23); each capacity leaves headroom.
OPTIX_INSTANCES = {False: 400_000, True: 800_000}


class _FillingRoomAction(argparse.BooleanOptionalAction):
    """Toggle filling-room physics and size the OptiX instance capacity for it."""

    def __call__(self, parser, namespace, values, option_string=None):
        super().__call__(parser, namespace, values, option_string)
        enabled = bool(getattr(namespace, self.dest))
        # Keep a capacity the caller chose; replace only the other mode's default.
        if getattr(namespace, "optix_max_instances", None) == OPTIX_INSTANCES[not enabled]:
            namespace.optix_max_instances = OPTIX_INSTANCES[enabled]


def _fleet_direct_contact_mask(joint_labels: list[str], exact: np.ndarray) -> np.ndarray | None:
    """Select fleet mechanisms for projected contact solves."""
    if not np.any(exact):
        return None
    fleet = "/Plant/Logistics/Fleet/"
    loaded_transfer = (fleet + "ab_U/", fleet + "ab_D/")
    # Keep fork support in the same temporal sweeps as the carton stack.
    return exact & np.asarray(
        [label.startswith(fleet) and not label.startswith(loaded_transfer) for label in joint_labels], dtype=bool
    )


def _local_fixture_drives(builder: newton.ModelBuilder) -> list[int]:
    """Use the contact timestep for independent rollers and stops on world-fixed packing fixtures."""
    incident = [[] for _ in builder.body_flags]
    for joint, (parent, child) in enumerate(zip(builder.joint_parent, builder.joint_child, strict=True)):
        if parent >= 0:
            incident[parent].append(joint)
        if child >= 0:
            incident[child].append(joint)

    local = []
    for weld, kind in enumerate(builder.joint_type):
        if kind != newton.JointType.FIXED or builder.joint_parent[weld] != -1 or not builder.joint_enabled[weld]:
            continue
        mount = builder.joint_child[weld]
        if mount < 0:
            continue
        drives = [joint for joint in incident[mount] if joint != weld]
        if not all(
            builder.joint_parent[j] == mount
            and builder.joint_type[j] in (newton.JointType.PRISMATIC, newton.JointType.REVOLUTE)
            and builder.joint_child[j] >= 0
            and len(incident[builder.joint_child[j]]) == 1
            for j in drives
        ):
            continue
        has_stop = any(
            builder.joint_type[j] == newton.JointType.PRISMATIC
            and builder.joint_target_ke[builder.joint_qd_start[j]] > 0
            for j in drives
        )
        if has_stop and any(builder.joint_type[j] == newton.JointType.REVOLUTE for j in drives):
            # The permanent world weld prescribes the mount pose. Its reaction
            # cannot couple the roller and stop velocities through the mount.
            builder.body_flags[mount] = int(newton.BodyFlags.KINEMATIC)
            local.extend(drives)
    return local


@wp.struct
class _PlantFilter:
    product: wp.array[wp.int32]
    collision_membership: wp.array[wp.uint64]
    collision_filtered: wp.array[wp.uint64]
    articulation: wp.array[wp.int32]
    shape_body: wp.array[wp.int32]
    held_by: wp.array[wp.int32]
    # The solver's filter culls pairs between sleeping bodies.
    sleeping: PhoenXSleepingFilterData


@wp.func
def _passive_shape(shape: int, data: _PlantFilter) -> bool:
    return phoenx_passive_shape(shape, data.sleeping)


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
    return phoenx_sleeping_filter(pair, data.sleeping)


@wp.kernel
def _sleep_colors(
    active: wp.array[wp.vec3],
    shape_body: wp.array[wp.int32],
    island_root: wp.array[wp.int32],
    output: wp.array[wp.vec3],
):
    shape = wp.tid()
    body = shape_body[shape]
    color = active[shape]
    if body >= 0 and island_root[body + 1] >= 0:
        color *= 0.55
    output[shape] = color


class Example:
    """One factory world, with CUDA graph capture for collision and stepping."""

    overlap_simulation_render = True

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.args = args
        if args.substeps < 1:
            raise ValueError("--substeps must be positive")
        if args.solver_substeps < 1:
            raise ValueError("--solver-substeps must be positive")
        if args.cartons_per_packer < 1:
            raise ValueError("--cartons-per-packer must be positive")
        if args.extra_bottles is not None and args.extra_bottles < 0:
            raise ValueError("--extra-bottles must be nonnegative")
        if args.max_colored_partitions < 1:
            raise ValueError("--max-colored-partitions must be positive")
        if not np.isfinite(args.sleeping_velocity_threshold) or args.sleeping_velocity_threshold < 0.0:
            raise ValueError("--sleeping-velocity-threshold must be finite and nonnegative")
        if not np.isfinite(args.sleeping_motion_threshold) or args.sleeping_motion_threshold < 0.0:
            raise ValueError("--sleeping-motion-threshold must be finite and nonnegative")
        if args.controllers and not args.motor_off and args.joint_mode != "maximal_pgs":
            raise ValueError("Physical runtime grasp joints require --joint-mode maximal_pgs")
        self.frame_dt = 1.0 / 60.0
        self.sim_time = 0.0
        self.scene = load_plant(
            Path(args.asset_root).expanduser(),
            cutaway=args.cutaway,
            visuals=args.visuals if args.visuals is not None else args.viewer != "null",
            motor_off=args.motor_off,
            cartons_per_packer=args.cartons_per_packer,
            filling_room_physics=args.filling_room_physics,
        )
        self.supply = None
        if args.controllers and not args.motor_off:
            carton_count = sum(len(cases) for cases in self.scene.metadata["case_inventory"].values())
            # Four tape bonds and one wrap bond per carton, plus packer/palletizer grasps.
            joints = max(320, 5 * carton_count + 36)
            # The filler runs at its rated output for the whole run, and at least
            # long enough to fill every carton.
            duration = max(_PRODUCTION_SECONDS, args.num_frames * self.frame_dt)
            demand = max(
                12 * carton_count - len(self.scene.bottle_bodies), math.ceil(BOTTLES_PER_MINUTE / 60.0 * duration)
            )
            extra = demand if args.extra_bottles is None else min(demand, args.extra_bottles)
            if extra:
                self.supply = BottleSupply(self.scene, extra, joints)
            else:
                self.scene.builder.reserve_dynamic_capacity(joints=joints)
        self.scene.builder.validate_inertia_detailed = args.diagnostics
        # Several source actuator tensors are positive definite but violate
        # the triangle inequality. Keep those authored approximations unless
        # the caller explicitly asks Newton to balance them.
        local_drives = []
        if args.temporal_iterations and args.joint_mode == "maximal_pgs" and not args.motor_off:
            local_drives = _local_fixture_drives(self.scene.builder)
        self.model = self.scene.builder.finalize(skip_validation_inertia=not args.correct_inertia)
        if args.viewer == "optix" and self.model.shape_count > args.optix_max_instances:
            raise ValueError(
                f"The plant has {self.model.shape_count} shapes, more than --optix-max-instances "
                f"{args.optix_max_instances} OptiX instances"
            )
        initial_state = self.model.state()
        newton.eval_ik(self.model, initial_state, self.model.joint_q, self.model.joint_qd)
        coordinates = self.model.joint_q.numpy()
        starts = self.model.joint_q_start.numpy()
        for joint, kind in enumerate(self.model.joint_type.numpy()):
            if kind == int(newton.JointType.REVOLUTE):
                index = starts[joint]
                coordinates[index] = (coordinates[index] + np.pi) % (2.0 * np.pi) - np.pi
        self.model.joint_q.assign(coordinates)
        for mesh, velocity, label in self.scene.belts:
            if label in self.scene.metadata.get("wrapper_belt_zones", {}):
                mesh.mesh.velocities.zero_()
            else:
                mesh.mesh.velocities.assign(velocity)
        product = np.zeros(self.model.shape_count, dtype=np.int32)
        groups = np.zeros(self.model.shape_count, dtype=np.int32)
        scopes = sorted(set(self.scene.self_collision_scopes), key=len)
        group_of = {scope: group for group, scope in enumerate(scopes, 1)}
        for shape, label in enumerate(self.model.shape_label):
            if label.startswith("/Plant/Sim/Product/Bottles/"):
                product[shape] = 2 if "/Collision/Hull_00/" in label else 1
            # The longest scope that contains the shape owns it.
            prefix = label
            while (cut := prefix.rfind("/")) > 0:
                prefix = prefix[:cut]
                if group := group_of.get(prefix):
                    groups[shape] = group
                    break
        self.filter_data = _PlantFilter()
        membership, filtered = self.scene.collision_masks(self.model.shape_label)
        if self.supply is not None:
            self.supply.copy_masks(membership, filtered)
        self.filter_data.collision_membership = wp.array(membership, dtype=wp.uint64, device=self.model.device)
        self.filter_data.collision_filtered = wp.array(filtered, dtype=wp.uint64, device=self.model.device)
        self.filter_data.product = wp.array(product, dtype=wp.int32, device=self.model.device)
        self.filter_data.articulation = wp.array(groups, dtype=wp.int32, device=self.model.device)
        self.filter_data.shape_body = self.model.shape_body
        self.filter_data.held_by = wp.zeros(self.model.body_count, dtype=int, device=self.model.device)
        self.filter_data.sleeping = build_phoenx_sleeping_filter_data(
            sleeping_enabled=False,
            phoenx_body_offset=1,
            shape_body=None,
            body_island_root=None,
            body_motion_type=None,
            device=self.model.device,
        )
        self.pipeline = newton.CollisionPipeline(
            self.model,
            contact_matching="sticky",
            # Each bottle of the filling room's packed beds touches about six others.
            rigid_contact_max=250_000 + 64 * self.scene.metadata.get("process_bottles", 0),
            broad_phase="sap" if self.model.device.is_cuda or self.supply is not None else "nxn",
            speculative_contact_gap_max=0.005,
            broad_phase_filter=(_filter_pair, _PlantFilter),
            max_triangle_pairs=1_000_000,
            shape_pairs_max=max(
                65536,
                32 * int(np.count_nonzero(self.model.shape_flags.numpy() & int(newton.ShapeFlags.COLLIDE_SHAPES))),
            ),
            include_static_kinematic_pairs=False,
        )
        if self.model.device.is_cuda:
            # A spatial tree keeps the many sleeping warehouse rows out of distant queries.
            self.pipeline.broad_phase = BroadPhaseImplicitTree(
                self.model.shape_world,
                self.model.shape_flags,
                dynamic_rows=self.pipeline._dynamic_rows,
                passive_shape=_passive_shape,
                filter_func=_filter_pair,
                filter_data_type=_PlantFilter,
                device=self.model.device,
            )
        self.pipeline.set_broad_phase_filter_data(self.filter_data)
        sleep_motion = args.sleeping_motion_threshold if args.sleeping_velocity_threshold > 0.0 else 0.0
        sleep_frames = 1 if sleep_motion > 0.0 else 30 * args.substeps
        if args.joint_mode == "maximal_pgs":
            mass_splitting = args.mass_splitting
            # Exact solves are faster for these driven scalar-joint mechanisms.
            # D6 closures and runtime grasp/tape joints remain local blocks.
            exact = np.asarray(
                [
                    label.startswith("/Plant/")
                    and not label.startswith("/Plant/Sim/Product/Cases/")
                    and kind != int(newton.JointType.D6)
                    for label, kind in zip(self.model.joint_label, self.model.joint_type.numpy(), strict=True)
                ],
                dtype=bool,
            )
            exact[local_drives] = False
            self.solver = SolverPhoenX(
                self.model,
                collision_pipeline=self.pipeline,
                substeps=args.solver_substeps,
                solver_iterations=args.iterations,
                temporal_iterations=args.temporal_iterations,
                contact_hertz=args.contact_hertz,
                contact_chunk_size=64,
                step_layout="single_world",
                joint_mode="maximal_pgs",
                direct_joint_mask=exact,
                direct_contact_mask=_fleet_direct_contact_mask(self.model.joint_label, exact),
                velocity_iterations=1,
                enable_body_pair_grouping=True,
                parallel_contact_prepare=self.model.device.is_cuda and not mass_splitting,
                mass_splitting=mass_splitting,
                mass_splitting_overflow_only=mass_splitting,
                max_colored_partitions=args.max_colored_partitions,
                mass_splitting_batch_size=1,
                mass_splitting_unrolled=mass_splitting,
                partitioner_algorithm="endpoint_owner" if mass_splitting else "greedy",
                sleeping_velocity_threshold=args.sleeping_velocity_threshold,
                sleeping_frames_required=sleep_frames,
                sleeping_motion_threshold=sleep_motion,
            )
        else:
            self.solver = SolverPhoenX(
                self.model,
                collision_pipeline=self.pipeline,
                step_layout="single_world",
                substeps=args.solver_substeps,
                solver_iterations=args.iterations,
                temporal_iterations=args.temporal_iterations,
                contact_hertz=args.contact_hertz,
                joint_mode=args.joint_mode,
                reduced_articulations=self.scene.articulations if args.joint_mode == "reduced" else None,
                velocity_iterations=1,
                sleeping_velocity_threshold=args.sleeping_velocity_threshold,
                sleeping_frames_required=sleep_frames,
                sleeping_motion_threshold=sleep_motion,
            )
        # Runtime joints install their default data during solver setup. This
        # filter also handles grasp pairs, alongside the source's hull groups.
        if self.solver._sleeping_filter_data is not None:
            self.filter_data.sleeping = self.solver._sleeping_filter_data
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
                self.supply,
            )
        self.joint_axes = self.model.joint_axis.numpy()
        self.joint_dofs = self.model.joint_qd_start.numpy()
        self.initial_poses = self.state_0.body_q.numpy()
        self.bottle_bodies = np.asarray(self.scene.bottle_bodies, dtype=np.int32)
        viewer.set_model(self.model)
        viewer.set_camera(wp.vec3(49.0, 48.0, 3.4), pitch=-12.0, yaw=10.0)
        if self.controller is not None:
            # Seed the first frame's sensors and host snapshot.
            self.controller.sense()
            self.controller.link.publish()
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
        self._render_contacts = [None, None]
        self._render_contact_snapshot = None
        self._sleep_color_active = wp.clone(self.model.shape_color) if self.solver._sleeping_enabled else None
        self._render_sleep_roots = (
            tuple(wp.empty_like(self.solver.bodies.island_root) for _ in range(2))
            if self._sleep_color_active is not None and overlap
            else None
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
        if self._render_sleep_roots is not None:
            wp.copy(self._render_sleep_roots[self._render_state_index], self.solver.bodies.island_root)
        self._render_contact_snapshot = None
        if self.viewer.show_contacts:
            wp.copy(self._render_states[self._render_state_index].body_qd, self.state_0.body_qd)
            contacts = self._render_contacts[self._render_state_index]
            if contacts is None:
                contacts = newton.Contacts(
                    self.contacts.rigid_contact_max,
                    0,
                    device=self.contacts.device,
                    requested_attributes={"force"} if self.contacts.force is not None else None,
                )
                self._render_contacts[self._render_state_index] = contacts
            # Contact glyphs share the pose snapshot's reuse event, so rendering
            # cannot race the next collision/solver update of the live arrays.
            for name in (
                "rigid_contact_count",
                "rigid_contact_shape0",
                "rigid_contact_shape1",
                "rigid_contact_point0",
                "rigid_contact_point1",
                "rigid_contact_offset0",
                "rigid_contact_normal",
                "force",
            ):
                source = getattr(self.contacts, name)
                if source is not None:
                    wp.copy(getattr(contacts, name), source)
            self._render_contact_snapshot = contacts
        self._render_time = self.sim_time
        self._render_state_prepared = True

    def simulate(self):
        self.state_0.clear_forces()
        if self.controller is not None:
            self.controller.simulate()
        self.viewer.apply_forces(self.state_0)
        self.solver.wake_on_external_input(self.state_0)
        dt = self.frame_dt / self.args.substeps
        for _ in range(self.args.substeps):
            # Refresh contacts at the physics rate: sealed cases have thin
            # floors that must keep supporting bottles while being lifted.
            self.pipeline.collide(self.state_0, self.contacts, dt=dt)
            self.solver.step(self.state_0, self.state_0, self.control, self.contacts, dt)
        if self.controller is not None:
            self.controller.transport.sense_pallet_support(self.contacts)
            self.controller.sense()

    def step(self):
        if self.supply is not None:
            self.supply.admit()
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        if self.controller is not None:
            # Host PLCs overlap this frame's GPU work; see PlantController.after_step.
            self.controller.after_step()
        self.sim_time += self.frame_dt

    def render(self):
        state = self._render_states[self._render_state_index] if self._render_state_prepared else self.state_0
        self.viewer.begin_frame(self._render_time if self._render_state_prepared else self.sim_time)
        if self._sleep_color_active is not None:
            roots = (
                self._render_sleep_roots[self._render_state_index]
                if self._render_state_prepared
                else self.solver.bodies.island_root
            )
            wp.launch(
                _sleep_colors,
                dim=self.model.shape_count,
                inputs=[self._sleep_color_active, self.model.shape_body, roots],
                outputs=[self.model.shape_color],
                device=self.model.device,
            )
        self.viewer.log_state(state)
        contacts = self._render_contact_snapshot if self._render_state_prepared else self.contacts
        if contacts is not None or not self.viewer.show_contacts:
            # A disabled call hides old glyphs without reading live buffers. If
            # enabled after snapshotting, defer glyphs until the next snapshot.
            self.viewer.log_contacts(contacts if contacts is not None else self.contacts, state)
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
        bodies = self.bottle_bodies
        if self.supply is not None:
            bodies = self.controller.bottles.numpy()
            bodies = bodies[bodies >= 0]
        bottles = poses[bodies]
        # Even floor-level pallets support the case contents above 0.1 m.
        lowest = int(np.argmin(bottles[:, 2]))
        assert float(bottles[lowest, 2]) > 0.1, (
            f"bottle {self.model.body_label[bodies[lowest]]} escaped the line or its case at {bottles[lowest, :3]}"
        )
        handoff = self.scene.metadata.get("process_handoff")
        if handoff is not None and not self.args.motor_off:
            point, support_length, half_width = handoff
            relative = bottles[:, :3] - point
            near = (-support_length < relative[:, 0]) & (relative[:, 0] < 0.80) & (np.abs(relative[:, 1]) < half_width)
            off_deck = bodies[near & (relative[:, 2] < -0.005)]
            assert len(off_deck) == 0, f"bottle {self.model.body_label[off_deck[0]]} sank below the wet-end deck"
            receiving = near & (relative[:, 0] > 0.04)
            tilted = np.linalg.norm(bottles[receiving, 3:5], axis=1) > np.sin(np.radians(7.5))
            fallen = bodies[receiving][tilted]
            assert len(fallen) == 0, f"bottle {self.model.body_label[fallen[0]]} tipped at the wet-end handoff"
        if self.controller is None:
            return

        def rotate(rotation, vectors):
            cross = 2.0 * np.cross(rotation[..., :3], vectors)
            return vectors + rotation[..., 3, None] * cross + np.cross(rotation[..., :3], cross)

        centers = bottles[:, :3] + rotate(bottles[:, 3:], self.model.body_com.numpy()[bodies])
        design = self.controller.case_modules.design.CASE
        half_width = np.array([design["L"], design["W"]]) * 0.5
        expected = int(np.prod(design["cells"]))
        sealed = self.controller.sealed.numpy()
        for index, case_bodies in enumerate(self.controller.case_body_groups.values()):
            if not sealed[index]:
                continue
            case = poses[case_bodies[0]]
            inverse = case[3:] * np.array([-1.0, -1.0, -1.0, 1.0])
            local = rotate(inverse, centers - case[:3])
            inside = np.all(np.abs(local[:, :2]) < half_width, axis=1) & (local[:, 2] > 0.0)
            count = np.count_nonzero(inside & (local[:, 2] < design["H"]))
            label = self.model.body_label[case_bodies[0]]
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
        process_bottles = self.scene.metadata.get("process_bottles", 0)
        if process_bottles and not self.args.motor_off and self.sim_time >= 2.0:
            assert travel[-process_bottles:].max() > 0.1, "wet-end bottles must advance on the process route"
            if self.sim_time + 0.5 * self.frame_dt >= 30.0:
                wet_end = bottles[-process_bottles:]
                infeed_x = self.scene.frames["/Plant/Line/IFM"][3, 0]
                upright = np.linalg.norm(wet_end[:, 3:5], axis=1) < np.sin(np.radians(7.5))
                assert np.any(upright & (wet_end[:, 0] > infeed_x + 0.05)), (
                    "an upright wet-end bottle must reach the IFM"
                )
        live = int(self.controller.bottle_count.numpy()[0]) if self.controller is not None else len(bottles)
        print(
            f"PhoenX bottling plant: {live} bottles, max initial-stock travel {travel.max():.3f} m, "
            f"minimum base height {bottles[:, 2].min():.3f} m, "
            f"{self.contacts.rigid_contact_count.numpy()[0]} contacts, max anchor error {drift:.4f} m at {self.sim_time:.2f} s"
        )
        if self.controller is not None:
            print(json.dumps(self.controller.report(), indent=2))
            assert all(plc.fault is None for plc in self.controller.plcs), "an inlet PLC faulted"
            assert all(line.station != "FAULT" for line in self.controller.case_lines), "a case station faulted"
            assert all(sealer.state != "FAULT" for sealer in self.controller.sealers), "a sealer faulted"
            assert all(p.fault is None for p in self.controller.transport.palletizers.values()), "a palletizer faulted"
            assert all(line["fault"] is None for line in self.controller.wrapper.lines.values()), "a wrapper faulted"
            if self.sim_time >= 22.0:
                assert (self.controller.placed.numpy() >= 12).all(), "each packer must fill at least one case"
            if self.sim_time + 0.5 * self.frame_dt >= 130.0:
                placed = self.controller.placed.numpy()
                for name, count in zip(self.controller.names, placed, strict=True):
                    assert count >= 36, (
                        f"{name} must fill at least three cases within 130 seconds; placed {count} bottles"
                    )
                for name, palletizer in self.controller.transport.palletizers.items():
                    assert palletizer.placed, f"{name} must place at least one case within 130 seconds"
            assert int(self.solver.dynamics.error_count.numpy()[0]) == 0, "a runtime constraint could not be created"

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
            "device": str(self.model.device),
            "physics_cuda_graph": self.graph is not None,
            "includes_host_controllers": self.controller is not None,
            "bodies": self.model.body_count,
            "bottles": int(self.controller.bottle_count.numpy()[0])
            if self.controller is not None
            else len(self.bottle_bodies),
            "extra_bottle_capacity": self.supply.capacity if self.supply is not None else 0,
            "shapes": self.model.shape_count,
            "joints": self.model.joint_count,
            "frames": self.args.timing_frames,
            "physics_ms": elapsed * 1000.0 / self.args.timing_frames,
            "physics_fps": self.args.timing_frames / elapsed,
            "substeps": self.args.substeps,
            "iterations": self.args.iterations,
            "temporal_iterations": self.args.temporal_iterations,
            "mass_splitting": self.solver.world.mass_splitting_enabled,
            "max_colored_partitions": self.args.max_colored_partitions,
            "num_colors": int(self.solver.world._partitioner.num_colors.numpy()[0]),
            "overflow_size": self.solver.step_report().overflow_size,
            "contact_hertz": self.args.contact_hertz,
            "sleeping_velocity_threshold": self.args.sleeping_velocity_threshold,
            "sleeping_bodies": int(
                np.count_nonzero(self.solver.bodies.island_root.numpy()[1 : self.model.body_count + 1] >= 0)
            ),
            "parallel_contact_prepare": self.solver.world.parallel_contact_prepare,
            "joint_mode": self.args.joint_mode,
            "warmup_frames": self.args.timing_warmup,
        }
        self.test_final()
        print(json.dumps(result, indent=2))
        if self.args.timing_output:
            Path(self.args.timing_output).write_text(json.dumps(result, indent=2) + "\n")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        # OptiX also allocates instances for hidden collision geometry.
        parser.set_defaults(viewer="optix", optix_max_instances=OPTIX_INSTANCES[False])
        parser.add_argument(
            "--asset-root",
            default=newton.examples.get_local_asset_directory("bottling_plant"),
            help="Local bottling_plant directory containing assets/, plant/, scenarios/, and sim/.",
        )
        parser.add_argument("--visuals", action=argparse.BooleanOptionalAction, default=None)
        parser.add_argument("--cutaway", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument("--motor-off", action="store_true", help="Leave conveyor surfaces stationary.")
        parser.add_argument(
            "--extra-bottles",
            type=int,
            default=None,
            help="Maximum additional bottles admitted at the line entrance. Defaults to the rated output over the run (at least 330 s) or the remaining carton demand, whichever is larger. Set to zero for the initial stock only.",
        )
        parser.add_argument("--controllers", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument(
            "--cartons-per-packer",
            type=int,
            default=50,
            help="Total cartons per packer, including the initial carton at the filling station.",
        )
        parser.add_argument(
            "--mass-splitting",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Enable experimental overflow-only mass splitting. Off by default for stable pallet placement.",
        )
        parser.add_argument(
            "--max-colored-partitions",
            type=int,
            default=12,
            help="Regular colors before the overflow partition when --mass-splitting is enabled.",
        )
        parser.add_argument("--diagnostics", action="store_true")
        parser.add_argument(
            "--filling-room-physics",
            action=_FillingRoomAction,
            default=False,
            help="Enable physical bottle transport from the depalletiser through the filling room to the IFM.",
        )
        parser.add_argument(
            "--correct-inertia", action="store_true", help="Balance source actuator inertia approximations."
        )
        parser.add_argument("--substeps", type=int, default=2, help="Collision and solver steps per 60 Hz frame.")
        parser.add_argument(
            "--solver-substeps", type=int, default=3, help="Internal solver substeps per collision pass."
        )
        parser.add_argument("--iterations", type=int, default=4, help="Biased solver sweeps per internal substep.")
        parser.add_argument(
            "--sleeping-velocity-threshold",
            type=float,
            default=0.01,
            help="Island velocity cutoff [m/s] for waking and moving kinematics. Zero disables sleeping.",
        )
        parser.add_argument(
            "--sleeping-motion-threshold",
            type=float,
            default=0.0002,
            help="Equivalent material displacement [m] for pose-based settling. Zero uses the velocity test.",
        )
        parser.add_argument(
            "--temporal-iterations",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use experimental contact timesteps within solver sweeps.",
        )
        parser.add_argument(
            "--contact-hertz",
            type=float,
            default=80.0,
            help="Contact spring frequency [Hz].",
        )
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
