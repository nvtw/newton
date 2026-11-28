# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Exercise grouped CUDA joint RHS policy and scalar equivalence."""

import functools
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.phoenx import simulation as phoenx_simulation
from newton._src.solvers.phoenx.body import BodyContainer
from newton._src.solvers.phoenx.constraints.bilateral_joint import get_iterate_bilateral_joint_block
from newton._src.solvers.phoenx.constraints.constraint_container import ConstraintContainer
from newton._src.solvers.phoenx.dispatch.color_groups import (
    get_sweep_block_dim,
    get_sweep_kernel,
    use_cooperative_joint_rhs,
)
from newton._src.solvers.phoenx.mass_splitting.copy_state import CopyStateContainer
from newton._src.solvers.phoenx.particle import ParticleContainer
from newton._src.solvers.phoenx.tests.test_block_joint_policy import make_model, make_solver
from newton._src.solvers.phoenx.tests.test_direct_drive import _cuda_with_graph_capture


@unittest.skipUnless(_cuda_with_graph_capture(), "Cooperative joint RHS requires CUDA")
class TestCooperativeJointRHS(unittest.TestCase):
    def test_grouped_block_joint_launch_uses_cooperative_lanes(self):
        """Select eight lanes per joint while leaving preparation scalar."""
        model = make_model(100.0)
        solver = make_solver(model, mass_splitting=True, mass_splitting_color_group_size=4)
        state = model.state()
        original = wp.launch
        widths = []

        def launch(kernel, *args, **kwargs):
            if kernel.key == "get_sweep_kernel__locals__sweep":
                widths.append(kwargs["block_dim"])
            return original(kernel, *args, **kwargs)

        with patch.object(wp, "launch", launch):
            solver.step(state, state, model.control(), None, 0.01)
        self.assertIn(32, widths)
        self.assertIn(256, widths)

    def test_row_counts_disabled_rows_and_partial_tiles(self):
        """Match scalar outputs for zero to six rows and a partial final warp."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        for pair in range(39):
            parent = builder.add_link(
                xform=wp.transform(wp.vec3(3.0 * pair, 0.0, 0.0), wp.quat_identity()),
                mass=1.0,
                inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0),
            )
            child = builder.add_link(
                xform=wp.transform(wp.vec3(3.0 * pair + 1.0, 0.0, 0.0), wp.quat_identity()),
                mass=2.0,
                inertia=wp.mat33(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0),
            )
            root = builder.add_joint_free(parent)
            fixed = builder.add_joint_fixed(
                parent,
                child,
                parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
            )
            builder.add_articulation([root, fixed])
        model = builder.finalize(device="cuda:0")
        solver = make_solver(model, mass_splitting=True, mass_splitting_color_group_size=4)
        state = model.state()
        solver.step(state, state, model.control(), None, 0.001)
        world = solver.world
        data = world.constraints.bilateral
        counts = data.row_count.numpy()
        owned = np.flatnonzero(counts)
        self.assertEqual(len(owned), 39)
        self.assertTrue(np.all(counts[owned] == 6))
        counts[owned] = np.arange(len(owned)) % 7
        data.row_count.assign(counts)
        valid = data.valid.numpy()
        valid[owned[8]] = 0
        data.valid.assign(valid)
        enabled = world._joint_pgs_enabled.numpy()
        enabled[owned[12]] = 0
        enabled[owned[13]] = 2
        world._joint_pgs_enabled.assign(enabled)
        arrays = [
            world.bodies.velocity,
            world.bodies.angular_velocity,
            world._copy_state.velocity,
            world._copy_state.angular_velocity,
            world.constraints.multipliers,
            data.accumulated,
        ]
        rng = np.random.default_rng(321)
        for array in arrays[:4]:
            array.assign(rng.normal(0.0, 0.1, array.numpy().shape).astype(np.float32))
        saved = [wp.clone(array) for array in arrays]
        topology = world._color_group_data
        inputs = [
            world.constraints,
            world._contact_cols,
            world.bodies,
            world._particles_or_sentinel(),
            world._contact_container,
            world._active_contact_views(),
            world._copy_state,
            world.num_joints,
            world._joint_pgs_enabled,
            world.num_bodies,
            wp.float32(1000.0),
            topology["ids"],
            topology["starts"],
            topology["num_colors"],
            world.mass_splitting_color_group_size,
        ]
        for globally_enabled in (1, 0):
            data.enabled = globally_enabled
            world.constraints.bilateral = data
            for phase in ("iterate", "relax"):
                expected = None
                for cooperative in (False, True):
                    for destination, source in zip(arrays, saved, strict=True):
                        wp.copy(destination, source)
                    block_dim = get_sweep_block_dim(cooperative)
                    wp.launch(
                        get_sweep_kernel(phase, False, cooperative_joints=cooperative),
                        (64, block_dim),
                        inputs,
                        block_dim=block_dim,
                        device=world.device,
                    )
                    actual = [array.numpy().copy() for array in arrays]
                    if expected is None:
                        expected = actual
                    else:
                        for a, b in zip(expected, actual, strict=True):
                            self.assertEqual(a.tobytes(), b.tobytes())

    def test_tile_scatter_matches_scalar_with_and_without_accumulation(self):
        """The tile's gathered wrench scatter reproduces the scalar block bit for bit.

        Covers the biased iterate, which accumulates row impulses, and the
        structural projection, which skips spring rows and must leave the
        accumulated impulses untouched, for zero to six rows.
        """
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        for pair in range(28):
            parent, child = (
                builder.add_link(
                    xform=wp.transform(wp.vec3(3.0 * pair + x, 0.0, 0.0), wp.quat_identity()),
                    mass=mass,
                    inertia=wp.mat33(mass, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0),
                )
                for x, mass in ((0.0, 1.0), (1.0, 2.0))
            )
            options = {
                "parent_xform": wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
                "child_xform": wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
            }
            if pair % 2:
                # A driven hinge orders its spring row after the structural rows.
                builder.add_joint_revolute(
                    parent,
                    child,
                    axis=newton.Axis.Z,
                    target_ke=40.0,
                    target_kd=1.0,
                    target_q=0.2,
                    actuator_mode=newton.JointTargetMode.POSITION,
                    **options,
                )
            else:
                builder.add_joint_fixed(parent, child, **options)
        model = builder.finalize(device="cuda:0", skip_validation_joints=True)
        solver = make_solver(model, sleeping_velocity_threshold=0.0)
        state = model.state()
        solver.step(state, state, model.control(), None, 0.001)
        world = solver.world
        data = world.constraints.bilateral
        counts = data.row_count.numpy()
        self.assertEqual(int(np.count_nonzero(counts == 6)), model.joint_count)
        self.assertTrue(data.row_dynamic.numpy().any())
        counts[1::2] = np.arange(model.joint_count // 2) % 7
        data.row_count.assign(counts)
        rng = np.random.default_rng(7)
        arrays = [world.bodies.velocity, world.bodies.angular_velocity, data.accumulated]
        for array in arrays:
            array.assign(rng.normal(0.0, 0.1, array.numpy().shape).astype(np.float32))
        saved = [wp.clone(array) for array in arrays]
        inputs = [world.constraints, world.bodies, world._particles_or_sentinel(), world._copy_state, model.joint_count]
        for structural_only in (False, True):
            results = []
            for cooperative in (False, True):
                for destination, source in zip(arrays, saved, strict=True):
                    wp.copy(destination, source)
                wp.launch(
                    _make_block_kernel(cooperative, structural_only),
                    model.joint_count * (8 if cooperative else 1),
                    inputs,
                    device=world.device,
                )
                results.append([array.numpy().copy() for array in arrays])
            with self.subTest(structural_only=structural_only):
                scalar, tile = results
                self.assertFalse(np.array_equal(scalar[0], saved[0].numpy()))
                for a, b in zip(scalar, tile, strict=True):
                    self.assertEqual(a.tobytes(), b.tobytes())
                accumulated_changed = not np.array_equal(tile[2], saved[2].numpy())
                self.assertEqual(accumulated_changed, not structural_only)


@functools.cache
def _make_block_kernel(cooperative: bool, structural_only: bool):
    """Solve every joint block once, eight lanes per joint or one."""
    iterate = get_iterate_bilateral_joint_block(
        cooperative, structural_only=structural_only, accumulate=not structural_only
    )
    width = 8 if cooperative else 1

    @wp.kernel(module="unique", enable_backward=False)
    def run(
        constraints: ConstraintContainer,
        bodies: BodyContainer,
        particles: ParticleContainer,
        copies: CopyStateContainer,
        num_joints: wp.int32,
    ):
        tid = wp.tid()
        cid = tid / width
        if cid < num_joints:
            iterate(constraints, cid, bodies, particles, copies, bodies.position.shape[0], -1, True, tid % width)

    return run


def _simulate_single_world_chains(
    *, cooperative: bool, steps: int = 30, fused_threshold: int | None = None
) -> np.ndarray:
    """Drop hinged box chains onto the ground using fused or unrolled dispatch."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for chain in range(24):
        previous = -1
        joints = []
        for link in range(4):
            body = builder.add_link(
                xform=wp.transform(
                    wp.vec3(0.5 * link, 0.6 * chain, 0.3 + 0.05 * chain),
                    wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.2 * chain),
                ),
            )
            builder.add_shape_box(body, hx=0.2, hy=0.1, hz=0.05)
            if previous < 0:
                joints.append(builder.add_joint_free(body))
            else:
                joints.append(
                    builder.add_joint_revolute(
                        previous,
                        body,
                        axis=wp.vec3(0.0, 1.0, 0.0),
                        parent_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
                        child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()),
                    )
                )
            previous = body
        builder.add_articulation(joints)
    model = builder.finalize(device="cuda:0")
    pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
    contacts = pipeline.contacts()

    original = phoenx_simulation.get_singleworld_kernel

    def get_kernel(*args, **kwargs):
        if "cooperative_joints" in kwargs:
            kwargs["cooperative_joints"] = kwargs["cooperative_joints"] and cooperative
            requested.append(kwargs["cooperative_joints"])
            if kwargs.get("fused"):
                fused_requested.append(kwargs["cooperative_joints"])
        return original(*args, **kwargs)

    requested = []
    fused_requested = []
    with patch.object(phoenx_simulation, "get_singleworld_kernel", get_kernel):
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            joint_mode="maximal_pgs",
            step_layout="single_world",
            substeps=2,
            solver_iterations=4,
            velocity_iterations=1,
            mass_splitting=fused_threshold is None,
            mass_splitting_overflow_only=fused_threshold is None,
            mass_splitting_batch_size=1,
            mass_splitting_unrolled=fused_threshold is None,
            max_colored_partitions=2,
            partitioner_algorithm="endpoint_owner" if fused_threshold is None else "greedy",
            colored_contact_headers=fused_threshold is None,
            colored_contact_rows=fused_threshold is None,
        )
        if fused_threshold is not None:
            solver.world._fuse_threshold = fused_threshold
        state = model.state()
        control = model.control()

        def simulate():
            pipeline.collide(state, contacts)
            solver.step(state, state, control, contacts, 1.0 / 120.0)

        simulate()
        with wp.ScopedCapture() as capture:
            simulate()
    assert any(requested) == cooperative
    if fused_threshold is not None:
        assert any(fused_requested) == cooperative
    for _ in range(steps):
        wp.capture_launch(capture.graph)
    return np.concatenate((state.body_q.numpy(), state.body_qd.numpy()), axis=1)


@unittest.skipUnless(_cuda_with_graph_capture(), "Cooperative joint RHS requires CUDA")
class TestSingleWorldCooperativeJoints(unittest.TestCase):
    def test_single_world_cooperative_joints_match_scalar(self):
        """Mixed joint/contact colors keep the scalar trajectory bit for bit."""
        scalar = _simulate_single_world_chains(cooperative=False)
        cooperative = _simulate_single_world_chains(cooperative=True)
        self.assertTrue(np.isfinite(scalar).all())
        self.assertEqual(scalar.tobytes(), cooperative.tobytes())

    def test_fused_cooperative_joints_match_scalar(self):
        """Preserve mixed-color dynamics with fusion thresholds below the block width."""
        for threshold in (32, 256):
            with self.subTest(threshold=threshold):
                scalar = _simulate_single_world_chains(cooperative=False, fused_threshold=threshold)
                cooperative = _simulate_single_world_chains(cooperative=True, fused_threshold=threshold)
                self.assertTrue(np.isfinite(scalar).all())
                self.assertEqual(scalar.tobytes(), cooperative.tobytes())


class TestCooperativeJointPolicy(unittest.TestCase):
    def test_scalar_fallback_selection(self):
        """Keep CPU, contact-only scenes and preparation on the scalar mapping."""
        for phase in ("prepare", "cached_prepare", "iterate", "relax"):
            for is_cuda in (False, True):
                for has_blocks in (False, True):
                    expected = is_cuda and has_blocks and phase in ("iterate", "relax")
                    actual = use_cooperative_joint_rhs(
                        phase=phase,
                        is_cuda=is_cuda,
                        has_bilateral_blocks=has_blocks,
                    )
                    self.assertEqual(actual, expected)
                    self.assertEqual(get_sweep_block_dim(actual), 256 if expected else 32)


if __name__ == "__main__":
    unittest.main()
