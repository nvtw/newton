# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from enum import IntFlag, auto

import warp as wp
import warp.examples

import newton
from newton import GeoType
from newton.examples import test_body_state
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


class TestLevel(IntFlag):
    VELOCITY_X = auto()
    VELOCITY_YZ = auto()
    VELOCITY_LINEAR = VELOCITY_X | VELOCITY_YZ
    VELOCITY_ANGULAR = auto()
    STRICT = VELOCITY_LINEAR | VELOCITY_ANGULAR


def type_to_str(shape_type: GeoType):
    if shape_type == GeoType.SPHERE:
        return "sphere"
    elif shape_type == GeoType.BOX:
        return "box"
    elif shape_type == GeoType.CAPSULE:
        return "capsule"
    elif shape_type == GeoType.CYLINDER:
        return "cylinder"
    elif shape_type == GeoType.MESH:
        return "mesh"
    elif shape_type == GeoType.CONVEX_MESH:
        return "convex_hull"
    else:
        return "unknown"


class CollisionSetup:
    def __init__(
        self,
        viewer,
        device,
        shape_type_a,
        shape_type_b,
        solver_fn,
        sim_substeps,
        use_unified_pipeline=False,
        broad_phase_mode=newton.BroadPhaseMode.EXPLICIT,
    ):
        self.sim_substeps = sim_substeps
        self.frame_dt = 1 / 60
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.shape_type_a = shape_type_a
        self.shape_type_b = shape_type_b
        self.use_unified_pipeline = use_unified_pipeline

        self.builder = newton.ModelBuilder(gravity=0.0)
        self.builder.add_articulation()
        body_a = self.builder.add_body(xform=wp.transform(wp.vec3(-1.0, 0.0, 0.0)))
        self.add_shape(shape_type_a, body_a)
        self.builder.add_joint_free(body_a)

        self.init_velocity = 5.0
        self.builder.joint_qd[0] = self.builder.body_qd[-1][0] = self.init_velocity

        self.builder.add_articulation()
        body_b = self.builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.0)))
        self.add_shape(shape_type_b, body_b)
        self.builder.add_joint_free(body_b)

        self.model = self.builder.finalize(device=device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Initialize collision pipeline
        if use_unified_pipeline:
            self.collision_pipeline = newton.CollisionPipelineUnified.from_model(
                self.model,
                rigid_contact_max_per_pair=20,
                rigid_contact_margin=0.01,
                broad_phase_mode=broad_phase_mode,
            )
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        else:
            self.collision_pipeline = None
            self.contacts = self.model.collide(self.state_0)

        self.solver = solver_fn(self.model)

        self.viewer = viewer
        self.viewer.set_model(self.model)

        self.graph = None
        if wp.get_device(device).is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def add_shape(self, shape_type: GeoType, body: int):
        if shape_type == GeoType.BOX:
            self.builder.add_shape_box(body, key=type_to_str(shape_type))
        elif shape_type == GeoType.SPHERE:
            self.builder.add_shape_sphere(body, radius=0.5, key=type_to_str(shape_type))
        elif shape_type == GeoType.CAPSULE:
            self.builder.add_shape_capsule(body, radius=0.25, half_height=0.3, key=type_to_str(shape_type))
        elif shape_type == GeoType.CYLINDER:
            self.builder.add_shape_cylinder(body, radius=0.25, half_height=0.4, key=type_to_str(shape_type))
        elif shape_type == GeoType.MESH:
            # Use box mesh for unified pipeline (works correctly), sphere mesh for legacy pipeline (box mesh has issues)
            if self.use_unified_pipeline:
                vertices, indices = newton.utils.create_box_mesh(extents=(0.5, 0.5, 0.5))
            else:
                vertices, indices = newton.utils.create_sphere_mesh(radius=0.5)
            self.builder.add_shape_mesh(body, mesh=newton.Mesh(vertices[:, :3], indices), key=type_to_str(shape_type))
        elif shape_type == GeoType.CONVEX_MESH:
            # Use a sphere mesh as it's already convex
            vertices, indices = newton.utils.create_sphere_mesh(radius=0.5)
            mesh = newton.Mesh(vertices[:, :3], indices)
            self.builder.add_shape_convex_hull(body, mesh=mesh, key=type_to_str(shape_type))
        else:
            raise NotImplementedError(f"Shape type {shape_type} not implemented")

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        if self.use_unified_pipeline:
            self.contacts = self.model.collide(self.state_0, collision_pipeline=self.collision_pipeline)
        else:
            self.contacts = self.model.collide(self.state_0)

        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # apply forces to the model
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test(self, test_level: TestLevel, body: int, tolerance: float = 3e-3):
        body_name = f"body {body} ({self.model.shape_key[body]})"
        if test_level & TestLevel.VELOCITY_X:
            test_body_state(
                self.model,
                self.state_0,
                f"{body_name} is moving forward",
                lambda _q, qd: qd[0] > 0.03 and qd[0] <= wp.static(self.init_velocity),
                indices=[body],
                show_body_qd=True,
            )
        if test_level & TestLevel.VELOCITY_YZ:
            test_body_state(
                self.model,
                self.state_0,
                f"{body_name} has correct linear velocity",
                lambda _q, qd: abs(qd[1]) < tolerance and abs(qd[2]) < tolerance,
                indices=[body],
                show_body_qd=True,
            )
        if test_level & TestLevel.VELOCITY_ANGULAR:
            test_body_state(
                self.model,
                self.state_0,
                f"{body_name} has correct angular velocity",
                lambda _q, qd: abs(qd[3]) < tolerance and abs(qd[4]) < tolerance and abs(qd[5]) < tolerance,
                indices=[body],
                show_body_qd=True,
            )


devices = get_cuda_test_devices(mode="basic")


class TestCollisionPipeline(unittest.TestCase):
    pass


# Note that body A does sometimes bounce off body B or continue moving forward
# due to inertia differences, so we only test linear velocity along the Y and Z directions.
# Some collisions also cause unwanted angular velocity, so we only test linear velocity
# for those cases.
contact_tests = [
    (GeoType.SPHERE, GeoType.SPHERE, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.BOX, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.CAPSULE, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.BOX, GeoType.BOX, TestLevel.VELOCITY_YZ, TestLevel.VELOCITY_LINEAR),
    (GeoType.BOX, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.CAPSULE, GeoType.CAPSULE, TestLevel.VELOCITY_YZ, TestLevel.VELOCITY_LINEAR),
    (GeoType.CAPSULE, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.MESH, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
]


def test_collision_pipeline(
    _test, device, shape_type_a: GeoType, shape_type_b: GeoType, test_level_a: TestLevel, test_level_b: TestLevel
):
    viewer = newton.viewer.ViewerNull()
    setup = CollisionSetup(
        viewer=viewer,
        device=device,
        solver_fn=newton.solvers.SolverXPBD,
        sim_substeps=10,
        shape_type_a=shape_type_a,
        shape_type_b=shape_type_b,
    )
    for _ in range(200):
        setup.step()
        setup.render()
    setup.test(test_level_a, 0)
    setup.test(test_level_b, 1)


for shape_type_a, shape_type_b, test_level_a, test_level_b in contact_tests:
    add_function_test(
        TestCollisionPipeline,
        f"test_{type_to_str(shape_type_a)}_{type_to_str(shape_type_b)}",
        test_collision_pipeline,
        devices=devices,
        shape_type_a=shape_type_a,
        shape_type_b=shape_type_b,
        test_level_a=test_level_a,
        test_level_b=test_level_b,
    )


class TestUnifiedCollisionPipeline(unittest.TestCase):
    pass


# Unified collision pipeline tests - now supports both MESH and CONVEX_MESH
# Note: MESH vs MESH is not yet supported
unified_contact_tests = [
    (GeoType.SPHERE, GeoType.SPHERE, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.BOX, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.CAPSULE, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.SPHERE, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.BOX, GeoType.BOX, TestLevel.VELOCITY_YZ, TestLevel.VELOCITY_LINEAR),
    (GeoType.BOX, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.BOX, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.CAPSULE, GeoType.CAPSULE, TestLevel.VELOCITY_YZ, TestLevel.VELOCITY_LINEAR),
    (GeoType.CAPSULE, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.CAPSULE, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    # (GeoType.MESH, GeoType.MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),  # Not yet supported
    (GeoType.MESH, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
    (GeoType.CONVEX_MESH, GeoType.CONVEX_MESH, TestLevel.VELOCITY_YZ, TestLevel.STRICT),
]


def test_unified_collision_pipeline(
    _test,
    device,
    shape_type_a: GeoType,
    shape_type_b: GeoType,
    test_level_a: TestLevel,
    test_level_b: TestLevel,
    broad_phase_mode: newton.BroadPhaseMode,
):
    viewer = newton.viewer.ViewerNull()
    setup = CollisionSetup(
        viewer=viewer,
        device=device,
        solver_fn=newton.solvers.SolverXPBD,
        sim_substeps=10,
        shape_type_a=shape_type_a,
        shape_type_b=shape_type_b,
        use_unified_pipeline=True,
        broad_phase_mode=broad_phase_mode,
    )
    for _ in range(200):
        setup.step()
        setup.render()
    setup.test(test_level_a, 0)
    setup.test(test_level_b, 1)


# Wrapper functions for each broad phase mode
def test_unified_collision_pipeline_explicit(
    _test, device, shape_type_a: GeoType, shape_type_b: GeoType, test_level_a: TestLevel, test_level_b: TestLevel
):
    test_unified_collision_pipeline(
        _test, device, shape_type_a, shape_type_b, test_level_a, test_level_b, newton.BroadPhaseMode.EXPLICIT
    )


def test_unified_collision_pipeline_nxn(
    _test, device, shape_type_a: GeoType, shape_type_b: GeoType, test_level_a: TestLevel, test_level_b: TestLevel
):
    test_unified_collision_pipeline(
        _test, device, shape_type_a, shape_type_b, test_level_a, test_level_b, newton.BroadPhaseMode.NXN
    )


def test_unified_collision_pipeline_sap(
    _test, device, shape_type_a: GeoType, shape_type_b: GeoType, test_level_a: TestLevel, test_level_b: TestLevel
):
    test_unified_collision_pipeline(
        _test, device, shape_type_a, shape_type_b, test_level_a, test_level_b, newton.BroadPhaseMode.SAP
    )


for shape_type_a, shape_type_b, test_level_a, test_level_b in unified_contact_tests:
    # EXPLICIT broad phase tests
    add_function_test(
        TestUnifiedCollisionPipeline,
        f"test_{type_to_str(shape_type_a)}_{type_to_str(shape_type_b)}_explicit",
        test_unified_collision_pipeline_explicit,
        devices=devices,
        shape_type_a=shape_type_a,
        shape_type_b=shape_type_b,
        test_level_a=test_level_a,
        test_level_b=test_level_b,
    )
    # NXN broad phase tests
    add_function_test(
        TestUnifiedCollisionPipeline,
        f"test_{type_to_str(shape_type_a)}_{type_to_str(shape_type_b)}_nxn",
        test_unified_collision_pipeline_nxn,
        devices=devices,
        shape_type_a=shape_type_a,
        shape_type_b=shape_type_b,
        test_level_a=test_level_a,
        test_level_b=test_level_b,
    )
    # SAP broad phase tests
    add_function_test(
        TestUnifiedCollisionPipeline,
        f"test_{type_to_str(shape_type_a)}_{type_to_str(shape_type_b)}_sap",
        test_unified_collision_pipeline_sap,
        devices=devices,
        shape_type_a=shape_type_a,
        shape_type_b=shape_type_b,
        test_level_a=test_level_a,
        test_level_b=test_level_b,
    )


class TestPerShapeContactMargin(unittest.TestCase):
    pass


def test_per_shape_contact_margin(test, device):
    """
    Test that per-shape contact margins work correctly by testing two spheres
    with different margins approaching a plane.
    """
    from newton._src.geometry.narrow_phase import NarrowPhase
    from newton._src.geometry.types import GeoType

    narrow_phase = NarrowPhase(max_candidate_pairs=100, max_triangle_pairs=1000, device=device)

    # Create geometries: plane + 2 spheres with different margins
    geom_types = wp.array([int(GeoType.PLANE), int(GeoType.SPHERE), int(GeoType.SPHERE)], dtype=wp.int32, device=device)
    geom_data = wp.array(
        [
            wp.vec4(0.0, 0.0, 1.0, 0.0),  # Plane (infinite)
            wp.vec4(0.2, 0.2, 0.2, 0.0),  # Sphere A radius=0.2
            wp.vec4(0.2, 0.2, 0.2, 0.0),  # Sphere B radius=0.2
        ],
        dtype=wp.vec4,
        device=device,
    )
    geom_source = wp.zeros(3, dtype=wp.uint64, device=device)
    geom_collision_radius = wp.array([1e6, 0.2, 0.2], dtype=wp.float32, device=device)

    # Contact margins: plane=0.01, sphereA=0.02, sphereB=0.06
    shape_contact_margin = wp.array([0.01, 0.02, 0.06], dtype=wp.float32, device=device)

    # Allocate output arrays
    max_contacts = 10
    contact_pair = wp.zeros(max_contacts, dtype=wp.vec2i, device=device)
    contact_position = wp.zeros(max_contacts, dtype=wp.vec3, device=device)
    contact_normal = wp.zeros(max_contacts, dtype=wp.vec3, device=device)
    contact_penetration = wp.zeros(max_contacts, dtype=float, device=device)
    contact_tangent = wp.zeros(max_contacts, dtype=wp.vec3, device=device)
    contact_count = wp.zeros(1, dtype=int, device=device)

    # Test 1: Sphere A at z=0.25 (outside combined margin 0.03) - no contact
    geom_transform = wp.array(
        [
            wp.transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            wp.transform((0.0, 0.0, 0.25), (0.0, 0.0, 0.0, 1.0)),
            wp.transform((10.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        ],
        dtype=wp.transform,
        device=device,
    )
    pairs = wp.array([wp.vec2i(0, 1)], dtype=wp.vec2i, device=device)
    num_pairs = wp.array([1], dtype=wp.int32, device=device)

    contact_count.zero_()
    narrow_phase.launch(
        pairs,
        num_pairs,
        geom_types,
        geom_data,
        geom_transform,
        geom_source,
        shape_contact_margin,
        geom_collision_radius,
        contact_pair,
        contact_position,
        contact_normal,
        contact_penetration,
        contact_count,  # contact_count comes BEFORE contact_tangent
        contact_tangent,
    )
    wp.synchronize()
    test.assertEqual(contact_count.numpy()[0], 0, "Sphere A outside margin should have no contact")

    # Test 2: Sphere A at z=0.15 (inside margin) - contact!
    geom_transform = wp.array(
        [
            wp.transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            wp.transform((0.0, 0.0, 0.15), (0.0, 0.0, 0.0, 1.0)),
            wp.transform((10.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
        ],
        dtype=wp.transform,
        device=device,
    )

    contact_count.zero_()
    narrow_phase.launch(
        pairs,
        num_pairs,
        geom_types,
        geom_data,
        geom_transform,
        geom_source,
        shape_contact_margin,
        geom_collision_radius,
        contact_pair,
        contact_position,
        contact_normal,
        contact_penetration,
        contact_count,
        contact_tangent,
    )
    wp.synchronize()
    test.assertGreater(contact_count.numpy()[0], 0, "Sphere A inside margin should have contact")

    # Test 3: Sphere B at z=0.23 (inside its larger margin 0.07) - contact!
    geom_transform = wp.array(
        [
            wp.transform((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
            wp.transform((10.0, 0.0, 1.0), (0.0, 0.0, 0.0, 1.0)),
            wp.transform((0.0, 0.0, 0.23), (0.0, 0.0, 0.0, 1.0)),
        ],
        dtype=wp.transform,
        device=device,
    )
    pairs = wp.array([wp.vec2i(0, 2)], dtype=wp.vec2i, device=device)

    contact_count.zero_()
    narrow_phase.launch(
        pairs,
        num_pairs,
        geom_types,
        geom_data,
        geom_transform,
        geom_source,
        shape_contact_margin,
        geom_collision_radius,
        contact_pair,
        contact_position,
        contact_normal,
        contact_penetration,
        contact_count,
        contact_tangent,
    )
    wp.synchronize()
    test.assertGreater(contact_count.numpy()[0], 0, "Sphere B with larger margin should have contact")


def test_per_shape_contact_margin_broad_phase(test, device):
    """
    Test that all broad phase modes correctly handle per-shape contact margins
    by applying them during AABB overlap checks (not pre-expanded).

    Setup two spheres (A and B) at different separations from a ground plane:
    - Sphere A: small margin, should NOT be detected by broad phase when far
    - Sphere B: large margin, SHOULD be detected by broad phase when at same distance

    This tests that the broad phase kernels correctly expand AABBs by the provided
    margins during overlap testing, not requiring pre-expanded AABBs.
    """
    from newton._src.geometry.broad_phase_nxn import BroadPhaseAllPairs
    from newton._src.geometry.broad_phase_sap import BroadPhaseSAP
    from newton._src.geometry.types import GeoType

    # Create UNEXPANDED AABBs for: ground plane + 2 spheres
    # The margins will be passed separately to test that broad phase applies them correctly

    # Ground plane AABB (infinite in XY, at z=0) WITHOUT margin
    ground_aabb_lower = wp.vec3(-1000.0, -1000.0, 0.0)
    ground_aabb_upper = wp.vec3(1000.0, 1000.0, 0.0)

    # Sphere A (radius=0.2, center at z=0.24) WITHOUT margin
    # AABB: z range = [0.24-0.2, 0.24+0.2] = [0.04, 0.44]
    sphere_a_aabb_lower = wp.vec3(-0.2, -0.2, 0.04)
    sphere_a_aabb_upper = wp.vec3(0.2, 0.2, 0.44)

    # Sphere B (radius=0.2, center at z=0.24) WITHOUT margin
    # AABB: z range = [0.04, 0.44]
    sphere_b_aabb_lower = wp.vec3(10.0 - 0.2, -0.2, 0.04)
    sphere_b_aabb_upper = wp.vec3(10.0 + 0.2, 0.2, 0.44)

    aabb_lower = wp.array([ground_aabb_lower, sphere_a_aabb_lower, sphere_b_aabb_lower], dtype=wp.vec3, device=device)
    aabb_upper = wp.array([ground_aabb_upper, sphere_a_aabb_upper, sphere_b_aabb_upper], dtype=wp.vec3, device=device)

    # Pass per-shape margins to broad phase - it will apply them during overlap checks
    # ground=0.01, sphereA=0.02, sphereB=0.06
    # With margins applied:
    # - Ground AABB becomes [-0.01, 0.01] in z
    # - Sphere A AABB becomes [0.04-0.02, 0.44+0.02] = [0.02, 0.46] - does NOT overlap ground
    # - Sphere B AABB becomes [0.04-0.06, 0.44+0.06] = [-0.02, 0.50] - DOES overlap ground
    shape_contact_margin = wp.array([0.01, 0.02, 0.06], dtype=wp.float32, device=device)

    # Use collision group 1 for all shapes (group -1 collides with everything, group 0 means no collision)
    collision_group = wp.array([1, 1, 1], dtype=wp.int32, device=device)
    shape_world = wp.array([0, 0, 0], dtype=wp.int32, device=device)

    # Test NXN broad phase
    nxn_bp = BroadPhaseAllPairs(shape_world, device=device)
    pairs_nxn = wp.zeros(100, dtype=wp.vec2i, device=device)
    pair_count_nxn = wp.zeros(1, dtype=wp.int32, device=device)

    nxn_bp.launch(
        aabb_lower,
        aabb_upper,
        shape_contact_margin,
        collision_group,
        shape_world,
        3,
        pairs_nxn,
        pair_count_nxn,
        device=device,
    )
    wp.synchronize()

    pairs_np = pairs_nxn.numpy()
    count_nxn = pair_count_nxn.numpy()[0]

    # Check that sphere B-ground pair is detected, but sphere A-ground is not
    has_sphere_b_ground = any((p[0] == 0 and p[1] == 2) or (p[0] == 2 and p[1] == 0) for p in pairs_np[:count_nxn])
    has_sphere_a_ground = any((p[0] == 0 and p[1] == 1) or (p[0] == 1 and p[1] == 0) for p in pairs_np[:count_nxn])

    test.assertTrue(has_sphere_b_ground, "NXN: Sphere B (large margin) should overlap ground")
    test.assertFalse(has_sphere_a_ground, "NXN: Sphere A (small margin) should NOT overlap ground")

    # Test SAP broad phase
    sap_bp = BroadPhaseSAP(shape_world, device=device)
    pairs_sap = wp.zeros(100, dtype=wp.vec2i, device=device)
    pair_count_sap = wp.zeros(1, dtype=wp.int32, device=device)

    sap_bp.launch(
        aabb_lower,
        aabb_upper,
        shape_contact_margin,
        collision_group,
        shape_world,
        3,
        pairs_sap,
        pair_count_sap,
        device=device,
    )
    wp.synchronize()

    pairs_np = pairs_sap.numpy()
    count_sap = pair_count_sap.numpy()[0]

    has_sphere_b_ground = any((p[0] == 0 and p[1] == 2) or (p[0] == 2 and p[1] == 0) for p in pairs_np[:count_sap])
    has_sphere_a_ground = any((p[0] == 0 and p[1] == 1) or (p[0] == 1 and p[1] == 0) for p in pairs_np[:count_sap])

    test.assertTrue(has_sphere_b_ground, "SAP: Sphere B (large margin) should overlap ground")
    test.assertFalse(has_sphere_a_ground, "SAP: Sphere A (small margin) should NOT overlap ground")


# Register per-shape contact margin tests
add_function_test(
    TestPerShapeContactMargin,
    "test_per_shape_contact_margin",
    test_per_shape_contact_margin,
    devices=devices,
)
add_function_test(
    TestPerShapeContactMargin,
    "test_per_shape_contact_margin_broad_phase",
    test_per_shape_contact_margin_broad_phase,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
