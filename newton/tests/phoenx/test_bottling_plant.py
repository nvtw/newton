# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the example's USD adapter without requiring the private plant assets."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.phoenx._bottling_control import _seal_cases, _target_coordinates
from newton.examples.phoenx._bottling_io import _case_drives, _photoeyes, _valves
from newton.examples.phoenx._bottling_scene import load_plant
from newton.examples.phoenx._bottling_wrap import build_wrapper, mesh_components
from newton.examples.phoenx.example_phoenx_bottling_plant import Example
from newton.tests.unittest_utils import requires
from newton.viewer import ViewerNull


@requires("pxr.Usd")
class TestBottlingPlant(unittest.TestCase):
    def test_contact_overlay_uses_render_snapshot(self):
        """Render contact geometry and keep it independent of advancing physics."""
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.49), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.5)
            builder.add_ground_plane()
            model = builder.finalize()
            example = Example.__new__(Example)
            example.model = model
            example.state_0 = model.state()
            pipeline = newton.CollisionPipeline(model)
            example.contacts = pipeline.contacts()
            pipeline.collide(example.state_0, example.contacts)
            count = int(example.contacts.rigid_contact_count.numpy()[0])
            self.assertGreater(count, 0)
            example.viewer = ViewerNull()
            example.viewer.set_model(model)
            example.viewer.show_contacts = True
            example.viewer.show_contact_disks = False
            example.viewer.show_contact_forces = False
            example._render_states = (model.state(), model.state())
            example._render_state_done = (None, None)
            example._render_contacts = [None, None]
            example._render_contact_snapshot = None
            example._render_state_index = 0
            example._render_state_prepared = False
            example.sim_time = 0.0
            example.render()
            self.assertIsNotNone(example.viewer._contact_points0)
            expected = example.viewer._contact_points0.numpy()[:count].copy()
            lengths = np.linalg.norm(example.viewer._contact_points1.numpy()[:count] - expected, axis=1)
            self.assertTrue(np.all(lengths > 0.0))
            example.prepare_render_state()
            snapshot = example._render_contact_snapshot
            self.assertIsNot(snapshot, example.contacts)
            # Physics may overwrite its buffers while the render stream consumes
            # the snapshot. The displayed arrows must retain the old geometry.
            example.contacts.rigid_contact_count.zero_()
            example.contacts.rigid_contact_point0.fill_(wp.vec3(100, 100, 100))
            example.state_0.body_q.fill_(wp.transform(wp.vec3(100, 0, 0), wp.quat_identity()))
            example.render()
            np.testing.assert_allclose(example.viewer._contact_points0.numpy()[:count], expected)
            self.assertEqual(int(snapshot.rigid_contact_count.numpy()[0]), count)
            example.viewer.show_contacts = False
            example.prepare_render_state()
            self.assertIsNone(example._render_contact_snapshot)
            # Enabling in begin-frame UI after snapshotting must defer contacts,
            # not read live buffers on the concurrently running physics stream.
            example.viewer.show_contacts = True
            example.render()
            np.testing.assert_allclose(example.viewer._contact_points0.numpy()[:count], expected)

    def test_offset_box_hulls_preserve_exact_geometry(self):
        """Recognize float32 box corners without replacing tapered hulls."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cpu"):
            root = Path(directory)
            path = root / "scenarios/line_sim/line_sim.usda"
            path.parent.mkdir(parents=True)
            stage = Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            corners = np.asarray(
                [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
            )
            points = np.asarray(corners * (0.142875, 0.0015875, 0.127) + (0, 0.1063625, 0.130175), dtype=np.float32)
            for name in ("Box", "Tapered"):
                mesh = UsdGeom.Mesh.Define(stage, f"/Plant/{name}")
                vertices = points.copy()
                if name == "Tapered":
                    vertices[4:, 0] *= 0.99
                mesh.CreatePointsAttr(vertices.tolist())
                mesh.CreateFaceVertexCountsAttr([4] * 6)
                mesh.CreateFaceVertexIndicesAttr(
                    [0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7]
                )
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
            stage.GetRootLayer().Save()

            builder = load_plant(root, visuals=False, motor_off=True).builder
            box = builder.shape_label.index("/Plant/Box/collision")
            tapered = builder.shape_label.index("/Plant/Tapered/collision")
            self.assertEqual(builder.shape_type[box], newton.GeoType.BOX)
            self.assertEqual(builder.shape_type[tapered], newton.GeoType.CONVEX_MESH)
            center = np.asarray(builder.shape_transform[box].p)
            half = np.asarray(builder.shape_scale[box])
            np.testing.assert_allclose(center - half, points.min(axis=0), rtol=0, atol=2e-8)
            np.testing.assert_allclose(center + half, points.max(axis=0), rtol=0, atol=2e-8)

    def test_wrapper_components_preserve_material_seams(self):
        """Split disconnected parts without welding away authored UV seams."""
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [3, 0, 0], [4, 0, 0], [3, 1, 0]],
            dtype=np.float32,
        )
        uvs = np.arange(18, dtype=np.float32).reshape(-1, 2) / 18
        normals = np.tile([0, 0, 1], (9, 1)).astype(np.float32)
        mesh = newton.Mesh(
            vertices,
            np.arange(9),
            normals=normals,
            uvs=uvs,
            color=(0.2, 0.4, 0.6),
            roughness=0.3,
            metallic=0.7,
            opacity=0.8,
            texture="wrapper.png",
            texture_transform=((2, 0, 0.1), (0, 3, 0.2)),
            compute_inertia=False,
        )
        parts = mesh_components(mesh)
        self.assertEqual(len(parts), 2)
        self.assertEqual(len(parts[0].vertices), 6)
        for part, indices in zip(parts, (np.arange(6), np.arange(6, 9)), strict=True):
            np.testing.assert_array_equal(part.vertices, vertices[indices])
            np.testing.assert_array_equal(part.uvs, uvs[indices])
            np.testing.assert_array_equal(part.normals, normals[indices])
            np.testing.assert_array_equal(part.color, mesh.color)
            np.testing.assert_array_equal(part.texture_transform, mesh.texture_transform)
            self.assertEqual(part.texture, mesh.texture)
            self.assertEqual(part.roughness, mesh.roughness)
            self.assertEqual(part.metallic, mesh.metallic)
            self.assertEqual(part.opacity, mesh.opacity)
        triangles = np.concatenate([part.vertices[part.indices] for part in parts])
        np.testing.assert_array_equal(triangles, mesh.vertices[mesh.indices])
        np.testing.assert_array_equal(mesh.vertices, vertices)
        np.testing.assert_array_equal(mesh.uvs, uvs)

    @unittest.skipUnless(wp.is_cuda_available(), "Wrapper direct-contact validation requires CUDA")
    def test_wrapper_homes_and_holds_lift(self):
        """Finite drives park the arm and support the carriage under gravity."""
        with wp.ScopedDevice("cuda:0"):
            components = []
            for name, center, half in (
                ("steel", (0, 0, 3), (0.34, 0.34, 0.0585)),
                ("steel", (-0.95, -0.95, 1.7), (0.065, 0.065, 1.105)),
                ("wrapline_film_roll", (-1.05, -0.53, 0.927), (0.125, 0.125, 0.25)),
            ):
                mesh = newton.Mesh.create_box(*half, compute_inertia=False)
                components.append((name, mesh.copy(vertices=mesh.vertices + np.asarray(center))))
            builder = newton.ModelBuilder()
            bodies, joints, _ = build_wrapper(builder, components, label="/Wrapper", visuals=False)
            model = builder.finalize()
            self.assertTrue(np.all(model.body_mass.numpy() > 0.0))
            self.assertTrue(np.all(np.linalg.eigvalsh(model.body_inertia.numpy()) > 0.0))
            solver = newton.solvers.SolverPhoenX(model, substeps=4, solver_iterations=4)
            state = model.state()
            control = model.control()
            targets = control.joint_target_q.numpy()
            targets[model.joint_target_q_start.numpy()[joints[1]]] = 0.3
            control.joint_target_q.assign(targets)
            contacts = model._collision_pipeline.contacts()
            for _ in range(240):
                solver.collide(state, contacts)
                solver.step(state, state, control, contacts, 1.0 / 60.0)
            newton.eval_ik(model, state, model.joint_q, model.joint_qd)
            q = model.joint_q.numpy()
            self.assertAlmostEqual(float(q[model.joint_q_start.numpy()[joints[0]]]), np.pi / 4, delta=0.03)
            self.assertAlmostEqual(float(q[model.joint_q_start.numpy()[joints[1]]]), 0.3, delta=0.005)
            np.testing.assert_allclose(model.joint_qd.numpy(), 0.0, atol=0.02)
            self.assertEqual(len(bodies), 2)
            self.assertTrue(np.isfinite(state.body_q.numpy()).all())

            # Continuous targets must retain phase through two revolutions.
            # Position-only actuation ignores velocity feed-forward and trails
            # by kd * angular_speed / ke even after acceleration has settled.
            rotor_target = model.joint_target_q_start.numpy()[joints[0]]
            speeds = control.joint_target_qd.numpy()
            speeds[model.joint_qd_start.numpy()[joints[0]]] = np.pi / 2
            speeds[model.joint_qd_start.numpy()[joints[1]]] = 0.1
            control.joint_target_qd.assign(speeds)
            for frame in range(480):
                targets[rotor_target] = np.pi / 4 + (frame + 1) / 60 * np.pi / 2
                targets[model.joint_target_q_start.numpy()[joints[1]]] = 0.3 + (frame + 1) / 60 * 0.1
                control.joint_target_q.assign(targets)
                solver.collide(state, contacts)
                solver.step(state, state, control, contacts, 1.0 / 60.0)
            newton.eval_ik(model, state, model.joint_q, model.joint_qd)
            self.assertAlmostEqual(
                float(model.joint_q.numpy()[model.joint_q_start.numpy()[joints[1]]]), 1.1, delta=0.005
            )
            angle = float(model.joint_q.numpy()[model.joint_q_start.numpy()[joints[0]]])
            error = angle - targets[rotor_target]
            self.assertAlmostEqual(float(np.arctan2(np.sin(error), np.cos(error))), 0.0, delta=0.05)
            self.assertAlmostEqual(
                float(model.joint_qd.numpy()[model.joint_qd_start.numpy()[joints[0]]]), np.pi / 2, delta=0.03
            )

    def test_disabled_cube_belt_can_start(self):
        """Retain a stopped USD cube conveyor's runtime surface-velocity support."""
        from pxr import Sdf, Usd

        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cpu"):
            root = Path(directory)
            path = root / "scenarios/line_sim/line_sim.usda"
            path.parent.mkdir(parents=True)
            path.write_text("""#usda 1.0
(defaultPrim = "Plant"; metersPerUnit = 1; upAxis = "Z")
def Xform "Plant" {
 def Xform "Belt" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
  bool physics:kinematicEnabled = true
  custom bool physxSurfaceVelocity:surfaceVelocityEnabled = false
  custom bool physxSurfaceVelocity:surfaceVelocityLocalSpace = true
  custom vector3f physxSurfaceVelocity:surfaceVelocity = (0.2, 0, 0)
  double3 xformOp:scale = (2, 1, 1)
  uniform token[] xformOpOrder = ["xformOp:scale"]
  def Cube "Strand" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
   double size = 2
   double3 xformOp:scale = (2, 0.025, 0.01)
   uniform token[] xformOpOrder = ["xformOp:scale"]
  }
 }
}
""")
            scene = load_plant(root, visuals=False, motor_off=True)
            self.assertEqual(len(scene.belts), 1)
            mesh, initial_velocity, label = scene.belts[0]
            self.assertEqual(label, "/Plant/Belt")
            np.testing.assert_array_equal(initial_velocity, 0.0)
            shape = scene.builder.shape_source.index(mesh)
            body = scene.builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.02), wp.quat_identity()))
            scene.builder.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01)
            model = scene.builder.finalize()
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            state = model.state()
            for speed in (0.0, 0.2, 0.0):
                velocity = np.zeros_like(initial_velocity)
                velocity[:, 0] = speed / scene.builder.shape_scale[shape][0]
                mesh.mesh.velocities.assign(velocity)
                pipeline.collide(state, contacts)
                count = int(contacts.rigid_contact_count.numpy()[0])
                self.assertGreater(count, 0)
                first = contacts.rigid_contact_shape0.numpy()[:count] == shape
                signs = np.where(first, 1.0, -1.0)
                np.testing.assert_allclose(
                    signs * contacts.rigid_contact_surface_velocity.numpy()[:count, 0], -speed, atol=1e-6
                )
            # Starting enabled must also honor authored linear speed and the
            # schema's degrees/second angular speed, rather than zeroing them.
            stage = Usd.Stage.Open(str(path))
            belt = stage.GetPrimAtPath("/Plant/Belt")
            belt.GetAttribute("physxSurfaceVelocity:surfaceVelocityEnabled").Set(True)
            belt.CreateAttribute("physxSurfaceVelocity:surfaceAngularVelocity", Sdf.ValueTypeNames.Vector3f).Set(
                (0.0, 0.0, 90.0)
            )
            stage.GetRootLayer().Save()
            vendor = root / "sim/exts/bottling_plant.runtime/bottling_plant_runtime/vendor"
            for module, data in (("ifm_v01", '{"belts":[],"speeds_mps":{}}'), ("ctp_v01", '{"modules":{}}')):
                folder = vendor / module
                folder.mkdir(parents=True)
                (folder / "interface.json").write_text(data)
            enabled_scene = load_plant(root, visuals=False)
            moving, velocity, _ = enabled_scene.belts[0]
            scale = np.asarray(enabled_scene.builder.shape_scale[0])
            expected = np.array([0.4, 0.0, 0.0]) + np.cross(
                np.array([0.0, 0.0, np.pi / 2]), np.asarray(moving.vertices) * scale
            )
            np.testing.assert_allclose(velocity * scale, expected, atol=1e-7)

    def test_authored_collision_groups(self):
        """Keep vehicle obstacles while honoring the line's USD exclusions."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice("cpu"):
            root = Path(directory)
            path = root / "scenarios/line_sim/line_sim.usda"
            path.parent.mkdir(parents=True)
            stage = Usd.Stage.CreateNew(str(path))
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdGeom.SetStageUpAxis(stage, "Z")
            paths = ["/Plant/Line/Item", "/Plant/Line/Allowed", "/Plant/Fleet/Truck", "/Plant/Keepouts/Box"]
            for index, label in enumerate(paths):
                cube = UsdGeom.Cube.Define(stage, label)
                cube.CreateSizeAttr(0.2)
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                if index < 3:
                    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
                    UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1.0)
            line = UsdPhysics.CollisionGroup.Define(stage, "/Plant/Groups/Line")
            line.GetCollidersCollectionAPI().GetIncludesRel().SetTargets(["/Plant/Line"])
            line.GetCollidersCollectionAPI().GetExcludesRel().SetTargets([paths[1]])
            obstacles = UsdPhysics.CollisionGroup.Define(stage, "/Plant/Groups/Obstacles")
            obstacles.GetCollidersCollectionAPI().GetIncludesRel().SetTargets(["/Plant/Keepouts"])
            obstacles.CreateFilteredGroupsRel().SetTargets([line.GetPath()])
            stage.GetRootLayer().Save()
            args = Example.create_parser().parse_args(
                [
                    "--asset-root",
                    str(root),
                    "--viewer",
                    "null",
                    "--no-controllers",
                    "--motor-off",
                ]
            )
            example = Example(ViewerNull(), args)
            example.pipeline.collide(example.state_0, example.contacts, dt=1.0 / 240.0)
            count = int(example.contacts.rigid_contact_count.numpy()[0])
            first = example.contacts.rigid_contact_shape0.numpy()[:count]
            second = example.contacts.rigid_contact_shape1.numpy()[:count]
            labels = example.model.shape_label
            pairs = {frozenset((labels[a], labels[b])) for a, b in zip(first, second, strict=True)}
            paths = [path + "/collision" for path in paths]
            self.assertNotIn(frozenset((paths[0], paths[3])), pairs)
            self.assertIn(frozenset((paths[1], paths[3])), pairs)
            self.assertIn(frozenset((paths[2], paths[3])), pairs)
            self.assertIn(frozenset((paths[0], paths[2])), pairs)

    def test_valves_hold_stroke_limits_under_load(self):
        """Hold both cylinder stops under opposing loads with the source valve law."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for _ in range(2):
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_prismatic(
                -1,
                body,
                axis=newton.Axis.X,
                limit_lower=0.0,
                limit_upper=0.05,
                target_kd=100.0,
                effort_limit=10.0,
                actuator_mode=newton.JointTargetMode.VELOCITY,
            )
            builder.add_articulation([joint])
        model = builder.finalize()
        state, control = model.state(), model.control()
        solver = newton.solvers.SolverPhoenX(
            model, joint_mode="maximal_pgs", step_layout="single_world", solver_iterations=4
        )
        wp.launch(
            _valves,
            dim=(1, 2),
            inputs=[
                wp.array([[0, 1]], dtype=int),
                wp.array([[0.05, 0.0]], dtype=float),
                wp.array([0.2, 0.2], dtype=float),
                control.joint_target_qd,
            ],
        )
        state.body_f.assign(np.array([[-2, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0]], dtype=np.float32))
        for _ in range(240):
            solver.step(state, state, control, None, 1.0 / 240.0)
        np.testing.assert_allclose(state.joint_q.numpy(), (0.05, 0.0), atol=1e-5)

    def test_grasp_ray_respects_physical_reach(self):
        """Acquisition requires a real surface hit within the authored 6 mm."""
        mesh = newton.Mesh.create_box(0.03, 0.03, 0.115, compute_inertia=False)
        mesh_id = mesh.finalize()
        # Exercise partial warps and products assigned to different ray lanes.
        product_count, sensor_count = 37, 77
        origins = np.asarray(
            [(0.1 * ((i * 7) % product_count), 0.0, 0.2305) for i in range(sensor_count)], dtype=np.float32
        )
        origins[1::3, 2] = 0.24
        origins[2::3, 0] += 0.035
        hits = wp.zeros((sensor_count // 7, 7), dtype=float)
        wp.launch(
            _photoeyes,
            dim=(sensor_count, 32),
            block_dim=32,
            inputs=[
                wp.array(
                    [wp.transform(wp.vec3(0.1 * i, 0.0, 0.0), wp.quat_identity()) for i in range(product_count)],
                    dtype=wp.transform,
                ),
                wp.array(np.arange(product_count), dtype=int),
                wp.array(origins, dtype=wp.vec3),
                wp.array([(0.0, 0.0, -1.0)] * sensor_count, dtype=wp.vec3),
                wp.full(sensor_count, 0.006, dtype=float),
                wp.array([mesh_id], dtype=wp.uint64),
                wp.array([wp.transform(wp.vec3(0.0, 0.0, 0.115), wp.quat_identity())], dtype=wp.transform),
                wp.array([(1.0, 1.0, 1.0)], dtype=wp.vec3),
                7,
                hits,
            ],
        )
        np.testing.assert_array_equal(hits.numpy().reshape(-1), np.arange(sensor_count) % 3 == 0)

    @unittest.skipUnless(wp.is_cuda_available(), "Runtime sealing requires CUDA")
    def test_tape_bonds_require_closed_flaps(self):
        """Open flaps stay free; closed, taped flaps resist hinge torque."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        root = builder.add_link(mass=0.2, inertia=wp.mat33(np.eye(3) * 0.01))
        joints = [builder.add_joint_fixed(-1, root)]
        for i, point in enumerate(((0.2, 0.0, 0.0), (-0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, -0.2, 0.0))):
            rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2) if i == 0 else wp.quat_identity()
            flap = builder.add_link(
                xform=wp.transform(wp.vec3(point), rotation), mass=0.015, inertia=wp.mat33(np.eye(3) * 0.001)
            )
            joint = builder.add_joint_revolute(
                root, flap, axis=newton.Axis.X, parent_xform=wp.transform(wp.vec3(point), wp.quat_identity())
            )
            joints.append(joint)
            if i == 0:
                builder.joint_q[builder.joint_q_start[joint]] = np.pi / 2
        builder.add_articulation(joints)
        builder.reserve_dynamic_capacity(joints=4)
        model = builder.finalize()
        exact = np.zeros(model.joint_count, dtype=bool)
        exact[joints] = True
        pipeline = newton.CollisionPipeline(model, contact_matching="sticky")
        solver = newton.solvers.SolverPhoenX(
            model,
            collision_pipeline=pipeline,
            joint_mode="maximal_pgs",
            step_layout="single_world",
            direct_joint_mask=exact,
            substeps=4,
            solver_iterations=4,
        )
        state = model.state()
        bodies = wp.array([[0, 1, 2, 3, 4]], dtype=int)
        requests, sealed = wp.ones(1, dtype=int), wp.zeros(1, dtype=int)
        attachments = wp.full((1, 4), -1, dtype=int)
        inputs = [solver.dynamics.data, state.body_q, state.body_qd, bodies, requests, sealed, attachments]
        wp.launch(_seal_cases, dim=1, inputs=inputs)
        self.assertEqual(int(sealed.numpy()[0]), 0)
        np.testing.assert_array_equal(attachments.numpy(), -1)
        poses = state.body_q.numpy()
        poses[1, 3:] = (0.0, 0.0, 0.0, 1.0)
        state.body_q.assign(poses)
        wp.launch(_seal_cases, dim=1, inputs=inputs)
        self.assertEqual(int(sealed.numpy()[0]), 1)
        self.assertTrue((attachments.numpy() >= 0).all())
        force = np.zeros((model.body_count, 6), dtype=np.float32)
        force[1, 3] = 0.01
        for _ in range(20):
            state.body_f.assign(force)
            solver.step(state, state, model.control(), None, 1.0 / 60.0)
        np.testing.assert_allclose(state.body_q.numpy()[1, 3:], (0.0, 0.0, 0.0, 1.0), atol=1e-3)
        self.assertEqual(int(solver.dynamics.error_count.numpy()[0]), 0)

    def test_scaled_belt_preserves_metres_per_second(self):
        """Preserve a moving belt's authored hull and its scaled 0.2 m/s speed."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "scenarios/line_sim/line_sim.usda"
            path.parent.mkdir(parents=True)
            path.write_text("""#usda 1.0
(defaultPrim = "Plant"; metersPerUnit = 1; upAxis = "Z")
def Xform "Plant" {
 def Xform "Line" {
  def Scope "Packers" {
   def Xform "PU1" {
    def Xform "Inlet" {
     def Scope "Belts" {
      def Xform "IN_A" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
       bool physics:kinematicEnabled = true
       custom bool physxSurfaceVelocity:surfaceVelocityEnabled = true
       def Mesh "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        uniform token physics:approximation = "convexHull"
        point3f[] points = [(-0.5,-0.5,-0.01), (0.5,-0.5,-0.01), (0.5,0.5,-0.01), (-0.5,0.5,-0.01),
                           (-0.5,-0.5,0.01), (0.5,-0.5,0.01), (0.5,0.5,0.01), (-0.5,0.5,0.01)]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0,1,2,3]
        double3 xformOp:scale = (4,0.05,1)
        uniform token[] xformOpOrder = ["xformOp:scale"]
       }
      }
     }
    }
   }
  }
 }
}
""")
            vendor = root / "sim/exts/bottling_plant.runtime/bottling_plant_runtime/vendor"
            for module, data in (("ifm_v01", '{"belts":[],"speeds_mps":{}}'), ("ctp_v01", '{"modules":{}}')):
                folder = vendor / module
                folder.mkdir(parents=True)
                (folder / "interface.json").write_text(data)
            scene = load_plant(root, cutaway=False, visuals=False)
            self.assertEqual(len(scene.belts), 1)
            mesh, velocities, _ = scene.belts[0]
            shape = scene.builder.shape_source.index(mesh)
            np.testing.assert_allclose(
                velocities.mean(axis=0) * scene.builder.shape_scale[shape], (0.2, 0.0, 0.0), atol=1e-7
            )
            # Convex cooking uses all points, even when source triangles only
            # describe the bottom. Surface velocity must not bypass cooking.
            body = scene.builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.02), wp.quat_identity()))
            scene.builder.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01)
            model = scene.builder.finalize()
            mesh.mesh.velocities.assign(velocities)
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            pipeline.collide(model.state(), contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            self.assertGreater(count, 0, "the authored convex belt must have a top support face")
            belt_first = contacts.rigid_contact_shape0.numpy()[:count] == shape
            signs = np.where(belt_first, 1.0, -1.0)
            self.assertTrue(np.all(signs * contacts.rigid_contact_normal.numpy()[:count, 2] > 0.99))
            belt_points = np.where(
                belt_first[:, None],
                contacts.rigid_contact_point0.numpy()[:count],
                contacts.rigid_contact_point1.numpy()[:count],
            )
            np.testing.assert_allclose(belt_points[:, 2], 0.01, atol=1e-5)
            np.testing.assert_allclose(
                signs * contacts.rigid_contact_surface_velocity.numpy()[:count, 0], -0.2, atol=1e-6
            )

    def test_position_drive_after_free_root(self):
        """A case clamp still receives its position command after a free root."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        free = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        builder.add_articulation([builder.add_joint_free(free)])
        slider = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_prismatic(
            -1,
            slider,
            axis=newton.Axis.X,
            target_ke=1000.0,
            target_kd=50.0,
            actuator_mode=newton.JointTargetMode.POSITION,
        )
        builder.add_articulation([joint])
        model = builder.finalize()
        dof = int(model.joint_qd_start.numpy()[joint])
        coordinate = int(_target_coordinates(model)[dof])
        self.assertNotEqual(dof, coordinate)
        control, state = model.control(), model.state()
        wp.launch(
            _case_drives,
            dim=1,
            inputs=[
                wp.array([dof], dtype=int),
                wp.array([coordinate], dtype=int),
                wp.array([0.1], dtype=float),
                wp.zeros(1, dtype=float),
                control.joint_target_q,
                control.joint_target_qd,
            ],
        )
        solver = newton.solvers.SolverPhoenX(
            model,
            joint_mode="maximal_pgs",
            step_layout="single_world",
            substeps=2,
            solver_iterations=2,
        )
        for _ in range(30):
            state.clear_forces()
            solver.step(state, state, control, None, 1.0 / 60.0)
        self.assertAlmostEqual(float(state.body_q.numpy()[slider, 0]), 0.1, delta=1e-3)
        np.testing.assert_allclose(state.body_q.numpy()[free], model.body_q.numpy()[free], atol=1e-6)

    def test_composed_geometry_and_physics(self):
        """Preserve inertial frames, static joint anchors, and point instancing."""
        from pxr import Usd, UsdGeom, UsdPhysics

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "scenarios/line_sim/line_sim.usda"
            path.parent.mkdir(parents=True)
            stage = Usd.Stage.CreateNew(str(path))
            layer = stage.GetRootLayer()
            layer.ImportFromString("""#usda 1.0
(
    defaultPrim = "Plant"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Plant" {
    def Xform "Machine" {
        double3 xformOp:translate = (4, 2, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Xform "Mount" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
            double3 xformOp:translate = (0, 0, 1)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            float physics:mass = 20
            float3 physics:diagonalInertia = (1, 2, 3)
            point3f physics:centerOfMass = (0, 0, 0.1)
            def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
                double size = 0.4
            }
        }
        def Xform "Arm" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
            double3 xformOp:translate = (1, 0, 1)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            float physics:mass = 2
            float3 physics:diagonalInertia = (0.1, 0.2, 0.3)
            def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
                double size = 0.2
            }
        }
        def Scope "Joints" {
        def PhysicsFixedJoint "Anchor" (prepend apiSchemas = ["PhysicsArticulationRootAPI"]) {
            custom bool physxArticulation:enabledSelfCollisions = false
            rel physics:body0 = </Plant/Machine>
            rel physics:body1 = </Plant/Machine/Mount>
            point3f physics:localPos0 = (0, 0, 1)
        }
        }
        def PhysicsRevoluteJoint "Hinge" (prepend apiSchemas = ["PhysicsDriveAPI:angular"]) {
            rel physics:body0 = </Plant/Machine/Mount>
            rel physics:body1 = </Plant/Machine/Arm>
            point3f physics:localPos0 = (1, 0, 0)
            uniform token physics:axis = "Z"
            float physics:lowerLimit = -45
            float physics:upperLimit = 45
            float drive:angular:physics:stiffness = 2
            float drive:angular:physics:targetPosition = 10
        }
    }
    def Xform "Flap" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        float physics:mass = 0.015
        def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
            double size = 1
            double3 xformOp:scale = (0.2, 0.08, 0.0015)
            double3 xformOp:translate = (0.1, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }
    }
    def PointInstancer "Copies" {
        rel prototypes = </Plant/Copies/Prototype>
        int[] protoIndices = [0, 0]
        point3f[] positions = [(10, 0, 0), (20, 0, 0)]
        def Xform "Prototype" {
            double3 xformOp:translate = (1, 0, 0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
            def Mesh "Triangle" {
                point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
                int[] faceVertexCounts = [3]
                int[] faceVertexIndices = [0, 1, 2]
                uniform token subdivisionScheme = "none"
                color3f[] primvars:displayColor = [(0.2, 0.4, 0.6)]
            }
            def Mesh "Guide" {
                point3f[] points = [(0, 0, 100), (1, 0, 100), (0, 1, 100)]
                int[] faceVertexCounts = [3]
                int[] faceVertexIndices = [0, 1, 2]
                uniform token purpose = "guide"
            }
        }
    }
    def Scope "Sim" {
        def Scope "Product" {
            def Scope "Bottles" {
                def Xform "Parked" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
                    double3 xformOp:translate = (0, 0, -3)
                    uniform token[] xformOpOrder = ["xformOp:translate"]
                    def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {}
                }
            }
        }
    }
}
""")
            blade = UsdGeom.Xform.Define(stage, "/Plant/OffsetBlade").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(blade)
            UsdPhysics.MassAPI.Apply(blade).CreateMassAttr(0.35)
            collider = UsdGeom.Mesh.Define(stage, "/Plant/OffsetBlade/Collider")
            UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
            half = np.asarray((0.003, 0.06, 0.008))
            corners = np.asarray(
                [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]
            )
            collider.CreatePointsAttr((corners * half + (5.0, 0.1, 1.0)).tolist())
            collider.CreateFaceVertexCountsAttr([3] * 12)
            collider.CreateFaceVertexIndicesAttr(
                [
                    0,
                    2,
                    1,
                    0,
                    3,
                    2,
                    4,
                    5,
                    6,
                    4,
                    6,
                    7,
                    0,
                    1,
                    5,
                    0,
                    5,
                    4,
                    1,
                    2,
                    6,
                    1,
                    6,
                    5,
                    2,
                    3,
                    7,
                    2,
                    7,
                    6,
                    3,
                    0,
                    4,
                    3,
                    4,
                    7,
                ]
            )
            layer.Save()
            scene = load_plant(root, cutaway=False, motor_off=True)
            builder = scene.builder
            self.assertEqual(len(scene.bodies), 4)
            mount = scene.bodies["/Plant/Machine/Mount"]
            np.testing.assert_allclose(builder.body_q[mount].p, (4, 2, 1))
            self.assertEqual(builder.body_mass[mount], 20.0)
            np.testing.assert_allclose(np.asarray(builder.body_inertia[mount]).reshape(3, 3), np.diag((1, 2, 3)))
            np.testing.assert_allclose(builder.body_com[mount], (0, 0, 0.1), atol=1e-7)
            self.assertIn("/Plant/Machine", scene.self_collision_scopes)
            anchor = scene.joints["/Plant/Machine/Joints/Anchor"]
            np.testing.assert_allclose(builder.joint_X_p[anchor].p, (4, 2, 1))
            hinge = scene.joints["/Plant/Machine/Hinge"]
            dof = builder.joint_qd_start[hinge]
            self.assertAlmostEqual(builder.joint_limit_lower[dof], -np.pi / 4)
            self.assertAlmostEqual(builder.joint_target_q[dof], np.pi / 18)
            flap = scene.bodies["/Plant/Flap"]
            self.assertAlmostEqual(builder.body_mass[flap], 0.015, places=7)
            np.testing.assert_allclose(builder.body_com[flap], (0.1, 0, 0), atol=1e-7)
            dims = np.asarray((0.2, 0.08, 0.0015)) ** 2
            expected = 0.015 / 12.0 * (dims.sum() - dims)
            np.testing.assert_allclose(
                np.diag(np.asarray(builder.body_inertia[flap]).reshape(3, 3)), expected, rtol=1e-6
            )
            blade = scene.bodies["/Plant/OffsetBlade"]
            tensor = np.asarray(builder.body_inertia[blade]).reshape(3, 3)
            expected = 0.35 / 3.0 * ((half**2).sum() - half**2)
            np.testing.assert_allclose(np.diag(tensor), expected, rtol=2e-4)
            self.assertGreater(float(np.linalg.eigvalsh(tensor).min()), 0.0)
            np.testing.assert_allclose(builder.body_com[blade], (5.0, 0.1, 1.0), atol=1e-6)
            copies = [i for i, name in enumerate(builder.shape_label) if name == "/Plant/Copies/instance"]
            self.assertEqual(len(copies), 2)
            self.assertIs(builder.shape_source[copies[0]], builder.shape_source[copies[1]])
            positions = []
            for shape in copies:
                mesh = builder.shape_source[shape]
                self.assertEqual(len(mesh.vertices), 3)
                positions.append(np.min(mesh.vertices[:, 0]) + builder.shape_transform[shape].p[0])
            np.testing.assert_allclose(positions, (11, 21))


if __name__ == "__main__":
    unittest.main()
