# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PhoenX Marbles USD integration helpers."""

import unittest
import warnings

import numpy as np

import newton
from newton._src.solvers.phoenx.examples.example_marbles_usd import (
    Example,
    _normalize_stage_units_for_newton,
    _physics_ignore_paths,
    _scale_shape_contact_gaps,
)


class TestMarblesUsd(unittest.TestCase):
    """Verify tolerant USD physics import preparation."""

    def test_ignore_composition_errors_warns_and_continues(self):
        """Allow an explicitly tolerant import to retain valid physics."""
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        root = stage.DefinePrim("/Root", "Xform")
        stage.SetDefaultPrim(root)
        root.GetReferences().AddReference("missing_asset.usda")
        UsdPhysics.Scene.Define(stage, "/PhysicsScene")

        builder = newton.ModelBuilder()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = builder.add_usd(stage, ignore_composition_errors=True)

        self.assertEqual(result["physics_scene_path"], "/PhysicsScene")
        self.assertTrue(any("composition errors" in str(item.message) for item in caught))

    def test_normalize_stage_units_scales_descendants_to_meters(self):
        """Convert centimeter-authored descendant transforms to meters."""
        from pxr import Gf, Usd, UsdGeom

        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, "/Root")
        stage.SetDefaultPrim(root.GetPrim())
        child = UsdGeom.Xform.Define(stage, "/Root/Child")
        child.AddTranslateOp().Set(Gf.Vec3d(100.0, 0.0, 0.0))
        UsdGeom.SetStageMetersPerUnit(stage, 0.01)

        authored_unit = _normalize_stage_units_for_newton(stage)
        world = child.ComputeLocalToWorldTransform(0)

        self.assertAlmostEqual(authored_unit, 0.01)
        self.assertAlmostEqual(UsdGeom.GetStageMetersPerUnit(stage), 1.0)
        np.testing.assert_allclose(world.ExtractTranslation(), (1.0, 0.0, 0.0))

    def test_contact_gaps_follow_stage_units(self):
        """Scale imported contact gaps consistently with collider geometry."""
        builder = newton.ModelBuilder()
        builder.shape_gap[:] = [0.1, 2.0]

        _scale_shape_contact_gaps(builder, 0.01)

        np.testing.assert_allclose(builder.shape_gap, (0.001, 0.02))

    def test_invalid_and_trigger_colliders_are_skipped(self):
        """Exclude colliders with missing topology or PhysX trigger scripts."""
        from pxr import Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        missing = UsdGeom.Mesh.Define(stage, "/Missing")
        UsdPhysics.CollisionAPI.Apply(missing.GetPrim())
        trigger = UsdGeom.Mesh.Define(stage, "/Trigger")
        trigger.CreatePointsAttr([(0.0, 0.0, 0.0)])
        trigger.CreateFaceVertexCountsAttr([])
        trigger.CreateFaceVertexIndicesAttr([])
        UsdPhysics.CollisionAPI.Apply(trigger.GetPrim())
        trigger.GetPrim().CreateAttribute("physxTrigger:onEnterScript", Sdf.ValueTypeNames.String).Set("trigger.py")

        ignored = _physics_ignore_paths(stage)

        self.assertIn("/Missing", ignored)
        self.assertIn("/Trigger", ignored)
        self.assertIn("topology", ignored["/Missing"])
        self.assertIn("trigger", ignored["/Trigger"].lower())

    def test_reset_in_place_reuses_loaded_scene(self):
        """Restore PhoenX state without reconstructing the USD example."""

        class StateStub:
            def __init__(self):
                self.assigned_from = None
                self.forces_cleared = False

            def assign(self, other):
                self.assigned_from = other

            def clear_forces(self):
                self.forces_cleared = True

        class CollisionPipelineStub:
            def __init__(self):
                self.reset_count = 0

            def reset_contact_matching(self):
                self.reset_count += 1

        example = Example.__new__(Example)
        example.sim_time = 12.0
        example.physics_enabled = True
        example.state = StateStub()
        example.initial_state = StateStub()
        example.collision_pipeline = CollisionPipelineStub()
        sync_count = []
        example._sync_dynamic_render_transforms = lambda: sync_count.append(1)

        example.reset_in_place()

        self.assertEqual(example.sim_time, 0.0)
        self.assertIs(example.state.assigned_from, example.initial_state)
        self.assertTrue(example.state.forces_cleared)
        self.assertEqual(example.collision_pipeline.reset_count, 1)

        self.assertEqual(sync_count, [1])


if __name__ == "__main__":
    unittest.main()
