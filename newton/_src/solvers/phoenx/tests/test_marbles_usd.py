# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the PhoenX Marbles USD integration helpers."""

import unittest
import warnings

import numpy as np

import newton
from newton._src.solvers.phoenx.examples.example_marbles_usd import (
    _normalize_stage_units_for_newton,
    _physics_ignore_paths,
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


if __name__ == "__main__":
    unittest.main()
