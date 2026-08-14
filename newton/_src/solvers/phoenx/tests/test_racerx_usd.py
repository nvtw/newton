# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused USD compatibility tests used by the PhoenX RacerX example."""

from __future__ import annotations

import unittest

import newton

try:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestRacerXUsd(unittest.TestCase):
    """Verify physics patterns required by the RacerX USD stage."""

    def test_coaxial_hard_limit_and_spring_merge_to_one_dof(self) -> None:
        """Merge stacked travel and spring joints into one driven coordinate."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/PhysicsScene")

        parent = UsdGeom.Cube.Define(stage, "/World/Parent")
        child = UsdGeom.Cube.Define(stage, "/World/Child")
        for body in (parent, child):
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())

        travel = self._add_prismatic(stage, "/World/Travel", parent, child, -0.9, 0.4)
        spring = self._add_prismatic(stage, "/World/Spring", parent, child, 0.0, 0.01)
        spring.GetPrim().CreateAttribute(
            "physxLimit:linear:stiffness",
            Sdf.ValueTypeNames.Float,
        ).Set(80000.0)
        spring.GetPrim().CreateAttribute(
            "physxLimit:linear:damping",
            Sdf.ValueTypeNames.Float,
        ).Set(600.0)

        builder = newton.ModelBuilder()
        result = builder.add_usd(
            stage,
            load_visual_shapes=False,
            schema_resolvers=[newton.usd.SchemaResolverPhysx()],
        )

        self.assertEqual(builder.joint_count, 1)
        self.assertEqual(builder.joint_dof_dim, [(1, 0)])
        self.assertEqual(result["path_joint_map"][str(travel.GetPath())], 0)
        self.assertEqual(result["path_joint_map"][str(spring.GetPath())], 0)
        self.assertAlmostEqual(builder.joint_limit_lower[0], -0.9)
        self.assertAlmostEqual(builder.joint_limit_upper[0], 0.4)
        self.assertAlmostEqual(builder.joint_target_q[0], 0.005)
        self.assertAlmostEqual(builder.joint_target_ke[0], 80000.0)
        self.assertAlmostEqual(builder.joint_target_kd[0], 600.0)
        self.assertEqual(builder.joint_target_mode[0], newton.JointTargetMode.POSITION)

    def test_self_filtered_collision_group_excludes_member_pair(self) -> None:
        """Translate a self-filtered USD collision group into shape exclusions."""
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdPhysics.Scene.Define(stage, "/PhysicsScene")

        members = []
        for name in ("A", "B"):
            body = UsdGeom.Cube.Define(stage, f"/World/{name}")
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.CollisionAPI.Apply(body.GetPrim())
            members.append(body.GetPath())

        group = UsdPhysics.CollisionGroup.Define(stage, "/World/CollisionGroup")
        collection = Usd.CollectionAPI.Apply(group.GetPrim(), "colliders")
        collection.CreateIncludesRel().SetTargets(members)
        group.CreateFilteredGroupsRel().SetTargets([group.GetPath()])

        builder = newton.ModelBuilder()
        builder.add_usd(stage, load_visual_shapes=False)

        self.assertEqual(builder.shape_count, 2)
        self.assertIn((0, 1), builder.shape_collision_filter_pairs)

    @staticmethod
    def _add_prismatic(stage, path, parent, child, lower, upper):
        """Author one X-axis prismatic joint with explicit anchors."""
        joint = UsdPhysics.PrismaticJoint.Define(stage, path)
        joint.CreateBody0Rel().SetTargets([parent.GetPath()])
        joint.CreateBody1Rel().SetTargets([child.GetPath()])
        joint.CreateAxisAttr().Set("X")
        joint.CreateLowerLimitAttr().Set(lower)
        joint.CreateUpperLimitAttr().Set(upper)
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        return joint


if __name__ == "__main__":
    unittest.main()
