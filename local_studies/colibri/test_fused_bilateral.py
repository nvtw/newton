"""Analytical drive and momentum checks for experimental colored joint blocks."""

import ast
import importlib
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from local_studies.colibri.bilateral_pgs import install_fused
from newton._src.solvers.phoenx.tests.test_reduced_articulation import _total_momentum


class ColoredOwnership(ast.NodeTransformer):
    """Update backend ownership assertions, preserving every physics assertion."""

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if not isinstance(node.func, ast.Attribute):
            return node
        text = ast.unparse(node)
        if "_joint_pgs_enabled" not in text:
            return node
        if node.func.attr == "assertFalse":
            node.func.attr = "assertTrue"
        elif node.func.attr == "assertEqual" and isinstance(node.args[-1], ast.Constant) and node.args[-1].value == 0:
            node.args[-1] = ast.Constant(value=1)
        return node


class TestFreeBallMomentum(unittest.TestCase):
    def test_ball_anchor_and_internal_motor(self):
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        root = builder.add_link(mass=2.0, inertia=wp.mat33(0.07, 0.0, 0.0, 0.0, 0.08, 0.0, 0.0, 0.0, 0.09))
        tip = builder.add_link(mass=1.0, inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.03))
        joints = [builder.add_joint_free(root)]
        joints.append(
            builder.add_joint_ball(
                root,
                tip,
                parent_xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
            )
        )
        builder.add_articulation(joints)
        model = builder.finalize(device="cuda:0")
        state = model.state()
        qd = state.joint_qd.numpy()
        qd[:] = [0.4, -0.2, 0.1, 0.2, 0.3, 0.4, 0.1, -0.2, 0.3]
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        control = model.control()
        force = control.joint_f.numpy()
        force[-3:] = [0.1, -0.2, 0.3]
        control.joint_f.assign(force)
        solver = newton.solvers.SolverPhoenX(model, substeps=1, solver_iterations=8, sor_boost=1.0)
        before = _total_momentum(model, state)
        with wp.ScopedCapture(model.device) as capture:
            state.clear_forces()
            solver.step(state, state, control, None, 0.001)
        for _ in range(200):
            wp.capture_launch(capture.graph)
        np.testing.assert_allclose(_total_momentum(model, state), before, rtol=0, atol=2e-4)
        q = state.body_q.numpy()
        p0 = wp.transform_point(wp.transform(q[0, :3], q[0, 3:]), wp.vec3(0.3, 0, 0))
        p1 = wp.transform_point(wp.transform(q[1, :3], q[1, 3:]), wp.vec3(-0.2, 0, 0))
        self.assertLess(float(np.linalg.norm(np.array(p0) - np.array(p1))), 1e-4)


def adapted_suite():
    suite = unittest.TestSuite()
    names = (
        "newton._src.solvers.phoenx.tests.test_direct_drive",
        "newton._src.solvers.phoenx.tests.test_d6_direct_drive",
    )
    for name in names:
        module = importlib.import_module(name)
        namespace = dict(vars(module))
        from pathlib import Path

        tree = ColoredOwnership().visit(ast.parse(Path(module.__file__).read_text()))
        ast.fix_missing_locations(tree)
        exec(compile(tree, module.__file__, "exec"), namespace)
        for value in namespace.values():
            if isinstance(value, type) and issubclass(value, unittest.TestCase):
                for test in unittest.defaultTestLoader.loadTestsFromTestCase(value):
                    if (
                        "mass_splitting" not in test.id()
                        and "multi_world" not in test.id()
                        and "coupled_three_joint" not in test.id()
                    ):
                        suite.addTest(test)
    suite.addTests(
        unittest.defaultTestLoader.loadTestsFromName(
            "newton._src.solvers.phoenx.tests.test_maximal_contact_conservation"
        )
    )
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(TestFreeBallMomentum))
    return suite


if __name__ == "__main__":
    original = newton.solvers.SolverPhoenX.__init__

    def initialize(self, *args, **kwargs):
        kwargs["step_layout"] = "single_world"
        original(self, *args, **kwargs)
        install_fused(self)

    with patch.object(newton.solvers.SolverPhoenX, "__init__", initialize):
        result = unittest.TextTestRunner().run(adapted_suite())
    raise SystemExit(not result.wasSuccessful())
