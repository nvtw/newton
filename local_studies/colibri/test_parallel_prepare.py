"""Compare serial versus point-parallel preparation on identical physical state."""

import types
import unittest

import numpy as np
import warp as wp

import newton
from local_studies.colibri.bilateral_pgs import install_fused
from local_studies.colibri.parallel_prepare import install


def run(parallel, recycle=False):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, -9.81))
    b0 = builder.add_link()
    b1 = builder.add_link()
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.5, gap=0.001)
    builder.add_shape_sphere(b0, radius=0.1, cfg=cfg)
    builder.add_shape_sphere(b1, radius=0.1, cfg=cfg)
    joints = [builder.add_joint_free(b0)]
    joints.append(
        builder.add_joint_revolute(
            b0,
            b1,
            axis=newton.Axis.Z,
            parent_xform=wp.transform(wp.vec3(0.15, 0, 0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.15, 0, 0), wp.quat_identity()),
            target_vel=2.0,
            target_kd=1.0,
        )
    )
    builder.add_articulation(joints)
    if recycle:
        cube = builder.add_link(xform=wp.transform(wp.vec3(0.6, 0, 2.0), wp.quat_identity()))
        builder.add_shape_mesh(cube, mesh=newton.Mesh.create_box(0.1, 0.1, 0.1), cfg=cfg)
        builder.add_articulation([builder.add_joint_free(cube)])
    builder.add_ground_plane()
    model = builder.finalize(device="cuda:0")
    state = model.state()
    q = state.joint_q.numpy()
    q[2] = 0.09
    state.joint_q.assign(q)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, contact_matching="sticky")
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverPhoenX(
        model,
        collision_pipeline=pipeline,
        articulation_mode="maximal",
        step_layout="single_world",
        substeps=2,
        solver_iterations=2,
        sor_boost=1.0,
        prepare_refresh_stride=1,
    )
    install_fused(solver)
    if parallel:
        install(solver)
    w = solver.world
    original = w._singleworld_head_plus_tail_sweep
    prepare = w._singleworld_kernels()[0]
    log = []

    def sweep(self, head, tail, idt, contact_container=None):
        result = original(head, tail, idt, contact_container)
        if head is prepare:
            cc = self._contact_container
            log.append(
                [
                    a.numpy().copy()
                    for a in (
                        cc.derived,
                        cc.impulses,
                        self.bodies.velocity,
                        self.bodies.angular_velocity,
                        self.constraints.data,
                        self._ingest_scratch.num_contact_columns,
                    )
                ]
            )
        return result

    w._singleworld_head_plus_tail_sweep = types.MethodType(sweep, w)
    for step in range(4):
        if recycle and step == 2:
            poses = state.body_q.numpy()
            poses[:2, 2] += 2.0
            poses[cube, 2] = 0.09
            state.body_q.assign(poses)
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, state, model.control(), contacts, 0.005)
    return log


class TestParallelPrepare(unittest.TestCase):
    def test_prepared_geometry_and_ordered_warmstart_match(self):
        """Parallel point geometry must preserve all prepared rows and velocities."""
        old = run(False)
        new = run(True)
        self.assertEqual(len(old), 8)
        self.assertEqual(len(old), len(new))
        for step, (a, b) in enumerate(zip(old, new, strict=False)):
            for field, (x, y) in enumerate(zip(a, b, strict=False)):
                np.testing.assert_array_equal(x, y, err_msg=f"prepare {step} field {field}")

    def test_recycled_contact_columns_match(self):
        """Inactive prior headers must not write reused live point slots."""
        old = run(False, recycle=True)
        new = run(True, recycle=True)
        self.assertGreater(int(old[0][-1][0]), int(old[-1][-1][0]))
        for step, (a, b) in enumerate(zip(old, new, strict=False)):
            for field, (x, y) in enumerate(zip(a, b, strict=False)):
                np.testing.assert_array_equal(x, y, err_msg=f"recycled prepare {step} field {field}")


if __name__ == "__main__":
    unittest.main()
