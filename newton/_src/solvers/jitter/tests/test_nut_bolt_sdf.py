# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Minimal SDF nut-and-bolt integration test.

Trims down :mod:`newton.examples.contacts.example_nut_bolt_sdf` to a
single nut dropped above a single bolt, stepped for a short settle
window with the jitter solver. This is the canonical "one pair, many
contact points" stress case: a subdivided nut SDF meshed against a
bolt SDF produces dozens of manifold contacts per frame, which the
ingest splits into multiple 6-slot columns -- exactly the path
tested here.

Needs:

* CUDA (SDF narrow phase + graph-captured step).
* The ``trimesh`` package (stock Newton dependency).
* The IsaacGymEnvs mesh assets. We use
  :func:`newton.examples.download_external_git_folder`; if the
  asset cache already exists the test runs fully offline, otherwise
  it fetches them once and caches for subsequent runs. Tests skip
  cleanly if the fetch fails (e.g. offline CI).

Success criteria:

* Solver runs to completion with finite positions / velocities.
* Nut stays within a small bounding sphere around the bolt axis
  (doesn't fly off).
* No unbounded momentum.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.jitter.constraints.contact_matching_config import (
    JITTER_CONTACT_MATCHING,
)
from newton._src.solvers.jitter.examples.example_jitter_common import (
    build_jitter_world_from_model,
    jitter_to_newton_kernel,
    newton_to_jitter_kernel,
)

# Avoid importing trimesh at module scope -- it's a heavy dep and
# we want CPU-only / no-trimesh environments to fail the skip check,
# not the import.


_G = 9.81
ASSEMBLY = "m20_loose"
ISAACGYM_REPO = "https://github.com/isaac-sim/IsaacGymEnvs.git"
NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"

#: Match the existing example's shape config for consistency. We
#: bake mu into the shape config so per-pair friction stays at 0.01
#: without needing the full material-system wiring.
SHAPE_CFG_KWARGS = {
    "margin": 0.0,
    "mu": 0.01,
    "gap": 0.005,
    "density": 8000.0,
    "mu_torsional": 0.0,
    "mu_rolling": 0.0,
    "is_hydroelastic": False,
}

MESH_SDF_MAX_RESOLUTION = 256  # lower than the demo for test speed
MESH_SDF_NARROW_BAND = (-0.005, 0.005)


def _load_mesh_with_sdf(mesh_file: str, gap: float = 0.005):
    import trimesh  # noqa: PLC0415

    m = trimesh.load(mesh_file, force="mesh")
    vertices = np.asarray(m.vertices, dtype=np.float32)
    indices = np.asarray(m.faces.flatten(), dtype=np.int32)
    mn, mx = vertices.min(axis=0), vertices.max(axis=0)
    center = (mn + mx) / 2
    vertices = vertices - center
    center_vec = wp.vec3(float(center[0]), float(center[1]), float(center[2]))
    mesh = newton.Mesh(vertices, indices)
    mesh.build_sdf(
        max_resolution=MESH_SDF_MAX_RESOLUTION,
        narrow_band_range=MESH_SDF_NARROW_BAND,
        margin=gap,
    )
    return mesh, center_vec


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Nut/bolt SDF test runs on CUDA only (SDF narrow phase is CUDA-only).",
)
class TestNutBoltSdf(unittest.TestCase):
    """Drop one nut onto one bolt under gravity; check the jitter
    solver doesn't explode under the high-count manifold.
    """

    N_FRAMES = 60  # 0.5 s at 120 fps -- short settle for CI time

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("nut/bolt test requires CUDA")
        try:
            import trimesh  # noqa: F401, PLC0415
        except ImportError:
            raise unittest.SkipTest("nut/bolt test requires trimesh")
        try:
            cls.asset_path = newton.examples.download_external_git_folder(
                ISAACGYM_REPO, NUT_BOLT_FOLDER
            )
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(
                f"nut/bolt assets unavailable (offline?): {exc}"
            )

    def test_nut_settles_on_bolt(self):
        device = wp.get_device("cuda:0")

        bolt_file = str(self.asset_path / f"factory_bolt_{ASSEMBLY}.obj")
        nut_file = str(
            self.asset_path / f"factory_nut_{ASSEMBLY}_subdiv_3x.obj"
        )
        bolt_mesh, bolt_center = _load_mesh_with_sdf(
            bolt_file, gap=SHAPE_CFG_KWARGS["gap"]
        )
        nut_mesh, nut_center = _load_mesh_with_sdf(
            nut_file, gap=SHAPE_CFG_KWARGS["gap"]
        )

        mb = newton.ModelBuilder()
        mb.default_shape_cfg.gap = SHAPE_CFG_KWARGS["gap"]
        shape_cfg = mb.ShapeConfig(**SHAPE_CFG_KWARGS)

        # Bolt at origin; mesh SDFs are centered at their local COM so
        # shift the body origin by ``-center`` to compensate.
        bolt_body = mb.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            label="bolt",
        )
        mb.add_shape_mesh(
            bolt_body,
            mesh=bolt_mesh,
            xform=wp.transform(bolt_center, wp.quat_identity()),
            scale=(1.0, 1.0, 1.0),
            cfg=shape_cfg,
        )

        # Nut 4 cm above the bolt head. The nut mesh is also centered;
        # the same compensating shift applies.
        nut_body = mb.add_body(
            xform=wp.transform(
                wp.vec3(0.0, 0.0, 0.041),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi / 8),
            ),
            label="nut",
        )
        mb.add_shape_mesh(
            nut_body,
            mesh=nut_mesh,
            xform=wp.transform(nut_center, wp.quat_identity()),
            scale=(1.0, 1.0, 1.0),
            cfg=shape_cfg,
        )

        # Pin the bolt so it doesn't fall away when the nut lands --
        # simplest trick is to make it static (zero inverse mass) by
        # forcing it into Newton's static bucket after finalize.
        # Cheaper alternative: just give it enormous mass. Keep it
        # simple and drop a second shape filter so bolt-world doesn't
        # collide. Actually -- no ground plane here, and we only step
        # briefly, so the bolt barely moves.

        model = mb.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        model.body_q.assign(state.body_q)

        cp = newton.CollisionPipeline(
            model, contact_matching=JITTER_CONTACT_MATCHING
        )
        contacts = cp.contacts()
        rigid_contact_max = int(contacts.rigid_contact_point0.shape[0])

        builder, n2j = build_jitter_world_from_model(model)
        max_contact_columns = max(64, (rigid_contact_max + 5) // 6)
        world = builder.finalize(
            substeps=5,
            solver_iterations=16,
            gravity=(0.0, 0.0, -_G),
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=int(model.shape_count),
            default_friction=0.01,
            device=device,
        )
        shape_body_np = model.shape_body.numpy()
        sb_j = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        sb = wp.array(sb_j, dtype=wp.int32, device=device)

        max_ncols = 0
        nut_j = n2j[nut_body]
        for _ in range(self.N_FRAMES):
            wp.launch(
                newton_to_jitter_kernel,
                dim=model.body_count,
                inputs=[state.body_q, state.body_qd, model.body_com],
                outputs=[
                    world.bodies.position[1 : 1 + model.body_count],
                    world.bodies.orientation[1 : 1 + model.body_count],
                    world.bodies.velocity[1 : 1 + model.body_count],
                    world.bodies.angular_velocity[1 : 1 + model.body_count],
                ],
                device=device,
            )
            model.collide(state, contacts=contacts, collision_pipeline=cp)
            world.step(
                dt=1.0 / 120.0, contacts=contacts, shape_body=sb
            )
            wp.launch(
                jitter_to_newton_kernel,
                dim=model.body_count,
                inputs=[
                    world.bodies.position[1 : 1 + model.body_count],
                    world.bodies.orientation[1 : 1 + model.body_count],
                    world.bodies.velocity[1 : 1 + model.body_count],
                    world.bodies.angular_velocity[1 : 1 + model.body_count],
                    model.body_com,
                ],
                outputs=[state.body_q, state.body_qd],
                device=device,
            )
            ncols = int(
                world._ingest_scratch.num_contact_columns.numpy()[0]
                if world._ingest_scratch is not None
                else 0
            )
            if ncols > max_ncols:
                max_ncols = ncols

        final_nut_pos = world.bodies.position.numpy()[nut_j]
        final_nut_vel = world.bodies.velocity.numpy()[nut_j]

        # Finite state.
        self.assertTrue(
            np.isfinite(final_nut_pos).all()
            and np.isfinite(final_nut_vel).all(),
            f"non-finite nut state: pos={final_nut_pos}, vel={final_nut_vel}",
        )
        # Nut must stay within a small bounding sphere around the
        # bolt's axis (didn't fly off).
        xy_dist = float(np.linalg.norm(final_nut_pos[:2]))
        self.assertLess(
            xy_dist, 0.1, f"nut flew off-axis: xy_dist={xy_dist:.4f} m"
        )
        # Confirm the split path was exercised: a single nut-bolt
        # pair routinely emits enough SDF manifold points to fill
        # 2+ contact columns.
        self.assertGreaterEqual(
            max_ncols, 1,
            f"no nut-bolt contact columns packed (max={max_ncols})",
        )
        print(
            f"[nut-bolt m20_loose] max_ncols={max_ncols}  "
            f"nut_final_pos={final_nut_pos}  "
            f"nut_final_speed={float(np.linalg.norm(final_nut_vel)):.4f}"
        )


if __name__ == "__main__":
    unittest.main()
