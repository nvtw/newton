# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Nut-rotation verification for the SDF nut/bolt scene.

Drops a single nut onto a single bolt under gravity and runs the
Jitter solver for a fixed simulation window. The physical setup
matches ``newton.examples.contacts.example_nut_bolt_sdf_benchmark``
at ``num_per_world=1``: same asset, same ``ShapeConfig``, same bolt
pinning pattern, same step rate.

Where the existing :mod:`test_nut_bolt_sdf` only verifies the nut
doesn't fly off, this test verifies the *dynamics* the upstream
benchmark checks:

    * Nut must accumulate non-trivial rotation about the bolt axis
      (>= 0.1 rad ~ 5.7 deg over the test window). A zero-rotation
      nut indicates the Jitter friction / anchor machinery is
      gripping the nut rigidly onto the bolt instead of letting
      it thread.
    * Nut must move downward (min z lower than initial). Threading
      rotation should translate into axial travel.
    * Bolt must stay approximately in place (pinned static; this
      catches a badly broken Coulomb cone that lets the bolt react
      through the ground).

Runs under a single ``wp.ScopedCapture`` so the whole 1000-step
window replays off a cached CUDA graph; the host-side rotation
tracking reads ``state.body_q`` once at the very end, not per step,
so the capture stays valid across the loop.

Skipped on CPU / no-CUDA, and on environments without ``trimesh`` or
the IsaacGymEnvs asset cache.
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


_G = 9.81
ASSEMBLY = "m20_loose"
ISAACGYM_REPO = "https://github.com/isaac-sim/IsaacGymEnvs.git"
NUT_BOLT_FOLDER = "assets/factory/mesh/factory_nut_bolt"

SHAPE_CFG_KWARGS = {
    "margin": 0.0,
    "mu": 0.01,
    "gap": 0.005,
    "density": 8000.0,
    "mu_torsional": 0.0,
    "mu_rolling": 0.0,
    "is_hydroelastic": False,
}

MESH_SDF_MAX_RESOLUTION = 256
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


def _yaw_from_quat_xyzw(quat_xyzw: np.ndarray) -> float:
    """Extract yaw (rotation about +Z) from an ``xyzw`` quaternion.

    Matches :func:`example_nut_bolt_sdf_benchmark._yaw_from_quat_xyzw`
    so this test's rotation metric is comparable with the MuJoCo /
    XPBD benchmark's report.
    """
    x, y, z, w = quat_xyzw
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


@unittest.skipUnless(
    wp.get_preferred_device().is_cuda,
    "Nut/bolt rotation test runs on CUDA only (SDF narrow phase is CUDA-only, graph-captured).",
)
class TestNutBoltRotation(unittest.TestCase):
    """Drop one nut onto one bolt and require it to thread.

    Runs ``N_FRAMES`` frames at 120 fps, 5 substeps / frame with
    the default solver iteration count. Graph-captured for speed.
    """

    N_FRAMES = 1000  # ~8.3 s at 120 fps
    FPS = 120
    SUBSTEPS = 5
    SOLVER_ITERS = 16

    @classmethod
    def setUpClass(cls):
        if not wp.is_cuda_available():
            raise unittest.SkipTest("requires CUDA")
        try:
            import trimesh  # noqa: F401, PLC0415
        except ImportError:
            raise unittest.SkipTest("requires trimesh")
        try:
            cls.asset_path = newton.examples.download_external_git_folder(
                ISAACGYM_REPO, NUT_BOLT_FOLDER
            )
        except Exception as exc:  # noqa: BLE001
            raise unittest.SkipTest(
                f"nut/bolt assets unavailable (offline?): {exc}"
            )

    def _build_scene(self, device):
        """Build the Newton model + Jitter world + collision pipeline.

        Returns a dict of handles the test driver needs to step the
        scene and read out the final nut pose.
        """
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

        model = mb.finalize()

        # Pin the bolt in place by zeroing its inverse mass / inertia.
        # Newton finalize would otherwise give the bolt a finite mass
        # from its mesh volume + density, and it would free-fall off
        # the origin alongside the nut.
        body_inv_mass_np = model.body_inv_mass.numpy()
        body_inv_inertia_np = model.body_inv_inertia.numpy()
        body_inv_mass_np[bolt_body] = 0.0
        body_inv_inertia_np[bolt_body] = np.zeros((3, 3), dtype=np.float32)
        model.body_inv_mass.assign(wp.array(body_inv_mass_np, dtype=wp.float32))
        model.body_inv_inertia.assign(
            wp.array(body_inv_inertia_np, dtype=wp.mat33)
        )

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
            substeps=self.SUBSTEPS,
            solver_iterations=self.SOLVER_ITERS,
            gravity=(0.0, 0.0, -_G),
            max_contact_columns=max_contact_columns,
            rigid_contact_max=rigid_contact_max,
            num_shapes=int(model.shape_count),
            default_friction=SHAPE_CFG_KWARGS["mu"],
            device=device,
        )
        shape_body_np = model.shape_body.numpy()
        sb_j = np.where(shape_body_np < 0, 0, shape_body_np + 1)
        sb = wp.array(sb_j, dtype=wp.int32, device=device)

        return {
            "model": model,
            "state": state,
            "cp": cp,
            "contacts": contacts,
            "world": world,
            "n2j": n2j,
            "bolt_body": bolt_body,
            "nut_body": nut_body,
            "shape_body": sb,
            "frame_dt": 1.0 / self.FPS,
        }

    def _run_and_track(self, scene: dict, n_frames: int) -> dict:
        """Step the scene ``n_frames`` times under a single captured
        graph, then read out the final state from the host. Tracks
        cumulative nut yaw, minimum z, bolt displacement, and peak
        number of contact columns.
        """
        model = scene["model"]
        state = scene["state"]
        cp = scene["cp"]
        contacts = scene["contacts"]
        world = scene["world"]
        sb = scene["shape_body"]
        frame_dt = scene["frame_dt"]
        device = wp.get_device("cuda:0")

        n = model.body_count
        nut_idx = scene["nut_body"]
        bolt_idx = scene["bolt_body"]

        body_q0 = state.body_q.numpy()
        initial_nut_pose = body_q0[nut_idx].copy()
        initial_bolt_pose = body_q0[bolt_idx].copy()
        initial_nut_yaw = _yaw_from_quat_xyzw(initial_nut_pose[3:7])
        initial_nut_z = float(initial_nut_pose[2])

        def _simulate_once():
            wp.launch(
                newton_to_jitter_kernel,
                dim=n,
                inputs=[state.body_q, state.body_qd, model.body_com],
                outputs=[
                    world.bodies.position[1 : 1 + n],
                    world.bodies.orientation[1 : 1 + n],
                    world.bodies.velocity[1 : 1 + n],
                    world.bodies.angular_velocity[1 : 1 + n],
                ],
                device=device,
            )
            model.collide(state, contacts=contacts, collision_pipeline=cp)
            world.step(dt=frame_dt, contacts=contacts, shape_body=sb)
            wp.launch(
                jitter_to_newton_kernel,
                dim=n,
                inputs=[
                    world.bodies.position[1 : 1 + n],
                    world.bodies.orientation[1 : 1 + n],
                    world.bodies.velocity[1 : 1 + n],
                    world.bodies.angular_velocity[1 : 1 + n],
                    model.body_com,
                ],
                outputs=[state.body_q, state.body_qd],
                device=device,
            )

        # Capture one frame -- follow ``example_pyramid`` / ``example_nut_bolt``
        # pattern where the first ``simulate()`` call happens inside
        # the ScopedCapture block.
        with wp.ScopedCapture(device=device) as capture:
            _simulate_once()
        graph = capture.graph
        use_graph = graph is not None

        # Host-side tracking loop: one host readback per frame to
        # unwrap yaw. The compute itself is 100% on-device and runs
        # off the captured graph; the host readback is the only
        # sync point.
        prev_yaw = initial_nut_yaw
        cumulative_yaw = 0.0
        min_nut_z = initial_nut_z
        max_bolt_displacement = 0.0
        for _ in range(n_frames):
            if use_graph:
                wp.capture_launch(graph)
            else:
                _simulate_once()
            body_q = state.body_q.numpy()
            nut_pose = body_q[nut_idx]
            bolt_pose = body_q[bolt_idx]
            yaw = _yaw_from_quat_xyzw(nut_pose[3:7])
            delta = float(
                np.arctan2(
                    np.sin(yaw - prev_yaw), np.cos(yaw - prev_yaw)
                )
            )
            cumulative_yaw += delta
            prev_yaw = yaw
            min_nut_z = min(min_nut_z, float(nut_pose[2]))
            bolt_disp = float(
                np.linalg.norm(bolt_pose[:3] - initial_bolt_pose[:3])
            )
            if bolt_disp > max_bolt_displacement:
                max_bolt_displacement = bolt_disp

        final_nut_pose = state.body_q.numpy()[nut_idx].copy()
        return {
            "initial_nut_pose": initial_nut_pose,
            "initial_bolt_pose": initial_bolt_pose,
            "final_nut_pose": final_nut_pose,
            "cumulative_yaw_rad": cumulative_yaw,
            "min_nut_z": min_nut_z,
            "initial_nut_z": initial_nut_z,
            "max_bolt_displacement": max_bolt_displacement,
        }

    def test_nut_threads_onto_bolt(self):
        """Nut must rotate non-trivially and move downward while
        settling; bolt must stay pinned.
        """
        device = wp.get_device("cuda:0")
        scene = self._build_scene(device)
        result = self._run_and_track(scene, self.N_FRAMES)

        cum_rot_deg = float(np.degrees(result["cumulative_yaw_rad"]))
        downward = result["initial_nut_z"] - result["min_nut_z"]
        print(
            f"[nut-bolt-rotation] "
            f"cumulative_yaw={cum_rot_deg:+.3f} deg  "
            f"downward={downward * 1000.0:.3f} mm  "
            f"bolt_max_disp={result['max_bolt_displacement'] * 1000.0:.4f} mm"
        )

        # Thresholds mirror the upstream MuJoCo/XPBD benchmark (at
        # 2000 steps those solvers achieve 30+ deg of rotation; we
        # require 5.7 deg over our 1000-step window, a loose bound
        # that still rules out "nut stuck on bolt").
        min_rotation_deg = 5.7
        self.assertGreater(
            abs(cum_rot_deg),
            min_rotation_deg,
            f"nut did not rotate enough: cumulative yaw "
            f"{cum_rot_deg:.3f} deg (expected > {min_rotation_deg:.1f} deg)",
        )

        # Nut must have moved downward at some point (threading
        # should convert rotation into axial travel). Loose 0.1 mm
        # threshold: catches a nut that's physically floating above
        # the bolt head without ever settling.
        self.assertGreater(
            downward,
            1.0e-4,
            f"nut did not move downward: delta_z={downward:.6f} m",
        )

        # Bolt must stay approximately pinned. 2 cm upstream test
        # tolerance; we use 1 mm because the bolt is genuinely
        # static in our scene.
        self.assertLess(
            result["max_bolt_displacement"],
            1.0e-3,
            f"bolt moved too much: "
            f"{result['max_bolt_displacement'] * 1000.0:.4f} mm",
        )


if __name__ == "__main__":
    unittest.main()
