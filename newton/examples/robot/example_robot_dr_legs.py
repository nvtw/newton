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

###########################################################################
# Example Robot DR Legs
#
# Loads the Disney Research bipedal-legs USD asset and simulates it with
# SolverXPBD. The asset has six closed kinematic loops (parallel-rod
# linkages and outer ankle brackets); XPBD's joint solver treats every
# joint -- whether on a tree edge or a loop closer -- as a generic
# positional constraint, so closed loops need no special equality
# constraint plumbing.
#
# What we do still need is an articulation tree rooted at the pelvis,
# so that ``add_usd(..., floating=True)`` can attach a FREE base joint
# to it (without a base joint, XPBD has no way to anchor the floating
# pelvis to inertial space and the system drifts). We give the
# importer a clean tree by:
#   - tagging the pelvis as the articulation root,
#   - tagging the six joints that would otherwise create multi-parent
#     bodies with ``physics:excludeFromArticulation`` -- they remain in
#     the model as loop-closer joint constraints,
#   - passing ``joint_ordering=None`` so the importer keeps the
#     remaining tree edges in their authored body0=parent / body1=child
#     orientation (no flipping needed).
#
# Drives the 12 actuated joints from the bundled walking animation. The
# animation is an open-loop joint-space gait that was authored for a
# more capable solver pipeline (Kamino's implicit-PD + ZMP-aware
# tracker); replaying it at full speed against pure XPBD position
# drives is reactively unstable -- the robot tracks the joint targets
# accurately but loses lateral balance and falls within a few seconds.
# We default to ``--animation-speed 0.25`` so the legs step in slow
# motion and the COM has time to remain over the support polygon. Set
# ``--animation-speed 1.0`` to attempt full-rate playback.
#
# Command: python -m newton.examples robot_dr_legs --world-count 16
#
###########################################################################

import argparse

import numpy as np
import warp as wp
from pxr import Sdf, Usd, UsdPhysics

import newton
import newton.examples
import newton.utils

# Joints whose authored ``body1`` is shared with another joint's
# ``body1`` (i.e. they would create a body with multiple parents in the
# tree). Tagging them with ``physics:excludeFromArticulation`` leaves
# them as ordinary joints in the model -- XPBD enforces them just like
# any other constraint -- but keeps the importer's articulation tree
# valid. Derived from the raw USD authored joint graph; every other
# joint stays on the pelvis-rooted tree.
_LOOP_CLOSER_JOINTS = (
    "/DR_Legs/Joints/j6_l_i",
    "/DR_Legs/Joints/j6_r_i",
    "/DR_Legs/Joints/j9_l_i",
    "/DR_Legs/Joints/j9_l_o",
    "/DR_Legs/Joints/j9_r_i",
    "/DR_Legs/Joints/j9_r_o",
)

# Animation channel -> joint path. The bundled .npy stores 12 columns
# in this order. With ``joint_ordering=None`` and the loop-closer set
# below, every actuated joint's authored axis already matches the
# .npy's sign convention -- a pinned-pelvis replay tracks each channel
# with corr > 0.99 -- so the per-channel sign array is identically +1.
_ANIMATION_JOINT_PATHS = (
    "/DR_Legs/Joints/j1_l_i",
    "/DR_Legs/Joints/j2_l_i",
    "/DR_Legs/Joints/j6_l_i",
    "/DR_Legs/Joints/j7_l_i",
    "/DR_Legs/Joints/j2_l_o",
    "/DR_Legs/Joints/j7_l_o",
    "/DR_Legs/Joints/j1_r_i",
    "/DR_Legs/Joints/j2_r_i",
    "/DR_Legs/Joints/j6_r_i",
    "/DR_Legs/Joints/j7_r_i",
    "/DR_Legs/Joints/j2_r_o",
    "/DR_Legs/Joints/j7_r_o",
)
_ANIMATION_CHANNEL_SIGN = np.ones(12, dtype=np.float32)


class Example:
    def __init__(self, viewer, args):
        # Match the animation's native rate: dr_legs_animation_100fps.npy is
        # sampled at 100 Hz, so running the example at fps=100 gives one
        # animation frame per sim frame and avoids aliasing the gait reference.
        self.fps = 100
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = args.sim_substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.world_count = args.world_count

        self.viewer = viewer

        dr_legs = newton.ModelBuilder(up_axis=newton.Axis.Z)
        # Leave the default joint PD gains at 0 -- non-actuated DOFs
        # (every loop-closer hinge, plus the unactuated outer-leg
        # joints) must not exert a spring/damper, otherwise XPBD
        # explodes when the measured joint angle is non-zero. PD gains
        # for the 12 actuated DOFs are set explicitly after add_usd.
        dr_legs.default_shape_cfg.ke = 2.0e3
        dr_legs.default_shape_cfg.kd = 1.0e2
        dr_legs.default_shape_cfg.kf = 1.0e3
        dr_legs.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("disneyresearch")
        asset_file = str(asset_path / "dr_legs/usd" / "dr_legs_with_meshes_and_boxes.usda")

        stage = Usd.Stage.Open(asset_file)
        if stage is None:
            raise RuntimeError(f"Failed to open dr_legs USD stage: {asset_file}")
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/DR_Legs/RigidBodies/pelvis"))
        for jp in _LOOP_CLOSER_JOINTS:
            stage.GetPrimAtPath(jp).CreateAttribute("physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool).Set(
                True
            )

        # Place the robot with the foot collision boxes just resting on
        # the ground plane (added below at z = -0.1). The lowest body
        # in neutral pose is the foot box at pelvis_z - 0.262, so a
        # pelvis offset of 0.165 puts the feet ~3 mm above the ground
        # at start-up -- enough to let contacts engage cleanly without
        # a long free-fall (which would otherwise impact the legs hard
        # before the PD drive can take effect). ``floating=True`` adds
        # the FREE base joint that XPBD needs to anchor the pelvis to
        # inertial space.
        dr_legs.add_usd(
            stage,
            xform=wp.transform(wp.vec3(0, 0, 0.165)),
            floating=True,
            joint_ordering=None,
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        # USD-authored drive stiffness on this asset is ~0.87 N.m/rad,
        # which is fine for MuJoCo's PD model but far too compliant when
        # XPBD treats target_ke directly as constraint stiffness. Override
        # every actuated DOF with PD gains tuned for XPBD; the
        # --animation-gain-scale knob then scales these.
        none_mode = int(newton.JointTargetMode.NONE)
        kp_scale = args.animation_gain_scale if args.animation else 1.0
        kd_scale = args.animation_kd_scale if (args.animation and args.animation_kd_scale is not None) else kp_scale
        for dof_i, mode in enumerate(dr_legs.joint_target_mode):
            if mode != none_mode:
                dr_legs.joint_target_ke[dof_i] = 2000.0 * kp_scale
                dr_legs.joint_target_kd[dof_i] = 50.0 * kd_scale

        # XPBD ignores joint armature, so we instead diagonally inflate
        # body inertia for solver stability. The DR Legs asset has very
        # light parallel-rod bodies (~6 g) feeding into much heavier
        # pelvis/leg bodies (~600 g), and the resulting mass ratio at
        # each loop-closure constraint causes explosive corrections
        # without this regularization.
        body_armature = 0.05
        for body in range(dr_legs.body_count):
            inertia_np = np.asarray(dr_legs.body_inertia[body], dtype=np.float32).reshape(3, 3)
            inertia_np += np.eye(3, dtype=np.float32) * body_armature
            dr_legs.body_inertia[body] = wp.mat33(inertia_np)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(dr_legs, self.world_count)

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.add_ground_plane(height=-0.1)

        self.model = builder.finalize()

        # A tiny non-zero compliance turns the joint constraints from
        # infinitely-stiff (compliance=0) into very stiff springs. The
        # six loop-closer hinges create an over-determined constraint
        # set whose residuals cannot all be zero with finite XPBD
        # iterations; infinite stiffness amplifies that residual into
        # an integrator-breaking impulse on frame 1.
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=args.solver_iterations,
            joint_linear_compliance=1.0e-6,
            joint_angular_compliance=1.0e-4,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.contacts = self.model.contacts()

        self._animation_data: np.ndarray | None = None
        self._animation_speed: float = args.animation_speed
        if args.animation:
            self._init_animation(asset_path)

        self.viewer.set_model(self.model)

        # Compile + capture the per-frame CUDA graph. On a cold Warp
        # cache the contact and XPBD kernels can take 30+ seconds to
        # compile; let the user know the process is working.
        print("Warming up Warp kernels and capturing CUDA graph...", flush=True)
        self.capture()
        print("Ready.", flush=True)

    def _init_animation(self, asset_path) -> None:
        anim_file = str(asset_path / "dr_legs/animation" / "dr_legs_animation_100fps.npy")
        anim = np.load(anim_file).astype(np.float32)
        if anim.shape[1] != len(_ANIMATION_JOINT_PATHS):
            raise RuntimeError(f"animation has {anim.shape[1]} channels, expected {len(_ANIMATION_JOINT_PATHS)}")
        joint_label = list(self.model.joint_label)
        joint_qd_start = self.model.joint_qd_start.numpy()
        try:
            channel_dofs = np.array(
                [joint_qd_start[joint_label.index(path)] for path in _ANIMATION_JOINT_PATHS],
                dtype=np.int64,
            )
        except ValueError as e:
            raise RuntimeError(f"animation joint not found in model.joint_label: {e}") from e
        n_dof_per_world = self.model.joint_dof_count // self.world_count
        world_offsets = np.arange(self.world_count, dtype=np.int64) * n_dof_per_world
        # 2-D fancy-index assignment broadcasts a (12,) RHS across worlds.
        self._animation_indices = channel_dofs[None, :] + world_offsets[:, None]
        self._animation_data = anim * _ANIMATION_CHANNEL_SIGN[None, :]
        self._animation_dt = 1.0 / 100.0
        self._target_pos_host = self.control.joint_target_pos.numpy()

    def _update_animation_targets(self):
        n_frames = self._animation_data.shape[0]
        frame = min(int(self.sim_time * self._animation_speed / self._animation_dt), n_frames - 1)
        self._target_pos_host[self._animation_indices] = self._animation_data[frame]
        self.control.joint_target_pos.assign(self._target_pos_host)

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self._animation_data is not None:
            self._update_animation_targets()

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

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        # body_q layout is [px, py, pz, qx, qy, qz, qw] per row.
        n_per_world = self.model.body_count // self.world_count
        body_label = list(self.model.body_label)
        pelvis_idx = next(
            (i for i, lbl in enumerate(body_label[:n_per_world]) if "pelvis" in str(lbl).lower()),
            0,
        )
        for w in range(self.world_count):
            row = w * n_per_world + pelvis_idx
            pz = float(body_q[row, 2])
            qx, qy, qz, qw = body_q[row, 3:7]
            sin_pitch = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
            pitch = float(np.arcsin(sin_pitch))
            roll = float(np.arctan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
            assert np.isfinite(pz), f"world {w}: pelvis z is non-finite ({pz})"
            assert pz > 0.05, f"world {w}: pelvis collapsed to z={pz:.3f} (expected > 0.05)"
            assert abs(roll) < np.pi / 4, f"world {w}: pelvis roll {np.degrees(roll):.1f} deg exceeds 45 deg"
            assert abs(pitch) < np.pi / 4, f"world {w}: pelvis pitch {np.degrees(pitch):.1f} deg exceeds 45 deg"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument(
            "--animation",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Drive the 12 USD-actuated joints from dr_legs_animation_100fps.npy.",
        )
        parser.add_argument(
            "--animation-gain-scale",
            type=float,
            default=1.0,
            help=("Multiplier on USD-authored kp/kd."),
        )
        parser.add_argument(
            "--animation-kd-scale",
            type=float,
            default=None,
            help="Optional separate multiplier on USD kd. Defaults to following --animation-gain-scale.",
        )
        parser.add_argument(
            "--animation-speed",
            type=float,
            default=0.25,
            help=(
                "Animation playback rate; 1.0 plays the gait at the authored 100 Hz."
                " Defaults to 0.25 because the open-loop gait is reactively"
                " unstable for pure XPBD position drives -- slowing playback"
                " keeps the COM over the support polygon. Set to 1.0 to attempt"
                " real-time playback (robot will fall over within ~2 s)."
            ),
        )
        parser.add_argument(
            "--sim-substeps",
            type=int,
            default=20,
            help="Inner solver steps per visualization frame; XPBD relies on substepping to converge stiff loop closures.",
        )
        parser.add_argument(
            "--solver-iterations",
            type=int,
            default=8,
            help="XPBD constraint iterations per substep; higher values tighten loop-closure residuals.",
        )
        parser.set_defaults(world_count=1)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)

    example = Example(viewer, args)

    newton.examples.run(example, args)
