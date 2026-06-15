# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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
# Example Robot DR Legs (PhoenX)
#
# Loads the Disney Research bipedal-legs USD asset and simulates it with
# :class:`SolverPhoenX`. The asset has six closed kinematic loops
# (parallel-rod linkages and outer ankle brackets).
#
# Topology choices follow the MuJoCo example
# (``example_robot_dr_legs_mujoco.py``) because the actuated joints
# must remain on the articulation tree -- if they end up as loop
# closers, the stiff USD-authored PD drive on a loop-closure
# constraint pins the bodies it spans (in the prior PhoenX
# experiment, ``foot_l`` was effectively glued to its inner-chain
# ankle bracket while the outer chain dangled, never moving).
#
# We therefore:
#   - flip ten joints so all hinges share a consistent
#     ``body0=parent`` orientation,
#   - exclude six unactuated outer/parallel joints from the
#     articulation tree (they remain in the model as ordinary joint
#     constraints; PhoenX treats them just like any other
#     constraint),
#   - tag the pelvis as the articulation root.
#
# Drives the 12 actuated joints from the bundled walking animation.
# Half of those joints were flipped, so their animation channels are
# negated to compensate. PhoenX handles substepping internally:
# per-frame we call ``solver.step(..., dt=frame_dt)`` exactly once
# and the solver advances ``substeps`` PGS substeps under the hood.
# ``--fps`` controls how often the broad/narrow-phase collision
# detection runs.
#
# Command: python -m newton.examples robot_dr_legs_phoenx --world-count 4
#
###########################################################################

import argparse

import numpy as np
import warp as wp
from pxr import Sdf, Usd, UsdPhysics

import newton
import newton.examples
import newton.utils

# Joints whose body0/body1 (and matching local pose attrs) are swapped
# before ``add_usd()`` so all hinges share a consistent
# ``body0=parent`` convention. The j1-j4 inner-chain joints in this
# asset are already authored with body0=parent (pelvis -> hip ->
# upperleg -> lowerleg -> ankle); only the ankle inner-loop hinges
# (j6_*_i: foot -> ankle_bracket_b) and the parallel-rod hinges
# (j9_*_*: rod -> lowerleg) are inverted in the USD and need
# flipping. Flipping a joint that is already in the right direction
# pushes the importer's tree root onto a leaf body (e.g.
# ankle_bracket_a_l_i) and leaves one side of the robot effectively
# pinned to the floating-base inertia, which is the "left foot
# glued to the world" symptom.
_FLIPPED_JOINTS = (
    "/DR_Legs/Joints/j6_l_i",
    "/DR_Legs/Joints/j6_r_i",
    "/DR_Legs/Joints/j9_l_i",
    "/DR_Legs/Joints/j9_l_o",
    "/DR_Legs/Joints/j9_r_i",
    "/DR_Legs/Joints/j9_r_o",
)

# Joints excluded from the articulation tree so the USD importer sees
# a clean parent-child graph. PhoenX still applies these as ordinary
# joint constraints. All six are unactuated; keeping the actuated
# joints on tree edges avoids stiff PD running through a
# loop-closure constraint.
_LOOP_CLOSER_JOINTS = (
    "/DR_Legs/Joints/j6_l_o",
    "/DR_Legs/Joints/j6_r_o",
    "/DR_Legs/Joints/j8_l_i",
    "/DR_Legs/Joints/j8_l_o",
    "/DR_Legs/Joints/j8_r_i",
    "/DR_Legs/Joints/j8_r_o",
)

# Animation channel -> joint path. The bundled .npy stores 12 columns
# in this order. Channels marked here with a sign of -1 must be
# negated because the corresponding joint was reoriented in
# ``_FLIPPED_JOINTS``.
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
_ANIMATION_CHANNEL_SIGN = np.array([+1, +1, -1, +1, +1, +1, +1, +1, -1, +1, +1, +1], dtype=np.float32)


def _parse_prepare_refresh_stride(value: str) -> int | str:
    if value.strip().lower() == "auto":
        return "auto"
    stride = int(value)
    if stride < 1:
        raise argparse.ArgumentTypeError("prepare refresh stride must be >= 1 or 'auto'")
    return stride


def _swap_attr_pair(prim, name_a: str, name_b: str) -> None:
    a = prim.GetAttribute(name_a)
    b = prim.GetAttribute(name_b)
    va, vb = a.Get(), b.Get()
    a.Set(vb)
    b.Set(va)


def _flip_joint(stage: Usd.Stage, joint_path: str) -> None:
    joint = stage.GetPrimAtPath(joint_path)
    body0 = joint.GetRelationship("physics:body0")
    body1 = joint.GetRelationship("physics:body1")
    t0, t1 = list(body0.GetTargets()), list(body1.GetTargets())
    body0.SetTargets(t1)
    body1.SetTargets(t0)
    _swap_attr_pair(joint, "physics:localPos0", "physics:localPos1")
    _swap_attr_pair(joint, "physics:localRot0", "physics:localRot1")


# ---------------------------------------------------------------------------
# Graph-captured animation pipeline. The per-frame target update used to be
# host-side (numpy fancy-index + ``assign``), which broke CUDA graph capture.
# These two kernels move the update onto the GPU so the entire per-frame
# pipeline can be captured into a single graph and replayed without any
# Python-side bookkeeping.
# ---------------------------------------------------------------------------


@wp.kernel(enable_backward=False)
def _scatter_animation_targets(
    sim_time: wp.array[wp.float32],
    animation_speed: wp.float32,
    animation_fps: wp.float32,
    n_frames: wp.int32,
    animation_data: wp.array2d[wp.float32],
    animation_indices: wp.array2d[wp.int32],
    joint_target_pos: wp.array[wp.float32],
):
    """Scatter the current animation frame's per-channel target into
    ``joint_target_pos``. ``dim=(world_count, n_channels)``."""
    world_idx, channel = wp.tid()
    frame = wp.int32(sim_time[0] * animation_speed * animation_fps)
    if frame >= n_frames:
        frame = n_frames - 1
    if frame < 0:
        frame = 0
    dof_idx = animation_indices[world_idx, channel]
    joint_target_pos[dof_idx] = animation_data[frame, channel]


@wp.kernel(enable_backward=False)
def _advance_sim_time(sim_time: wp.array[wp.float32], dt: wp.float32):
    """Increment the GPU-resident frame counter by ``dt``. ``dim=1``."""
    sim_time[0] = sim_time[0] + dt


class Example:
    def __init__(self, viewer, args):
        # ``--fps`` controls how often we enter the per-frame pipeline:
        # broad/narrow-phase collide() once, then a single
        # ``solver.step(..., dt=frame_dt)`` call. PhoenX runs
        # ``args.sim_substeps`` PGS substeps internally per step, so
        # the integrator dt is ``frame_dt / sim_substeps``.
        self.fps = args.fps
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.world_count = args.world_count

        self.viewer = viewer

        dr_legs = newton.ModelBuilder(up_axis=newton.Axis.Z)
        # Mirror the kamino DR Legs reference example: ``armature``
        # = 0.011 kg.m^2 is the Dynamixel XH540-V150 reflected rotor
        # inertia ("a_j" in kamino). Kamino additionally sets a
        # viscous joint damping ``b_j = 0.044 N.m.s/rad``; PhoenX has
        # no separate viscous-damping field on JointDofConfig, so
        # that contribution lives inside the USD-authored
        # ``target_kd`` PD term (effectively kd_total = kd_USD + b_j
        # at small angular velocities).
        dr_legs.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5, armature=args.armature
        )
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
        for jp in _FLIPPED_JOINTS:
            _flip_joint(stage, jp)
        for jp in _LOOP_CLOSER_JOINTS:
            stage.GetPrimAtPath(jp).CreateAttribute("physics:excludeFromArticulation", Sdf.ValueTypeNames.Bool).Set(
                True
            )

        # Place the robot with the foot collision boxes resting just
        # above the ground plane (added below at z = 0). The lowest
        # body in neutral pose is the foot box at pelvis_z - 0.262, so
        # a pelvis offset of 0.265 puts the feet ~3 mm above the
        # ground at start-up -- enough to let contacts engage cleanly
        # without a long free-fall.
        dr_legs.add_usd(
            stage,
            xform=wp.transform(wp.vec3(0, 0, 0.265)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )

        # Optionally rescale USD-authored kp/kd before replication.
        kp_scale = args.animation_gain_scale if args.animation else 1.0
        kd_scale = args.animation_kd_scale if (args.animation and args.animation_kd_scale is not None) else kp_scale
        if kp_scale != 1.0 or kd_scale != 1.0:
            none_mode = int(newton.JointTargetMode.NONE)
            for dof_i, mode in enumerate(dr_legs.joint_target_mode):
                if mode != none_mode:
                    dr_legs.joint_target_ke[dof_i] *= kp_scale
                    dr_legs.joint_target_kd[dof_i] *= kd_scale

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(dr_legs, self.world_count)

        builder.default_shape_cfg.ke = 1.0e3
        builder.default_shape_cfg.kd = 1.0e2
        builder.add_ground_plane()

        self.model = builder.finalize()

        # PhoenX handles substepping internally. ``substeps`` controls
        # the number of PGS substeps per :meth:`step` call;
        # ``solver_iterations`` is PGS iterations per substep;
        # ``velocity_iterations`` is the TGS-soft relax sweep count.
        # ``armature_mode`` defaults to ``"bake"``; ``"exact"`` is
        # physically the more principled per-joint virtual-rotor
        # treatment, but the PhoenX docstring warns it diverges for
        # dr_legs-class scenes (light parallel rods + stiff PD +
        # ground contact) because external-force paths still see the
        # raw 2e-7 kg.m^2 rod inertia, and a stiff contact impulse
        # against that raw inertia launches the chain instantly.
        # ``"bake"`` is the empirically working choice for this asset.
        self.solver = newton.solvers.SolverPhoenX(
            self.model,
            substeps=args.sim_substeps,
            solver_iterations=args.solver_iterations,
            velocity_iterations=args.velocity_iterations,
            prepare_refresh_stride=args.prepare_refresh_stride,
        )

        self.state_0 = self.model.state()
        self.control = self.model.control()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        # PhoenX always uses the model's CollisionPipeline contacts
        # buffer; the solver auto-attaches a sticky pipeline on
        # construction so ``model.contacts()`` returns the right
        # buffer.
        self.contacts = self.model.contacts()

        self._animation_enabled: bool = bool(args.animation)
        self._animation_speed: float = args.animation_speed
        self._sim_time_wp = wp.array([0.0], dtype=float)
        if self._animation_enabled:
            self._init_animation(asset_path)

        self.viewer.set_model(self.model)
        self._fix_picking_effective_mass()

        # Cap rendered worlds. Render time scales ~linearly with visible
        # robots (~0.1 ms/world for the dr_legs mesh-heavy asset on a
        # Blackwell-class GPU); at world_count=1000 the GPU spends ~95 ms
        # per frame on rasterisation alone (~10 fps), independent of
        # physics. Limiting visible worlds keeps physics simulating all
        # ``world_count`` robots while only drawing the first
        # ``visible_world_count`` of them. ``0`` means render all.
        n_visible = args.visible_world_count
        if 0 < n_visible < self.world_count:
            self.viewer.set_visible_worlds(list(range(n_visible)))

        self.capture()

    def _fix_picking_effective_mass(self) -> None:
        """Override Newton picking's per-body effective-mass with the
        true connected-component sum.

        Newton viewer picking derives per-body effective mass by writing
        ``body_art[child] = joint_art[j]``
        for every joint touching ``child`` -- last write wins. In the
        DR Legs asset six bodies (the two feet plus the four
        parallel-rods) are the *child* of both a tree-edge joint
        (articulation=0) and an excluded loop-closer joint
        (articulation=-1). The last-write-wins assignment leaves them
        flagged as free bodies, so their picking ``effective_mass``
        falls back to their own ``body_mass`` (foot 0.14 kg, parallel
        rod 0.016 kg). With the default ``pick_max_acceleration=5 g``
        the picking force on a parallel rod saturates at <1 N and on
        a foot at ~7 N -- not enough to overcome the chain's foot
        friction, which presents in the viewer as the picked body
        being trapped inside a tiny cube.

        We reseed ``Picking._pick_effective_mass`` with the
        connected-component total mass: every body in the same
        component as the pelvis gets the full articulation mass.
        """
        picking = getattr(self.viewer, "picking", None)
        if picking is None or picking._pick_effective_mass is None:
            return

        body_mass = self.model.body_mass.numpy()
        joint_parent = self.model.joint_parent.numpy()
        joint_child = self.model.joint_child.numpy()
        adj: dict[int, list[int]] = {b: [] for b in range(-1, self.model.body_count)}
        for j in range(self.model.joint_count):
            p = int(joint_parent[j])
            c = int(joint_child[j])
            adj[p].append(c)
            adj[c].append(p)

        component_id = np.full(self.model.body_count, -1, dtype=np.int32)
        cid = 0
        component_mass: list[float] = []
        for start in range(self.model.body_count):
            if component_id[start] >= 0:
                continue
            stack = [start]
            mass = 0.0
            while stack:
                b = stack.pop()
                if b < 0 or component_id[b] >= 0:
                    continue
                component_id[b] = cid
                mass += float(body_mass[b])
                for n in adj[b]:
                    if n >= 0 and component_id[n] < 0:
                        stack.append(n)
            component_mass.append(mass)
            cid += 1

        eff = np.zeros(self.model.body_count, dtype=np.float32)
        for b in range(self.model.body_count):
            eff[b] = component_mass[component_id[b]]
        picking._pick_effective_mass.assign(eff)

    def _init_animation(self, asset_path) -> None:
        # The full trajectory is uploaded once into a GPU-resident
        # ``wp.array2d[float]`` of shape ``(n_frames, n_channels)``.
        # Per-frame target updates are then a single Warp kernel
        # launch that reads the current frame from
        # ``self._sim_time_wp`` and scatters it into
        # ``control.joint_target_pos`` -- no host->device copy and no
        # Python work in the hot loop.
        anim_file = str(asset_path / "dr_legs/animation" / "dr_legs_animation_100fps.npy")
        anim = np.load(anim_file).astype(np.float32)
        if anim.shape[1] != len(_ANIMATION_JOINT_PATHS):
            raise RuntimeError(f"animation has {anim.shape[1]} channels, expected {len(_ANIMATION_JOINT_PATHS)}")
        joint_label = list(self.model.joint_label)
        joint_qd_start = self.model.joint_qd_start.numpy()
        try:
            channel_dofs = np.array(
                [joint_qd_start[joint_label.index(path)] for path in _ANIMATION_JOINT_PATHS],
                dtype=np.int32,
            )
        except ValueError as e:
            raise RuntimeError(f"animation joint not found in model.joint_label: {e}") from e
        n_dof_per_world = self.model.joint_dof_count // self.world_count
        world_offsets = np.arange(self.world_count, dtype=np.int32) * n_dof_per_world
        animation_indices = channel_dofs[None, :] + world_offsets[:, None]

        self._animation_n_frames = anim.shape[0]
        self._animation_fps = 100.0
        self._animation_data_wp = wp.array(anim * _ANIMATION_CHANNEL_SIGN[None, :], dtype=float)
        self._animation_indices_wp = wp.array(animation_indices, dtype=int)

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        # One collide() per frame -- ``--fps`` chooses how often
        # broad/narrow-phase fires. PhoenX advances
        # ``args.sim_substeps`` PGS substeps inside the single
        # ``solver.step`` below. PhoenX imports ``state_0`` into its
        # internal body container before advancing, so in-place export
        # is graph-safe and avoids copy-back kernels.
        #
        # Animation targets are scattered into
        # ``control.joint_target_pos`` by a Warp kernel that reads
        # the GPU-resident frame counter ``self._sim_time_wp``; the
        # trajectory itself lives in ``self._animation_data_wp``.
        # The advance kernel at the end bumps the counter so
        # successive graph replays walk the clip without any
        # Python-side bookkeeping.
        if self._animation_enabled:
            wp.launch(
                _scatter_animation_targets,
                dim=(self.world_count, len(_ANIMATION_JOINT_PATHS)),
                inputs=[
                    self._sim_time_wp,
                    self._animation_speed,
                    self._animation_fps,
                    self._animation_n_frames,
                    self._animation_data_wp,
                    self._animation_indices_wp,
                    self.control.joint_target_pos,
                ],
            )
        self.model.collide(self.state_0, self.contacts)
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.solver.step(self.state_0, self.state_0, self.control, self.contacts, self.frame_dt)
        if self._animation_enabled:
            wp.launch(_advance_sim_time, dim=1, inputs=[self._sim_time_wp, self.frame_dt])

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

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.add_argument(
            "--fps",
            type=int,
            default=60,
            choices=(60, 120),
            help=(
                "Frame rate (Hz) at which collision detection runs and the"
                " solver is stepped. PhoenX advances ``--sim-substeps``"
                " PGS substeps internally per step, so the integrator dt"
                " is ``1 / (fps * sim_substeps)``."
            ),
        )
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
            help="Multiplier on USD-authored kp/kd.",
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
            default=1.0,
            help=(
                "Animation playback rate; 1.0 plays the gait at the authored 100 Hz."
                " The bundled animation is open-loop -- without a"
                " balance / posture feedback term the robot eventually"
                " falls and never alternates stance / swing legs. Lower"
                " values (e.g. 0.25) let the COM track the support"
                " polygon longer."
            ),
        )
        parser.add_argument(
            "--sim-substeps",
            type=int,
            default=40,
            help=(
                "PhoenX internal PGS substeps per ``solver.step`` call."
                " The asset has ~6 g parallel-rod / ankle-bracket bodies"
                " with min principal inertia ~2e-7 kg.m^2. The actuated"
                " joints use a small ``--armature`` rotor, while the"
                " unactuated parallel-rod / ankle-bracket loop closers"
                " still run on the raw body inertia and set the stiffest"
                " local timestep bound. At fps=60 the default gives a"
                " 0.42 ms substep and is validated by the walking"
                " regression."
            ),
        )
        parser.add_argument(
            "--solver-iterations",
            type=int,
            default=8,
            help="PhoenX PGS iterations per substep.",
        )
        parser.add_argument(
            "--velocity-iterations",
            type=int,
            default=1,
            help="PhoenX TGS-soft velocity-relaxation sweeps per substep.",
        )
        parser.add_argument(
            "--prepare-refresh-stride",
            type=_parse_prepare_refresh_stride,
            default="auto",
            help=(
                "Refresh PhoenX prepared row data every N internal substeps,"
                " or 'auto' for the graph-capture-safe default. Auto keeps"
                " rigid contact worlds at a refresh stride of at most 3 and"
                " is the validated Dr Legs default."
            ),
        )
        parser.add_argument(
            "--armature",
            type=float,
            default=0.001,  # 0.011 is the physical XH540 reflected rotor inertia.
            help=(
                "Per-joint axial armature [kg*m^2] applied as the"
                " builder default. 0.001 is the validated PhoenX walking"
                " setting. 0.011 is the Dynamixel XH540-V150 reflected"
                " rotor inertia used by the Kamino reference example"
                " (``a_j``), but needs a more robust PhoenX armature path"
                " before it is a low-substep default."
            ),
        )
        parser.add_argument(
            "--visible-world-count",
            type=int,
            default=2000,
            help=(
                "Render only the first N replicated robots. Physics still"
                " simulates all ``--world-count`` worlds; the cap only"
                " limits GL draw work. Defaults to 32 because the"
                " dr_legs mesh asset is rasterisation-bound at scale --"
                " on a Blackwell-class GPU 1000 visible robots take ~95"
                " ms/frame (10 fps) regardless of how fast physics runs,"
                " 32 visible takes ~10 ms (100 fps). ``0`` means render"
                " every world."
            ),
        )

        parser.set_defaults(world_count=2000)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    # viewer._paused = True
    viewer.show_visual = False
    viewer.show_inertia_boxes = True

    example = Example(viewer, args)

    newton.examples.run(example, args)
