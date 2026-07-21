# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Robot control via keyboard
#
# Shows how to control robots pretrained in IsaacLab with RL. Policies are
# loaded from ONNX files and run with Warp-NN's Warp-backed runtime, so PyTorch
# is not required for policy inference.
#
# Press "p" to reset the robot.
# Press "i", "j", "k", "l", "u", "o" to move the robot.
# Run this example with:
# python -m newton.examples robot_policy --robot g1_29dof
# python -m newton.examples robot_policy --robot g1_23dof
# python -m newton.examples robot_policy --robot go2
# python -m newton.examples robot_policy --robot anymal
# python -m newton.examples robot_policy --robot anymal --physx
# python -m newton.examples robot_policy --robot g1_29dof --solver phoenx
# to run the example with a PhysX-trained policy run with --physx
# to run the example with the PhoenX solver instead of MuJoCo run with --solver phoenx
###########################################################################

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp
import yaml
from warp_nn.runtime import OnnxRuntime

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode
from newton.examples.robot.onnx_policy_utils import validate_policy_io_shapes


@dataclass
class RobotConfig:
    """Configuration for a robot including asset paths and policy paths."""

    asset_dir: str
    policy_path: dict[str, str]
    asset_path: str
    yaml_path: str


ROBOT_CONFIGS = {
    "anymal": RobotConfig(
        asset_dir="anybotics_anymal_c",
        policy_path={"mjw": "rl_policies/mjw_anymal.onnx", "physx": "rl_policies/physx_anymal.onnx"},
        asset_path="usd/anymal_c.usda",
        yaml_path="rl_policies/anymal.yaml",
    ),
    "go2": RobotConfig(
        asset_dir="unitree_go2",
        policy_path={"mjw": "rl_policies/mjw_go2.onnx", "physx": "rl_policies/physx_go2.onnx"},
        asset_path="usd/go2.usda",
        yaml_path="rl_policies/go2.yaml",
    ),
    "g1_29dof": RobotConfig(
        asset_dir="unitree_g1",
        policy_path={"mjw": "rl_policies/mjw_g1_29DOF.onnx"},
        asset_path="usd/g1_isaac.usd",
        yaml_path="rl_policies/g1_29dof.yaml",
    ),
    "g1_23dof": RobotConfig(
        asset_dir="unitree_g1",
        policy_path={"mjw": "rl_policies/mjw_g1_23DOF.onnx", "physx": "rl_policies/physx_g1_23DOF.onnx"},
        asset_path="usd/g1_minimal.usd",
        yaml_path="rl_policies/g1_23dof.yaml",
    ),
}


@wp.kernel
def _compute_obs_kernel(
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_pos_initial: wp.array[float],
    physx_to_mjc_idx: wp.array[int],
    gravity_w: wp.vec3,
    command: wp.vec3,
    prev_act: wp.array2d[float],
    num_dofs: int,
    obs: wp.array2d[float],
):
    q = wp.quat(joint_q[3], joint_q[4], joint_q[5], joint_q[6])

    lin_w = wp.vec3(joint_qd[0], joint_qd[1], joint_qd[2])
    ang_w = wp.vec3(joint_qd[3], joint_qd[4], joint_qd[5])

    vel_b = wp.quat_rotate_inv(q, lin_w)
    avel_b = wp.quat_rotate_inv(q, ang_w)
    grav_b = wp.quat_rotate_inv(q, gravity_w)

    obs[0, 0] = vel_b[0]
    obs[0, 1] = vel_b[1]
    obs[0, 2] = vel_b[2]
    obs[0, 3] = avel_b[0]
    obs[0, 4] = avel_b[1]
    obs[0, 5] = avel_b[2]
    obs[0, 6] = grav_b[0]
    obs[0, 7] = grav_b[1]
    obs[0, 8] = grav_b[2]
    obs[0, 9] = command[0]
    obs[0, 10] = command[1]
    obs[0, 11] = command[2]

    for k in range(num_dofs):
        idx = physx_to_mjc_idx[k]
        obs[0, 12 + k] = joint_q[7 + idx] - joint_pos_initial[idx]
        obs[0, 12 + num_dofs + k] = joint_qd[6 + idx]
        obs[0, 12 + 2 * num_dofs + k] = prev_act[0, k]


@wp.kernel
def _build_joint_target_q_kernel(
    act: wp.array2d[float],
    joint_pos_initial: wp.array[float],
    reorder: wp.array[int],
    action_scale: float,
    num_prefix_zeros: int,
    out: wp.array[float],
):
    i = wp.tid()
    if i < num_prefix_zeros:
        out[i] = 0.0
    else:
        j = i - num_prefix_zeros
        idx = reorder[j]
        out[i] = joint_pos_initial[j] + action_scale * act[0, idx]


def load_policy_and_setup_arrays(example: Any, policy_path: str, num_dofs: int, joint_pos_slice: slice):
    """Load ONNX policy and setup device buffers for the policy step."""
    print("[INFO] Loading policy from:", policy_path)
    example.policy = OnnxRuntime(policy_path, device=example.device)
    example.policy_input_name = example.policy.input_names[0]
    example.policy_output_name = example.policy.output_names[0]

    if example.state_0.joint_q is not None:
        example._joint_pos_initial_wp = wp.clone(example.state_0.joint_q[joint_pos_slice])
    else:
        example._joint_pos_initial_wp = wp.zeros(num_dofs, dtype=wp.float32, device=example.device)

    expected = 12 + 3 * num_dofs
    obs_dim = int(example.config["num_observations"]) if "num_observations" in example.config else expected
    if obs_dim != expected:
        raise ValueError(
            f"load_policy_and_setup_arrays: config num_observations={obs_dim} does not match the expected "
            f"layout (12 + 3*num_dofs = {expected})"
        )
    validate_policy_io_shapes(
        policy_path,
        example.policy_input_name,
        example.policy_output_name,
        obs_width=obs_dim,
        action_width=num_dofs,
        context="load_policy_and_setup_arrays",
    )
    example._obs_wp = wp.zeros((1, obs_dim), dtype=wp.float32, device=example.device)
    example._prev_act_wp = wp.zeros((1, num_dofs), dtype=wp.float32, device=example.device)

    example._physx_to_mjc_wp = wp.array(
        np.asarray(example.physx_to_mjc_indices, dtype=np.int32), dtype=wp.int32, device=example.device
    )
    example._mjc_to_physx_wp = wp.array(
        np.asarray(example.mjc_to_physx_indices, dtype=np.int32), dtype=wp.int32, device=example.device
    )
    example._num_dofs = num_dofs


def find_physx_mjwarp_mapping(mjwarp_joint_names, physx_joint_names):
    mjc_to_physx = []
    physx_to_mjc = []
    for j in mjwarp_joint_names:
        if j in physx_joint_names:
            mjc_to_physx.append(physx_joint_names.index(j))

    for j in physx_joint_names:
        if j in mjwarp_joint_names:
            physx_to_mjc.append(mjwarp_joint_names.index(j))

    return mjc_to_physx, physx_to_mjc


class Example:
    def __init__(self, viewer, args):
        if args.robot not in ROBOT_CONFIGS:
            raise ValueError(f"Unknown robot: {args.robot}. Available: {list(ROBOT_CONFIGS.keys())}")
        robot_config = ROBOT_CONFIGS[args.robot]
        print(f"[INFO] Selected robot: {args.robot}")

        asset_directory = str(newton.utils.download_asset(robot_config.asset_dir))
        print(f"[INFO] Asset directory: {asset_directory}")

        yaml_file_path = f"{asset_directory}/{robot_config.yaml_path}"
        with open(yaml_file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        num_dofs = config["num_dofs"]
        print(f"[INFO] Loaded config with {num_dofs} DOFs")

        mjc_to_physx = list(range(num_dofs))
        physx_to_mjc = list(range(num_dofs))

        if args.physx:
            if "physx" not in robot_config.policy_path or "physx_joint_names" not in config:
                physx_robots = [name for name, cfg in ROBOT_CONFIGS.items() if "physx" in cfg.policy_path]
                raise ValueError(
                    f"PhysX policy not available for robot '{args.robot}'. Robots with PhysX support: {physx_robots}"
                )
            policy_path = f"{asset_directory}/{robot_config.policy_path['physx']}"
            mjc_to_physx, physx_to_mjc = find_physx_mjwarp_mapping(
                config["mjw_joint_names"], config["physx_joint_names"]
            )
            if len(mjc_to_physx) != num_dofs or len(physx_to_mjc) != num_dofs:
                missing_mjw = sorted(set(config["physx_joint_names"]) - set(config["mjw_joint_names"]))
                missing_physx = sorted(set(config["mjw_joint_names"]) - set(config["physx_joint_names"]))
                raise ValueError(
                    "PhysX/MJWarp joint mapping is incomplete: "
                    f"expected {num_dofs} DOFs, got {len(mjc_to_physx)} MJWarp-to-PhysX and "
                    f"{len(physx_to_mjc)} PhysX-to-MJWarp entries. "
                    f"Missing from MJWarp: {missing_mjw}; missing from PhysX: {missing_physx}"
                )
        else:
            policy_path = f"{asset_directory}/{robot_config.policy_path['mjw']}"

        solver_name = getattr(args, "solver", "mujoco")

        # Setup simulation parameters first
        fps = 200
        self.frame_dt = 1.0e0 / fps
        self.decimation = 4
        self.cycle_time = 1 / fps * self.decimation

        self.sim_time = 0.0
        self.sim_step = 0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        self.use_mujoco = False
        self.config = config
        self.robot_config = robot_config
        self.solver_name = solver_name

        self.device = wp.get_device()

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        # MuJoCo's custom attributes (joint armature etc.) are read by
        # the importer regardless of which backend ends up stepping the
        # model -- registering them is harmless on the PhoenX path.
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e2,
            limit_kd=1.0e0,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75
        builder.rigid_gap = 0.0

        if self.solver_name == "phoenx":
            # Newton's default ``rigid_gap = 0.1`` (10 cm) is the
            # speculative-contact band the :class:`CollisionPipeline`
            # uses to inflate AABBs and pre-generate contacts before
            # the surfaces actually touch. MuJoCo Warp runs its own
            # collision pipeline (driven by ``geom_margin`` /
            # ``geom_gap`` set up in ``solver_mujoco`` from the same
            # shape arrays) and doesn't lean on this band the same
            # way. For the G1 standing pose with feet at z = 0 a 10 cm
            # gap means PhoenX sees speculative contacts on bodies
            # that aren't actually touching the ground -- contributing
            # to the contact-impulse asymmetry observed at step 0/1.
            # Force gap = 0 on the PhoenX path so the contact set
            # contains only real surface contacts.
            builder.default_shape_cfg.gap = 0.01

        builder.add_usd(
            newton.examples.get_asset(asset_directory + "/" + robot_config.asset_path),
            xform=wp.transform(wp.vec3(0, 0, 0.8)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
            hide_collision_shapes=True,
        )
        builder.approximate_meshes("convex_hull")

        builder.add_ground_plane()

        builder.joint_q[:3] = [0.0, 0.0, 0.76]
        builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
        builder.joint_q[7:] = config["mjw_joint_pos"]

        for i in range(len(config["mjw_joint_stiffness"])):
            builder.joint_target_ke[i + 6] = config["mjw_joint_stiffness"][i]
            builder.joint_target_kd[i + 6] = config["mjw_joint_damping"][i]
            builder.joint_armature[i + 6] = config["mjw_joint_armature"][i]
            builder.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        if self.solver_name == "phoenx":
            # PhoenX enforces stiction strictly (matches the analytic
            # Coulomb law to within a few micrometres of creep over
            # 10 s; see ``newton/_src/solvers/phoenx/tests/
            # _actuator_compare.py``). MuJoCo Warp's solver lets the
            # foot creep ~1 m / 10 s on a 30 deg incline at mu = 0.75
            # and a MuJoCo-trained policy implicitly assumes that
            # micro-creep is available. Without it, the entire ground-
            # reaction torque funnels into the lowest-stiffness DoF
            # (G1 ankle pitch, target_ke = 20 N*m/rad) and bends it,
            # producing the "toes push into the floor" symptom.
            #
            # Scaling PhoenX's foot friction down lets a small amount
            # of slip back into the contact and unloads the ankle.
            # Tune this in-place; values < 1 reduce stiction, > 1
            # increase it. 1.0 = no scaling.
            phoenx_foot_friction_scale = 1.0
            if phoenx_foot_friction_scale != 1.0:
                self._scale_foot_friction(phoenx_foot_friction_scale)

        if self.solver_name == "phoenx":
            # PhoenX runs its own contacts (sticky CollisionPipeline
            # auto-attached). 4 substeps + 8 PGS iterations matches the
            # MuJoCo-parity sweep used by the Anymal walk demo and is
            # the right cadence for PhysX-trained PD policies whose
            # joint stiffnesses sit in the 100-1000 N*m/rad range.
            #
            # ``velocity_readout="finite_difference"`` stamps
            # ``state.body_qd`` with the outer-step pose-delta velocity
            # instead of PhoenX's substep-end value. Matches the
            # post-integration ``qvel`` convention MuJoCo Warp returns
            # (which is what the policy was trained against), and
            # specifically removes the multi-rad/s per-step ringing
            # that the stiff foot-ground contact resolution produces
            # on the ankle DoFs (most visible after the first 1-2
            # steps; see ``_g1_obs_diff.py``).
            # ``velocity_readout="substep_end"`` was empirically the
            # best of the three modes on the G1 standing pose: the
            # finite_difference and substep_average modes both invert
            # the trunk-pitch direction relative to MuJoCo Warp, which
            # turned the policy's projected-gravity observation into
            # an off-distribution signal. ``substep_end`` keeps the
            # right sign (just at higher magnitude than MJ's
            # post-integration qvel). See _g1_obs_diff.py for numbers.
            self.solver = newton.solvers.SolverPhoenX(
                self.model,
                substeps=20,
                solver_iterations=8,
                velocity_iterations=2,
                velocity_readout="substep_end",
            )
        else:
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                use_mujoco_cpu=self.use_mujoco,
                solver="newton",
                nconmax=30,
                njmax=100,
            )

        self.state_temp = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        # PhoenX drives ``model.collide`` from the per-frame loop and
        # needs the standard contacts buffer; MuJoCo gets to size its
        # own from ``get_max_contact_count``.
        if self.solver_name == "phoenx":
            self.contacts = self.model.contacts()
        else:
            self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)

        self.viewer.set_model(self.model)
        self.viewer.vsync = True

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)

        self.physx_to_mjc_indices = np.asarray(physx_to_mjc, dtype=np.int64)
        self.mjc_to_physx_indices = np.asarray(mjc_to_physx, dtype=np.int64)
        self._gravity_w = wp.vec3(0.0, 0.0, -1.0)
        self._command = wp.vec3(0.0, 0.0, 0.0)
        self._reset_key_prev = False

        self.policy = None
        self.policy_input_name = None
        self.policy_output_name = None
        self._joint_pos_initial_wp = None
        self._obs_wp = None
        self._prev_act_wp = None
        self._physx_to_mjc_wp = None
        self._mjc_to_physx_wp = None
        self._num_dofs = None

        load_policy_and_setup_arrays(self, policy_path, config["num_dofs"], slice(7, None))

        self.capture()

    def capture(self):
        self.graph = None
        self.use_graph = False
        if self.device.is_cpu or self.device.is_mempool_enabled:
            print("[INFO] Using graph capture")
            self.use_graph = True
            self.control.joint_target_q = wp.zeros(self.config["num_dofs"] + 6, dtype=wp.float32, device=self.device)
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        need_state_copy = self.use_graph and self.sim_substeps % 2 == 1

        for i in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.viewer.apply_forces(self.state_0)

            # PhoenX needs explicit collision detection per substep; MuJoCo
            # runs its own internal contact solver and gets ``contacts=None``.
            if self.solver_name == "phoenx":
                self.model.collide(self.state_0, self.contacts)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            else:
                self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

            if need_state_copy and i == self.sim_substeps - 1:
                self.state_0.assign(self.state_1)
            else:
                self.state_0, self.state_1 = self.state_1, self.state_0

        # ``update_contacts`` is the MuJoCo viewer-side pull of the
        # internal solver's contact buffer; PhoenX already populated
        # ``self.contacts`` via ``model.collide`` above.
        if self.solver_name != "phoenx":
            self.solver.update_contacts(self.contacts, self.state_0)

    def reset(self):
        print("[INFO] Resetting example")
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_1.joint_q, self._initial_joint_q)
        wp.copy(self.state_1.joint_qd, self._initial_joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)
        if self._prev_act_wp is not None:
            self._prev_act_wp.zero_()

    def step(self):
        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)
            self._command = wp.vec3(float(fwd), float(lat), float(rot))
            reset_down = bool(self.viewer.is_key_down("p"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down

        wp.launch(
            _compute_obs_kernel,
            dim=1,
            inputs=[
                self.state_0.joint_q,
                self.state_0.joint_qd,
                self._joint_pos_initial_wp,
                self._physx_to_mjc_wp,
                self._gravity_w,
                self._command,
                self._prev_act_wp,
                self._num_dofs,
                self._obs_wp,
            ],
            device=self.device,
        )
        out = self.policy({self.policy_input_name: self._obs_wp})
        act_wp = out[self.policy_output_name]

        wp.launch(
            _build_joint_target_q_kernel,
            dim=6 + self._num_dofs,
            inputs=[
                act_wp,
                self._joint_pos_initial_wp,
                self._mjc_to_physx_wp,
                float(self.config["action_scale"]),
                6,
                self.control.joint_target_q,
            ],
            device=self.device,
        )

        wp.copy(self._prev_act_wp, act_wp)

        for _ in range(self.decimation):
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

    def _scale_foot_friction(self, scale: float) -> None:
        """Multiply ``model.shape_material_mu`` for shapes whose body
        is a foot / ankle-roll link by ``scale``. Keyword matching
        on the body label catches both Anymal (``*_FOOT``) and G1
        (``*_ankle_roll_link``) without per-robot wiring.
        """
        body_labels = self.model.body_label
        # Names suffixes that mark a "foot" body across the supported
        # robots. Anymal: LF_FOOT/RF_FOOT/LH_FOOT/RH_FOOT. G1 / Go2:
        # left_ankle_roll_link / right_ankle_roll_link.
        foot_keywords = ("foot", "ankle_roll", "FOOT")
        foot_bodies = {i for i, lbl in enumerate(body_labels) if any(kw in lbl for kw in foot_keywords)}
        if not foot_bodies:
            print("[WARN] foot-friction-scale: no foot bodies matched; flag is a no-op")
            return

        shape_body_np = self.model.shape_body.numpy()
        mu_np = self.model.shape_material_mu.numpy()
        n_scaled = 0
        for shape_idx, body_idx in enumerate(shape_body_np):
            if int(body_idx) in foot_bodies:
                mu_np[shape_idx] *= float(scale)
                n_scaled += 1
        self.model.shape_material_mu.assign(mu_np)
        print(
            f"[INFO] foot-friction-scale: {scale:.2f} applied to {n_scaled} foot shapes "
            f"(across {len(foot_bodies)} foot bodies)"
        )

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--robot",
            type=str,
            default="g1_29dof",
            choices=list(ROBOT_CONFIGS.keys()),
            help="Robot name to load",
        )
        parser.add_argument(
            "--physx",
            action="store_true",
            help="Run physX policy instead of MJWarp.",
        )
        parser.add_argument(
            "--solver",
            choices=["mujoco", "phoenx"],
            default="phoenx",
            help=(
                "Rigid-body solver backend. 'phoenx' (default) steps the policy model with SolverPhoenX. "
                "'mujoco' uses the backend the policies were trained against. PhoenX is useful for transfer "
                "experiments, but the policy may behave differently since PhoenX's contact and "
                "PD-drive dynamics are not bit-identical to MuJoCo's."
            ),
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
