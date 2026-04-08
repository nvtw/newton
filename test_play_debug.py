"""Debug play script: run 200 frames headless, print diagnostics."""
import warp as wp
wp.init()
import numpy as np
import newton, newton.utils
from newton import GeoType, State
from newton._src.onnx_runtime import OnnxRuntime

_LAB_TO_MUJOCO = np.array([0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11], dtype=np.intp)
_MUJOCO_TO_LAB = np.array([0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11], dtype=np.intp)

def qri(q, v):
    qw = q[..., 3:4]; qv = q[..., :3]
    return v * (2 * qw * qw - 1) - np.cross(qv, v, axis=-1) * qw * 2 + qv * np.sum(qv * v, axis=-1, keepdims=True) * 2

# Build robot
builder = newton.ModelBuilder()
newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.06, limit_ke=1e3, limit_kd=1e1)
builder.default_shape_cfg.ke = 5e4; builder.default_shape_cfg.kd = 5e2
builder.default_shape_cfg.kf = 1e3; builder.default_shape_cfg.mu = 0.75
p = newton.utils.download_asset("anybotics_anymal_c")
builder.add_urdf(str(p / "urdf" / "anymal.urdf"), floating=True, enable_self_collisions=False,
                 collapse_fixed_joints=True, ignore_inertial_definitions=False)
for i in range(len(builder.shape_type)):
    if builder.shape_type[i] == GeoType.SPHERE:
        r = builder.shape_scale[i][0]; builder.shape_scale[i] = (r * 2, 0, 0)
for name, val in {"LF_HAA": 0, "LF_HFE": 0.4, "LF_KFE": -0.8, "RF_HAA": 0, "RF_HFE": 0.4, "RF_KFE": -0.8,
                   "LH_HAA": 0, "LH_HFE": -0.4, "LH_KFE": 0.8, "RH_HAA": 0, "RH_HFE": -0.4, "RH_KFE": 0.8}.items():
    idx = next(i for i, l in enumerate(builder.joint_label) if l.endswith(f"/{name}"))
    builder.joint_q[idx + 6] = val
for i in range(len(builder.joint_target_ke)):
    builder.joint_target_ke[i] = 150; builder.joint_target_kd[i] = 5
builder.add_ground_plane()
model = builder.finalize()
solver = newton.solvers.SolverMuJoCo(model, use_mujoco_contacts=False, solver="newton",
                                      ls_parallel=False, ls_iterations=50, njmax=50, nconmax=100)
state_0 = model.state(); state_1 = model.state()
control = model.control()
contacts = model.contacts()
newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

policy = OnnxRuntime("anymal_best.onnx", device=str(wp.get_device()))
joint_pos_initial = state_0.joint_q.numpy()[7:].reshape(1, 12).astype(np.float32)
act_raw = np.zeros((1, 12), dtype=np.float32)
command = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

print(f"Joint pos initial: {joint_pos_initial[0]}")
print(f"Control joint_target_pos shape: {control.joint_target_pos.shape}")
print(f"Control joint_target_pos dtype: {control.joint_target_pos.dtype}")
print(f"Control joint_target_pos: {control.joint_target_pos.numpy()}")

for frame in range(200):
    q = state_0.joint_q.numpy(); qd = state_0.joint_qd.numpy()
    root_quat = q[3:7].reshape(1, 4)
    vel_b = qri(root_quat, qd[:3].reshape(1, 3))
    avel_b = qri(root_quat, qd[3:6].reshape(1, 3))
    grav = qri(root_quat, np.array([[0, 0, -1]], dtype=np.float32))
    jp_rel = (q[7:].reshape(1, 12) - joint_pos_initial)[:, _LAB_TO_MUJOCO]
    jv = qd[6:].reshape(1, 12)[:, _LAB_TO_MUJOCO]
    obs = np.concatenate([vel_b, avel_b, grav, command, jp_rel, jv, act_raw], axis=1).astype(np.float32)

    obs_wp = wp.array(obs, dtype=wp.float32, device=wp.get_device())
    act_wp = policy({"observation": obs_wp})["action"]
    act_raw = act_wp.numpy()
    rearranged = act_raw[:, _MUJOCO_TO_LAB]
    targets = joint_pos_initial + 0.5 * rearranged

    # Write targets to control
    target_np = control.joint_target_pos.numpy()
    target_np[6:18] = targets.squeeze(0)
    wp.copy(control.joint_target_pos, wp.array(target_np, dtype=wp.float32, device=wp.get_device()))

    # Simulate
    for _ in range(4):
        state_0.clear_forces()
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 0.005)
        state_0, state_1 = state_1, state_0

    if frame % 20 == 0:
        pos = state_0.joint_q.numpy()
        vel = state_0.joint_qd.numpy()
        print(f"Frame {frame:3d}: pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}] "
              f"vel=[{vel[0]:.3f},{vel[1]:.3f},{vel[2]:.3f}] "
              f"act_raw=[{act_raw[0,0]:.3f},{act_raw[0,1]:.3f},{act_raw[0,2]:.3f}] "
              f"targets=[{targets[0,0]:.3f},{targets[0,1]:.3f},{targets[0,2]:.3f}] "
              f"joint_q=[{pos[7]:.3f},{pos[8]:.3f},{pos[9]:.3f}]")

print("\nFinal position:", state_0.joint_q.numpy()[:3])
print("Distance traveled:", np.linalg.norm(state_0.joint_q.numpy()[:2]))
