"""Compare working IsaacLab policy vs our trained policy."""
import warp as wp
wp.init()
import numpy as np
import os
from newton._src.onnx_runtime import OnnxRuntime
from newton import GeoType

_LAB = np.array([0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11], dtype=np.intp)
_MJ = np.array([0, 4, 8, 2, 6, 10, 1, 5, 9, 3, 7, 11], dtype=np.intp)


def qri(q, v):
    qw = q[..., 3:4]; qv = q[..., :3]
    return v * (2 * qw * qw - 1) - np.cross(qv, v, axis=-1) * qw * 2 + qv * np.sum(qv * v, axis=-1, keepdims=True) * 2


def build_single_robot():
    import newton, newton.utils
    bb = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(bb)
    bb.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.06, limit_ke=1e3, limit_kd=1e1)
    bb.default_shape_cfg.ke = 5e4; bb.default_shape_cfg.kd = 5e2
    bb.default_shape_cfg.kf = 1e3; bb.default_shape_cfg.mu = 0.75
    pa = newton.utils.download_asset("anybotics_anymal_c")
    bb.add_urdf(str(pa / "urdf" / "anymal.urdf"), floating=True, enable_self_collisions=False,
                collapse_fixed_joints=True, ignore_inertial_definitions=False)
    for i in range(len(bb.shape_type)):
        if bb.shape_type[i] == GeoType.SPHERE:
            r = bb.shape_scale[i][0]; bb.shape_scale[i] = (r * 2, 0, 0)
    for n2, v2 in {"LF_HAA": 0, "LF_HFE": 0.4, "LF_KFE": -0.8, "RF_HAA": 0, "RF_HFE": 0.4, "RF_KFE": -0.8,
                    "LH_HAA": 0, "LH_HFE": -0.4, "LH_KFE": 0.8, "RH_HAA": 0, "RH_HFE": -0.4, "RH_KFE": 0.8}.items():
        idx = next(i for i, l in enumerate(bb.joint_label) if l.endswith(f"/{n2}"))
        bb.joint_q[idx + 6] = v2
    for i in range(len(bb.joint_target_ke)):
        bb.joint_target_ke[i] = 150; bb.joint_target_kd[i] = 5
    bb.add_ground_plane()
    return bb.finalize()


def run_policy(pol, name, n_steps=500):
    mm = build_single_robot()
    sol = newton.solvers.SolverMuJoCo(mm, use_mujoco_contacts=False, solver="newton",
                                       ls_parallel=False, ls_iterations=50, njmax=50, nconmax=100)
    s0 = mm.state(); s1 = mm.state(); cc = mm.control(); ct = mm.contacts()
    newton.eval_fk(mm, s0.joint_q, s0.joint_qd, s0)
    ji = s0.joint_q.numpy()[7:].reshape(1, 12).astype(np.float32)
    ar = np.zeros((1, 12), dtype=np.float32)
    cmd = np.array([[1, 0, 0]], dtype=np.float32)
    all_actions = []; all_heights = []; all_fwd_vel = []
    for f in range(n_steps):
        q = s0.joint_q.numpy(); qd = s0.joint_qd.numpy(); rq = q[3:7].reshape(1, 4)
        obs = np.concatenate([
            qri(rq, qd[:3].reshape(1, 3)), qri(rq, qd[3:6].reshape(1, 3)),
            qri(rq, np.array([[0, 0, -1]], dtype=np.float32)), cmd,
            (q[7:].reshape(1, 12) - ji)[:, _LAB], qd[6:].reshape(1, 12)[:, _LAB], ar,
        ], axis=1).astype(np.float32)
        aw = pol({"observation": wp.array(obs, dtype=wp.float32, device="cuda:0")})["action"]
        ar = aw.numpy()
        all_actions.append(ar[0].copy())
        tp = np.zeros(18, dtype=np.float32)
        tp[6:] = (ji + 0.5 * ar[:, _MJ]).squeeze(0)
        wp.copy(cc.joint_target_pos, wp.array(tp, dtype=wp.float32, device="cuda:0"))
        for _ in range(4):
            s0.clear_forces(); mm.collide(s0, ct)
            sol.step(s0, s1, cc, ct, 0.005); s0, s1 = s1, s0
        all_heights.append(q[2])
        # Forward is +y for anymal (rotated 90 deg)
        all_fwd_vel.append(qd[1])
    aa = np.array(all_actions)
    q = s0.joint_q.numpy()
    print(f"\n{name}:")
    print(f"  action: range=[{aa.min():.3f}, {aa.max():.3f}] mean_abs={np.abs(aa).mean():.3f} std={aa.std():.3f}")
    print(f"  action_rate: {np.abs(np.diff(aa, axis=0)).mean():.4f}")
    print(f"  height: min={min(all_heights):.3f} mean={np.mean(all_heights[50:]):.3f} final={all_heights[-1]:.3f}")
    print(f"  fwd_vel (y): mean={np.mean(all_fwd_vel[50:]):.3f}")
    print(f"  final_pos: [{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}]  dist={np.linalg.norm(q[:2]):.2f}")


import newton, newton.utils

good = OnnxRuntime("/home/twidmer/Documents/git/newton/newton/_src/anymal_c_walking_policy.onnx", device="cuda:0")
run_policy(good, "IsaacLab (WORKING)")

if os.path.exists("anymal_best.onnx"):
    our = OnnxRuntime("anymal_best.onnx", device="cuda:0")
    run_policy(our, "Ours (best checkpoint)")
