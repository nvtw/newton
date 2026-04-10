"""v11: Higher initial exploration (logstd=0) + more envs for GPU util."""
import os, resource, threading, subprocess
_stop = threading.Event()
def wd():
    while not _stop.is_set():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024 > 15000: os._exit(1)
        _stop.wait(2.0)
threading.Thread(target=wd, daemon=True).start()
import warp as wp; wp.init()
import numpy as np
from newton._src.pufferlib.envs.anymal import AnymalEnv
from newton._src.pufferlib.trainer import PPOConfig, PPOTrainer
from newton._src.pufferlib.onnx_export import export_policy_to_onnx, save_checkpoint
best = [-1e9]
def ckpt(p, i, s, br, norm):
    if br > best[0]:
        best[0] = br; save_checkpoint(p, "anymal_checkpoint.npz")
        om, ov = (None, None) if norm is None else norm.get_mean_var()
        export_policy_to_onnx(p, 48, 12, "anymal_best.onnx", obs_mean=om, obs_var=ov)
        print(f"  -> Saved {br:.1f}", flush=True)
N = 4096
config = PPOConfig(
    num_envs=N, horizon=24, obs_dim=48, num_actions=12, hidden=256,
    total_timesteps=100_000_000, seed=42,
    optimizer="adamw", lr=3e-4, anneal_lr=False, activation="elu",
    gamma=0.99, gae_lambda=0.95, clip_coef=0.2, vf_coef=1.0, vf_clip_coef=0.2,
    ent_coef=0.01, max_grad_norm=1.0, momentum=0.9,
    replay_ratio=5.0, minibatch_size=8192,
    continuous=True, init_logstd=-1.0, normalize_obs=True, reward_clamp=30.0,
    env_name="ANYmal v13", best_return_init=-1e9, device="cuda:0",
    log_interval=10, checkpoint_fn=ckpt,
    format_return_fn=lambda a,b,l,s: f"Ret {a:8.1f} | Best {b:8.1f} | Ent {l[3]:7.4f} | SPS {s:>9,.0f} | GPU {float(subprocess.run(['nvidia-smi','--query-gpu=power.draw','--format=csv,noheader,nounits','-i','0'],capture_output=True,text=True).stdout.strip()):.0f}W",
)
PPOTrainer(config, lambda d: AnymalEnv(N, d, 42, use_mujoco_contacts=True)).train()
# Test
print("Testing...", flush=True)
from newton._src.onnx_runtime import OnnxRuntime
from newton import GeoType
import newton, newton.utils
_LAB = np.array([0,6,3,9,1,7,4,10,2,8,5,11], dtype=np.intp)
_MJ = np.array([0,4,8,2,6,10,1,5,9,3,7,11], dtype=np.intp)
def qri(q,v):
    qw=q[...,3:4];qv=q[...,:3]
    return v*(2*qw*qw-1)-np.cross(qv,v,axis=-1)*qw*2+qv*np.sum(qv*v,axis=-1,keepdims=True)*2
bb=newton.ModelBuilder();newton.solvers.SolverMuJoCo.register_custom_attributes(bb)
bb.default_joint_cfg=newton.ModelBuilder.JointDofConfig(armature=0.06,limit_ke=1e3,limit_kd=1e1)
bb.default_shape_cfg.ke=5e4;bb.default_shape_cfg.kd=5e2;bb.default_shape_cfg.kf=1e3;bb.default_shape_cfg.mu=0.75
pa=newton.utils.download_asset("anybotics_anymal_c")
bb.add_urdf(str(pa/"urdf"/"anymal.urdf"),floating=True,enable_self_collisions=False,collapse_fixed_joints=True,ignore_inertial_definitions=False)
for i in range(len(bb.shape_type)):
    if bb.shape_type[i]==GeoType.SPHERE: r=bb.shape_scale[i][0]; bb.shape_scale[i]=(r*2,0,0)
for n2,v2 in {"LF_HAA":0,"LF_HFE":0.4,"LF_KFE":-0.8,"RF_HAA":0,"RF_HFE":0.4,"RF_KFE":-0.8,"LH_HAA":0,"LH_HFE":-0.4,"LH_KFE":0.8,"RH_HAA":0,"RH_HFE":-0.4,"RH_KFE":0.8}.items():
    idx=next(i for i,l in enumerate(bb.joint_label) if l.endswith(f"/{n2}")); bb.joint_q[idx+6]=v2
for i in range(len(bb.joint_target_ke)): bb.joint_target_ke[i]=150; bb.joint_target_kd[i]=5
bb.add_ground_plane(); mm=bb.finalize()
sol2=newton.solvers.SolverMuJoCo(mm,use_mujoco_contacts=False,solver="newton",ls_parallel=False,ls_iterations=50,njmax=50,nconmax=100)
s0=mm.state();s1=mm.state();cc=mm.control();ct=mm.contacts()
newton.eval_fk(mm,s0.joint_q,s0.joint_qd,s0)
pol=OnnxRuntime("anymal_best.onnx",device=str(wp.get_device()))
ji=s0.joint_q.numpy()[7:].reshape(1,12).astype(np.float32);ar=np.zeros((1,12),dtype=np.float32)
cmd=np.array([[1,0,0]],dtype=np.float32)
for f in range(500):
    q=s0.joint_q.numpy();qd=s0.joint_qd.numpy();rq=q[3:7].reshape(1,4)
    obs=np.concatenate([qri(rq,qd[:3].reshape(1,3)),qri(rq,qd[3:6].reshape(1,3)),qri(rq,np.array([[0,0,-1]],dtype=np.float32)),cmd,(q[7:].reshape(1,12)-ji)[:,_LAB],qd[6:].reshape(1,12)[:,_LAB],ar],axis=1).astype(np.float32)
    aw=pol({"observation":wp.array(obs,dtype=wp.float32,device=wp.get_device())})["action"];ar=aw.numpy()
    tp=np.zeros(18,dtype=np.float32);tp[6:]=(ji+0.5*ar[:,_MJ]).squeeze(0)
    wp.copy(cc.joint_target_pos,wp.array(tp,dtype=wp.float32,device=wp.get_device()))
    for _ in range(4): s0.clear_forces();mm.collide(s0,ct);sol2.step(s0,s1,cc,ct,0.005);s0,s1=s1,s0
    if f%100==0: print(f"F{f}: pos=[{q[0]:.2f},{q[1]:.2f},{q[2]:.2f}]", flush=True)
q=s0.joint_q.numpy(); dist=np.linalg.norm(q[:2])
print(f"Final: pos=[{q[0]:.2f},{q[1]:.2f},{q[2]:.2f}] dist={dist:.2f} h={q[2]:.2f}", flush=True)
print(f"WALKING: {q[2]>0.3 and dist>3.0}", flush=True)
_stop.set()
