"""v10: 4096 envs + mujoco_contacts for max throughput, 50M steps."""
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
    total_timesteps=50_000_000, seed=42,
    optimizer="muon", lr=0.005, anneal_lr=True, activation="elu",
    gamma=0.99, gae_lambda=0.95, clip_coef=0.2, vf_coef=2.0, vf_clip_coef=0.2,
    ent_coef=0.01, max_grad_norm=1.5, momentum=0.95,
    replay_ratio=1.0, minibatch_size=32768,
    continuous=True, init_logstd=-1.0, normalize_obs=True, reward_clamp=30.0,
    env_name="ANYmal v10", best_return_init=-1e9, device="cuda:0",
    log_interval=10, checkpoint_fn=ckpt,
    format_return_fn=lambda a,b,l,s: f"Ret {a:8.1f} | Best {b:8.1f} | Ent {l[3]:7.4f} | SPS {s:>9,.0f} | GPU {float(subprocess.run(['nvidia-smi','--query-gpu=power.draw','--format=csv,noheader,nounits','-i','0'],capture_output=True,text=True).stdout.strip()):.0f}W",
)
PPOTrainer(config, lambda d: AnymalEnv(N, d, 42, use_mujoco_contacts=True)).train()
_stop.set()
