"""Training test with memory monitoring. Self-kills at 12GB RSS."""
import os, signal, resource, subprocess, threading, time

RSS_LIMIT = 12000
_stop = threading.Event()

def watchdog():
    """Background thread that kills process if RSS exceeds limit."""
    while not _stop.is_set():
        m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
        if m > RSS_LIMIT:
            print(f"\nWATCHDOG KILL: RSS {m} MB > {RSS_LIMIT} MB", flush=True)
            os._exit(1)
        _stop.wait(2.0)

t = threading.Thread(target=watchdog, daemon=True)
t.start()

import warp as wp
wp.init()

def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024

def gpu_power():
    r = subprocess.run(["nvidia-smi","--query-gpu=power.draw","--format=csv,noheader,nounits","-i","0"], capture_output=True, text=True)
    return float(r.stdout.strip())

from newton._src.pufferlib.envs.anymal import AnymalEnv
from newton._src.pufferlib.trainer import PPOConfig, PPOTrainer

N = 4096
print(f"baseline: {rss()} MB", flush=True)

config = PPOConfig(
    num_envs=N, horizon=24, obs_dim=48, num_actions=12, hidden=128,
    total_timesteps=N * 24 * 100, seed=42, lr=0.005, anneal_lr=True,
    gamma=0.99, gae_lambda=0.95, clip_coef=0.2, vf_coef=2.0, vf_clip_coef=0.2,
    ent_coef=0.01, max_grad_norm=1.5, momentum=0.95, replay_ratio=1.0,
    minibatch_size=32768, continuous=True, reward_clamp=10.0,
    env_name="ANYmal C Walk", best_return_init=-1000.0, device="cuda:0",
    log_interval=10,
    format_return_fn=lambda avg_ret, best_ret, loss_np, sps: (
        f"Ret {avg_ret:8.1f} | SPS {sps:>9,.0f} | RSS {rss()} MB | GPU {gpu_power():.0f}W"
    ),
)

def make_env(device):
    return AnymalEnv(num_envs=N, device=device, seed=config.seed)

PPOTrainer(config, make_env).train()
_stop.set()
