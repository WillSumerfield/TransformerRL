"""Benchmark full PPO loop throughput: rollout + update, across (mode, num_envs, rollout_size).

Mirrors the SKRL PPO training loop:
  rollout: preprocess obs -> policy.act() -> env.step()
  update:  LEARNING_EPOCHS x MINI_BATCHES forward+backward through policy + value
"""
import functools
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import gymnasium as gym
import envs  # registers CustomAnt-v5
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from skrl.resources.preprocessors.torch import RunningStandardScaler

from transformer_rl import Policy, Value

MODES = ["sync", "async"]
NUM_ENVS_LIST = [8, 16, 32, 64, 128, 256, 512]
ROLLOUT_SIZES = [256, 512, 1024, 2048]
LEARNING_EPOCHS = 10
MINI_BATCHES = 32
N_WARMUP = 2
N_MEASURE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _poll_gpu_util(samples: list, stop: threading.Event):
    while not stop.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2,
            ).stdout.strip().split("\n")[0]
            samples.append(float(out))
        except Exception:
            pass
        stop.wait(0.5)


def _make_env(env_id: str):
    import envs  # ensure registered in subprocess
    return gym.make(env_id)


def benchmark(mode: str, n: int, rollout_size: int) -> tuple[float, float]:
    factories = [functools.partial(_make_env, "CustomAnt-v5") for _ in range(n)]
    venvs = AsyncVectorEnv(factories) if mode == "async" else SyncVectorEnv(factories)

    obs_space = venvs.single_observation_space
    act_space = venvs.single_action_space
    obs_dim = obs_space.shape[0]
    act_dim = act_space.shape[0]
    total = rollout_size * n
    mb_size = total // MINI_BATCHES

    policy = Policy(obs_space, act_space, device=DEVICE).to(DEVICE)
    value = Value(obs_space, act_space, device=DEVICE).to(DEVICE)
    scaler = RunningStandardScaler(size=obs_space, device=DEVICE)
    opt = torch.optim.Adam(
        list(policy.parameters()) + list(value.parameters()), lr=1e-4
    )

    buf_obs      = torch.zeros(rollout_size, n, obs_dim, device=DEVICE)
    buf_acts     = torch.zeros(rollout_size, n, act_dim, device=DEVICE)
    buf_log_prob = torch.zeros(rollout_size, n, 1,       device=DEVICE)
    buf_vals     = torch.zeros(rollout_size, n, 1,       device=DEVICE)

    obs_np, _ = venvs.reset(seed=0)

    def one_cycle():
        nonlocal obs_np

        # -- Rollout --
        policy.eval()
        value.eval()
        with torch.no_grad():
            for t in range(rollout_size):
                obs_t = torch.from_numpy(obs_np).float().to(DEVICE)
                obs_scaled = scaler(obs_t, train=True)
                acts, act_out = policy.act({"observations": obs_scaled}, role="policy")
                vals, _       = value.act({"observations": obs_scaled}, role="value")

                buf_obs[t]      = obs_t
                buf_acts[t]     = acts
                buf_log_prob[t] = act_out["log_prob"]
                buf_vals[t]     = vals

                obs_np, _, _, _, _ = venvs.step(acts.cpu().numpy())

        # -- Update --
        policy.train()
        value.train()
        flat_obs      = buf_obs.view(total, obs_dim)
        flat_acts     = buf_acts.view(total, act_dim)
        flat_log_prob = buf_log_prob.view(total, 1)
        flat_vals     = buf_vals.view(total, 1)

        for _ in range(LEARNING_EPOCHS):
            idx = torch.randperm(total, device=DEVICE)
            for mb in range(MINI_BATCHES):
                mb_idx = idx[mb * mb_size : (mb + 1) * mb_size]
                mb_obs  = scaler(flat_obs[mb_idx], train=False)
                mb_acts = flat_acts[mb_idx]

                _, new_out = policy.act({"observations": mb_obs, "taken_actions": mb_acts}, role="policy")
                new_vals, _ = value.act({"observations": mb_obs}, role="value")

                ratio = (new_out["log_prob"] - flat_log_prob[mb_idx]).exp()
                surrogate = torch.min(
                    ratio,
                    torch.clamp(ratio, 1 - 0.2, 1 + 0.2),
                )
                loss = -surrogate.mean() + 0.5 * (new_vals - flat_vals[mb_idx]).pow(2).mean()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(policy.parameters()) + list(value.parameters()), 0.5
                )
                opt.step()

    try:
        for _ in range(N_WARMUP):
            one_cycle()

        gpu_samples: list[float] = []
        stop_evt = threading.Event()
        t = threading.Thread(target=_poll_gpu_util, args=(gpu_samples, stop_evt), daemon=True)
        t.start()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_MEASURE):
            one_cycle()
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        stop_evt.set()
        t.join(timeout=1.0)

        steps_per_sec = (N_MEASURE * rollout_size * n) / elapsed
        gpu_util = float(np.mean(gpu_samples)) if gpu_samples else float("nan")
        return steps_per_sec, gpu_util
    finally:
        del buf_obs, buf_acts, buf_log_prob, buf_vals
        venvs.close()


if __name__ == "__main__":
    import csv, datetime

    results = []
    for mode in MODES:
        for n in NUM_ENVS_LIST:
            for rollout_size in ROLLOUT_SIZES:
                print(f"  {mode:5s} x {n:4d} envs x rollout {rollout_size:5d} ...", end=" ", flush=True)
                try:
                    sps, gpu = benchmark(mode, n, rollout_size)
                    results.append((mode, n, rollout_size, sps, gpu))
                    print(f"{sps:10.0f} steps/sec  GPU {gpu:.0f}%")
                except Exception as e:
                    results.append((mode, n, rollout_size, 0.0, float("nan")))
                    print(f"FAILED: {e}")

    print("\n--- sorted by steps/sec ---")
    print(f"{'mode':6} {'n_envs':>7} {'rollout':>8} {'steps/sec':>12} {'gpu_util%':>10}")
    print("-" * 50)
    for mode, n, rollout_size, sps, gpu in sorted(results, key=lambda x: -x[3]):
        gpu_s = f"{gpu:.0f}" if not np.isnan(gpu) else "n/a"
        print(f"{mode:6} {n:7d} {rollout_size:8d} {sps:12.0f} {gpu_s:>10}")

    out = Path("logs") / f"benchmark_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "n_envs", "rollout_size", "steps_per_sec", "gpu_util_pct"])
        w.writerows(results)
    print(f"\nsaved -> {out}")
