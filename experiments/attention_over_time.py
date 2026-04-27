"""Measure per-token attention weights over time for a trained LegTransformer ant policy."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
import torch
import yaml
from tqdm import tqdm

from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler

from transformer_rl import Policy, Value


def _build_agent(cfg, obs_space, act_space, device, policy_model, value_model):
    memory = RandomMemory(memory_size=1, num_envs=1, device=device)
    pc = PPO_CFG()
    for k, v in cfg["ppo"].items():
        setattr(pc, k, v)
    pc.observation_preprocessor = RunningStandardScaler
    pc.observation_preprocessor_kwargs = {"size": obs_space, "device": device}
    pc.value_preprocessor = RunningStandardScaler
    pc.value_preprocessor_kwargs = {"size": 1, "device": device}
    pc.experiment.directory = cfg["experiment"]["directory"]
    pc.experiment.experiment_name = cfg["experiment"]["name"]
    pc.experiment.write_interval = 0
    pc.experiment.checkpoint_interval = 0
    return PPO(
        models={"policy": policy_model, "value": value_model},
        memory=memory, cfg=pc,
        observation_space=obs_space, action_space=act_space, device=device,
    )


def _patch_attn(encoder):
    """Patch each layer's _sa_block to capture per-head attention weights."""
    captures = [[] for _ in encoder.layers]
    for i, layer in enumerate(encoder.layers):
        def _make(lay, cap):
            def patched_sa_block(x, attn_mask, key_padding_mask, is_causal=False):
                out, w = lay.self_attn(x, x, x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                    is_causal=is_causal)
                cap.append(w.detach().cpu())  # [1, n_heads, 9, 9]
                return lay.dropout1(out)
            return patched_sa_block
        layer._sa_block = _make(layer, captures[i])
    return captures


def run(seed, episodes, checkpoint, config, cpu, output):
    cfg = yaml.safe_load(config.read_text())
    device = torch.device("cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    env = gym.make("Ant-v5")
    policy_model = Policy(env.observation_space, env.action_space, device=device)
    value_model = Value(env.observation_space, env.action_space, device=device)
    agent = _build_agent(cfg, env.observation_space, env.action_space, device, policy_model, value_model)
    agent.load(str(checkpoint))
    # agent.load() calls .eval() which enables a C++ fast path that bypasses _sa_block.
    # dropout=0.0, so train mode is identical numerically but forces the Python slow path.
    policy_model.train()
    value_model.train()
    agent.training = True  # so act() calls value model (gated on agent.training)

    actor_captures  = _patch_attn(policy_model.net.encoder)
    critic_captures = _patch_attn(value_model.net.encoder)
    n_layers = len(actor_captures)

    all_actor, all_critic, all_rewards, episode_ends = [], [], [], []

    for ep in tqdm(range(episodes), desc="episodes"):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, outputs = agent.act(obs_t, None, timestep=0, timesteps=1)
            action = outputs["mean_actions"].squeeze(0).cpu().numpy().astype(np.float32)

            # captures[l][-1]: [1, n_heads, 9, 9] — grab batch=0
            all_actor.append(np.stack([actor_captures[l][-1][0].numpy()  for l in range(n_layers)]))
            all_critic.append(np.stack([critic_captures[l][-1][0].numpy() for l in range(n_layers)]))

            obs, reward, terminated, truncated, _ = env.step(action)
            all_rewards.append(float(reward))
            done = terminated or truncated
        episode_ends.append(len(all_actor))

    env.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        actor_attn  =np.array(all_actor,  dtype=np.float32),  # [T, n_layers, n_heads, 9, 9]
        critic_attn =np.array(all_critic, dtype=np.float32),  # [T, n_layers, n_heads, 9, 9]
        rewards     =np.array(all_rewards, dtype=np.float32), # [T]
        episode_ends=np.array(episode_ends, dtype=np.int64),  # [E]
    )
    T = len(all_actor)
    print(f"saved {T} steps across {episodes} episodes → {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--config", type=Path, default=Path("configs/ppo_ant.yaml"))
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    if args.checkpoint is None:
        args.checkpoint = Path(f"runs/ant_transformer_ppo_seed{args.seed}/checkpoints/best_agent.pt")
    if args.output is None:
        args.output = Path(f"data/attention_over_time_seed{args.seed}.npz")

    run(args.seed, args.episodes, args.checkpoint, args.config, args.cpu, args.output)
