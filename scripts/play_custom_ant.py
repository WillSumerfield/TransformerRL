"""Play a trained CustomAnt-v5 PPO agent with the MuJoCo viewer."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gymnasium as gym
from utils import find_latest_run
import numpy as np
import torch
import yaml
import envs  # registers CustomAnt-v5
from envs import sample_valid_mask, LimbMaskObsWrapper

from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler

import torch
from transformer_rl import Policy, Value, OBS_DIM


class LimbMaskScaler(RunningStandardScaler):
    """Normalizes obs[:OBS_DIM]; passes mask dims through unchanged."""
    def forward(self, x, *, train=False, inverse=False, no_grad=True):
        if x is None:
            return None
        core = super().forward(x[..., :OBS_DIM], train=train,
                               inverse=inverse, no_grad=no_grad)
        return torch.cat([core, x[..., OBS_DIM:]], dim=-1)


def build_agent(cfg: dict, obs_space, act_space, device: torch.device):
    models = {
        "policy": Policy(obs_space, act_space, device=device),
        "value":  Value(obs_space, act_space, device=device),
    }
    memory = RandomMemory(memory_size=1, num_envs=1, device=device)

    pc = PPO_CFG()
    for k, v in cfg["ppo"].items():
        setattr(pc, k, v)
    pc.observation_preprocessor = LimbMaskScaler
    pc.observation_preprocessor_kwargs = {
        "size": gym.spaces.Box(-float("inf"), float("inf"), shape=(OBS_DIM,)),
        "device": device,
    }
    pc.value_preprocessor = RunningStandardScaler
    pc.value_preprocessor_kwargs = {"size": 1, "device": device}
    pc.experiment.directory = cfg["experiment"]["directory"]
    pc.experiment.experiment_name = cfg["experiment"]["name"]
    pc.experiment.write_interval = 0
    pc.experiment.checkpoint_interval = 0

    agent = PPO(models=models, memory=memory, cfg=pc,
                observation_space=obs_space, action_space=act_space, device=device)
    return agent


def run(episodes: int, seed: int, env_id: str, render: bool,
        checkpoint: Path, config: Path, cpu: bool, fps: float, prob_mask: float,
        record: Path | None) -> None:
    cfg = yaml.safe_load(config.read_text())
    device = torch.device("cpu" if cpu else
                          "cuda" if torch.cuda.is_available() else "cpu")

    if record is not None:
        record.mkdir(parents=True, exist_ok=True)
        env = gym.make(env_id, render_mode="rgb_array")
        env = LimbMaskObsWrapper(env)
        env = gym.wrappers.RecordVideo(env, video_folder=str(record),
                                       episode_trigger=lambda i: True,
                                       name_prefix="custom_ant")
    else:
        env = gym.make(env_id, render_mode="human" if render else None)
        env = LimbMaskObsWrapper(env)
    agent = build_agent(cfg, env.observation_space, env.action_space, device)
    agent.load(str(checkpoint))

    dt = 1.0 / fps if (render and record is None and fps > 0) else 0.0
    for ep in range(episodes):
        mask = sample_valid_mask() if np.random.rand() < prob_mask else np.ones(8, dtype=bool)
        print(f"ep {ep} mask: {mask.astype(int)}")
        obs, _ = env.reset(seed=seed + ep, options={"limb_mask": mask})
        ret, steps, done = 0.0, 0, False
        while not done:
            t0 = time.perf_counter()
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, outputs = agent.act(obs_t, None, timestep=0, timesteps=1)
            action = outputs["mean_actions"].squeeze(0).cpu().numpy().astype(np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            ret += float(reward); steps += 1
            done = terminated or truncated or (steps > 100)
            if dt > 0:
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
        print(f"ep {ep}: return={ret:.2f}  steps={steps}")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env", type=str, default="CustomAnt-v5", dest="env_id")
    p.add_argument("--no-render", action="store_false", dest="render")
    p.add_argument("--checkpoint", type=Path,
                   default=find_latest_run("custom_ant_transformer_ppo"))
    p.add_argument("--config", type=Path, default=Path("configs/ppo_custom_ant.yaml"))
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--fps", type=float, default=60.0)
    p.add_argument("--prob_mask", type=float, default=1.0, help="Prob of using a new random limb mask each episode")
    p.add_argument("--record", type=Path, default=None,
                   help="dir to save mp4 per episode; forces rgb_array render")
    run(**vars(p.parse_args()))
