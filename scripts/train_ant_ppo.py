"""PPO training on Ant-v5 w/ the per-joint LegTransformer."""
import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import mujoco
import torch
import yaml

from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch.sequential import SequentialTrainerCfg
from skrl.utils import set_seed

from transformer_rl import Policy, Value


def route_mujoco_warnings(logs_dir: Path, seed: int) -> Path:
    """Redirect MuJoCo's C-level warnings (stderr) to a per-run file.
    Works with vectorization_mode='sync' (single process). For 'async' each
    subprocess would need its own call."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"mujoco_seed{seed}_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    fh = path.open("a", buffering=1)
    def _cb(msg):
        fh.write(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}\n")
    mujoco.set_mju_user_warning(_cb)
    return path


def build(cfg: dict, device: torch.device, seed: int):
    ec = cfg["env"]
    venvs = gym.make_vec(ec["id"], num_envs=ec["num_envs"], vectorization_mode="sync")
    venvs.reset(seed=seed)
    env = wrap_env(venvs)

    models = {
        "policy": Policy(env.observation_space, env.action_space, device=device),
        "value":  Value(env.observation_space, env.action_space, device=device),
    }

    memory = RandomMemory(memory_size=cfg["ppo"]["rollouts"],
                          num_envs=ec["num_envs"], device=device)

    pc = PPO_CFG()
    for k, v in cfg["ppo"].items():
        setattr(pc, k, v)
    pc.observation_preprocessor = RunningStandardScaler
    pc.observation_preprocessor_kwargs = {"size": env.observation_space, "device": device}
    pc.value_preprocessor = RunningStandardScaler
    pc.value_preprocessor_kwargs = {"size": 1, "device": device}
    pc.experiment.directory = cfg["experiment"]["directory"]
    pc.experiment.experiment_name = f"{cfg['experiment']['name']}_seed{seed}"

    agent = PPO(models=models, memory=memory, cfg=pc,
                observation_space=env.observation_space,
                action_space=env.action_space, device=device)
    return agent, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/ppo_ant.yaml"))
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if args.num_envs is not None:
        cfg["env"]["num_envs"] = args.num_envs

    device = torch.device("cpu" if args.cpu else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    set_seed(args.seed)

    log_path = route_mujoco_warnings(Path("logs"), args.seed)
    print(f"mujoco warnings -> {log_path}")

    agent, env = build(cfg, device, seed=args.seed)

    tc = SequentialTrainerCfg()
    tc.timesteps = args.timesteps or cfg["trainer"]["timesteps"]
    tc.headless = cfg["trainer"]["headless"]
    SequentialTrainer(env=env, agents=agent, cfg=tc).train()


if __name__ == "__main__":
    main()
