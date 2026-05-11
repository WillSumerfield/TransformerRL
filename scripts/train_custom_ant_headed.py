"""PPO training on CustomAnt-v5 w/ the per-joint LegTransformer (headed/rendered mode)."""
import argparse
import datetime
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import mujoco
import torch
import yaml
import envs  # registers CustomAnt-v5

from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.trainers.torch import SequentialTrainer
from skrl.trainers.torch.sequential import SequentialTrainerCfg
from skrl.utils import set_seed

from transformer_rl import Policy, Value, OBS_DIM
from envs import LimbMaskObsWrapper
from envs.curriculum_wrapper import CurriculumWrapper


class LimbMaskScaler(RunningStandardScaler):
    """Normalizes obs[:OBS_DIM]; passes mask dims through unchanged."""
    def __init__(self, size, device, **kwargs):
        core = gym.spaces.Box(-float("inf"), float("inf"), shape=(OBS_DIM,))
        super().__init__(size=core, device=device, **kwargs)

    def forward(self, x, *, train=False, inverse=False, no_grad=True):
        if x is None:
            return None
        core = super().forward(x[..., :OBS_DIM], train=train,
                               inverse=inverse, no_grad=no_grad)
        return torch.cat([core, x[..., OBS_DIM:]], dim=-1)


class _PPO(PPO):
    def __init__(self, *args, onset_start: int, onset_end: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._onset_start = onset_start
        self._onset_end   = onset_end

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        if timestep < self._onset_start:
            p = 0.0
        elif timestep >= self._onset_end:
            p = 1.0
        else:
            p = (timestep - self._onset_start) / (self._onset_end - self._onset_start)
        self.track_data("Curriculum/p_mask", p)
        super().post_interaction(timestep=timestep, timesteps=timesteps)


def route_mujoco_warnings(logs_dir: Path, seed: int) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"mujoco_seed{seed}_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"
    fh = path.open("a", buffering=1)
    def _cb(msg):
        fh.write(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}\n")
    mujoco.set_mju_user_warning(_cb)
    return path


def build(cfg: dict, device: torch.device, seed: int):
    ec = cfg["env"]
    total = cfg["trainer"]["timesteps"]
    onset_start = int(ec["curriculum_onset_start"] * total)
    onset_end   = int(ec["curriculum_onset_end"]   * total)

    wrappers = [
        LimbMaskObsWrapper,
        functools.partial(CurriculumWrapper,
                          onset_start_steps=onset_start,
                          onset_end_steps=onset_end,
                          num_envs=1),
    ]
    venvs = gym.make_vec(ec["id"], num_envs=1, render_mode="human",
                         vectorization_mode="sync", wrappers=wrappers)
    venvs.reset(seed=seed)
    env = wrap_env(venvs)

    models = {
        "policy": Policy(env.observation_space, env.action_space, device=device),
        "value":  Value(env.observation_space, env.action_space, device=device),
    }

    memory = RandomMemory(memory_size=cfg["ppo"]["rollouts"], num_envs=1, device=device)

    pc = PPO_CFG()
    for k, v in cfg["ppo"].items():
        setattr(pc, k, v)
    pc.observation_preprocessor = LimbMaskScaler
    pc.observation_preprocessor_kwargs = {"size": env.observation_space, "device": device}
    pc.experiment.directory = cfg["experiment"]["directory"]
    pc.experiment.experiment_name = f"{cfg['experiment']['name']}_seed{seed}"

    agent = _PPO(models=models, memory=memory, cfg=pc,
                 observation_space=env.observation_space,
                 action_space=env.action_space, device=device,
                 onset_start=onset_start, onset_end=onset_end)
    return agent, env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/ppo_custom_ant.yaml"))
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    device = torch.device("cpu" if args.cpu else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    set_seed(args.seed)

    log_path = route_mujoco_warnings(Path("logs"), args.seed)
    print(f"mujoco warnings -> {log_path}")

    agent, env = build(cfg, device, seed=args.seed)

    exp_dir = Path(cfg["experiment"]["directory"]) / f"{cfg['experiment']['name']}_seed{args.seed}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.yaml").write_text(yaml.dump(cfg, default_flow_style=False))

    tc = SequentialTrainerCfg()
    tc.timesteps = args.timesteps or cfg["trainer"]["timesteps"]
    tc.headless = False
    SequentialTrainer(env=env, agents=agent, cfg=tc).train()


if __name__ == "__main__":
    main()
