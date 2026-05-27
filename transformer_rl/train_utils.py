"""Shared boilerplate for rl_games-based ant training scripts."""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _str_to_bool(s: str) -> bool:
    return s.lower() == "true"


def _adjust_minibatch(cfg: dict, n_envs: int, h_len: int) -> None:
    mb = cfg["minibatch_size"]
    batch = h_len * n_envs
    n_batches = (batch + mb - 1) // mb
    mb = batch // n_batches if n_batches > 1 else batch
    if batch % mb != 0:
        print(f"Error: batch ({batch}) not divisible by minibatch ({mb})")
        sys.exit(1)
    cfg["minibatch_size"] = mb


def _run_random(env_class, args) -> None:
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_envs = args.num_envs or 1
    env = env_class(
        num_envs,
        device,
        rendering=True,
        raise_exception=True,
        seed=args.seed,
        onset_end=0,
        flip_prob=0.5,
    )
    act_low = torch.tensor(env.action_space.low, device=device)
    act_high = torch.tensor(env.action_space.high, device=device)
    total = getattr(env, "total_num_envs", num_envs)
    env.reset()
    while not env.render_finished:
        actions = act_low + torch.rand(total, act_low.shape[0], device=device) * (act_high - act_low)
        env.step(actions)


def run_training(
    default_config: str,
    train_dir: str,
    env_class,
    env_name: str,
    network: tuple | None = None,
    extra_args_fn=None,
    post_config_fn=None,
) -> None:
    import yaml
    import torch
    import gymnasium.spaces
    from argparse import ArgumentParser
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.ivecenv import IVecEnv
    from rl_games.algos_torch import model_builder as mb_module
    from rl_games.torch_runner import Runner
    from vlearn.spaces import Box, Discrete
    from vlearn.torch_utils.wrappers import NewToOldAPICompatilibity

    def convert_space(space):
        if isinstance(space, Box):
            return gymnasium.spaces.Box(low=space.low, high=space.high, shape=space.shape)
        if isinstance(space, Discrete):
            return gymnasium.spaces.Discrete(n=space.n)

    class VlearnEnv(IVecEnv):
        def __init__(self, config_dict, config_name, num_actors, **kwargs):
            self.envs = config_dict[config_name]["env_creator"](num_actors, **kwargs)
            self.num_actors = num_actors

        def step(self, actions):
            return self.envs.step(actions)

        def reset(self):
            return self.envs.reset()

        def get_env_info(self):
            env_info = {
                "observation_space": convert_space(self.envs.observation_space),
                "action_space": convert_space(self.envs.action_space),
            }
            if hasattr(self.envs, "state_space"):
                env_info["state_space"] = convert_space(self.envs.state_space)
            return env_info

    # --- Arg parsing ---
    parser = ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=["train", "play", "random"], default="train")
    parser.add_argument("checkpoint", nargs="?", default=None)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--headless", choices=["True", "False"], default=None)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--horizon_length", type=int)
    parser.add_argument("--config", type=Path, default=None)
    if extra_args_fn is not None:
        extra_args_fn(parser)
    args = parser.parse_args()

    mode = args.mode
    checkpoint = args.checkpoint

    if mode == "random":
        _run_random(env_class, args)
        return

    # --- Config loading ---
    config_path = args.config if args.config is not None \
        else _PROJECT_ROOT / "configs" / default_config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "player" not in config["params"]["config"]:
        config["params"]["config"]["player"] = {}
    config["params"]["config"]["player"]["use_vecenv"] = True
    cfg = config["params"]["config"]
    exp_name = cfg.get("name", "run").removeprefix("ant_")
    cfg["train_dir"] = f"{train_dir}/{exp_name}"
    cfg["full_experiment_name"] = datetime.now().strftime("%d-%H-%M-%S")

    # --- Seed ---
    if args.seed is not None:
        config["params"]["seed"] = args.seed
        torch.cuda.manual_seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(1.0)

    # --- CLI overrides ---
    if args.num_envs is not None:
        config["params"]["config"]["num_actors"] = args.num_envs
    if args.max_epochs is not None:
        config["params"]["config"]["max_epochs"] = args.max_epochs
    if args.horizon_length is not None:
        config["params"]["config"]["horizon_length"] = args.horizon_length

    # --- Script-specific mutations (snap, extra arg application, etc.) ---
    if post_config_fn is not None:
        post_config_fn(args, config)

    # --- Minibatch adjustment ---
    ppo_cfg = config["params"]["config"]
    if "horizon_length" in ppo_cfg:
        _adjust_minibatch(ppo_cfg, ppo_cfg["num_actors"], ppo_cfg["horizon_length"])
        if "central_value_config" in ppo_cfg:
            _adjust_minibatch(
                ppo_cfg["central_value_config"],
                ppo_cfg["num_actors"],
                ppo_cfg["horizon_length"],
            )

    # --- Rendering ---
    if args.headless is None:
        args.headless = "False" if mode == "play" else "True"
    rendering = not _str_to_bool(args.headless)

    env_kwargs = {
        "rendering": rendering,
        "raise_exception": rendering,
        "seed": args.seed,
        **config.get("env", {}),
    }

    # --- Env + vecenv registration ---
    def create_envs(n, **kw):
        assert torch.cuda.is_available()
        device = torch.device("cuda:0")
        envs = env_class(n, device, **env_kwargs)
        if mode == "play":
            envs.inference_mode_post_init_callback()
        return NewToOldAPICompatilibity(envs)

    def make_vecenv(config_name, num_actors, **kw):
        return VlearnEnv(env_configurations.configurations, config_name, num_actors, **kw)

    env_configurations.register(
        env_name,
        {"vecenv_type": "VLEARN", "env_creator": create_envs},
    )
    vecenv.register("VLEARN", make_vecenv)

    if network is not None:
        net_name, net_builder = network
        mb_module.register_network(net_name, net_builder)

    # --- Run ---
    run_args = {"train": mode == "train", "play": mode == "play"}
    if checkpoint:
        run_args["checkpoint"] = checkpoint

    runner = Runner()
    runner.load(config)
    runner.run(run_args)
