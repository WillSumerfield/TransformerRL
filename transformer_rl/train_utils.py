"""Shared boilerplate for rl_games-based ant training scripts."""
from __future__ import annotations

import math
import os
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VIDEOS_DIR = _PROJECT_ROOT / "videos"

# Title/class fragments used to identify the vsim render window.
# SDL2 sets WM_CLASS to "SDL_app" by default; also try common vsim/vlearn names.
_VSIM_WINDOW_TITLES = ["vsim", "vlearn", "SDL_app"]


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


def _find_xwindow(display, title_fragments):
    """Find the vsim render window by title/class fragment, falling back to largest mapped window."""
    from Xlib import X, error as Xerror

    root = display.screen().root
    candidates = []

    def collect(win, depth=0):
        try:
            attrs = win.get_attributes()
            if attrs.map_state != X.IsViewable:
                return
            name = win.get_wm_name() or ""
            cls = win.get_wm_class() or ()
            combined = (name + " " + " ".join(cls)).lower()
            if any(t.lower() in combined for t in title_fragments):
                geom = win.get_geometry()
                candidates.append((geom.width * geom.height, win))
                return  # don't recurse into matched window
        except Xerror.BadWindow:
            return
        try:
            for child in win.query_tree().children:
                collect(child, depth + 1)
        except Xerror.BadWindow:
            pass

    collect(root)
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # Fallback: largest viewable top-level window that isn't the root
    largest = None
    largest_area = 0
    try:
        for child in root.query_tree().children:
            try:
                attrs = child.get_attributes()
                if attrs.map_state != X.IsViewable:
                    continue
                geom = child.get_geometry()
                area = geom.width * geom.height
                if area > largest_area:
                    largest_area = area
                    largest = child
            except Xerror.BadWindow:
                pass
    except Xerror.BadWindow:
        pass
    return largest


class _VideoRecorder:
    """Captures the vsim X11 window each render_callback and writes to mp4."""

    def __init__(self, path: Path, max_frames: int, stop_env: bool):
        self.path = path
        self.max_frames = max_frames
        self.stop_env = stop_env
        self._writer = None
        self._display = None
        self._window = None
        self._size = None
        self._frames = 0
        self.done = False

    def _init(self):
        from Xlib import display as xdisplay
        self._display = xdisplay.Display()

    def _get_window(self):
        if self._window is not None:
            return self._window
        win = _find_xwindow(self._display, _VSIM_WINDOW_TITLES)
        if win:
            self._window = win
        return win

    def _capture(self):
        import numpy as np
        from Xlib import X
        win = self._get_window()
        if win is None:
            return None
        try:
            geom = win.get_geometry()
            raw = win.get_image(0, 0, geom.width, geom.height, X.ZPixmap, 0xFFFFFFFF)
        except Exception:
            self._window = None  # window may have moved; retry next frame
            return None
        img = np.frombuffer(raw.data, dtype=np.uint8).reshape(geom.height, geom.width, 4)
        # Drop alpha → BGR; copy to a writable C-contiguous array (VideoWriter requires it).
        # Crop to even dims so the mp4 encoder doesn't choke.
        h = geom.height - (geom.height & 1)
        w = geom.width - (geom.width & 1)
        return np.ascontiguousarray(img[:h, :w, :3])

    def make_callback(self, env):
        def callback():
            if self.done:
                return
            try:
                import cv2
                if self._display is None:
                    self._init()
                frame = self._capture()
                if frame is None:
                    return
                if self._writer is None:
                    h, w = frame.shape[:2]
                    self._size = (w, h)
                    fps = round(1.0 / env.dt) if hasattr(env, "dt") else 60
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    self._writer = cv2.VideoWriter(
                        str(self.path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        self._size,
                    )
                    print(f"Recording video → {self.path}  ({fps} fps, max {self.max_frames} frames)")
                if (frame.shape[1], frame.shape[0]) != self._size:
                    frame = cv2.resize(frame, self._size)
                self._writer.write(frame)
                self._frames += 1
                if self._frames >= self.max_frames:
                    self._writer.release()
                    self.done = True
                    print(f"Video saved: {self.path}")
                    if self.stop_env:
                        env.render_finished = True
            except Exception as e:
                print(f"[video] capture error: {e}")
        return callback


def _run_random(env_class, args, video_path=None, video_length=1) -> None:
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
    recorder = None
    if video_path is not None:
        max_ep = getattr(env, "max_episode_length", 1000)
        # stop_env=False: we own this loop, so break on recorder.done instead of
        # tripping env.render_finished (which would raise via raise_exception).
        recorder = _VideoRecorder(video_path, max_frames=video_length * max_ep, stop_env=False)
        env.render_callback = recorder.make_callback(env)
    while not env.render_finished:
        if recorder is not None and recorder.done:
            break
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
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-length", type=int, default=1, dest="video_length")
    if extra_args_fn is not None:
        extra_args_fn(parser)
    args = parser.parse_args()

    mode = args.mode
    checkpoint = args.checkpoint

    video_path = None
    if args.video:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = _VIDEOS_DIR / f"{timestamp}_{mode}.mp4"

    if mode == "random":
        _run_random(env_class, args, video_path=video_path, video_length=args.video_length)
        return

    # --- Config loading ---
    config_path = args.config if args.config is not None \
        else _PROJECT_ROOT / "configs" / default_config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "player" not in config["params"]["config"]:
        config["params"]["config"]["player"] = {}
    config["params"]["config"]["player"]["use_vecenv"] = True
    config["params"]["config"]["player"]["print_stats"] = False
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
    if args.video:
        args.headless = "False"
    elif args.headless is None:
        args.headless = "False" if mode == "play" else "True"
    rendering = not _str_to_bool(args.headless)

    env_kwargs = {
        "rendering": rendering,
        "raise_exception": rendering,
        "seed": args.seed,
        **config.get("env", {}),
    }

    # --- Video recorder setup ---
    recorder = None
    if args.video:
        max_ep = env_kwargs.get("max_episode_length", 1000)
        horizon = ppo_cfg.get("horizon_length", 16)
        max_frames = args.video_length * max_ep
        if mode == "train":
            # Limit training to exactly N episodes worth of steps
            cfg["max_epochs"] = math.ceil(max_frames / horizon)
            recorder = _VideoRecorder(video_path, max_frames=max_frames, stop_env=False)
        else:  # play
            recorder = _VideoRecorder(video_path, max_frames=max_frames, stop_env=True)

    # --- Env + vecenv registration ---
    def create_envs(n, **kw):
        assert torch.cuda.is_available()
        device = torch.device("cuda:0")
        envs = env_class(n, device, **env_kwargs)
        if mode == "play":
            envs.inference_mode_post_init_callback()
        if recorder is not None:
            envs.render_callback = recorder.make_callback(envs)
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

    # Mask-passthrough normalizer variant (used by configs via model.name; harmless otherwise).
    from .models import TransformerMaskedNorm
    mb_module.register_model('transformer_masked_a2c_logstd', TransformerMaskedNorm)

    # --- Run ---
    run_args = {"train": mode == "train", "play": mode == "play"}
    if checkpoint:
        run_args["checkpoint"] = checkpoint

    if mode == "play":
        if checkpoint:
            print(f"[play] Loading model from checkpoint: {checkpoint}")
        else:
            print("[play] No checkpoint provided; running with randomly initialized model")

    runner = Runner()
    runner.load(config)
    try:
        runner.run(run_args)
    except Exception:
        # vsim's render() raises a bare Exception to signal shutdown. When the
        # recorder finished on purpose, that's a clean stop; otherwise re-raise.
        if recorder is not None and recorder.done:
            print("Recording complete, exiting.")
        else:
            raise
