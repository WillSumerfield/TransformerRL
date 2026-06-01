"""Shared boilerplate for rl_games-based ant training scripts."""
from __future__ import annotations

import os
import random
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VIDEOS_DIR = _PROJECT_ROOT / "videos"

# Camera position relative to the followed ant (world axes, Y up). The look
# direction is auto-derived to point back at the ant.
CAMERA_OFFSET = (2.5, 2.5, 2.5)

# Give the SDL render window a stable, unique WM_CLASS so _find_xwindow can
# target it. Without this SDL defaults WM_CLASS to the interpreter name
# ("python3.11"), which matches nothing and makes the recorder grab the wrong
# window (e.g. the editor). Must be set before vlearn creates the window.
os.environ.setdefault("SDL_VIDEO_X11_WMCLASS", "vsim_render")

# Title/class fragments used to identify the vsim render window.
_VSIM_WINDOW_TITLES = ["vsim_render", "vsim", "vlearn"]

_LEG_CODE = {1: "F", 2: "FR", 3: "R", 4: "BR", 5: "B", 6: "BL", 7: "L", 8: "FL"}


def _morph_label(legs) -> str:
    return "·".join(_LEG_CODE[n] for n in sorted(legs))


class _TestDone(Exception):
    pass


class _TestTracker:
    """Wraps AntMultiMorphEnv; tracks per-env episode rewards, raises _TestDone when done."""

    def __init__(self, env, num_episodes: int):
        from tqdm import tqdm
        self._env = env
        self._num_ep = num_episodes
        n = env.total_num_envs
        self._epm = env.envs_per_morph
        self._n_morphs = len(env.groups)
        self._ep_counts = [0] * n
        self._cur_rew = [0.0] * n
        self._scores: list[list[float]] = [[] for _ in range(self._n_morphs)]
        self.done = False
        self._bar = tqdm(total=n * num_episodes, unit="ep", desc="testing")

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, actions):
        result = self._env.step(actions)
        rew = self._env._rew_buf.cpu()
        done = (self._env._term_buf | self._env._trunc_buf).cpu()
        for i in range(len(done)):
            self._cur_rew[i] += float(rew[i])
            if done[i]:
                gi = i // self._epm
                if self._ep_counts[i] < self._num_ep:
                    self._scores[gi].append(self._cur_rew[i])
                    self._bar.update(1)
                self._ep_counts[i] += 1
                self._cur_rew[i] = 0.0
        if all(c >= self._num_ep for c in self._ep_counts):
            self._bar.close()
            self.done = True
            raise _TestDone
        return result

    def get_scores(self) -> list[list[float]]:
        return self._scores


class _PlayLimiter:
    """Wraps env; stops play after max_steps total steps."""

    def __init__(self, env, max_steps: int):
        self._env = env
        self._max = max_steps
        self._count = 0

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, actions):
        result = self._env.step(actions)
        self._count += 1
        if self._count >= self._max:
            self._env.render_finished = True
        return result


def _print_and_save_test_results(
    scores: list[list[float]],
    groups: list[dict],
    checkpoint_path,
    split_labels: dict[int, str] | None = None,
) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg', force=False)
    import matplotlib.pyplot as plt
    import csv as _csv
    from matplotlib.patches import Patch

    morphs = [sorted(g['morph']) for g in groups]
    labels = [_morph_label(m) for m in morphs]
    leg_counts = [len(m) for m in morphs]
    has_split = split_labels is not None

    rows = []
    for gi, ep_scores in enumerate(scores):
        if not ep_scores:
            continue
        arr = np.array(ep_scores, dtype=np.float32)
        row = {
            'gi': gi, 'morph': labels[gi], 'legs': leg_counts[gi],
            'n': len(arr), 'mean': float(arr.mean()),
            'median': float(np.median(arr)), 'std': float(arr.std()),
            'min': float(arr.min()), 'max': float(arr.max()),
        }
        if has_split:
            row['split'] = split_labels.get(gi, 'train')
        rows.append(row)
    rows.sort(key=lambda r: r['mean'], reverse=True)

    all_scores = np.array([s for ep in scores for s in ep], dtype=np.float32)

    def _stats(arr):
        a = np.array(arr, dtype=np.float32)
        return {'n': len(a), 'mean': a.mean(), 'median': float(np.median(a)),
                'std': a.std(), 'min': a.min(), 'max': a.max()}

    # Per-leg group summaries, optionally broken down by split
    by_legs: dict[tuple, list[float]] = {}  # key = (split, legs) or (legs,)
    for r in rows:
        key = (r.get('split', ''), r['legs']) if has_split else (r['legs'],)
        by_legs.setdefault(key, []).extend(scores[r['gi']])
    leg_rows = {k: _stats(v) for k, v in sorted(by_legs.items())}

    # Split-level summaries
    split_rows: dict[str, dict] = {}
    if has_split:
        by_split: dict[str, list[float]] = {}
        for r in rows:
            by_split.setdefault(r['split'], []).extend(scores[r['gi']])
        split_rows = {sp: _stats(v) for sp, v in sorted(by_split.items())}

    # Console
    W = 20
    hdr = f"{'morph':<{W}} {'legs':>4} {'n':>5} {'mean':>8} {'med':>8} {'std':>8} {'min':>8} {'max':>8}"
    sep = '-' * len(hdr)
    print(f"\n{'='*len(hdr)}\n{hdr}\n{sep}")
    for r in rows:
        print(f"{r['morph']:<{W}} {r['legs']:>4} {r['n']:>5} {r['mean']:>8.2f} "
              f"{r['median']:>8.2f} {r['std']:>8.2f} {r['min']:>8.2f} {r['max']:>8.2f}")
    print(sep)
    for key, s in leg_rows.items():
        lbl = f"{key[0]} {key[1]}L" if has_split else f"{key[0]}L group"
        print(f"{lbl:<{W}} {key[-1]:>4} {s['n']:>5} {s['mean']:>8.2f} "
              f"{s['median']:>8.2f} {s['std']:>8.2f} {s['min']:>8.2f} {s['max']:>8.2f}")
    print(sep)
    if has_split:
        for sp, s in split_rows.items():
            print(f"{sp.upper():<{W}} {'':>4} {s['n']:>5} {s['mean']:>8.2f} "
                  f"{s['median']:>8.2f} {s['std']:>8.2f} {s['min']:>8.2f} {s['max']:>8.2f}")
        print(sep)
    a = all_scores
    print(f"{'GLOBAL':<{W}} {'':>4} {len(a):>5} {a.mean():>8.2f} "
          f"{np.median(a):>8.2f} {a.std():>8.2f} {a.min():>8.2f} {a.max():>8.2f}")
    print('=' * len(hdr), '\n')

    if checkpoint_path is None:
        return

    out_dir = Path(checkpoint_path).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"test_{timestamp}"

    def _f(v): return f"{v:.2f}"

    # Markdown summary
    def _md_stats_row(prefix, s):
        return f"| {prefix} | {s['n']} | {_f(s['mean'])} | {_f(s['median'])} | {_f(s['std'])} | {_f(s['min'])} | {_f(s['max'])} |"

    md_lines = [
        f"# Test Results — {timestamp}",
        f"",
        f"**Checkpoint**: `{checkpoint_path}`",
        f"",
        f"## Summary",
        f"",
        f"| group | n | mean | median | std | min | max |",
        f"|:------|--:|-----:|-------:|----:|----:|----:|",
    ]
    if has_split:
        for sp, s in split_rows.items():
            md_lines.append(_md_stats_row(sp, s))
    md_lines.append(_md_stats_row("**global**", _stats(all_scores)))
    leg_hdr = "| split | legs |" if has_split else "| legs |"
    leg_sep = "|:------|-----:|" if has_split else "|-----:|"
    md_lines += [
        f"",
        f"## By Leg Count",
        f"",
        f"{leg_hdr} n | mean | median | std | min | max |",
        f"{leg_sep}--:|-----:|-------:|----:|----:|----:|",
    ]
    for key, s in leg_rows.items():
        prefix = f"{key[0]} | {key[1]}" if has_split else str(key[0])
        md_lines.append(_md_stats_row(prefix, s))
    split_col = " split |" if has_split else ""
    split_sep = ":------:|" if has_split else ""
    md_lines += [
        f"",
        f"## Per Morphology",
        f"",
        f"| morph | legs |{split_col} n | mean | median | std | min | max |",
        f"|:------|-----:|{split_sep}--:|-----:|-------:|----:|----:|----:|",
    ]
    for r in rows:
        sc = f" {r['split']} |" if has_split else ""
        md_lines.append(
            f"| {r['morph']} | {r['legs']} |{sc} {r['n']} | {_f(r['mean'])} |"
            f" {_f(r['median'])} | {_f(r['std'])} | {_f(r['min'])} | {_f(r['max'])} |"
        )
    md_path = out_dir / f"{stem}.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"[test] MD   → {md_path}")

    # CSV
    csv_path = out_dir / f"{stem}.csv"
    fields = ['morph', 'legs', 'split', 'n', 'mean', 'median', 'std', 'min', 'max'] if has_split \
        else ['morph', 'legs', 'n', 'mean', 'median', 'std', 'min', 'max']
    with open(csv_path, 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"[test] CSV  → {csv_path}")

    # Bar chart
    png_path = out_dir / f"{stem}.png"
    n = len(rows)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.45), 5))

    if not has_split:
        unique_legs = sorted(set(r['legs'] for r in rows))
        cmap = plt.cm.get_cmap('tab10', len(unique_legs))
        leg_color = {l: cmap(i) for i, l in enumerate(unique_legs)}
        colors = [leg_color[r['legs']] for r in rows]
        legend_handles = [Patch(color=leg_color[l], label=f'{l} legs') for l in unique_legs]
    else:
        def _shade(i, n): return 0.45 + 0.4 * (i / max(1, n - 1))
        train_legs = sorted(set(r['legs'] for r in rows if r.get('split') == 'train'))
        test_legs  = sorted(set(r['legs'] for r in rows if r.get('split') == 'test'))
        train_color = {l: plt.cm.Blues(_shade(i, len(train_legs)))   for i, l in enumerate(train_legs)}
        test_color  = {l: plt.cm.Oranges(_shade(i, len(test_legs)))  for i, l in enumerate(test_legs)}
        colors = [
            train_color.get(r['legs'], plt.cm.Blues(0.6)) if r.get('split') == 'train'
            else test_color.get(r['legs'], plt.cm.Oranges(0.6))
            for r in rows
        ]
        legend_handles = (
            [Patch(color=train_color[l], label=f'train {l}L') for l in train_legs] +
            [Patch(color=test_color[l],  label=f'test {l}L')  for l in test_legs]
        )

    means = [r['mean'] for r in rows]
    stds  = [r['std']  for r in rows]
    ax.bar(range(n), means, yerr=stds, color=colors, capsize=3, alpha=0.85)
    ax.set_xticks(range(n))
    ax.set_xticklabels([r['morph'] for r in rows], rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('mean episode reward ± std')
    suffix = ' (train vs test)' if has_split else ''
    ax.set_title(f'test results{suffix}  ({n} morphs, {len(a)} total episodes)')
    ax.legend(handles=legend_handles)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[test] Chart → {png_path}")


def _str_to_bool(s: str) -> bool:
    return s.lower() == "true"


def _adjust_minibatch(cfg: dict, n_envs: int, h_len: int) -> None:
    requested = cfg["minibatch_size"]
    batch = h_len * n_envs
    n_batches = (batch + requested - 1) // requested
    mb = batch // n_batches if n_batches > 1 else batch
    if batch % mb != 0:
        print(f"Error: batch ({batch}) not divisible by minibatch ({mb})")
        sys.exit(1)
    if mb != requested:
        print(f"Warning: minibatch_size {requested} does not divide batch {batch} "
              f"(num_actors={n_envs} * horizon={h_len}); snapped to {mb} ({n_batches} minibatches).")
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

    # No match. Deliberately no "largest window" fallback: it grabs whatever
    # else is on screen (e.g. the editor). A missing match means the window is
    # on a non-visible workspace (unviewable) or SDL_VIDEO_X11_WMCLASS didn't
    # apply.
    return None


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
        self._warned = False

    def _warn_once(self, msg):
        if not self._warned:
            print(f"[video] {msg}")
            self._warned = True

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
            self._warn_once(
                "render window not found — is it on a visible workspace? "
                "(matching WM_CLASS 'vsim_render')"
            )
            return None
        try:
            # Unviewable windows (on a non-visible workspace) yield black frames;
            # skip them so the video only contains real, rendered frames.
            if win.get_attributes().map_state != X.IsViewable:
                self._warn_once(
                    "render window is hidden (non-visible workspace) — frames "
                    "skipped; keep it on a visible workspace to record."
                )
                return None
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


class FollowCamera:
    """Drives the viewer camera to follow one ant, switching each episode.

    Picks a new environment-set in a no-repeat random order (reshuffles once all
    sets are exhausted), then a random ant within it. Switches when the followed
    ant's episode ends (its progress counter resets).
    """

    MAX_FRAMES_PER_MORPH = 250

    def __init__(self, env, offset_xyz=CAMERA_OFFSET):
        import vlearn as v
        self._v = v
        self.env = env
        self.offset = v.Vec3(*offset_xyz)
        n = (offset_xyz[0] ** 2 + offset_xyz[1] ** 2 + offset_xyz[2] ** 2) ** 0.5
        self.look = v.Vec3(-offset_xyz[0] / n, -offset_xyz[1] / n, -offset_xyz[2] / n)
        self.sets = env.follow_sets()
        self._order = []
        self._last_set = None
        self._cur_idx = None
        self._last_progress = -1
        self._start_time = 0
        self._pick_new()

    def _next_set(self) -> int:
        if not self._order:
            self._order = list(range(len(self.sets)))
            random.shuffle(self._order)
            if len(self._order) > 1 and self._order[0] == self._last_set:
                self._order.append(self._order.pop(0))
        return self._order.pop(0)

    def _pick_new(self) -> None:
        si = self._next_set()
        self._last_set = si
        self._cur_idx = random.choice(self.sets[si])
        self._last_progress = -1

    def update(self) -> None:
        prog = int(self.env.progress_buf[self._cur_idx].item())
        time_w_current_morph = prog - self._start_time
        env_reset = prog < self._last_progress
        if time_w_current_morph > self.MAX_FRAMES_PER_MORPH or env_reset:
            self._pick_new()
            prog = int(self.env.progress_buf[self._cur_idx].item())
            self._start_time = prog
        self._last_progress = prog
        eye = self.env.follow_world_pos(self._cur_idx) + self.offset
        self.env.gym_render.reset_camera(eye, self.look)


def _attach_render_callback(env, recorder=None) -> None:
    """Compose follow-camera (if env supports it and is rendering) + video capture."""
    hooks = []
    if getattr(env, "rendering", False) and hasattr(env, "follow_sets"):
        hooks.append(FollowCamera(env).update)
    if recorder is not None:
        hooks.append(recorder.make_callback(env))
    if not hooks:
        return

    def composite():
        for h in hooks:
            h()

    env.render_callback = composite


def _run_random(env_class, args, video_path=None, num_episodes=1) -> None:
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
        recorder = _VideoRecorder(video_path, max_frames=num_episodes * max_ep, stop_env=False)
    _attach_render_callback(env, recorder)
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
    morphology_set: list | None = None,
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
    parser.add_argument("mode", nargs="?", choices=["train", "play", "random", "test"], default="train")
    parser.add_argument("checkpoint", nargs="?", default=None)
    parser.add_argument("--num-episodes", type=int, default=None, dest="num_episodes")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--headless", choices=["True", "False"], default=None)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--horizon_length", type=int)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--train-pct", type=float, default=None, dest="train_pct",
                        help="Fraction of morphologies for training (rest = test set).")
    parser.add_argument("--test-set", action="store_true", default=None, dest="test_set",
                        help="Use held-out test morphologies (play/test only).")
    parser.add_argument("--compare", action="store_true", default=False,
                        help="Test mode: run on full morph set, chart train vs test morphs.")
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
        _run_random(env_class, args, video_path=video_path, num_episodes=args.num_episodes or 1)
        return

    if mode == "test" and not checkpoint:
        print("Error: checkpoint required for test mode")
        sys.exit(1)

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
    cfg.setdefault("use_diagnostics", True)  # enables diagnostics/exp_var, clip_frac, rms_value
    exp_name = cfg.get("name", "run").removeprefix("ant_")
    cfg.setdefault("train_dir", f"{train_dir}/{exp_name}")
    cfg.setdefault("full_experiment_name", datetime.now().strftime("%d-%H-%M-%S"))

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

    # --- Morph split (multi-morph envs only) ---
    compare_labels_ref: list[dict | None] = [None]
    if morphology_set is not None:
        from envs.ant_envs.ant_multimorph import morph_split
        env_cfg = config.setdefault("env", {})
        train_pct = args.train_pct if args.train_pct is not None else env_cfg.get("train_pct", 1.0)
        test_set = args.test_set if args.test_set is not None else env_cfg.get("test_set", False)

        if args.compare and mode != "test":
            raise SystemExit("--compare is only valid in test mode")
        if args.compare and test_set:
            raise SystemExit("--compare is incompatible with --test-set")
        if args.compare and train_pct >= 1.0:
            raise SystemExit("--compare requires --train-pct < 1.0")

        if train_pct < 1.0:
            seed = config["params"].get("seed") or args.seed
            if seed is None:
                raise SystemExit("--seed is required when --train-pct < 1.0")
            if args.compare:
                train_morphs = set(morph_split(morphology_set, train_pct, seed, False))
                compare_labels_ref[0] = {
                    gi: ('train' if morphology_set[gi] in train_morphs else 'test')
                    for gi in range(len(morphology_set))
                }
                effective = morphology_set
                env_cfg["train_pct"] = 1.0
                env_cfg["test_set"] = False
            else:
                effective = morph_split(morphology_set, train_pct, seed, test_set)
        else:
            effective = morphology_set

        if not args.compare:
            env_cfg["train_pct"] = train_pct
            env_cfg["test_set"] = test_set

        n_total = len(morphology_set)
        n_morphs = len(effective)
        n_train = n_total - n_morphs if test_set else n_morphs
        n_test = n_total - n_train
        label = "compare (train+test)" if args.compare else ("test" if test_set else "train")
        print(f"[morphs] {label} set: {n_morphs}/{n_total}  "
              f"(train={n_train}, test={n_test}, train_pct={train_pct:.2f})")
        if test_set and mode == "train":
            print("[WARNING] Training on test morphologies — omit --test-set when evaluating to measure generalization.")

        n_envs = config["params"]["config"]["num_actors"]
        epm = max(1, n_envs // n_morphs)
        config["params"]["config"]["num_actors"] = n_morphs * epm

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
        # test is headless by default (already covered by "True" above)
    rendering = not _str_to_bool(args.headless)

    resolved_seed = args.seed if args.seed is not None else config["params"].get("seed")
    env_kwargs = {
        "rendering": rendering,
        "raise_exception": rendering,
        "seed": resolved_seed,
        **config.get("env", {}),
    }

    # --- Video recorder setup ---
    recorder = None
    if args.video:
        if mode == "train":
            print("Error: --video is not supported in train mode")
            sys.exit(1)
        max_ep = env_kwargs.get("max_episode_length", 1000)
        max_frames = (args.num_episodes or 1) * max_ep
        recorder = _VideoRecorder(video_path, max_frames=max_frames, stop_env=(mode == "play"))

    # --- Env + vecenv registration ---
    num_episodes = args.num_episodes
    tracker_ref: list[_TestTracker | None] = [None]

    def create_envs(n, **kw):
        assert torch.cuda.is_available()
        device = torch.device("cuda:0")
        envs = env_class(n, device, **env_kwargs)
        if mode in ("play", "test"):
            envs.inference_mode_post_init_callback()
        if mode == "test":
            ep = num_episodes if num_episodes is not None else 10
            tracker = _TestTracker(envs, ep)
            tracker_ref[0] = tracker
            return NewToOldAPICompatilibity(tracker)
        _attach_render_callback(envs, recorder)
        if mode == "play" and num_episodes is not None:
            max_ep = env_kwargs.get("max_episode_length", 1000)
            limiter = _PlayLimiter(envs, num_episodes * max_ep)
            return NewToOldAPICompatilibity(limiter)
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
    run_args = {"train": mode == "train", "play": mode in ("play", "test")}
    if checkpoint:
        run_args["checkpoint"] = checkpoint

    if mode == "play":
        if checkpoint:
            print(f"[play] Loading model from checkpoint: {checkpoint}")
        else:
            print("[play] No checkpoint provided; running with randomly initialized model")
    elif mode == "test":
        ep = num_episodes if num_episodes is not None else 10
        print(f"[test] {ep} episodes per env slot  |  checkpoint: {checkpoint}")

    runner = Runner()
    # Swap in the metrics-logging agent for all continuous PPO runs (see logging_agent.py).
    from .logging_agent import LoggingA2CAgent
    runner.algo_factory.register_builder(
        'a2c_continuous', lambda **kwargs: LoggingA2CAgent(**kwargs)
    )
    runner.load(config)
    try:
        runner.run(run_args)
    except _TestDone:
        tracker = tracker_ref[0]
        if tracker is not None:
            _print_and_save_test_results(
                tracker.get_scores(), tracker._env.groups, checkpoint,
                split_labels=compare_labels_ref[0],
            )
    except Exception:
        # vsim's render() raises a bare Exception to signal shutdown. When the
        # recorder finished on purpose, that's a clean stop; otherwise re-raise.
        if recorder is not None and recorder.done:
            print("Recording complete, exiting.")
        else:
            raise
