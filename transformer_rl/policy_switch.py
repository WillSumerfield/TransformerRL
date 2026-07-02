"""Live policy switching for `play` (see scripts/CONTEXT.md "Policy switching").

`play` takes a directory instead of a single .pth:
  - run dir   (has nn/)           -> switch epoch within one run
  - model dir (dir of run folders) -> switch run + epoch

`resolve_source` enumerates the checkpoints (best first, then ep_N ascending,
best = the bare <name>.pth rl_games writes on a new best mean reward, identified
via config.name). `PolicySwitch` is the shared state the FollowCamera panel drives
and the player reads; `SwitchMixin` applies a pending switch inside the player loop
(restore new weights + full env reset -> codesign resamples bodies with the new
generator)."""
import re
from pathlib import Path

import torch


def _enumerate_run(run_dir: Path, base_name: str):
    """[(label, path)] for one run's nn/: best (<name>.pth) first, then ep_N ascending."""
    nn = run_dir / "nn"
    snaps: dict[int, Path] = {}
    pat = re.compile(rf"^last_{re.escape(base_name)}_ep_(\d+)_rew_.*\.pth$")
    for p in nn.glob("*.pth"):
        m = pat.match(p.name)
        if m:
            snaps[int(m.group(1))] = p          # dedupe rare double-underscore twins by epoch
    out = []
    best = nn / f"{base_name}.pth"
    if best.exists():
        out.append(("best", best))
    out += [(f"ep {ep}", snaps[ep]) for ep in sorted(snaps)]
    return out


def resolve_source(path: Path, base_name: str):
    """Classify the play arg. Returns a dict:
      {"mode": "file", "checkpoint": str}                         # single .pth (legacy)
      {"mode": "run",   "runs": [(run_name, [(label, path)])]}    # one run
      {"mode": "model", "runs": [(run_name, [(label, path)]), ...]}  # runs under a <model> dir
    """
    path = Path(path)
    if path.is_file():
        return {"mode": "file", "checkpoint": str(path)}
    if (path / "nn").is_dir():
        eps = _enumerate_run(path, base_name)
        if not eps:
            raise SystemExit(f"No checkpoints matching '{base_name}' in {path}/nn")
        return {"mode": "run", "runs": [(path.name, eps)]}
    runs = []
    for d in sorted(p for p in path.iterdir() if p.is_dir() and (p / "nn").is_dir()):
        eps = _enumerate_run(d, base_name)
        if eps:
            runs.append((d.name, eps))
    if not runs:
        raise SystemExit(f"No runs with '{base_name}' checkpoints under {path}")
    return {"mode": "model", "runs": runs}


class PolicySwitch:
    """Shared state between the FollowCamera panel (UI) and the player (apply).

    `runs` is always a list of (run_name, [(label, path)]) -- run mode is just a
    single-run list. The UI sets `pending`; the player applies it, commits
    run_idx/epoch_idx, then clears it."""

    def __init__(self, source: dict):
        self.mode = source["mode"]              # "run" | "model"
        self.runs = source["runs"]
        self.run_idx = 0
        self.epoch_idx = 0                       # 0 = best
        self.pending: tuple[int, int] | None = None

    @property
    def has_runs(self) -> bool:
        return self.mode == "model"

    def run_labels(self):
        return [name for name, _ in self.runs]

    def epoch_labels(self, run_idx: int | None = None):
        return [lbl for lbl, _ in self.runs[self.run_idx if run_idx is None else run_idx][1]]

    def n_epochs(self, run_idx: int | None = None) -> int:
        return len(self.runs[self.run_idx if run_idx is None else run_idx][1])

    def current_path(self) -> str:
        return str(self.runs[self.run_idx][1][self.epoch_idx][1])

    def current_label(self) -> str:
        name, eps = self.runs[self.run_idx]
        return f"{name} / {eps[self.epoch_idx][0]}"

    def request(self, run_idx: int, epoch_idx: int) -> None:
        self.pending = (run_idx, epoch_idx)

    def filter_compatible(self, obs_dim: int, action_dim: int):
        """Drop runs whose checkpoints don't fit the current architecture (a <model>
        dir can span arch generations). Compat = the checkpoint's obs normalizer and
        action log_std match the live env's dims. Resets selection to the first
        compatible run's best; raises if none. Returns [(run_name, reason)] skipped."""
        def _find(sd, suffix):
            # checkpoints from compiled training carry a "_orig_mod." key prefix
            # that rl_games strips on load -- match by suffix to be prefix-agnostic.
            return next((v for k, v in sd.items() if k.endswith(suffix)), None)

        kept, skipped = [], []
        for name, eps in self.runs:
            sd = torch.load(eps[0][1], map_location="cpu")["model"]   # best checkpoint
            o = _find(sd, "running_mean_std.running_mean")
            a = _find(sd, "a2c_network.log_std_param")
            if o is not None and o.shape[0] != obs_dim:
                skipped.append((name, f"obs {o.shape[0]}!={obs_dim}"))
            elif a is not None and a.shape[0] != action_dim:
                skipped.append((name, f"act {a.shape[0]}!={action_dim}"))
            else:
                kept.append((name, eps))
        if not kept:
            raise SystemExit(
                f"[play] no runs compatible with current arch (obs={obs_dim}, act={action_dim})")
        self.runs, self.run_idx, self.epoch_idx = kept, 0, 0
        return skipped


class SwitchMixin:
    """Player mix-in: applies a pending PolicySwitch inside the rollout loop.

    On a pending switch it restores the target checkpoint and forces a full
    env_reset (CodesignPlayer's env_reset resamples bodies from the freshly
    loaded generator), marking every env done so the loop's reward/step
    accumulators reset cleanly. `attach_switch(None)` (or never attaching) is an
    inert pass-through -- identical to the base player."""

    _switch: "PolicySwitch | None" = None
    _initial_restored = False

    def attach_switch(self, switch) -> None:
        self._switch = switch

    def restore(self, fn):
        # rl_games' initial restore uses the checkpoint arg, which may name a run
        # filtered out as incompatible (filtering runs at env-build, before this).
        # Redirect the first restore to the committed (first compatible) path.
        if self._switch is not None and not self._initial_restored:
            self._initial_restored = True
            fn = self._switch.current_path()
        super().restore(fn)

    def env_step(self, env, actions):
        obs, r, done, info = super().env_step(env, actions)
        if self._switch is not None and self._switch.pending is not None:
            obs = self._apply_switch(env)
            done = torch.ones_like(done)
        return obs, r, done, info

    @torch.no_grad()
    def _apply_switch(self, env):
        s = self._switch
        s.run_idx, s.epoch_idx = s.pending
        s.pending = None
        path = s.current_path()
        print(f"[policy-switch] -> {s.current_label()}  ({path})", flush=True)
        self.restore(path)                       # new control (+ generator) weights
        return self.env_reset(env)               # full reset; codesign rebuilds bodies
