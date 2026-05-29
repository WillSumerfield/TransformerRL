#!/usr/bin/env python3
"""Bayesian hyperparameter tuner — Optuna + Rich TUI.

Usage: python tune.py [tune_config.yaml]
"""

import json
import subprocess
import sys
import tempfile
import threading
import time
from copy import deepcopy
from pathlib import Path

import optuna
import yaml
from rich import box
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

optuna.logging.set_verbosity(optuna.logging.WARNING)

_SPARK  = "▁▂▃▄▅▆▇█"
_SPIN   = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_NEGINF = -float("inf")


# ── helpers ───────────────────────────────────────────────────────────

def _sparkline(scores: list) -> str:
    valid = [s for s in scores if s > _NEGINF]
    if not valid:
        return "—" * len(scores)
    lo, hi = min(valid), max(valid)
    out = []
    for s in scores:
        if s <= _NEGINF:
            out.append("—")
        elif lo == hi:
            out.append(_SPARK[7])
        else:
            out.append(_SPARK[min(7, int((s - lo) / (hi - lo) * 8))])
    return "".join(out)


def _set(d: dict, keys: list, value):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _fmt(v) -> str:
    return f"{v:.3g}" if isinstance(v, float) else str(v)


def _fmt_t(s: float) -> str:
    m, sec = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _read_tb(log_dir: Path, metric: str) -> float:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    if metric not in ea.Tags().get("scalars", []):
        return _NEGINF
    scalars = ea.Scalars(metric)
    return max(s.value for s in scalars) if scalars else _NEGINF


# ── state ─────────────────────────────────────────────────────────────

class _Trial:
    __slots__ = ("num", "params", "score", "status", "elapsed")

    def __init__(self, num: int, params: dict):
        self.num     = num
        self.params  = params
        self.score: float | None = None
        self.status  = "running"
        self.elapsed = 0.0


class _State:
    def __init__(self, n_trials: int, param_names: list[str]):
        self.n_trials   = n_trials
        self.param_names = param_names
        self.trials: list[_Trial] = []
        self.current: _Trial | None = None
        self._t0: float | None = None
        self._done_times: list[float] = []

    def begin(self, num: int, params: dict):
        t = _Trial(num, params)
        self.trials.append(t)
        self.current = t
        self._t0 = time.time()

    def retry(self, num: int):
        for t in self.trials:
            if t.num == num:
                t.status = "retrying"
        self._t0 = time.time()

    def end(self, num: int, score: float):
        elapsed = time.time() - self._t0 if self._t0 else 0.0
        for t in self.trials:
            if t.num == num:
                t.score   = score
                t.status  = "done" if score > _NEGINF else "failed"
                t.elapsed = elapsed
        self._done_times.append(elapsed)
        self.current = None
        self._t0 = None

    def elapsed_now(self) -> float:
        return time.time() - self._t0 if self._t0 else 0.0

    def mean_time(self) -> float:
        return sum(self._done_times) / len(self._done_times) if self._done_times else 0.0

    def best(self) -> _Trial | None:
        done = [t for t in self.trials if t.score is not None and t.score > _NEGINF]
        return max(done, key=lambda t: t.score) if done else None

    def worst(self) -> _Trial | None:
        done = [t for t in self.trials if t.score is not None and t.score > _NEGINF]
        return min(done, key=lambda t: t.score) if done else None

    def all_scores(self) -> list[float]:
        return [t.score for t in self.trials if t.score is not None]

    def n_done(self) -> int:
        return sum(1 for t in self.trials if t.status in ("done", "failed"))

    def seed(self, study) -> None:
        """Pre-populate from an existing study so the TUI shows prior trials on resume."""
        TS = optuna.trial.TrialState
        for t in sorted(study.trials, key=lambda x: x.number):
            if t.state in (TS.RUNNING, TS.WAITING):
                continue  # interrupted/stale; not a finished result
            tr = _Trial(t.number, dict(t.params))
            if t.state == TS.COMPLETE and t.value is not None and t.value > _NEGINF:
                tr.score, tr.status = t.value, "done"
            else:
                tr.score, tr.status = _NEGINF, "failed"
            if t.datetime_complete and t.datetime_start:
                tr.elapsed = (t.datetime_complete - t.datetime_start).total_seconds()
                self._done_times.append(tr.elapsed)
            self.trials.append(tr)


# ── TUI ───────────────────────────────────────────────────────────────

def _build(state: _State, tick: int) -> Layout:
    sp    = _SPIN[tick % len(_SPIN)]
    n_done = state.n_done()
    pct   = n_done / state.n_trials if state.n_trials else 0.0
    bw    = 26
    filled = int(pct * bw)
    bar   = f"[cyan]{'█' * filled}[/cyan][dim]{'░' * (bw - filled)}[/dim]"

    # ── progress ──
    prog = Panel(
        Text.from_markup(
            f"Trial [bold white]{n_done}[/bold white] / [bold white]{state.n_trials}[/bold white]"
            f"   {bar}   [bold white]{pct * 100:.0f}%[/bold white]"
        ),
        title="[bold blue]Progress[/bold blue]",
        border_style="blue",
    )

    # ── current trial ──
    cur = state.current
    if cur:
        pstr = "   ".join(
            f"[cyan]{k.split('.')[-1]}[/cyan]=[yellow]{_fmt(v)}[/yellow]"
            for k, v in cur.params.items()
        )
        ic  = "↺" if cur.status == "retrying" else sp
        lbl = "Retrying" if cur.status == "retrying" else "Running"
        cur_body = Text.from_markup(
            f"[bold yellow]{ic} {lbl}[/bold yellow]   {pstr}\n\n"
            f"  [dim]Elapsed[/dim] [white]{_fmt_t(state.elapsed_now())}[/white]"
            f"    [dim]Mean[/dim] [white]"
            + (_fmt_t(state.mean_time()) if state._done_times else "——")
            + "[/white]"
        )
    else:
        cur_body = Text("[dim]——[/dim]")
    cur_panel = Panel(cur_body, title="[bold yellow]Current Trial[/bold yellow]", border_style="yellow")

    # ── all trials table ──
    tbl = Table(box=box.SIMPLE_HEAD, expand=True, header_style="bold dim", show_edge=False)
    tbl.add_column("#",     justify="right", width=4)
    tbl.add_column("Score", justify="right", width=9)
    for name in state.param_names:
        tbl.add_column(name.split(".")[-1], justify="right")
    tbl.add_column("", width=2)

    best = state.best()
    for t in reversed(state.trials):
        if t.score is not None and t.score > _NEGINF:
            sc_str = f"{t.score:.1f}"
        elif t.status in ("running", "retrying"):
            sc_str = sp if t.status == "running" else "↺"
        else:
            sc_str = "—"

        if   t.status in ("running", "retrying"): row_style, icon = "yellow",    ""
        elif t.status == "failed":                row_style, icon = "dim red",   "✗"
        elif best and t.num == best.num:          row_style, icon = "bold green", "★"
        else:                                     row_style, icon = "",           "✓"

        tbl.add_row(
            str(t.num), sc_str,
            *[_fmt(t.params.get(n, "—")) for n in state.param_names],
            icon, style=row_style,
        )

    trials_panel = Panel(tbl, title="[bold]All Trials[/bold]", border_style="blue")

    # ── best ──
    best = state.best()
    if best:
        p = "\n".join(
            f"  [dim]{k.split('.')[-1]}[/dim] = [bold cyan]{_fmt(v)}[/bold cyan]"
            for k, v in best.params.items()
        )
        best_body = Text.from_markup(
            f"[bold green]Trial {best.num}[/bold green]   [bold white]{best.score:.2f}[/bold white]\n\n{p}"
        )
    else:
        best_body = Text("[dim]——[/dim]")
    best_panel = Panel(best_body, title="[bold green]Best[/bold green]", border_style="green")

    # ── worst ──
    worst = state.worst()
    if worst:
        p = "\n".join(
            f"  [dim]{k.split('.')[-1]}[/dim] = [bold cyan]{_fmt(v)}[/bold cyan]"
            for k, v in worst.params.items()
        )
        worst_body = Text.from_markup(
            f"[bold red]Trial {worst.num}[/bold red]   [bold white]{worst.score:.2f}[/bold white]\n\n{p}"
        )
    else:
        worst_body = Text("[dim]——[/dim]")
    worst_panel = Panel(worst_body, title="[bold red]Worst[/bold red]", border_style="red")

    # ── score graph ──
    scores = [s for s in state.all_scores() if s > _NEGINF]
    if scores:
        lo, hi = min(scores), max(scores)
        graph_body = Text.from_markup(
            f"[dim]{hi:.2f}[/dim]\n\n"
            f"[bold magenta]{_sparkline(scores)}[/bold magenta]\n\n"
            f"[dim]{lo:.2f}[/dim]"
        )
    else:
        graph_body = Text("[dim]——[/dim]")
    graph_panel = Panel(graph_body, title="[bold magenta]Score over Trials[/bold magenta]", border_style="magenta")

    # ── assemble ──
    layout = Layout()
    layout.split_row(
        Layout(name="left",  ratio=3),
        Layout(name="right", ratio=2),
    )

    left = Layout()
    left.split_column(
        Layout(prog,         name="progress", size=3),
        Layout(cur_panel,    name="current",  size=6),
        Layout(trials_panel, name="trials"),
    )

    right = Layout()
    right.split_column(
        Layout(best_panel,  name="best"),
        Layout(worst_panel, name="worst"),
        Layout(graph_panel, name="graph"),
    )

    layout["left"].update(left)
    layout["right"].update(right)
    return layout


# ── results ───────────────────────────────────────────────────────────

def _show_results(study: optuna.Study, tune_cfg: dict, base_cfg: dict, output_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from rich.console import Console
    from rich.table import Table
    from rich import box as rbox

    c = Console()
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None and t.value > _NEGINF
    ]

    if not completed:
        c.print("[dim]No successful trials.[/dim]")
        return

    # save best_params.yaml
    bt = study.best_trial
    best_cfg = deepcopy(base_cfg)
    for path, value in bt.params.items():
        _set(best_cfg, path.split("."), value)
    with open(output_dir / "best_params.yaml", "w") as f:
        yaml.dump(best_cfg, f)

    # print top-10 table
    top = sorted(completed, key=lambda t: t.value, reverse=True)
    tbl = Table(title="Top Trials", box=rbox.SIMPLE_HEAD, header_style="bold dim")
    tbl.add_column("Rank", justify="right", width=5)
    tbl.add_column("#",    justify="right", width=4)
    tbl.add_column("Score", justify="right", width=10)
    for p in tune_cfg["params"]:
        tbl.add_column(p["path"].split(".")[-1], justify="right")
    for rank, t in enumerate(top[:10], 1):
        tbl.add_row(
            str(rank), str(t.number), f"{t.value:.4f}",
            *[_fmt(t.params.get(p["path"], "—")) for p in tune_cfg["params"]],
            style="bold green" if rank == 1 else "",
        )
    c.print(tbl)
    c.print(f"[dim]Output: {output_dir}/[/dim]")

    # plots: 2 rows (reward, time) × N params columns
    params_cfg = tune_cfg["params"]
    n          = len(params_cfg)

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7), squeeze=False)
    fig.suptitle(
        f"{tune_cfg['study']['name']}  ({len(completed)} successful / {len(study.trials)} total)",
        fontsize=12,
    )

    def _duration(t) -> float | None:
        if t.datetime_complete and t.datetime_start:
            return (t.datetime_complete - t.datetime_start).total_seconds()
        return None

    bp_kw = dict(patch_artist=True, medianprops=dict(color="white", linewidth=2))

    for idx, p in enumerate(params_cfg):
        ax_r = axes[0, idx]
        ax_t = axes[1, idx]

        path  = p["path"]
        label = path.split(".")[-1]
        pvals = [t.params[path] for t in completed]

        # discrete: categorical OR int/float with an explicit step > 1
        is_discrete = (
            p["type"] == "categorical"
            or (p["type"] == "int"   and p.get("step", 1) > 1)
            or (p["type"] == "float" and "step" in p)
        )

        if is_discrete:
            if p["type"] == "categorical":
                groups = p["choices"]
            else:
                groups = list(range(p["low"], p["high"] + 1, p.get("step", 1)))
            g_scores = {g: [t.value for t in completed if t.params[path] == g] for g in groups}
            g_times  = {g: [d for t in completed
                            if t.params[path] == g and (d := _duration(t)) is not None]
                        for g in groups}
            act_labels = [str(g) for g in groups]

            bps = ax_r.boxplot([g_scores[g] for g in groups], tick_labels=act_labels, **bp_kw)
            for patch in bps["boxes"]:
                patch.set(facecolor="#4c72b0", alpha=0.7)

            bps = ax_t.boxplot([g_times[g] for g in groups], tick_labels=act_labels, **bp_kw)
            for patch in bps["boxes"]:
                patch.set(facecolor="#dd8452", alpha=0.7)
        else:
            scores    = [t.value for t in completed]
            dur_pairs = [(t.params[path], d) for t in completed if (d := _duration(t)) is not None]

            if len(set(pvals)) > 1:
                sns.kdeplot(x=pvals, y=scores, ax=ax_r, fill=True,
                            color="#4c72b0", alpha=0.7, warn_singular=False)
            ax_r.scatter(pvals, scores, s=18, color="#4c72b0", alpha=0.6, zorder=3)

            if dur_pairs:
                tvs, tds = zip(*dur_pairs)
                if len(set(tvs)) > 1:
                    sns.kdeplot(x=list(tvs), y=list(tds), ax=ax_t, fill=True,
                                color="#dd8452", alpha=0.7, warn_singular=False)
                ax_t.scatter(list(tvs), list(tds), s=18, color="#dd8452", alpha=0.6, zorder=3)

            if p.get("log", False):
                ax_r.set_xscale("log")
                ax_t.set_xscale("log")

        ax_r.set_title(label, fontsize=10)
        ax_r.grid(True, alpha=0.3)
        ax_t.grid(True, alpha=0.3)
        if idx == 0:
            ax_r.set_ylabel("Reward")
            ax_t.set_ylabel("Time (s)")

    plt.tight_layout()
    plot_path = output_dir / "results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    c.print(f"[dim]Plot saved to {plot_path}[/dim]")


# ── trial runner ──────────────────────────────────────────────────────

def _run(num: int, params: dict, base_cfg: dict, tune_cfg: dict, output_dir: Path, attempt: int = 0) -> float:
    sc = tune_cfg["study"]
    trial_cfg = deepcopy(base_cfg)
    for path, value in params.items():
        _set(trial_cfg, path.split("."), value)
    ns = {path.split(".")[-1]: val for path, val in params.items()}
    for dp in tune_cfg.get("derived_params", []):
        val = eval(dp["expr"], {"__builtins__": {}}, ns)
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        _set(trial_cfg, dp["path"].split("."), val)
    exp_name_path = sc.get("experiment_name_path", "experiment.name").split(".")
    train_dir_path = sc.get("train_dir_path", "experiment.directory").split(".")
    _set(trial_cfg, exp_name_path, f"tune_trial_{num}")
    _set(trial_cfg, train_dir_path, str(output_dir / "runs"))

    log_path = output_dir / "logs" / f"trial_{num}_attempt{attempt}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mktemp(suffix=".yaml", prefix=f"tune_{num}_"))
    try:
        with open(tmp, "w") as f:
            yaml.dump(trial_cfg, f)
        timed_out = False
        try:
            r = subprocess.run(
                [sys.executable, sc["script"], "--config", str(tmp)],
                timeout=sc.get("trial_timeout_seconds", 300),
                capture_output=True, text=True,
            )
            ok = r.returncode == 0
        except subprocess.TimeoutExpired as e:
            timed_out = True
            ok = False
            r = e

        with open(log_path, "w") as f:
            f.write(f"# trial {num}  attempt {attempt}  params: {params}\n")
            f.write(f"# {'TIMED OUT' if timed_out else f'exit {r.returncode}'}\n\n")
            f.write("=== stdout ===\n")
            f.write((r.stdout or "") if not timed_out else "(timed out)\n")
            f.write("\n=== stderr ===\n")
            f.write((r.stderr or "") if not timed_out else "(timed out)\n")

        if timed_out:
            return None  # no retry on timeout
        if not ok:
            return _NEGINF
        summaries_subdir = sc.get("summaries_subdir", "")
        log_dir = output_dir / "runs" / f"tune_trial_{num}"
        if summaries_subdir:
            log_dir = log_dir / summaries_subdir
        return _read_tb(log_dir, sc.get("metric_key", "Reward / Total reward (mean)"))
    finally:
        tmp.unlink(missing_ok=True)


# ── entry point ───────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Bayesian hyperparameter tuner")
    ap.add_argument("config", nargs="?", default="configs/tune_config.yaml")
    ap.add_argument("--reset", action="store_true", help="delete study DB before starting")
    args = ap.parse_args()

    with open(args.config) as f:
        tune_cfg = yaml.safe_load(f)
    with open(tune_cfg["study"]["base_config"]) as f:
        base_cfg = yaml.safe_load(f)

    sc = tune_cfg["study"]
    output_dir = Path(sc["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.reset:
        import shutil
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True, exist_ok=True)

    for p in tune_cfg["params"]:
        d, keys = base_cfg, p["path"].split(".")
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                raise KeyError(f"param path '{p['path']}' not found in base config (failed at '{k}')")
            d = d[k]
    for dp in tune_cfg.get("derived_params", []):
        d, keys = base_cfg, dp["path"].split(".")
        for k in keys:
            if not isinstance(d, dict) or k not in d:
                raise KeyError(f"derived_param path '{dp['path']}' not found in base config (failed at '{k}')")
            d = d[k]

    param_names = [p["path"] for p in tune_cfg["params"]]
    state = _State(sc["n_trials"], param_names)

    study = optuna.create_study(
        study_name=sc["name"],
        storage=f"sqlite:///{output_dir}/study.db",
        direction="maximize",
        load_if_exists=True,
    )

    if not args.reset and len(study.trials) >= sc["n_trials"]:
        _show_results(study, tune_cfg, base_cfg, output_dir)
        return

    state.seed(study)  # show prior trials in the TUI on resume
    n_remaining = max(0, sc["n_trials"] - len(study.trials))

    if len(study.trials) == 0:
        defaults = {}
        for p in tune_cfg["params"]:
            d = base_cfg
            for k in p["path"].split("."):
                d = d[k]
            defaults[p["path"]] = d
        study.enqueue_trial(defaults)

    log_file = open(output_dir / "trials.jsonl", "a")
    tick = 0

    with Live(_build(state, tick), refresh_per_second=4, screen=True) as live:
        stop = threading.Event()

        def _updater():
            nonlocal tick
            while not stop.is_set():
                live.update(_build(state, tick))
                tick += 1
                time.sleep(0.25)

        th = threading.Thread(target=_updater, daemon=True)
        th.start()

        def objective(trial):
            params = {}
            for p in tune_cfg["params"]:
                path = p["path"]
                match p["type"]:
                    case "float":
                        params[path] = trial.suggest_float(
                            path, p["low"], p["high"], log=p.get("log", False)
                        )
                    case "int":
                        params[path] = trial.suggest_int(
                            path, p["low"], p["high"], step=p.get("step", 1)
                        )
                    case "categorical":
                        params[path] = trial.suggest_categorical(path, p["choices"])

            state.begin(trial.number, params)
            score = _run(trial.number, params, base_cfg, tune_cfg, output_dir, attempt=0)
            if score is not None and score <= _NEGINF:
                state.retry(trial.number)
                score = _run(trial.number, params, base_cfg, tune_cfg, output_dir, attempt=1)
            if score is None:
                score = _NEGINF
            state.end(trial.number, score)

            log_file.write(json.dumps({"trial": trial.number, "params": params, "score": score}) + "\n")
            log_file.flush()
            return score

        try:
            study.optimize(objective, n_trials=n_remaining)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set()
            th.join()

    log_file.close()
    _show_results(study, tune_cfg, base_cfg, output_dir)


if __name__ == "__main__":
    main()
