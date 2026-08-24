#!/usr/bin/env python3
"""Bayesian hyperparameter tuner — Optuna + Rich TUI.

Usage: python tune.py <tune_config.yaml> [--calibrate N | --confirm N | --report]
"""

import json
import random
import signal
import sys
import threading
import time
from collections import Counter
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transformer_rl.train_utils import _load_config   # ONE `extends:` resolver, shared with training

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


class _TB:
    """One cached EventAccumulator per trial. Building one costs ~0.34 s of directory scan;
    Reload()ing an existing one costs ~0 s, and it picks up event files that appear later — which
    is what makes a resumed trial's second event file readable through the same object. Constructing
    a fresh accumulator per poll (as this did before) is what made polling look expensive at 8-wide."""

    __slots__ = ("log_dir", "_ea")

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self._ea = None

    def series(self, metric: str) -> list[float]:
        """The scalar series for `metric` in logging order (one point per epoch). [] until the
        trainer has created the log dir and written its first event."""
        if self._ea is None:
            if not self.log_dir.exists():
                return []
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            self._ea = EventAccumulator(str(self.log_dir))
        try:
            self._ea.Reload()
        except Exception:
            return []   # event file mid-creation or partially written this poll
        if metric not in self._ea.Tags().get("scalars", []):
            return []
        return [s.value for s in self._ea.Scalars(metric)]


def _tb_series(log_dir: Path, metric: str) -> list[float]:
    """One-shot read, for callers outside the poll loop (reporting)."""
    return _TB(log_dir).series(metric)


def _final_score(series: list[float], mode: str, window: int) -> float:
    """Completed-trial objective: recovered-level (mean of last `window` epochs) when pruning
    is on, else max-over-training. Empty series (no metric logged) -> _NEGINF (failure)."""
    if not series:
        return _NEGINF
    if mode == "max":
        return max(series)
    tail = series[-window:]
    return sum(tail) / len(tail)


def _gate_factor(job_tb, sc: dict) -> float:
    """The collapse gate: a SATURATING floor on generator diversity, `min(1, D / floor)` (ADR-0020).

    Above `floor` it is exactly 1, so ranking stays pure reward; below, it discounts linearly. It
    exists because `quality/R_mean` averages over the sampled body population and so structurally
    rewards collapse -- a diverse population holds more bad bodies and scores lower -- which makes
    "maximise reward" partly an instruction to stop exploring. Saturation (rather than a product) is
    what keeps it a guard: it needs no reward-per-body trade rate, and above the floor extra
    generator entropy buys exactly zero score while still costing reward.

    `window: 0`/absent averages the WHOLE series. For `build/n_modes` that is the intended reading --
    it is logged RL-only, so the whole series is exactly the RL phase, and the pathology being caught
    is "never searched", not "stopped searching" (late commitment is success, not collapse).

    Returns 1.0 when no gate is configured or `floor` is null (the pre-calibration state, where the
    floor has not been measured yet). Returns None when the gate metric is configured but ABSENT --
    a run that logged no diversity at all never reached the RL phase, which is a failure rather than
    a free pass."""
    gate = sc.get("gate")
    if not gate or not gate.get("floor"):
        return 1.0
    series = job_tb.series(gate["key"])
    if not series:
        return None
    win = int(gate.get("window", 0)) or len(series)
    tail = series[-win:]
    return min(1.0, (sum(tail) / len(tail)) / float(gate["floor"]))


def _param_or_base(params: dict, base_cfg: dict, path: str):
    """Value of a config path: the swept value if this trial tuned it, else the base-config default."""
    if path in params:
        return params[path]
    d = base_cfg
    for k in path.split("."):
        d = d[k]
    return d


def _final_frac_score(series: list[float], params: dict, sc: dict, base_cfg: dict) -> float:
    """Objective when resample_interval is swept (windows vary in length). Mean of the last
    max(eval_min, min(floor(score_frac*N), N - n_pretrain)) logged windows, N = total logged. Capping
    the tail at the RL (non-pretrain) window count keeps warmup points out of the score even when a
    trial ends up with few windows. Empty series (no metric logged) -> _NEGINF (failure)."""
    if not series:
        return _NEGINF
    n = len(series)
    npre = int(_param_or_base(params, base_cfg, "params.config.generator.n_pretrain"))
    frac = sc.get("score_frac", 0.20)
    eval_min = sc.get("eval_min", 1)
    window = max(eval_min, min(int(frac * n), n - npre))
    window = max(1, min(window, n))
    tail = series[-window:]
    return sum(tail) / len(tail)


# ── slot layer ────────────────────────────────────────────────────────
# Slot discovery, pinned launch, exit classification and checkpoint bookkeeping live in
# experiments/harness/slots.py: the paper-experiment launcher schedules onto the same 8-slice machine,
# and which GPU a process lands on must have ONE implementation or two runs will share a slice.
from experiments.harness.slots import (   # noqa: E402
    _Slot, _classify_exit, _kill, _launch, _make_slots, _newest_ckpt, _purge_ckpts,
)


# ── state ─────────────────────────────────────────────────────────────

class _Trial:
    __slots__ = ("num", "params", "score", "status", "elapsed",
                 "slot", "t0", "epoch", "ema", "attempt")

    def __init__(self, num: int, params: dict, slot: str = ""):
        self.num     = num
        self.params  = params
        self.score: float | None = None
        self.status  = "running"
        self.elapsed = 0.0
        self.slot    = slot     # slot name it occupies (display only; "" once finished)
        self.t0      = 0.0      # wall-clock start of the CURRENT attempt
        self.epoch   = 0        # last epoch seen in TB (progress readout)
        self.ema     = None     # smoothed metric, for the slot table and for pruning
        self.attempt = 0

    def elapsed_now(self) -> float:
        return time.time() - self.t0 if self.t0 else 0.0


class _State:
    """What the TUI renders. Trials are keyed by number and several run at once, so nothing here
    assumes a single current trial: `running` is the live set, one entry per occupied slot."""

    def __init__(self, n_trials: int, param_names: list[str], slots: list[str] | None = None):
        self.n_trials   = n_trials
        self.param_names = param_names
        self.slot_names = slots or []
        self.trials: list[_Trial] = []
        self.running: dict[int, _Trial] = {}
        self._done_times: list[float] = []

    def get(self, num: int) -> _Trial | None:
        for t in self.trials:
            if t.num == num:
                return t
        return None

    def begin(self, num: int, params: dict, slot: str = ""):
        t = _Trial(num, params, slot)
        t.t0 = time.time()
        self.trials.append(t)
        self.running[num] = t
        return t

    def retry(self, num: int, attempt: int, kind: str = "retrying"):
        """A crashed trial is going round again on the same slot: `kind` distinguishes a resume
        from checkpoint from a from-scratch retry, and the clock restarts either way."""
        t = self.running.get(num) or self.get(num)
        if t:
            t.status, t.attempt, t.t0 = kind, attempt, time.time()
            t.epoch = 0 if kind == "retrying" else t.epoch

    def end(self, num: int, score: float):
        t = self.running.pop(num, None) or self.get(num)
        if t:
            t.score   = score
            t.status  = "done" if score > _NEGINF else "failed"
            t.elapsed = t.elapsed_now()
            t.slot    = ""
            self._done_times.append(t.elapsed)

    def prune(self, num: int):
        t = self.running.pop(num, None) or self.get(num)
        if t:
            t.status  = "pruned"
            t.score   = None   # killed early -> no recovered-level score; excluded from best/worst
            t.elapsed = t.elapsed_now()
            t.slot    = ""
            self._done_times.append(t.elapsed)

    def by_slot(self) -> dict[str, _Trial]:
        return {t.slot: t for t in self.running.values() if t.slot}

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
        return sum(1 for t in self.trials if t.status in ("done", "failed", "pruned"))

    def seed(self, study) -> None:
        """Pre-populate from an existing study so the TUI shows prior trials on resume."""
        TS = optuna.trial.TrialState
        for t in sorted(study.trials, key=lambda x: x.number):
            # FAIL is not a result: it marks a trial an interrupt killed, whose params were
            # re-enqueued and will run again under a new number. Replaying it here would show it
            # twice and count it against the budget the DB has already decided not to charge.
            if t.state in (TS.RUNNING, TS.WAITING, TS.FAIL):
                continue
            tr = _Trial(t.number, dict(t.params))
            if t.state == TS.COMPLETE and t.value is not None and t.value > _NEGINF:
                tr.score, tr.status = t.value, "done"
            elif t.state == TS.PRUNED:
                tr.score, tr.status = None, "pruned"
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

    # ── slot table ──
    # One row per slot, always: an idle row is information (the pool is not saturated), so the
    # table is built from the pool rather than from whatever happens to be running.
    occ = state.by_slot()
    slot_tbl = Table(box=box.SIMPLE_HEAD, expand=True, header_style="bold dim", show_edge=False)
    slot_tbl.add_column("Slot",  justify="left",  width=6)
    slot_tbl.add_column("Trial", justify="right", width=6)
    slot_tbl.add_column("Epoch", justify="right", width=7)
    slot_tbl.add_column("EMA",   justify="right", width=10)
    slot_tbl.add_column("Elapsed", justify="right", width=9)
    for name in state.slot_names:
        t = occ.get(name)
        if t is None:
            slot_tbl.add_row(name, "—", "—", "—", "—", style="dim")
            continue
        ic = {"resuming": "⇄", "retrying": "↺"}.get(t.status, sp)
        slot_tbl.add_row(
            name, f"{ic} {t.num}", str(t.epoch),
            f"{t.ema:.1f}" if t.ema is not None else "—",
            _fmt_t(t.elapsed_now()),
            style="magenta" if t.status in ("resuming", "retrying") else "yellow",
        )
    mean_str = _fmt_t(state.mean_time()) if state._done_times else "——"
    cur_panel = Panel(
        slot_tbl,
        title=f"[bold yellow]Slots[/bold yellow] [dim]({len(occ)}/{len(state.slot_names)} busy, "
              f"mean {mean_str})[/dim]",
        border_style="yellow",
    )

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
        elif t.status == "running":
            sc_str = sp
        elif t.status in ("retrying", "resuming"):
            sc_str = "↺" if t.status == "retrying" else "⇄"
        elif t.status == "pruned":
            sc_str = "⊘"
        else:
            sc_str = "—"

        if   t.status in ("running", "retrying", "resuming"):
            row_style, icon = "yellow", ""
        elif t.status == "pruned":                row_style, icon = "dim magenta",  "⊘"
        elif t.status == "failed":                row_style, icon = "dim red",      "✗"
        elif best and t.num == best.num:          row_style, icon = "bold green",   "★"
        else:                                     row_style, icon = "",             "✓"

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
        Layout(cur_panel,    name="current",  size=max(6, len(state.slot_names) + 5)),
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


def _headless_view(state: _State, stop: threading.Event, every: float = 60.0):
    """Line logging for `--headless`. Live(screen=True) drives an alternate screen buffer and needs
    a TTY, which a nohup'd/ssh'd sweep on the tuning box does not have. Prints one block per minute
    and, in between, a line whenever a slot changes hands — the transitions are the part worth
    keeping in a log file, the periodic block is just a heartbeat."""
    last_seen: dict[str, str] = {}
    last_block = 0.0
    while not stop.is_set():
        occ = state.by_slot()
        now = {n: f"{t.num}:{t.status}" for n, t in occ.items()}
        for name in state.slot_names:
            was, isnow = last_seen.get(name), now.get(name)
            if was != isnow:
                print(f"[slot {name}] {was or 'idle'} -> {isnow or 'idle'}", flush=True)
        last_seen = now

        if time.time() - last_block >= every:
            last_block = time.time()
            best = state.best()
            rows = "  ".join(
                f"{n}=" + (f"{t.num}@{t.epoch}" + (f"/{t.ema:.0f}" if t.ema is not None else "")
                           if (t := occ.get(n)) else "idle")
                for n in state.slot_names
            )
            print(f"[tune] {state.n_done()}/{state.n_trials} done  "
                  f"best={f'{best.score:.1f} (t{best.num})' if best else '——'}  "
                  f"mean={_fmt_t(state.mean_time()) if state._done_times else '——'}  | {rows}",
                  flush=True)
        stop.wait(5.0)


# ── winner selection ──────────────────────────────────────────────────

def _topk(completed: list, k: int) -> list:
    return sorted(completed, key=lambda t: t.value, reverse=True)[:k]


def _median(v: list[float]) -> float:
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _centre_and_spread(top: list, tune_cfg: dict) -> tuple[dict, dict]:
    """The winner as the CENTRE of the top-k cluster, plus how tight that cluster actually is.

    Argmax picks the luckiest draw, and at a noise floor near the effect size that bias grows as the
    sampler concentrates: the single best score is mostly a good seed. The per-parameter median (mode
    for categoricals) over the top k averages that draw noise out.

    The spread is not decoration — it is what says whether the centre means anything. A coordinate-
    wise summary describes one blob; if the top-k is bimodal in a parameter the "centre" lands
    between two clusters and assembles a config no trial ever ran. Reported per parameter as the
    fraction of the search range the top-k covers (continuous, IQR-based) or the modal share
    (categorical); wide spread means the parameter is undetermined, not that the centre is optimal."""
    centre, spread = {}, {}
    for p in tune_cfg["params"]:
        path = p["path"]
        vals = [t.params[path] for t in top if path in t.params]
        if not vals:
            continue
        if p["type"] == "categorical":
            counts = Counter(vals)
            val, n = counts.most_common(1)[0]
            centre[path] = val
            spread[path] = (n / len(vals), f"{n}/{len(vals)} {_fmt(val)}"
                            + (f"  (+{len(counts) - 1} other)" if len(counts) > 1 else ""))
        else:
            centre[path] = _median(vals)
            if p["type"] == "int":
                centre[path] = int(round(centre[path]))
            lo, hi = float(p["low"]), float(p["high"])
            s = sorted(vals)
            q1, q3 = s[len(s) // 4], s[min(len(s) - 1, (3 * len(s)) // 4)]
            if p.get("log", False) and lo > 0:
                import math
                frac = (math.log(max(q3, lo)) - math.log(max(q1, lo))) / (math.log(hi) - math.log(lo))
            else:
                frac = (q3 - q1) / (hi - lo) if hi > lo else 0.0
            spread[path] = (1.0 - frac, f"IQR {_fmt(q1)}–{_fmt(q3)}  ({frac * 100:.0f}% of range)")
    return centre, spread


def _importances(completed: list):
    """fANOVA importance over the SUCCESSFUL trials only, or None if it cannot be computed.

    Built on a throwaway in-memory study rather than run against the real one: `get_param_importances`
    reads every COMPLETE trial, and OOM trials are COMPLETE-with-0.0 by design (so the sampler learns
    the region is unavailable). Leaving them in would let a memory limit masquerade as a
    hyperparameter effect -- and a 0.0 against rewards in the thousands would dominate the variance
    decomposition, so the most "important" parameter would be whichever one drives memory."""
    try:
        import optuna.importance as oimp
        tmp = optuna.create_study(direction="maximize")
        tmp.add_trials(completed)
        return oimp.get_param_importances(tmp, evaluator=oimp.FanovaImportanceEvaluator(seed=0))
    except Exception as e:
        print(f"[importance] unavailable: {e}")
        return None


def _apply_params(cfg: dict, params: dict, tune_cfg: dict) -> dict:
    """Overlay a trial's swept params onto `cfg`, then the derived params computed from them. Shared
    by the trial materialiser and the winner export so the config a trial RAN and the config the
    report EXPORTS can never disagree."""
    for path, value in params.items():
        _set(cfg, path.split("."), value)
    ns = {path.split(".")[-1]: val for path, val in params.items()}
    for dp in tune_cfg.get("derived_params", []):
        val = eval(dp["expr"], {"__builtins__": {}}, ns)
        if isinstance(val, float) and val.is_integer():
            val = int(val)
        _set(cfg, dp["path"].split("."), val)
    return cfg


def _write_params(cfg_path: Path, base_cfg: dict, params: dict, tune_cfg: dict):
    """The export must be RUNNABLE as-is: a focus stage takes best_params.yaml as its base_config, so
    omitting the derived pass would hand it the BASE's max_epochs (3000) rather than the winner's
    (1500 at horizon_length 16) -- twice the frame budget, silently."""
    with open(cfg_path, "w") as f:
        yaml.dump(_apply_params(deepcopy(base_cfg), params, tune_cfg), f)


# ── noise floor ───────────────────────────────────────────────────────

def _noise_metrics(tune_cfg: dict) -> list[tuple[str, int]]:
    """(metric, tail window) pairs for the noise report. Each metric needs its OWN window: the study
    objective is per-epoch (~2000 points, tail 100) while quality/R_mean is logged once per resample
    window (~31 points), where a 100-point tail would silently average the entire run including
    pretraining. Defaults cover the two metrics the codesign runs actually log."""
    sc = tune_cfg["study"]
    m = sc.get("noise_metrics")
    if m:
        return [(d["key"], int(d["window"])) for d in m]
    return [(sc.get("metric_key", "rewards/iter"), sc.get("score_window", 100)),
            ("quality/R_mean", 5)]


def _cv(vals: list[float]) -> tuple[float, float, float]:
    n = len(vals)
    mean = sum(vals) / n
    sd = (sum((v - mean) ** 2 for v in vals) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mean, sd, (sd / mean if mean else float("nan"))


def _wave_groups(study) -> dict[str, list]:
    """Trials grouped into replicate sets — same hyperparameters, different seeds. Explicit wave
    tags win; otherwise any params repeated across trials form a group, so a noise estimate is
    available even from a sweep that happened to resample a point."""
    tagged, byparams = {}, {}
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE or t.value is None or t.value <= _NEGINF:
            continue
        w = t.user_attrs.get("wave")
        if w:
            tagged.setdefault(w, []).append(t)
        byparams.setdefault(tuple(sorted(t.params.items())), []).append(t)
    if tagged:
        return {k: v for k, v in tagged.items() if len(v) > 1}
    return {f"replicates@{i}": v for i, v in enumerate(
        [v for v in byparams.values() if len(v) > 1])}


def _noise_report(study, tune_cfg: dict, output_dir: Path, c) -> None:
    """The number every downstream decision depends on: the CV of the objective across seeds at
    FIXED hyperparameters. If the across-hyperparameter spread of a sweep is not comfortably larger
    than this, the sweep is ranking seeds rather than configs."""
    from rich.table import Table
    from rich import box as rbox

    groups = _wave_groups(study)
    if not groups:
        return
    metrics = _noise_metrics(tune_cfg)
    sub = tune_cfg["study"].get("summaries_subdir", "")

    tbl = Table(title="Noise floor (same config, different seeds)", box=rbox.SIMPLE_HEAD,
                header_style="bold dim")
    for col, j in (("Wave", "left"), ("Metric", "left"), ("n", "right"), ("Mean", "right"),
                   ("SD", "right"), ("CV", "right"), ("± on CV", "right"), ("Scores", "left")):
        tbl.add_column(col, justify=j)

    for wave, trials in sorted(groups.items()):
        for metric, window in metrics:
            vals = []
            for t in sorted(trials, key=lambda x: x.number):
                d = output_dir / "runs" / f"tune_trial_{t.number}"
                s = _tb_series(d / sub if sub else d, metric)
                if s:
                    vals.append(_final_score(s, "recovered", window))
            if len(vals) < 2:
                continue
            mean, sd, cv = _cv(vals)
            # An n-sample CV is itself an estimate: SE(CV) ~ CV/sqrt(2(n-1)). At n=8 that is ~27%
            # of the CV, so 8 seeds fix the order of magnitude, not the value.
            se = cv / (2 * (len(vals) - 1)) ** 0.5
            tbl.add_row(wave, metric, str(len(vals)), f"{mean:.1f}", f"{sd:.1f}",
                        f"{cv * 100:.0f}%", f"±{se * 100:.0f}%",
                        " ".join(f"{v:.0f}" for v in vals))
    if tbl.row_count:
        c.print(tbl)
        c.print("[dim]A sweep whose across-config spread is below its own CV is ranking seeds, "
                "not hyperparameters.[/dim]")


# ── results ───────────────────────────────────────────────────────────

def _kde(sns, ax, xs, ys, color):
    """Bivariate density behind a scatter, or nothing if the sample cannot support one.

    A KDE over a near-degenerate cloud raises from deep inside matplotlib ("Contour levels must be
    increasing"), which used to abort _show_results ENTIRELY -- losing the winner, the noise report
    and results.png over a decoration. That is most likely exactly when the report matters most: a
    study whose trials mostly OOM'd or failed has few points left to plot. Needing >=3 distinct
    values on both axes screens the common cases; the guard catches the rest."""
    if len(set(xs)) < 3 or len(set(ys)) < 3:
        return
    try:
        sns.kdeplot(x=xs, y=ys, ax=ax, fill=True, color=color, alpha=0.7, warn_singular=False)
    except Exception:
        pass


def _show_results(study: optuna.Study, tune_cfg: dict, base_cfg: dict, output_dir: Path):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from rich.console import Console
    from rich.table import Table
    from rich import box as rbox

    c = Console()
    # OOM trials are COMPLETE with a scored 0.0 so the sampler learns to avoid that region, but they
    # are not measurements of anything — keep them out of the winner, the top-k and the plots.
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None and t.value > _NEGINF
        and not t.user_attrs.get("failed")
    ]
    n_oom = sum(1 for t in study.trials if t.user_attrs.get("failed") == "oom")

    if not completed:
        c.print("[dim]No successful trials.[/dim]")
        return

    # ── winner: the centre of the top-k cluster, with the raw argmax kept alongside ──
    k = tune_cfg["study"].get("top_k") or max(3, min(len(completed), round(len(completed) * 0.15)))
    k = min(k, len(completed))
    top_k = _topk(completed, k)
    centre, spread = _centre_and_spread(top_k, tune_cfg)
    bt = study.best_trial

    _write_params(output_dir / "best_params.yaml", base_cfg, centre, tune_cfg)
    _write_params(output_dir / "best_params_argmax.yaml", base_cfg, bt.params, tune_cfg)

    wt = Table(title=f"Winner — top-{k} centre vs argmax", box=rbox.SIMPLE_HEAD, header_style="bold dim")
    wt.add_column("Parameter", justify="left")
    wt.add_column(f"Centre (top-{k})", justify="right")
    wt.add_column(f"Argmax (trial {bt.number})", justify="right")
    wt.add_column("Top-k spread", justify="left")
    for p in tune_cfg["params"]:
        path = p["path"]
        conc, desc = spread.get(path, (1.0, "—"))
        wt.add_row(path.split(".")[-1], _fmt(centre.get(path, "—")), _fmt(bt.params.get(path, "—")),
                   desc, style="" if conc >= 0.5 else "yellow")
    c.print(wt)
    c.print("[dim]best_params.yaml = centre (exported). best_params_argmax.yaml = single best trial.\n"
            "Yellow rows are undetermined: the top-k covers most of the search range, so the centre "
            "is a midpoint, not an optimum.[/dim]")

    # ── parameter importance ──  the other half of the screen->focus decision
    imps = _importances(completed)
    if imps:
        it = Table(title="Parameter importance (fANOVA, successful trials)", box=rbox.SIMPLE_HEAD,
                   header_style="bold dim")
        it.add_column("Parameter", justify="left")
        it.add_column("Share", justify="right")
        it.add_column("", justify="left")
        for path, v in imps.items():
            it.add_row(path.split(".")[-1], f"{v * 100:.1f}%", "█" * int(round(v * 40)))
        c.print(it)
        c.print("[dim]Importance says a parameter MATTERS (share of objective variance it explains); "
                "the top-k spread above says it was DETERMINED. Carry a parameter into a focus sweep "
                "only when both agree — high importance with a wide spread means it matters and the "
                "study did not pin it down.[/dim]")

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

    # The across-config spread, to be read directly against the noise floor below it.
    vals = [t.value for t in completed]
    mean, sd, cv = _cv(vals)
    n_gated = sum(1 for t in study.trials if t.user_attrs.get("gated"))
    c.print(f"[dim]Across-config spread: mean {mean:.1f}  SD {sd:.1f}  "
            f"CV {cv * 100:.0f}%  over {len(completed)} trials"
            + (f"   ({n_oom} OOM excluded)" if n_oom else "") + "[/dim]")
    if n_gated:
        gate = tune_cfg["study"].get("gate", {})
        c.print(f"[yellow]Collapse gate fired on {n_gated}/{len(study.trials)} trials "
                f"({gate.get('key')} < {gate.get('floor')}).[/yellow] [dim]The gate is insurance and "
                f"is expected to be INERT; if it is firing routinely the floor is wrong, not the "
                f"configs — it is then acting as a second objective (ADR-0020).[/dim]")
    _noise_report(study, tune_cfg, output_dir, c)
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

            _kde(sns, ax_r, pvals, scores, "#4c72b0")
            ax_r.scatter(pvals, scores, s=18, color="#4c72b0", alpha=0.6, zorder=3)

            if dur_pairs:
                tvs, tds = zip(*dur_pairs)
                _kde(sns, ax_t, list(tvs), list(tds), "#dd8452")
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

class _Job:
    """One trial in flight on one slot: the process plus everything needed to score, resume or
    kill it. Lives from the first launch to the final tell(), across any resumed attempts."""

    __slots__ = ("trial", "num", "params", "seed", "slot", "proc", "logf", "cfg_path",
                 "log_dir", "nn_dir", "t0", "attempt", "ema", "next_epoch", "tb")

    def __init__(self, trial, num: int, params: dict, seed: int, slot: _Slot,
                 cfg_path: Path, log_dir: Path, nn_dir: Path):
        self.trial    = trial
        self.num      = num
        self.params   = params
        self.seed     = seed
        self.slot     = slot
        self.cfg_path = cfg_path
        self.log_dir  = log_dir
        self.nn_dir   = nn_dir
        self.tb       = _TB(log_dir)
        self.proc     = None
        self.logf     = None
        self.t0       = 0.0
        self.attempt  = 0
        self.ema      = None
        self.next_epoch = 0

    def log_path(self, output_dir: Path) -> Path:
        return output_dir / "logs" / f"trial_{self.num}_attempt{self.attempt}.log"


def _write_trial_cfg(num: int, params: dict, seed: int, base_cfg: dict, tune_cfg: dict,
                     output_dir: Path) -> tuple[Path, Path, Path]:
    """Materialise this trial's config and return (cfg_path, summaries_dir, nn_dir). The config is
    written under the study's own dir rather than to a temp file: a resumed attempt must relaunch
    with byte-identical settings, the seed included, and a file that outlives the attempt is the
    simplest way to guarantee that."""
    sc = tune_cfg["study"]
    trial_cfg = _apply_params(deepcopy(base_cfg), params, tune_cfg)

    # Seed goes through the CONFIG, never the trainer's --seed flag: --seed additionally forces
    # cudnn.deterministic, CUBLAS_WORKSPACE_CONFIG and set_num_threads(1) (train_utils.py:835-851),
    # which would make every trial slower and measure a machine the real runs never use.
    _set(trial_cfg, sc.get("seed_path", "params.seed").split("."), seed)

    _set(trial_cfg, sc.get("experiment_name_path", "experiment.name").split("."), f"tune_trial_{num}")
    _set(trial_cfg, sc.get("train_dir_path", "experiment.directory").split("."), str(output_dir / "runs"))

    cfg_path = output_dir / "configs" / f"trial_{num}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.dump(trial_cfg, f)

    run_dir = output_dir / "runs" / f"tune_trial_{num}"
    sub = sc.get("summaries_subdir", "")
    return cfg_path, (run_dir / sub if sub else run_dir), run_dir / "nn"


def _start(job: _Job, tune_cfg: dict, output_dir: Path, checkpoint: Path | None = None):
    """Launch (or relaunch) a job's process on its slot. `checkpoint` resumes rl_games from a saved
    state instead of starting over — the trainer takes it as the second positional argument."""
    sc  = tune_cfg["study"]
    cmd = [sys.executable, sc["script"]]
    if checkpoint is not None:
        cmd += ["train", str(checkpoint)]
    cmd += ["--config", str(job.cfg_path)]
    header = (f"# trial {job.num}  attempt {job.attempt}  slot {job.slot.name} ({job.slot.device})\n"
              f"# seed {job.seed}  params: {job.params}\n"
              + (f"# RESUMING from {checkpoint}\n" if checkpoint else "") + "\n")
    job.proc, job.logf = _launch(cmd, job.slot.device, job.log_path(output_dir), header)
    job.t0 = time.time()


def _score_job(job: _Job, tune_cfg: dict, base_cfg: dict) -> float:
    sc = tune_cfg["study"]
    pruning = tune_cfg.get("pruner", {}).get("enabled", False)
    # ADR-0009: default max objective when timing is tuned (pruning off). A study may force a
    # final-level objective even with pruning off via study.score_mode (e.g. "final" -> mean of
    # the last `score_window` logged points) — used for sparse per-window metrics like quality/R_mean.
    mode   = sc.get("score_mode", "recovered" if pruning else "max")
    window = sc.get("score_window", tune_cfg.get("pruner", {}).get("score_window", 50))
    series = job.tb.series(sc.get("metric_key", "Reward / Total reward (mean)"))
    if mode == "final_frac":
        score = _final_frac_score(series, job.params, sc, base_cfg)
    else:
        score = _final_score(series, mode, window)
    if score <= _NEGINF:
        return score
    gate = _gate_factor(job.tb, sc)
    if gate is None:
        return _NEGINF
    if gate < 1.0:      # recorded so _show_results can report how often the guard actually fired;
        job.trial.set_user_attr("gated", round(gate, 4))   # if that is often, the floor is wrong
    return score * gate


def _suggest(trial, tune_cfg: dict) -> dict:
    params = {}
    for p in tune_cfg["params"]:
        path = p["path"]
        match p["type"]:
            case "float":
                params[path] = trial.suggest_float(path, p["low"], p["high"], log=p.get("log", False))
            case "int":
                params[path] = trial.suggest_int(path, p["low"], p["high"], step=p.get("step", 1))
            case "categorical":
                params[path] = trial.suggest_categorical(path, p["choices"])
    return params


def _feasible(params: dict, tune_cfg: dict, base_cfg: dict) -> bool:
    """Reject configs whose derived eval window can't reach the RL phase BEFORE spending any GPU
    (the sampled params alone decide it). Expr sees each swept param by short name plus the injected
    budget constants."""
    sc = tune_cfg["study"]
    fexpr = sc.get("feasibility_expr")
    if not fexpr:
        return True
    cc  = base_cfg["params"]["config"]
    fns = {p["path"].split(".")[-1]: params[p["path"]] for p in tune_cfg["params"]}
    fns.update(max_epochs=cc["max_epochs"], horizon_length=cc["horizon_length"],
               max_episode_length=sc.get("max_episode_length", 1000),
               eval_min=sc.get("eval_min", 1))
    return bool(eval(fexpr, {"__builtins__": {}}, fns))


# ── scheduler ─────────────────────────────────────────────────────────

def _schedule(study, state: _State, slots: list[_Slot], tune_cfg: dict, base_cfg: dict,
              output_dir: Path, log_file, budget: int, rng: random.Random) -> None:
    """The control loop. Owns the slot pool and drives Optuna by hand: ask() whenever a slot is free
    and budget remains, launch a pinned process, poll every live trial, tell() on exit. The trials
    run in parallel; the bookkeeping does not, so SQLite needs no concurrency story at all."""
    TS = optuna.trial.TrialState
    sc, pc  = tune_cfg["study"], tune_cfg.get("pruner", {})
    pruning   = pc.get("enabled", False)
    alpha     = pc.get("ema_alpha", 0.3)
    poll_secs = pc.get("poll_seconds", 20)
    warmup    = pc.get("n_warmup_steps", 0)
    metric    = sc.get("metric_key", "Reward / Total reward (mean)")
    timeout   = sc.get("trial_timeout_seconds", 300)
    stagger   = sc.get("launch_stagger_seconds", 30)
    max_att   = sc.get("max_attempts", 3)
    keep_ck   = sc.get("keep_checkpoints", 2)

    free: list[_Slot] = list(slots)
    jobs: dict[int, _Job] = {}
    launched, last_launch = 0, 0.0

    def _record(num, params, score):
        log_file.write(json.dumps({"trial": num, "params": params, "score": score}) + "\n")
        log_file.flush()

    def _release(job, note: str):
        if job.logf:
            job.logf.write(f"\n# {note}\n")
            job.logf.close()
            job.logf = None

    def _retire(job, score: float, failed: str | None = None, note: str | None = None):
        """Close a job out: tell Optuna, hand the slot back, and drop the trial's checkpoints —
        they exist only to make a resume possible and this trial will never be resumed again."""
        _release(job, note or (f"exit {job.proc.returncode}" if job.proc else "closed"))
        if failed:
            job.trial.set_user_attr("failed", failed)
        study.tell(job.trial, score)
        state.end(job.num, score)
        _record(job.num, job.params, "oom" if failed == "oom" else score)
        _purge_ckpts(job.nn_dir, keep=0)
        free.append(job.slot)
        jobs.pop(job.num, None)

    try:
        while launched < budget or jobs:
            # ── fill idle slots ──
            while free and launched < budget and time.time() - last_launch >= stagger:
                trial = study.ask()
                params = _suggest(trial, tune_cfg)
                if not _feasible(params, tune_cfg, base_cfg):
                    # No GPU spent, so no slot is taken and no stagger is owed.
                    state.begin(trial.number, params)
                    state.prune(trial.number)
                    study.tell(trial, state=TS.PRUNED)
                    _record(trial.number, params, "infeasible")
                    launched += 1
                    continue
                seed = rng.randrange(1, 2 ** 31 - 1)
                trial.set_user_attr("seed", seed)
                slot = free.pop(0)
                trial.set_user_attr("slot", slot.name)
                cfg_path, log_dir, nn_dir = _write_trial_cfg(
                    trial.number, params, seed, base_cfg, tune_cfg, output_dir)
                job = _Job(trial, trial.number, params, seed, slot, cfg_path, log_dir, nn_dir)
                jobs[trial.number] = job
                state.begin(trial.number, params, slot.name)
                _start(job, tune_cfg, output_dir)
                launched += 1
                last_launch = time.time()

            # ── poll live trials ──
            for num, job in list(jobs.items()):
                ret = job.proc.poll()
                t   = state.running.get(num)

                series = job.tb.series(metric)
                if len(series) > job.next_epoch:
                    for e in range(job.next_epoch, len(series)):
                        x = series[e]
                        job.ema = x if job.ema is None else alpha * x + (1 - alpha) * job.ema
                        if pruning:
                            job.trial.report(job.ema, e)
                            if e >= warmup and job.trial.should_prune():
                                _kill(job.proc)
                                _release(job, f"PRUNED at epoch {e} (ema={job.ema:.4g})")
                                study.tell(job.trial, state=TS.PRUNED)
                                state.prune(num)
                                _record(num, job.params, "pruned")
                                _purge_ckpts(job.nn_dir, keep=0)
                                free.append(job.slot)
                                jobs.pop(num, None)
                                break
                    job.next_epoch = len(series)
                    if t:
                        t.epoch, t.ema = len(series), job.ema
                if num not in jobs:
                    continue
                _purge_ckpts(job.nn_dir, keep=keep_ck)

                if ret is None:
                    if time.time() - job.t0 > timeout:
                        # Soft cap: score the partial run rather than discard it. The timeout is a
                        # wall-clock safety net, not a verdict on the hyperparameters.
                        _kill(job.proc)
                        _retire(job, _score_job(job, tune_cfg, base_cfg), note="TIMED OUT")
                    continue

                if ret == 0:
                    _retire(job, _score_job(job, tune_cfg, base_cfg))
                    continue

                # ── it died: which kind of death decides whether it runs again ──
                cause = _classify_exit(job.log_path(output_dir))
                if cause == "oom":
                    # A real property of these hyperparameters, not bad luck. Score it (rewards are
                    # strictly positive, so 0.0 is below every real result and TPE is rank-based
                    # anyway) so the sampler learns the region is unavailable, and never relaunch.
                    _retire(job, 0.0, failed="oom")
                    continue
                if job.attempt + 1 >= max_att:
                    _retire(job, _NEGINF, failed=cause)
                    continue
                ckpt = _newest_ckpt(job.nn_dir)
                _release(job, f"exit {job.proc.returncode} ({cause}) -> "
                              f"{'resume from ' + ckpt.name if ckpt else 'retry from scratch'}")
                job.attempt += 1
                state.retry(num, job.attempt, "resuming" if ckpt else "retrying")
                if ckpt is None:
                    job.next_epoch, job.ema = 0, None
                _start(job, tune_cfg, output_dir, checkpoint=ckpt)

            # Sleep only as long as nothing is owed: a free slot waiting out its stagger should
            # not also wait out a full poll interval.
            nap = poll_secs
            if free and launched < budget:
                nap = min(nap, max(1.0, stagger - (time.time() - last_launch)))
            time.sleep(nap)
    except (KeyboardInterrupt, SystemExit):
        # Kill the whole cohort, then mark every in-flight trial FAIL and re-enqueue its exact
        # params. Without the re-enqueue an interrupted trial is lost budget: it is neither a result
        # nor a pending trial, and at 8-wide every Ctrl-C would silently drop eight of them.
        for job in list(jobs.values()):
            _kill(job.proc)
            _release(job, "INTERRUPTED")
            study.tell(job.trial, state=TS.FAIL)
            state.end(job.num, _NEGINF)
            if job.params:
                study.enqueue_trial(job.params, skip_if_exists=False)
        raise


# ── entry point ───────────────────────────────────────────────────────

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Bayesian hyperparameter tuner")
    # Mandatory, no default: a tune config names a base_config, which names a TASK. Defaulting would
    # pick a task by omission -- the same reason --config is mandatory for the train scripts.
    ap.add_argument("config", help="tune study yaml, e.g. configs/tune_codesign_ant_screen.yaml")
    ap.add_argument("--reset", action="store_true", help="delete study DB before starting")
    ap.add_argument("--slots", default=None,
                    help="'auto' (every idle device), an int N (first N idle), or a comma-separated "
                         "list of CUDA_VISIBLE_DEVICES strings used verbatim. Overrides study.slots.")
    ap.add_argument("--allow-busy", action="store_true",
                    help="schedule onto devices that already carry a foreign compute app "
                         "(a workstation's desktop session); never wanted on the tuning server")
    ap.add_argument("--headless", action="store_true",
                    help="line logging instead of the live TUI (screen mode needs a TTY)")
    ap.add_argument("--calibrate", type=int, metavar="N",
                    help="calibration wave: run the anchor config at N seeds and stop, to measure "
                         "the study's noise floor before committing budget to a sweep")
    ap.add_argument("--confirm", type=int, metavar="N",
                    help="confirmation wave: run the exported top-k centre at N seeds and stop, to "
                         "test the winner rather than describe it")
    ap.add_argument("--report", action="store_true",
                    help="re-print results for an existing study without running anything")
    args = ap.parse_args()

    with open(args.config) as f:
        tune_cfg = yaml.safe_load(f)
    # RESOLVED, not raw: the base config is an `extends:` leaf (task + seed body over an algorithm
    # parent), so a raw read would see only those two keys -- every param path below would miss, and
    # each trial_N.yaml would be written with no algorithm in it. Resolving also flattens the chain,
    # which the trial configs need: they are written under the study dir, where a relative `extends:`
    # no longer points anywhere.
    base_cfg = _load_config(Path(tune_cfg["study"]["base_config"]))

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

    # ── slot pool ──  (skipped for --report: reading a finished study needs no GPU at all)
    slots: list[_Slot] = []
    if not args.report:
        spec = args.slots if args.slots is not None else sc.get("slots")
        if isinstance(spec, str):
            spec = None if spec == "auto" else (int(spec) if spec.isdigit() else spec.split(","))
        slots = _make_slots(spec, allow_busy=args.allow_busy)
        print(f"[slots] {len(slots)}: " + ", ".join(f"{s.name}={s.device}" for s in slots), flush=True)
        # Forgetting the MIG setup is SILENT otherwise: `slots: auto` finds 2 whole cards instead of
        # 8 slices and the study runs 4x slower for days without a single warning. Declaring the
        # expected width turns that into a refusal at second zero. Non-destructive on purpose -- the
        # partitioning itself needs sudo and lives in `experiments/harness/mig.py`.
        # Only when the width came from DISCOVERY: an explicit --slots is the operator stating the
        # width deliberately (the local `--slots 1 --allow-busy` path), and _make_slots already
        # treats an explicit spec as an override to be honoured verbatim.
        if (want := sc.get("expect_slots")) and args.slots is None and len(slots) != want:
            raise RuntimeError(
                f"expect_slots: {want} but discovered {len(slots)}. On the tuning box run "
                f"`sudo python -m experiments.harness.mig --gpus 0,1 --profile 67 --count 4`; "
                f"elsewhere override with --slots or drop expect_slots from the config.")

    param_names = [p["path"] for p in tune_cfg["params"]]
    state = _State(sc["n_trials"], param_names, [s.name for s in slots])

    pc = tune_cfg.get("pruner", {})
    if pc.get("enabled", False) and pc.get("type", "median") == "median":
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=pc.get("n_startup_trials", 4),
            n_warmup_steps=pc.get("n_warmup_steps", 200),
            interval_steps=pc.get("interval_steps", 25),
        )
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=sc["name"],
        storage=f"sqlite:///{output_dir}/study.db",
        direction="maximize",
        load_if_exists=True,
        # constant_liar is mandatory above one slot: in-flight trials are otherwise invisible to the
        # sampler, which keeps proposing near-duplicates of whatever is already running.
        sampler=optuna.samplers.TPESampler(seed=sc.get("sampler_seed", 42),
                                           constant_liar=len(slots) > 1),
        pruner=pruner,
    )

    # Trials left RUNNING by a previous session are neither results nor pending work. Mark them FAIL
    # and re-enqueue their params, or they would count against the budget forever.
    TS = optuna.trial.TrialState
    stale = [] if args.report else [t for t in study.trials if t.state == TS.RUNNING]
    for t in stale:
        study.tell(t.number, state=TS.FAIL)
        if t.params:
            study.enqueue_trial(t.params, skip_if_exists=False)
    if stale:
        print(f"[resume] {len(stale)} interrupted trial(s) marked FAIL and re-enqueued", flush=True)

    # THE budget counts finished results only. Counting len(study.trials) (as this did) counts
    # interrupted and re-enqueued trials too, so every Ctrl-C silently shrank the sweep — 8x worse
    # at 8-wide, where one interrupt strands a whole cohort.
    n_done_db = sum(1 for t in study.trials if t.state in (TS.COMPLETE, TS.PRUNED))
    wave = args.calibrate or args.confirm
    if args.report or (not args.reset and not wave and n_done_db >= sc["n_trials"]):
        _show_results(study, tune_cfg, base_cfg, output_dir)
        return

    if not wave:
        state.seed(study)  # show prior trials in the TUI on resume; a wave shows only its own

    def _defaults() -> dict:
        d = {}
        for p in tune_cfg["params"]:
            v = base_cfg
            for k in p["path"].split("."):
                v = v[k]
            d[p["path"]] = v
        return d

    if wave:
        # A wave is N ordinary trials pinned to ONE point in the space, differing only by seed, run
        # in the search study rather than a side DB so its runs stay comparable to every other trial
        # and resume/reporting need no special case. The cost is real: N observations stacked on one
        # coordinate skew the sampler's density model, which is why a wave is a deliberate act.
        if args.calibrate:
            point, tag = (sc.get("enqueue_trials") or [None])[0] or _defaults(), "calibration"
        else:
            done = [t for t in study.trials
                    if t.state == TS.COMPLETE and t.value is not None and t.value > _NEGINF
                    and not t.user_attrs.get("failed")]
            if not done:
                raise SystemExit("--confirm needs completed trials to take a winner from")
            k = sc.get("top_k") or max(3, min(len(done), round(len(done) * 0.15)))
            point, tag = _centre_and_spread(_topk(done, min(k, len(done))), tune_cfg)[0], "confirmation"
        for _ in range(wave):
            study.enqueue_trial(point, user_attrs={"wave": tag}, skip_if_exists=False)
        n_remaining = wave
        state.n_trials = wave
        print(f"[{tag}] {wave} seeds of {point}", flush=True)
    else:
        n_remaining = max(0, sc["n_trials"] - n_done_db)
        if len(study.trials) == 0:
            enqueue = sc.get("enqueue_trials")
            if enqueue:
                for t in enqueue:          # explicit anchors (fully-specified path->value dicts)
                    study.enqueue_trial(t)
            else:
                study.enqueue_trial(_defaults())

    log_file = open(output_dir / "trials.jsonl", "a")
    # Per-trial seeds come from one study-level RNG, so the study as a whole stays reproducible
    # while no two trials share a seed. Advanced past the results already in the DB so a resumed
    # session does not re-issue seeds the finished trials already used.
    rng = random.Random(sc.get("sampler_seed", 42))
    for _ in range(n_done_db):
        rng.randrange(1, 2 ** 31 - 1)

    stop = threading.Event()

    # Take over BOTH stop signals so the cohort is always cleaned up, however the tuner is stopped.
    # Two reasons the default handling is not enough: a remote sweep is stopped with `kill` (SIGTERM),
    # which by default kills the tuner outright and leaves eight orphaned trials holding every slot;
    # and a tuner launched in the background or under nohup inherits SIGINT as SIG_IGN, so Python
    # never installs its KeyboardInterrupt handler and Ctrl-C/-INT is silently ignored. Installing
    # handlers explicitly overrides the inherited disposition in both cases.
    def _on_signal(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for _sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(_sig, _on_signal)

    def _run_loop():
        _schedule(study, state, slots, tune_cfg, base_cfg, output_dir, log_file, n_remaining, rng)

    try:
        if args.headless:
            # Live(screen=True) needs a TTY; the tuning box is driven over ssh/nohup.
            th = threading.Thread(target=_headless_view, args=(state, stop), daemon=True)
            th.start()
            try:
                _run_loop()
            finally:
                stop.set()
                th.join(timeout=2)
        else:
            tick = 0
            with Live(_build(state, tick), refresh_per_second=4, screen=True) as live:
                def _updater():
                    nonlocal tick
                    while not stop.is_set():
                        live.update(_build(state, tick))
                        tick += 1
                        time.sleep(0.25)

                th = threading.Thread(target=_updater, daemon=True)
                th.start()
                try:
                    _run_loop()
                finally:
                    stop.set()
                    th.join(timeout=2)
    except KeyboardInterrupt:
        print("\n[tune] interrupted; in-flight trials killed and re-enqueued", flush=True)

    log_file.close()
    _show_results(study, tune_cfg, base_cfg, output_dir)


if __name__ == "__main__":
    main()
