"""Wave scheduler for the paper experiments: a fixed queue of runs onto a fixed pool of slots.

    python -m experiments.harness.launch aux                      # one study
    python -m experiments.harness.launch baseline aux clone       # several, as ONE mixed pool
    python -m experiments.harness.launch clone --dry-run          # print the plan, launch nothing
    python -m experiments.harness.launch attention --slots 4 --arms full,self_cls

The tuner (`scripts/tune.py`) schedules an *open* set of trials it invents as it goes and scores
them; this schedules a *closed* set someone already wrote down in a build plan, and scores nothing --
`scrape.py` does that afterwards from the run dirs. What the two share is the device layer, which
lives in `slots.py`.

Three properties the experiments depend on:

**The queue is seed-major, arm-minor.** Runs go out interleaved by condition, never one arm at a
time, so slot degradation, another tenant, or thermal drift over a 9-hour session cannot land wholly
on one arm -- where it would be invisible as a uniform per-arm shift. This is the "mixed waves" rule
in every build plan and it is a property of the *order*, not of any synchronisation: slots refill
from the queue as they free, since a lock-step wave would idle 7 slots waiting for the slowest run.

**Several studies queue as one pool, and usually should.** The rule above is about conditions that
get COMPARED, and since the baseline was split into its own study those conditions now span studies:
running `baseline` in one session and `clone` hours later puts the shared comparison point on the
far side of exactly the drift mixed waves exist to cancel. Naming several studies interleaves them
seed-major across all of them, which restores the property. `--arms` is single-study only, since an
arm name means nothing across a set.

**Seeds go through the config, never `--seed`.** The trainer's `--seed` flag additionally forces
cudnn.deterministic, CUBLAS_WORKSPACE_CONFIG and single-threaded torch (train_utils.py:888-894),
which measures a machine the paper's numbers do not come from. `params.seed` alone reaches
rl_games' `Runner.load_config`, which seeds torch/numpy/random -- the seed-to-seed variation the
error bars are made of -- and nothing else.

**A crashed run resumes; it never silently restarts.** The gym-rebuild race
(docs/troubleshooting/resample_rebuild_crash.md) kills long codesign runs intermittently, and a
3-hour run lost at hour 2 costs a wave. Resume is from the run's own newest checkpoint. A run dir
that exists with *no* checkpoint is refused rather than reused: relaunching into it would append a
second TensorBoard event stream to the same summaries dir and quietly fuse two runs into one curve.
A legitimate resume also leaves a second event file, whose steps *overlap* the first's tail (the crash
came after the last save) -- `scrape.py` owns reconciling that, and must prefer the later file where
they overlap.
"""
import argparse
import inspect
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from codesigner.metrics.record import RECORD     # noqa: E402  the package's run-dir names,
from codesigner.rundir import LATEST             # noqa: E402  imported so they cannot drift

from experiments.harness.slots import (          # noqa: E402
    _Slot, _classify_exit, _kill, _launch, _make_slots, _newest_ckpt, _ckpt_epoch, parse_slot_spec,
)
from transformer_rl.train_utils import _load_config, _resolve_task   # noqa: E402  ONE extends: resolver
# The experiment dir experiment 5 writes to, taken from the module that CONSTRUCTS those
# algorithms rather than restated here -- see EXPERIMENT there for why a second copy is a bug.
from experiments.harness.baselines import EXPERIMENT as _BASELINES_EXPERIMENT  # noqa: E402

SEEDS = tuple(range(42, 50))            # 42-49, series-wide (ADR-0021)
WINDOWS = 48                            # 8 pretrain + 40 RL (ADR-0021); epochs are DERIVED from it
EPOCHS_NO_GEN = 3000                    # studies with no windows to count (experiment 4)


# ── studies ───────────────────────────────────────────────────────────
# Data, not code: an arm is a name plus the `--set` overrides that define it, and the build plans in
# temp/experiment_plan_*.md are the source. An arm may override the entry point too, which is what
# experiment 1 needs -- its three arms are three different agents, hence three scripts and three
# `experiment` dirs.

@dataclass(frozen=True)
class Arm:
    sets: tuple[str, ...] = ()
    script: str | None = None
    config: str | None = None
    experiment: str | None = None      # run dir leaf: runs/<task>_<family>/<experiment>/<run>
    args: tuple[str, ...] = ()         # positional args, before the flags -- the arm's own name,
                                       # where one script serves several arms


@dataclass(frozen=True)
class Study:
    script: str
    config: str
    experiment: str
    arms: dict[str, Arm]
    common: tuple[str, ...] = ()       # applied to every arm, before the arm's own overrides
    family: str = "codesign"
    seeds: tuple[int, ...] = SEEDS
    windows: int = WINDOWS             # budget in resample windows; epochs are derived from it
    epochs: int | None = None          # only for a study with no windows (experiment 4)
    # Which loop owns the run, which decides where its checkpoints are and what a resume means.
    # "rlgames": the trainer writes nn/*.pth and a resume is this launcher handing the newest one
    # back on the command line. "codesigner": `codesigner.optimize` writes a RunDirectory and
    # `resume=True` means *run or continue*, so the resume is internal and re-issuing the identical
    # command is the whole of it. Done-detection differs with it -- see `_classify_pending`.
    driver: str = "rlgames"


# The TUNED config -- the FOCUS study's exported winner, so every arm is a `--set` delta off a
# well-set algorithm rather than off defaults the sweep superseded. NOT ppo_ant_codesign_focus_base,
# which is that study's INPUT (the screen winner): the two differ in exactly the 7 parameters focus
# swept, `generator.lam` by 15x -- and lam is one of the two coefficients experiment 3 ablates, so
# basing the arms on the input would have measured the removal of a badly-set coefficient.
_SINGLE = dict(script="scripts/train_codesign_single.py",
               config="configs/ppo_ant_codesign_tuned.yaml",
               experiment="codesign_single_transformer")

# Stripping the codesign scaffolding for a no-generator run takes THREE overrides, not one: FD and FK
# are per-step control-side terms fused into the PPO loss, so they survive `resample_interval=0` and
# a run that only zeroes the interval silently carries experiment 2's treatment (ADR-0021).
_NO_GEN = ("params.config.resample_interval=0",
           "params.config.fd.enabled=false", "params.config.fk.enabled=false")

STUDIES: dict[str, Study] = {
    # THE BASELINE: the tuned algorithm with no overrides at all, run ONCE for the whole series.
    # Every ablation below is a `--set` delta off exactly this, so its unmodified cell is this study
    # rather than an arm of its own -- `aux/aux` and `clone/both` were byte-identical run sets, so
    # launching both experiments trained the same 8 seeds twice. Scraped with the union of the
    # ablations' extra tags (see scrape.EXTRA_TAGS), since it is the comparison point for all of them.
    #
    # NOT the baseline for experiment 4, which strips the generator entirely: `attention/full` is a
    # different configuration, not this one, and keeps its own in-study control arm.
    "baseline": Study(**_SINGLE, arms={"tuned": Arm()}),

    # Experiment 2 -- auxiliary prediction. The unmodified arm is `baseline`.
    "aux": Study(**_SINGLE, arms={
        "none": Arm(("params.config.fd.enabled=false", "params.config.fk.enabled=false")),
    }),

    # Experiment 3 -- control clone. 2x2 on the two clone coefficients; the both-on cell is
    # `baseline`, so only the three ablated cells run here.
    "clone": Study(**_SINGLE, arms={
        "kl_only":  Arm(("params.config.generator.lam=0",)),
        "mse_only": Arm(("params.config.generator.beta=0",)),
        "none":     Arm(("params.config.generator.beta=0", "params.config.generator.lam=0")),
    }),

    # Experiment 4 -- control attention. No generator at all, one fixed body; `full` must pass NO
    # attn_scope override, so its network path stays bit-identical to the shipped one. Its budget is
    # in plain epochs because it has no windows to count.
    "attention": Study(**_SINGLE, common=_NO_GEN, epochs=EPOCHS_NO_GEN, arms={
        "full":     Arm(),
        "self_cls": Arm(("params.network.transformer.attn_scope=self_cls",)),
        "self":     Arm(("params.network.transformer.attn_scope=self",)),
    }),

    # Experiment 1 -- shared backbone. Blocked on the Phase5 port: two of the three scripts and
    # configs below do not exist on HEAD yet, which `_resolve` reports as a missing entry point
    # rather than as a launch failure three hours in.
    "backbone": Study(**_SINGLE, arms={
        "single":    Arm(),
        "split":     Arm(script="scripts/train_codesign_split.py",
                         config="configs/ppo_ant_codesign_split.yaml",
                         experiment="codesign_split_transformer"),
        "decoupled": Arm(script="scripts/train_codesign_decoupled.py",
                         config="configs/ppo_ant_codesign_decoupled.yaml",
                         experiment="codesign_decoupled_transformer"),
    }),

    # Experiment 5 -- cross-method baselines, the two conditions with no learned generator. One
    # entry point, one config, one budget; the arm name is the only thing that differs, because
    # that is the only thing that differs. `fixed_body` is also the series NORMALIZER, so it is run
    # on the same 8 seeds as everything else rather than once.
    #
    # No `_NO_GEN` here, and that is the point. Experiment 4 strips the codesign scaffolding to
    # measure attention; these arms KEEP it -- FD and FK are control-side and belong to the control
    # stack under comparison, and `resample_interval` stays at 1 because the window is what bounds
    # one `Algorithm.run()`. What each arm removes is the generator's UPDATE, and neither a config
    # key nor an override expresses that: it is which body source the algorithm was built with.
    "baselines": Study(script="scripts/optimize_baselines.py",
                       config="configs/ppo_ant_codesign_tuned.yaml",
                       experiment=_BASELINES_EXPERIMENT, driver="codesigner", arms={
        "fixed_body":       Arm(args=("fixed_body",)),
        "random_generator": Arm(args=("random_generator",)),
    }),
}


# ── runs ──────────────────────────────────────────────────────────────

@dataclass
class Run:
    """One process to schedule. `sets` are dotted config overrides; `checkpoint` is a WARM START
    (specialization's 250-epoch fine-tune begins from a training checkpoint), distinct from a resume,
    which the pool derives from the run's own nn/ dir after a crash."""
    name: str
    script: str
    config: str
    train_dir: str
    sets: tuple[str, ...] = ()
    args: tuple[str, ...] = ()          # positional args, ahead of the flags
    driver: str = "rlgames"             # see Study.driver
    checkpoint: str | None = None
    target_epoch: int = 0               # for done-detection; 0 disables it
    # Whether a run dir with no checkpoint may be relaunched. False for training runs, where the only
    # honest recovery is a human looking at it. True for a warm-started pass (specialization), whose
    # start state is a file on disk and therefore reproducible: relaunching is exactly a resume, and
    # a 15-minute run is not worth an operator's attention.
    restartable: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return _ROOT / self.train_dir / self.name

    @property
    def nn_dir(self) -> Path:
        return self.run_dir / "nn"

    def cmd(self, checkpoint: str | None) -> list[str]:
        cmd = [sys.executable, str(_ROOT / self.script), *self.args]
        if self.driver == "codesigner":
            # No checkpoint on the command line, ever: `resume=True` continues the run in the run
            # directory from the inside, so re-issuing this exact command IS the resume. A path
            # here would be a WARM START -- a different thing, and not what a crash wants.
            assert checkpoint is None, "the codesigner driver resumes from its own run dir"
            cmd += ["--config", str(_ROOT / self.config), "--name", self.name]
            for s in self.sets:
                cmd += ["--set", s]
            return cmd
        if checkpoint:
            cmd += ["train", str(checkpoint)]
        cmd += ["--config", str(_ROOT / self.config)]
        # The run's identity goes through the CONFIG rather than through `--name`: train_utils
        # refuses to start when `train_dir/<--name>` already exists (:882), which is the right guard
        # for a human but blocks every resume. Setting full_experiment_name leaves that check
        # looking at a timestamp dir that never exists, and is the same route the tuner takes.
        for s in (*self.sets, f"params.config.full_experiment_name={self.name}"):
            cmd += ["--set", s]
        return cmd


def _train_dir(study: Study, arm: Arm, config: Path, sets: tuple[str, ...]) -> str:
    """Where the entry point will put this run: runs/<task>_<family>/<experiment>. Composed the same
    way `_compose_identity` composes it, with the task read from the config rather than assumed, so a
    grasp config -- or a `--set env.task=grasp` -- lands under runs/grasp_codesign/ with no registry
    edit."""
    task, _, _ = _effective(config, sets)
    return f"runs/{task}_{study.family}/{arm.experiment or study.experiment}"


def _effective(config: Path, sets: tuple[str, ...]) -> tuple[str, type, dict]:
    """The config as the TRAINER will see it: resolved, then this run's `--set` overrides applied the
    same way train_utils applies them (:834-840, YAML-parsed values). Reading the file alone is wrong
    for any arithmetic the overrides can change -- experiment 4 sets `resample_interval=0`, so a
    file-only reading would give it a window cadence it does not have."""
    cfg = _load_config(config)
    for kv in sets:
        key, _, val = kv.partition("=")
        *path, leaf = key.split(".")
        d = cfg
        for p in path:
            d = d.setdefault(p, {})
        d[leaf] = yaml.safe_load(val)
    task, task_class = _resolve_task(cfg)
    return task, task_class, cfg


def _window_epochs(config: Path, sets: tuple[str, ...] = ()) -> int:
    """Epochs in one resample window, or 0 if this run never resamples.

    `resample_interval` is in episodes and an epoch is `horizon_length` env-steps, so a window spans
    ceil(interval * max_episode_length / horizon_length) epochs -- the same arithmetic the agent uses
    to size its LR warmup (codesign_agent.py:163) and `CodesignAlgorithm` uses to set checkpoint
    cadence (D24). `max_episode_length` has a Task-owned default a config need not restate, so it is
    read from the Task class rather than assumed; verified empirically against a live run, whose
    window boundaries land every 63 epochs with no drift (`_steps_since_resample` resets to 0 at the
    boundary rather than subtracting, so the +8-step overshoot does not accumulate).
    """
    _, task_class, cfg = _effective(config, sets)
    ppo = cfg["params"]["config"]
    interval = ppo.get("resample_interval", 0)
    if not interval:
        return 0
    ep_len = cfg.get("env", {}).get("max_episode_length")
    if ep_len is None:
        ep_len = inspect.signature(task_class.__init__).parameters["max_episode_length"].default
    return max(1, -(-interval * int(ep_len) // ppo["horizon_length"]))


def study_runs(name: str, arms: list[str] | None = None, seeds: tuple[int, ...] | None = None,
               epochs: int | None = None) -> list[Run]:
    """The study's queue, seed-major / arm-minor -- see the module docstring on why the order is the
    experiment's confound control."""
    study = STUDIES[name]
    picked = arms or list(study.arms)
    unknown = [a for a in picked if a not in study.arms]
    if unknown:
        raise SystemExit(f"[launch] unknown arm(s) for '{name}': {', '.join(unknown)}")
    runs = []
    for seed in (seeds or study.seeds):
        for arm_name in picked:
            arm = study.arms[arm_name]
            config = _ROOT / (arm.config or study.config)
            script = _ROOT / (arm.script or study.script)
            for p, what in ((config, "config"), (script, "entry point")):
                if not p.exists():
                    raise SystemExit(f"[launch] {name}/{arm_name}: missing {what} {p}")
            sets = (*study.common, *arm.sets)
            budget = _budget(study, config, sets, epochs)
            runs.append(Run(
                name=f"{name}_{arm_name}_s{seed}",
                script=arm.script or study.script,
                config=arm.config or study.config,
                train_dir=_train_dir(study, arm, config, sets),
                sets=(*sets, *budget, f"params.seed={seed}"),
                args=arm.args,
                driver=study.driver,
                target_epoch=int(budget[0].split("=")[1]),
                meta={"study": name, "arm": arm_name, "seed": seed},
            ))
    return runs


def queue_runs(studies: list[str], arms: list[str] | None = None,
               seeds: tuple[int, ...] | None = None, epochs: int | None = None) -> list[Run]:
    """Several studies as ONE queue, still seed-major so every wave spreads across all of them.

    Concatenating the studies instead would run each to completion in turn, which is the arm-major
    order the whole scheduler exists to avoid -- just at study granularity, and therefore invisible
    as a per-study shift in whatever the studies are compared on. Within a seed the order is the one
    given on the command line, so a wave carries a spread of every queued condition.
    """
    per_seed: dict[int, list[Run]] = {}
    for name in studies:
        for run in study_runs(name, arms, seeds, epochs):
            per_seed.setdefault(run.meta["seed"], []).append(run)
    return [run for seed in sorted(per_seed) for run in per_seed[seed]]


def _budget(study: Study, config: Path, sets: tuple[str, ...],
            epochs: int | None) -> tuple[str, ...]:
    """The two overrides that put a run's budget and its checkpoints on window boundaries.

    **max_epochs is a whole number of windows.** A window's metrics (`quality/R_mean`, `clone/*`,
    `build/*`) are written by the resample at its END, so a budget that stops mid-window silently
    drops that window: 3000 epochs is 47 windows plus 39 orphan epochs, and metric 1 would run to
    w=46 while ADR-0021 promises 48. `windows * epochs_per_window` is what the ADR actually means.

    **save_frequency is one window, not the config's 50.** The ladder and specialization passes read
    checkpoints AT boundaries (w=8/28/47); at 50 epochs against 63 the nearest save is 4-14 epochs
    into the *following* window, one generator update short of its own label. A window-aligned
    cadence also caps what the gym-rebuild crash can cost at one window, which is D24's reason for
    the same rule on the `codesigner` path. Costs ~620 MB/run in checkpoints.
    """
    per_window = _window_epochs(config, sets)
    if not per_window:                       # no generator: nothing to align to, leave cadence alone
        n = epochs or study.epochs
        if not n:
            raise SystemExit("[launch] this study never resamples, so its budget cannot be derived "
                             "from windows: set Study.epochs or pass --epochs")
        return (f"params.config.max_epochs={n}",)
    n = epochs or (study.windows * per_window)
    if n % per_window:
        print(f"[warn] max_epochs={n} is not a whole number of {per_window}-epoch windows; the "
              f"final {n % per_window} epochs form a window whose metrics are never written",
              flush=True)
    return (f"params.config.max_epochs={n}", f"params.config.save_frequency={per_window}")


# ── the pool ──────────────────────────────────────────────────────────

class _Job:
    __slots__ = ("run", "slot", "proc", "logf", "attempt", "t0", "resumed_from")

    def __init__(self, run: Run, slot: _Slot):
        self.run, self.slot = run, slot
        self.proc = self.logf = None
        self.attempt, self.t0 = 0, 0.0
        self.resumed_from = None

    def log_path(self, log_dir: Path) -> Path:
        return log_dir / f"{self.run.name}.attempt{self.attempt}.log"


def _classify_codesigner(run: Run) -> tuple[str, Path | None, str]:
    """`_classify_pending` for a run the package owns, which knows the same three states by
    different files.

    `latest.ckpt` is what `rundir.holds_run` keys off, so its presence is exactly "this directory
    can be continued" -- and its ABSENCE in a non-empty directory is the same refusal this launcher
    already makes, raised by `RunDirectory` itself, so blocking here only saves the slot.

    Done is read from `metrics.json`, whose provenance gains a `finished` stamp when the record is
    closed out -- a text file rather than the position inside `latest.ckpt`, so classifying a queue
    of 16 runs does not load 16 torch checkpoints. It has to be detected at all because resuming a
    finished run is a hard error in the package, which the pool would otherwise burn three attempts
    discovering.
    """
    if not run.run_dir.exists() or not any(run.run_dir.iterdir()):
        return "fresh", None, ""
    if not (run.run_dir / LATEST).exists():
        return "blocked", None, f"run dir exists with no {LATEST}: {run.run_dir}"
    record = run.run_dir / RECORD
    if record.exists():
        finished = json.loads(record.read_text()).get("provenance", {}).get("finished")
        if finished:
            return "done", None, f"{RECORD} finished at {finished}"
    return "resume", None, f"continuing the run in {run.run_dir.name}"


def _classify_pending(run: Run) -> tuple[str, Path | None, str]:
    """(state, checkpoint, note) for a run before it is scheduled. `blocked` is deliberate: a run dir
    with no checkpoint cannot be resumed AND must not be relaunched into, because rl_games would open
    a second event file in the same summaries dir and the two partial runs would scrape as one."""
    if run.driver == "codesigner":
        return _classify_codesigner(run)
    if not run.run_dir.exists():
        return "fresh", None, ""
    ckpt = _newest_ckpt(run.nn_dir)
    if ckpt is None:
        if run.restartable:
            return "fresh", None, "run dir exists with no checkpoint; relaunching from the warm start"
        return "blocked", None, f"run dir exists with no checkpoint: {run.run_dir}"
    ep = _ckpt_epoch(ckpt)
    if run.target_epoch and ep >= run.target_epoch - 1:
        return "done", ckpt, f"epoch {ep} >= target {run.target_epoch}"
    return "resume", ckpt, f"from epoch {ep}"


def run_pool(runs: list[Run], slots: list[_Slot], log_dir: Path, *, poll: float = 20.0,
             stagger: float = 30.0, max_attempts: int = 3, timeout: float | None = None,
             heartbeat: float = 300.0, redo: bool = False) -> dict:
    """Run `runs` across `slots`, in queue order, resuming crashes. Returns per-run final status.

    Never purges checkpoints: unlike a tuning trial's, a paper run's checkpoints ARE the artifact --
    the ladder and specialization passes read them.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    manifest = open(log_dir / "manifest.jsonl", "a")
    status: dict[str, str] = {}

    def _record(run: Run, state: str, **extra):
        status[run.name] = state
        manifest.write(json.dumps({"t": time.strftime("%F %T"), "run": run.name,
                                   "state": state, **run.meta, **extra}) + "\n")
        manifest.flush()

    queue: list[tuple[Run, Path | None]] = []
    for run in runs:
        state, ckpt, note = _classify_pending(run)
        if state == "done":
            if not redo:
                print(f"[skip] {run.name}: {note}", flush=True)
                status[run.name] = "done"
                continue
            state = "resume"                 # --redo continues it; it never restarts it from zero
        if state == "blocked":
            print(f"[BLOCK] {run.name}: {note}\n"
                  f"        move or delete that dir yourself, then re-run; the launcher will not "
                  f"touch it", flush=True)
            _record(run, "blocked", note=note)
            continue
        # A warm start (specialization) is the run's own `checkpoint`; the run's OWN progress
        # overrides it, since a dir that already has checkpoints must be continued rather than
        # relaunched -- a fresh process would open a second event stream in the same summaries dir.
        start = ckpt if state == "resume" else (Path(run.checkpoint) if run.checkpoint else None)
        if state == "resume":
            print(f"[resume] {run.name}: {note}", flush=True)
        queue.append((run, start))

    print(f"[launch] {len(queue)} run(s) on {len(slots)} slot(s): "
          + ", ".join(f"{s.name}={s.device}" for s in slots), flush=True)

    free, jobs = list(slots), {}
    idx, last_launch, last_beat = 0, 0.0, time.time()

    def _start(job: _Job, checkpoint: Path | None):
        header = (f"# {job.run.name}  attempt {job.attempt}  slot {job.slot.name} "
                  f"({job.slot.device})\n"
                  f"# {' '.join(job.run.cmd(str(checkpoint) if checkpoint else None))}\n"
                  + (f"# STARTING FROM {checkpoint}\n" if checkpoint else "") + "\n")
        job.proc, job.logf = _launch(job.run.cmd(str(checkpoint) if checkpoint else None),
                                     job.slot.device, job.log_path(log_dir), header)
        job.t0 = time.time()
        job.resumed_from = checkpoint

    def _release(job: _Job, note: str):
        if job.logf:
            job.logf.write(f"\n# {note}\n")
            job.logf.close()
            job.logf = None

    def _retire(job: _Job, state: str, note: str):
        _release(job, note)
        _record(job.run, state, note=note, minutes=round((time.time() - job.t0) / 60, 1))
        print(f"[{state}] {job.run.name} ({note})", flush=True)
        free.append(job.slot)
        jobs.pop(job.run.name, None)

    try:
        while idx < len(queue) or jobs:
            while free and idx < len(queue) and time.time() - last_launch >= stagger:
                run, start = queue[idx]
                job = _Job(run, free.pop(0))
                jobs[run.name] = job
                _record(run, "started", slot=job.slot.name)
                print(f"[start] {run.name} -> {job.slot.name}"
                      + (f" (from {start.name})" if start else ""), flush=True)
                _start(job, start)
                idx += 1
                last_launch = time.time()

            for name, job in list(jobs.items()):
                ret = job.proc.poll()
                if ret is None:
                    if timeout and time.time() - job.t0 > timeout:
                        _kill(job.proc)
                        _retire(job, "timeout", f"exceeded {timeout / 3600:.1f}h")
                    continue
                if ret == 0:
                    _retire(job, "done", "exit 0")
                    continue
                cause = _classify_exit(job.log_path(log_dir))
                if cause == "oom":
                    # Not bad luck: these settings do not fit this slot. Relaunching unchanged just
                    # burns the slot again.
                    _retire(job, "oom", "out of memory -- not relaunched")
                    continue
                if job.attempt + 1 >= max_attempts:
                    _retire(job, "failed", f"exit {ret} ({cause}), {max_attempts} attempts spent")
                    continue
                # rl_games' nn/ exists under a codesigner run too (its save_frequency is still
                # set), so the checkpoint has to be suppressed by DRIVER rather than by absence --
                # handing that path back would turn an internal resume into a warm start the entry
                # point does not even take an argument for.
                ckpt = None if job.run.driver == "codesigner" else _newest_ckpt(job.run.nn_dir)
                _release(job, f"exit {ret} ({cause}) -> "
                              + (f"resume from {ckpt.name}" if ckpt else
                                 "relaunching" if job.run.driver == "codesigner" else
                                 "no checkpoint, retrying"))
                job.attempt += 1
                _record(job.run, "retry", attempt=job.attempt, cause=cause,
                        ckpt=ckpt.name if ckpt else None)
                print(f"[retry] {name} attempt {job.attempt} ({cause})", flush=True)
                _start(job, ckpt)

            if heartbeat and time.time() - last_beat >= heartbeat:
                live = ", ".join(f"{j.slot.name}:{n.split('_', 1)[1]} "
                                 f"{(time.time() - j.t0) / 60:.0f}m" for n, j in jobs.items())
                print(f"[{time.strftime('%H:%M')}] {idx}/{len(queue)} launched | {live or 'idle'}",
                      flush=True)
                last_beat = time.time()

            nap = poll
            if free and idx < len(queue):
                nap = min(nap, max(1.0, stagger - (time.time() - last_launch)))
            time.sleep(nap)
    except (KeyboardInterrupt, SystemExit):
        # Kill the cohort, then leave every in-flight run recorded as interrupted. Re-invoking the
        # launcher picks each one up from its own newest checkpoint -- nothing is re-enqueued here,
        # because the queue is fixed and recoverable from the run dirs.
        for job in list(jobs.values()):
            _kill(job.proc)
            _release(job, "INTERRUPTED")
            _record(job.run, "interrupted")
        print(f"\n[launch] interrupted; {len(jobs)} in-flight run(s) killed. Re-run the same "
              f"command to resume them from their checkpoints.", flush=True)
        raise
    finally:
        manifest.close()
    return status


# ── entry point ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Launch a paper experiment's runs across GPU slots")
    ap.add_argument("studies", nargs="+", choices=sorted(STUDIES), metavar="STUDY",
                    help="one or more experiments; several queue as one mixed pool")
    ap.add_argument("--arms", help="comma-separated subset of the study's arms (default: all)")
    ap.add_argument("--seeds", help="comma-separated seeds (default: 42-49)")
    ap.add_argument("--epochs", type=int,
                    help=f"override max_epochs (default: {WINDOWS} windows' worth)")
    ap.add_argument("--slots", default=None,
                    help="'auto' (every idle device), an int N, or a comma-separated device list")
    ap.add_argument("--allow-busy", action="store_true",
                    help="schedule onto devices that already carry a foreign compute app")
    ap.add_argument("--timeout-hours", type=float, default=None,
                    help="kill a run that outlives this (default: no cap)")
    ap.add_argument("--redo", action="store_true",
                    help="continue runs already at their target epoch (e.g. after raising --epochs); "
                         "never restarts one from zero -- move its dir aside for that")
    ap.add_argument("--dry-run", action="store_true", help="print the queue and the plan, launch nothing")
    args = ap.parse_args()

    studies = list(dict.fromkeys(args.studies))       # de-dup, keep the order given
    if args.arms and len(studies) > 1:
        ap.error("--arms selects arms of ONE study; an arm name does not identify a run across a set")

    seeds = tuple(int(x) for x in args.seeds.split(",")) if args.seeds else None
    runs = queue_runs(studies, arms=args.arms.split(",") if args.arms else None,
                      seeds=seeds, epochs=args.epochs)
    label = "+".join(studies)

    if args.dry_run:
        print(f"[dry-run] {label}: {len(runs)} runs, queue order (seed-major, arm-minor)\n")
        for run in runs:
            state, _, note = _classify_pending(run)
            print(f"  {state:8} {run.name:28} -> {run.train_dir}")
            if note:
                print(f"           {note}")
        print(f"\n  command for {runs[0].name}:\n    "
              + " ".join(runs[0].cmd(runs[0].checkpoint)))
        return

    slots = _make_slots(parse_slot_spec(args.slots), allow_busy=args.allow_busy)
    log_dir = _ROOT / "logs" / "paper" / label
    try:
        status = run_pool(runs, slots, log_dir,
                          timeout=args.timeout_hours * 3600 if args.timeout_hours else None,
                          redo=args.redo)
    except KeyboardInterrupt:
        sys.exit(130)

    tally = {}
    for st in status.values():
        tally[st] = tally.get(st, 0) + 1
    print(f"\n[launch] {label}: " + ", ".join(f"{n} {s}" for s, n in sorted(tally.items())))
    bad = [n for n, s in status.items() if s not in ("done",)]
    if bad:
        print(f"[launch] not complete: {', '.join(sorted(bad))}")
        sys.exit(1)


if __name__ == "__main__":
    main()
