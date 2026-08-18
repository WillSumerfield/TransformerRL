"""Metric 5 (specialized return): strip the scaffolding, fine-tune control on the committed body
alone, and measure what that body is worth.

    python -m experiments.harness.specialize clone prepare   # committed bodies -> warm-start ckpts
    python -m experiments.harness.specialize clone launch    # the 250-epoch fine-tunes
    python -m experiments.harness.specialize clone measure   # mu rollout -> data/paper/spec_clone.npz

Three phases because the middle one is a training run and the outer two are not. The reason `prepare`
exists at all is that "warm-start on the committed body" is **not** expressible in overrides:

  * `resample_interval: 0` does not put the run on the committed body. Restoring a checkpoint calls
    `env.resample(...)` onto the checkpoint's own sampled population (`codesign_agent.py:993-996`),
    before any interval check — correct for a crash resume, wrong here. Left alone, metric 5 would
    fine-tune on the window's *population* and silently measure something else entirely.
  * `max_epochs: 250` stops the run after one epoch, because the checkpoint carries `epoch_num` and
    training halts at `epoch_num >= max_epochs`.

So `prepare` writes a **doctored copy** of the training checkpoint whose current-design arrays are the
committed body repeated across every env, and `launch` sets `max_epochs = ckpt_epoch + 250`. The
doctored copy goes to `data/paper/spec/`; the training checkpoint is never modified. Nothing new is
added to the trainer — the install rides the restore path the trainer already has, which is also the
path that would otherwise be the bug.

**The committed body is the MODAL greedy design, not one greedy draw.** `net.sample(N, 'greedy')` is
argmax at every step, but the MDP visits growable limbs in random order, so draws differ (`eval.py`
reports `best_n_unique` for this reason). The mode is stable across that ordering noise; a single draw
would be an arbitrary member of a family. `modal_share` is recorded so a generator with no real mode
is visible rather than assumed away.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from experiments.harness.launch import (            # noqa: E402
    STUDIES, Run, _window_epochs, run_pool, study_runs,
)
from experiments.harness.slots import _make_slots, _newest_ckpt, _ckpt_epoch, parse_slot_spec  # noqa: E402

CKPT_WINDOWS = (8, 28, 47)      # pretrain->RL boundary, mid-RL, final (ADR-0021's ladder points)
FINETUNE_EPOCHS = 250           # chosen to APPROACH the fixed-body ceiling, not to avoid it
SPEC_ROOT = _ROOT / "data" / "paper" / "spec"

# The strip. FD and FK are per-step control-side terms fused into the PPO loss, so they do not switch
# off with the generator and must be named explicitly (ADR-0021).
STRIP = ("params.config.resample_interval=0",
         "params.config.fd.enabled=false", "params.config.fk.enabled=false")


# ---- the training checkpoints being specialized ------------------------------------

def _boundary_ckpt(run: Run, epoch: int) -> Path | None:
    """The training checkpoint saved at the end of `epoch`. Exact, not nearest: `launch.py` sets the
    save cadence to one window, so a boundary save either exists or the run did not get that far."""
    hits = [p for p in run.nn_dir.glob(f"last_*_ep_{epoch}_*.pth")]
    return hits[0] if hits else None


def _targets(study: str, windows=CKPT_WINDOWS):
    """[(train_run, window, epoch, ckpt_path_or_None)] for every (run x ladder point) in a study."""
    out = []
    for run in study_runs(study):
        per_window = _window_epochs(_ROOT / run.config, run.sets)
        if not per_window:
            raise SystemExit(f"[spec] {study} never resamples, so it has no committed body")
        for w in windows:
            epoch = w * per_window
            out.append((run, w, epoch, _boundary_ckpt(run, epoch)))
    return out


def _spec_name(run: Run, w: int) -> str:
    return f"spec_{run.name}_w{w}"


def _warm_path(study: str, run: Run, w: int) -> Path:
    return SPEC_ROOT / study / f"{_spec_name(run, w)}.pth"


# ---- phase 1: committed body -> doctored warm-start checkpoint ---------------------

def prepare(study: str, *, population: int, windows=CKPT_WINDOWS, device="cuda:0") -> list[dict]:
    """Write one warm-start checkpoint per (run x ladder point), each holding the run's committed body.

    Opens a **minimal** task — the sim is needed only for the Task's published `obs_layout`, which is
    what sizes the network; sampling designs touches no simulator. Tiling to the training run's
    `num_actors` is what makes the restore-time `env.resample` land on the right number of bodies.
    """
    from experiments.harness.evalpass import load_net, modal_design, open_task, sample_bodies

    targets = _targets(study, windows)
    missing = [(r.name, w) for r, w, _, c in targets if c is None]
    if missing:
        print(f"[spec] {len(missing)} boundary checkpoint(s) absent, skipped: "
              + ", ".join(f"{n}@w{w}" for n, w in missing[:6])
              + (" ..." if len(missing) > 6 else ""), flush=True)
    targets = [t for t in targets if t[3] is not None]
    if not targets:
        raise SystemExit(f"[spec] no boundary checkpoints found for '{study}' — run it first")

    dev = torch.device(device)
    cfg = yaml.safe_load((targets[0][0].run_dir / "config.yaml").read_text())
    env, _, layout = open_task(cfg, 8, device=dev)       # 8 envs: the layout is what is wanted here
    print(f"[spec] {len(targets)} checkpoint(s); greedy population {population}", flush=True)

    records = []
    for run, w, epoch, ckpt in targets:
        run_cfg = yaml.safe_load((run.run_dir / "config.yaml").read_text())
        n_envs = int(run_cfg["params"]["config"]["num_actors"])
        net, _ = load_net(ckpt, run_cfg, layout, dev)
        pop = sample_bodies(net, population, "greedy")
        idx, stats = modal_design(pop)
        counts = pop["counts"][idx].long().cpu()
        eff, cap = pop["eff_sub"][idx].cpu(), pop["cap_sub"][idx].cpu()

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "cur_eff" not in state:
            raise SystemExit(f"[spec] {ckpt.name} predates the subtype axis; restore would skip the "
                             f"body install and train on the seed body")
        tile = lambda t: t.unsqueeze(0).expand(n_envs, *t.shape).clone()
        state["cur_counts"], state["cur_eff"], state["cur_cap"] = tile(counts), tile(eff), tile(cap)
        # No trace: these bodies were not sampled in a window, and a trace describing the population
        # they replaced would be a lie. Nothing reads it once `resample_interval` is 0 — the resample
        # update that would is never reached.
        state["cur_trace"] = None
        # Let the fine-tune write its own best-mean save: `last_mean_rewards` is the training run's
        # best, and keeping it would suppress every best-save in a run that starts from scratch on a
        # single body. The final save is guaranteed by the max-epochs exit either way.
        state["last_mean_rewards"] = -1e9
        out = _warm_path(study, run, w)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, out)

        rec = {"run": run.name, "spec_run": _spec_name(run, w), **run.meta, "window": w,
               "epoch": epoch, "source": str(ckpt.relative_to(_ROOT)),
               "warm_start": str(out.relative_to(_ROOT)), "n_envs": n_envs,
               "counts": counts.tolist(), "eff_sub": eff.tolist(), "cap_sub": cap.tolist(),
               "n_limbs": int((counts > 0).sum()), "n_modules": int(counts.sum()), **stats}
        records.append(rec)
        print(f"[spec] {rec['spec_run']}: {rec['n_limbs']} limbs / {rec['n_modules']} modules, "
              f"modal_share={stats['modal_share']:.2f} ({stats['n_unique']} unique)", flush=True)
        del net
        torch.cuda.empty_cache()

    manifest = SPEC_ROOT / study / "designs.json"
    manifest.write_text(json.dumps(records, indent=1))
    print(f"[spec] {len(records)} warm-start checkpoint(s) -> {manifest.parent}", flush=True)
    return records


# ---- phase 2: the fine-tune runs ---------------------------------------------------

def spec_runs(study: str, windows=CKPT_WINDOWS, epochs: int = FINETUNE_EPOCHS) -> list[Run]:
    """The fine-tune queue, ordered window-major so the arms stay interleaved within each ladder
    point — the same confound argument as the training queue, and the ladder points are also the
    natural checkpoint at which to stop and look."""
    runs = []
    for w in windows:
        for run, ww, epoch, _ in _targets(study, (w,)):
            warm = _warm_path(study, run, w)
            if not warm.exists():
                continue
            target = epoch + epochs
            runs.append(Run(
                name=_spec_name(run, w),
                script=run.script, config=run.config, train_dir=run.train_dir,
                # `max_epochs` is an OFFSET: restoring sets epoch_num to the source epoch.
                # `save_frequency` is coarse on purpose -- 96 of these runs exist and the final save
                # is guaranteed by the max-epochs exit, so periodic saves are only crash insurance.
                sets=(*STRIP, f"params.config.max_epochs={target}",
                      f"params.config.save_frequency={epochs}", f"params.seed={run.meta['seed']}"),
                checkpoint=str(warm),
                target_epoch=target,
                restartable=True,        # a crash before the first save relaunches from the warm start
                meta={**run.meta, "window": w, "phase": "specialize"},
            ))
    return runs


# ---- phase 3: measure --------------------------------------------------------------

def measure(study: str, *, envs: int, episodes: int, windows=CKPT_WINDOWS,
            device="cuda:0") -> Path:
    """Roll out each fine-tuned policy at mu on its own committed body; write the metric-5 artifact.

    Measured with the ladder's level-0 rollout rather than read off the training reward, so metric 5
    lands on the same axis as metric 3's level 0 and as `eval.py`'s numbers. Every env carries the
    identical body, so the across-env spread IS this metric's noise floor, measured in-band.
    """
    from experiments.harness.evalpass import install_design, load_net, open_task, rollout

    designs = {r["spec_run"]: r
               for r in json.loads((SPEC_ROOT / study / "designs.json").read_text())}
    arms = list(STUDIES[study].arms)
    seeds = list(STUDIES[study].seeds)
    shape = (len(arms), len(seeds), len(windows))
    ret = np.full(shape, np.nan)
    sd = np.full(shape, np.nan)
    fell = np.full(shape, np.nan)
    src_epoch = np.zeros(shape, int)

    dev = torch.device(device)
    env = layout = None
    for run in spec_runs(study, windows):
        rec = designs.get(run.name)
        final = _newest_ckpt(run.nn_dir)
        if rec is None or final is None:
            print(f"[spec] no finished run for {run.name}", flush=True)
            continue
        cfg = yaml.safe_load((run.run_dir / "config.yaml").read_text())
        if env is None:
            env, _, layout = open_task(cfg, envs, device=dev)
        net, obs_norm = load_net(final, cfg, layout, dev)
        install_design(env, torch.tensor(rec["counts"]), torch.tensor(rec["eff_sub"]),
                       torch.tensor(rec["cap_sub"]))
        r = rollout(net, obs_norm, env, dev, episodes=episodes, label=run.name)
        i = (arms.index(run.meta["arm"]), seeds.index(run.meta["seed"]),
             list(windows).index(run.meta["window"]))
        ret[i], sd[i] = float(r["return"].mean()), float(r["return"].std())
        fell[i] = float(r["fall_rate"].mean())
        src_epoch[i] = rec["epoch"]
        print(f"[spec] {run.name}: spec={ret[i]:.1f} +-{sd[i]:.1f} (noise floor), "
              f"fall={fell[i]:.2f}, from epoch {_ckpt_epoch(final)}", flush=True)
        del net, obs_norm
        torch.cuda.empty_cache()

    out = _ROOT / "data" / "paper" / f"spec_{study}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, spec=ret, spec_sd=sd, fall_rate=fell, src_epoch=src_epoch,
             arms=np.array(arms), seeds=np.array(seeds), windows=np.array(windows),
             episodes=episodes, envs=envs)
    print(f"[spec] {int(np.isfinite(ret).sum())}/{ret.size} cells -> {out}", flush=True)
    return out


# ---- entry point -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Metric 5: specialize control onto the committed body")
    ap.add_argument("study", choices=sorted(STUDIES))
    ap.add_argument("phase", choices=["prepare", "launch", "measure"])
    ap.add_argument("--windows", default=",".join(str(w) for w in CKPT_WINDOWS),
                    help="ladder points to specialize at (default 8,28,47)")
    ap.add_argument("--population", type=int, default=512,
                    help="greedy draws the committed body is the mode of (prepare)")
    ap.add_argument("--epochs", type=int, default=FINETUNE_EPOCHS, help="fine-tune length (launch)")
    ap.add_argument("--envs", type=int, default=256, help="identical bodies to measure over")
    ap.add_argument("--episodes", type=int, default=32, help="episodes per env (measure)")
    ap.add_argument("--slots", default=None, help="'auto', an int N, or a device list (launch)")
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    windows = tuple(int(w) for w in args.windows.split(","))

    if args.phase == "prepare":
        if args.dry_run:
            for run, w, epoch, ckpt in _targets(args.study, windows):
                print(f"  w{w:<3} epoch {epoch:<5} {run.name:28} "
                      f"{ckpt.name if ckpt else 'MISSING'}")
            return
        prepare(args.study, population=args.population, windows=windows)
        return

    if args.phase == "measure":
        measure(args.study, envs=args.envs, episodes=args.episodes, windows=windows)
        return

    runs = spec_runs(args.study, windows, args.epochs)
    if not runs:
        raise SystemExit(f"[spec] no warm-start checkpoints for '{args.study}' — run `prepare` first")
    if args.dry_run:
        print(f"[dry-run] {len(runs)} fine-tune(s), window-major\n")
        for run in runs:
            print(f"  {run.name:34} -> epoch {run.target_epoch}  (from {Path(run.checkpoint).name})")
        print(f"\n  command for {runs[0].name}:\n    " + " ".join(runs[0].cmd(runs[0].checkpoint)))
        return
    slots = _make_slots(parse_slot_spec(args.slots), allow_busy=args.allow_busy)
    try:
        run_pool(runs, slots, _ROOT / "logs" / "paper" / f"spec_{args.study}")
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
