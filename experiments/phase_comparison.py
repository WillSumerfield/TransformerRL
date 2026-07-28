"""Phase-comparison harness (ADR-0015): run the codesign algorithm at the fixed budget and emit one
frozen per-phase artifact `data/phase_comparison/<PHASE>.npz` that the shared notebook overlays.

Each phase, on its branch, sets PHASE + points SCRIPT/CONFIG at that phase's algorithm, then commits
its `.npz`. Phase 0 = the current presence-only ant codesign = the baseline series.

Measured (fixed env-step budget = 3000 epochs of the Phase-0 config, 5 seeds; all OUTPUTS):
  performance  eval-time det-mu control return on the converged generator, 3 views:
               top choice (mode/argmax-likelihood body), distribution average, top-K best bodies.
  runtime      perf/steps_per_sec (throughput), perf/peak_mem_mib.
  convergence  quality (R_mean -> 90% plateau) + morphology (per-limb presence stable, <eps).
  diversity    d_comp / d_struct / N_modes, within-run + between-seed (experiments/diversity.py).

Full run:  python experiments/phase_comparison.py
Dry run:   python experiments/phase_comparison.py --seeds 42 --max-epochs 3 --num-envs 256
Re-scrape: python experiments/phase_comparison.py --skip-train        # eval+scrape existing runs
"""
import os; os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import re
import sys
import shutil
import argparse
import subprocess
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from experiments.diversity import within_run_metrics, between_seed_metrics, counts_to_repr

# ---- phase identity (the ONE block a future phase edits) -------------------------
PHASE = "phase3_substrate"
BRANCH = "phase-3"             # harness aborts unless current git branch == this (stale-PHASE tripwire)
SCRIPT = "scripts/train_ant_codesign_single.py"
CONFIG = "configs/ppo_ant_codesign_single.yaml"   # +RoPE +AdamW-WD +short LR-warmup baked; FD+FK aux ON
RUN = "runs/ant_codesign/codesign_single_transformer/{name}"   # name = phase3_s{seed}
NAME = lambda seed: f"phase3_s{seed}"


def _git(args):
    return subprocess.check_output(["git", *args], cwd=_ROOT, text=True).strip()

DATA_DIR = _ROOT / "data" / "phase_comparison"
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
MAX_EPOCHS = 3000                 # the budget (Phase-0 config default; realized env-steps recorded)

# eval-pass knobs
N_SAMPLE = 4096                   # bodies drawn from the converged generator (dist-avg + diversity pop)
TOPK = 10                         # top-K best-performing generated bodies
MAX_BODIES = 128                  # cap distinct bodies actually built+rolled (most-frequent; bounds envs)
EVAL_EPM = 128                    # eval envs per body (episode-averaged -> low variance)
EVAL_SEED = 123
_LIMBS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]

# convergence knobs (ADR-0015; tune after first runs)
CONV_FRAC = 0.90                  # quality: fraction of final plateau
EPS_MORPH = 0.05                  # morphology: max per-slot |on-rate - final| tolerance
PLATEAU_TAIL = 5                  # windows averaged for the "final" plateau value

# curves stored per-seed for the notebook to overlay (step + val)
_FPS_TAG = "performance/step_inference_rl_update_fps"   # rl_games native throughput (env-steps/sec)
CURVE_TAGS = ["quality/R_mean", "quality/R_std", "build/limbcount", "build/modulecount",
              "rewards/step", _FPS_TAG, "perf/peak_mem_mib",
              "losses/fd", "losses/fk"]        # phase-2 aux diagnostics (absent on phase-0/1 overlays)


# ---- 1. training -----------------------------------------------------------------

def train_all(seeds, max_epochs, num_envs):
    for seed in seeds:
        name = NAME(seed)
        assert name.startswith("phase3_"), "safety: only ever delete phase3_* run dirs"
        run_dir = _ROOT / RUN.format(name=name)
        if run_dir.exists():
            shutil.rmtree(run_dir)                      # always retrain (only this exact phase2_ dir)
        # NB: no --timing -- it inserts cuda.synchronize()/epoch that kills GPU pipelining (~2x
        # slower). Throughput comes from rl_games' native performance/* tag; peak-mem logs passively.
        cmd = [sys.executable, str(_ROOT / SCRIPT), "train", "--headless", "True",
               "--seed", str(seed), "--name", name,
               "--max_epochs", str(max_epochs if max_epochs is not None else MAX_EPOCHS)]
        if num_envs is not None:
            cmd += ["--num_envs", str(num_envs)]
        print(f"\n[phase] TRAIN {name}: {' '.join(cmd)}", flush=True)
        # Retry the known intermittent gym-rebuild crash (docs/troubleshooting/resample_rebuild_crash.md) so one bad
        # resample doesn't abort a multi-hour multi-seed sweep. rmtree between attempts = fresh run.
        for attempt in range(1, 4):
            if subprocess.run(cmd, cwd=_ROOT).returncode == 0:
                break
            print(f"[phase] {name} attempt {attempt} FAILED", flush=True)
            if attempt == 3:
                raise RuntimeError(f"{name} failed 3x")
            if run_dir.exists():
                shutil.rmtree(run_dir)
            time.sleep(30)                                 # let the dying process release VRAM


# ---- 2. eval pass: converged generator -> 3 performance views + diversity ---------

def _final_ckpt(run_dir: Path) -> Path | None:
    nn = run_dir / "nn"
    if not nn.exists():
        return None
    epoched = [(int(m.group(1)), p) for p in nn.glob("*_ep_*.pth")
               if (m := re.search(r"_ep_(\d+)", p.name))]
    if epoched:
        return max(epoched)[1]                          # highest-epoch checkpoint
    pths = sorted(nn.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    return pths[-1] if pths else None                   # else newest .pth


def eval_all(seeds):
    import gc
    import torch
    import yaml
    import vlearn as v
    from experiments.ppg_parity import _load_policy, _rollout_return
    from envs.ant_envs.ant_multimorph import (AntMultiMorphEnv, _OBS_TOTAL as OBS_BASE,
                                              _MASK_DIM as MASK_DIM, _N_DOFS_FULL as N_ACT)
    from envs.ant_envs.build_vsim import Morphology

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    cfg = yaml.safe_load((_ROOT / CONFIG).read_text())
    value_size = int(cfg.get("env", {}).get("value_size", 1))
    net_params = cfg["params"]["network"]

    scal = {k: [] for k in ("perf_top", "perf_distavg", "perf_topk",
                            "div_comp", "div_comp_norm", "div_struct", "div_nmodes")}
    dominants, used_seeds = [], []
    for seed in seeds:
        ckpt = _final_ckpt(_ROOT / RUN.format(name=NAME(seed)))
        if ckpt is None:
            print(f"[phase] WARN no checkpoint for {NAME(seed)}"); continue
        net, obs_norm = _load_policy(ckpt, net_params, device, value_size=value_size,
                                     obs_base=OBS_BASE, n_act=N_ACT)

        with torch.no_grad():                            # sample the converged generator
            mods = net.net.sample(N_SAMPLE)["counts"].cpu().numpy()   # (N,8) per-limb module count 0..MAX
        patterns, counts = np.unique(mods, axis=0, return_counts=True)
        keep = np.argsort(counts)[::-1][:MAX_BODIES]     # most-frequent distinct bodies (mode first)
        kept_pat, kept_cnt = patterns[keep], counts[keep].astype(float)

        bodies = [Morphology.from_counts({i + 1: int(c) for i, c in enumerate(row) if c > 0})
                  for row in kept_pat]                   # 1-based limb ids; count 0 = absent
        env = AntMultiMorphEnv(len(bodies) * EVAL_EPM, device, morphologies=bodies,
                               sample_morphs=False, rendering=False, raise_exception=False,
                               seed=EVAL_SEED, with_window=False, value_size=value_size)
        epm = env.envs_per_morph
        ep = _rollout_return(net, obs_norm, env, device,
                             obs_base=OBS_BASE, mask_dim=MASK_DIM)    # (len*epm,) raw returns
        per_body = np.array([ep[i * epm:(i + 1) * epm].mean() for i in range(len(bodies))])

        w = kept_cnt / kept_cnt.sum()
        scal["perf_distavg"].append(float((w * per_body).sum()))     # freq-weighted mean
        scal["perf_top"].append(float(per_body[0]))                  # mode / argmax-likelihood body
        scal["perf_topk"].append(float(np.sort(per_body)[::-1][:TOPK].mean()))

        wm = within_run_metrics([counts_to_repr(p) for p in mods])   # full N sample (dedup inside)
        for k in ("div_comp", "div_comp_norm", "div_struct", "div_nmodes"):
            scal[k].append(wm[k])
        dominants.append(counts_to_repr(kept_pat[0]))                # dominant = mode body
        used_seeds.append(seed)
        print(f"[phase] eval {NAME(seed)}: top={scal['perf_top'][-1]:.1f} "
              f"distavg={scal['perf_distavg'][-1]:.1f} top{TOPK}={scal['perf_topk'][-1]:.1f} "
              f"| {len(patterns)} distinct, nmodes={wm['div_nmodes']:.2f}", flush=True)
        # vsim gym is a licensed per-process singleton: must tear down before the next seed's
        # create_gym (else "License validation failed"). Drain in-flight work first, mirroring
        # AntMultiMorphEnv._rebuild, so delete_gym doesn't free vsim buffers under a live async op.
        torch.cuda.synchronize()
        for _fn in ("end_streaming", "_check_for_cuda_errors"):
            try: getattr(env.gym, _fn)()
            except Exception: pass
        env.gym = None; del env; gc.collect()
        v.delete_gym()

    out = {f"seed_{k}": np.array(v, dtype=np.float32) for k, v in scal.items()}
    if dominants:
        bm = between_seed_metrics(dominants)             # single set over seeds' dominant bodies
        out.update({f"between_{k}": np.float32(v) for k, v in bm.items()})
    out["eval_seeds"] = np.array(used_seeds)
    return out


# ---- 3. curve scrape: convergence + runtime + overlay curves ---------------------

def _event_file(run_dir: Path) -> Path | None:
    files = sorted((run_dir / "summaries").glob("events.out.tfevents.*"))
    return files[-1] if files else None


def _scalar(ea, tag):
    if tag not in ea.Tags()["scalars"]:
        return None, None
    sc = ea.Scalars(tag)
    return (np.array([s.step for s in sc], dtype=np.int64),
            np.array([s.value for s in sc], dtype=np.float32))


def _quality_conv(steps, vals):
    """First env-step where R_mean reaches CONV_FRAC of its final-plateau mean."""
    if steps is None or len(steps) < PLATEAU_TAIL:
        return np.nan
    plateau = float(vals[-PLATEAU_TAIL:].mean())
    thr = CONV_FRAC * plateau
    hit = np.where(vals >= thr)[0]
    return float(steps[hit[0]]) if len(hit) else np.nan


def _morph_conv(ea):
    """First env-step after which every per-limb on-rate stays within EPS_MORPH of its final value."""
    cols, steps = [], None
    for l in _LIMBS:
        s, v = _scalar(ea, f"build/p/{l}")
        if s is None:
            return np.nan
        steps = s; cols.append(v)
    P = np.stack(cols, axis=1)                           # (T, 8) on-rates over windows
    final = P[-PLATEAU_TAIL:].mean(axis=0)
    dev = np.abs(P - final).max(axis=1)                  # (T,) worst-slot deviation
    below = dev < EPS_MORPH
    # earliest t from which all subsequent windows are below tolerance
    t = len(below)
    while t > 0 and below[t - 1]:
        t -= 1
    return float(steps[t]) if t < len(steps) else np.nan


def scrape_all(seeds):
    out = {"conv_quality": [], "conv_morph": [], "steps_per_sec": [], "peak_mem_mib": [],
           "env_steps": []}
    curves, used = {}, []
    for seed in seeds:
        ev = _event_file(_ROOT / RUN.format(name=NAME(seed)))
        if ev is None:
            print(f"[phase] WARN no summaries for {NAME(seed)}"); continue
        ea = EventAccumulator(str(ev)); ea.Reload()
        qs, qv = _scalar(ea, "quality/R_mean")
        out["conv_quality"].append(_quality_conv(qs, qv))
        out["conv_morph"].append(_morph_conv(ea))
        ss, sv = _scalar(ea, _FPS_TAG)         # rl_games native throughput (no --timing needed)
        out["steps_per_sec"].append(float(np.median(sv[1:])) if sv is not None and len(sv) > 1 else np.nan)
        ms, mv = _scalar(ea, "perf/peak_mem_mib")
        out["peak_mem_mib"].append(float(mv.max()) if mv is not None else np.nan)
        rs, _ = _scalar(ea, "rewards/step")
        out["env_steps"].append(float(rs[-1]) if rs is not None else float(qs[-1]) if qs is not None else np.nan)
        for tag in CURVE_TAGS:
            s, v = _scalar(ea, tag)
            if s is not None:
                key = f"curve__s{seed}__{tag.replace('/', '_')}"
                curves[key + "__step"] = s; curves[key + "__val"] = v
        used.append(seed)
    res = {f"seed_{k}": np.array(v, dtype=np.float32) for k, v in out.items()}
    res["scrape_seeds"] = np.array(used)
    res.update(curves)
    return res


# ---- 4. assemble the frozen artifact ---------------------------------------------

def _agg(d):
    """Add mean/std for every per-seed scalar array (seed_* keys)."""
    agg = {}
    for k, v in d.items():
        if k.startswith("seed_") and v.dtype != np.int64 and v.size and not np.all(np.isnan(v)):
            agg[k.replace("seed_", "mean_")] = np.float32(np.nanmean(v))
            agg[k.replace("seed_", "std_")] = np.float32(np.nanstd(v))
    return agg


def build_artifact(seeds):
    payload = {"phase": PHASE, "seeds": np.array(seeds), "n_sample": N_SAMPLE,
               "topk": TOPK, "max_epochs": MAX_EPOCHS,
               "git_branch": BRANCH, "git_commit": _git(["rev-parse", "--short", "HEAD"])}
    payload.update(eval_all(seeds))
    payload.update(scrape_all(seeds))
    payload.update(_agg(payload))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{PHASE}.npz"
    if path.exists():                                    # never clobber another phase/branch's frozen data
        _prev = np.load(path, allow_pickle=True)
        prev_b = str(_prev["git_branch"]) if "git_branch" in _prev else BRANCH
        assert prev_b == BRANCH, f"{path.name} written on {prev_b!r}; refusing to overwrite from {BRANCH!r}"
    np.savez_compressed(path, **payload)
    print(f"[phase] artifact -> {path}")
    for k in ("perf_top", "perf_distavg", "perf_topk", "conv_quality", "conv_morph",
              "steps_per_sec", "peak_mem_mib", "div_comp", "div_struct", "div_nmodes"):
        m, s = payload.get(f"mean_{k}"), payload.get(f"std_{k}")
        if m is not None:
            print(f"    {k:14s} {float(m):10.3f} ± {float(s):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--max-epochs", type=int, default=None, help="override budget (dry run)")
    ap.add_argument("--num-envs", type=int, default=None, help="override num_actors (dry run)")
    ap.add_argument("--skip-train", action="store_true", help="eval+scrape existing phase1_ runs only")
    a = ap.parse_args()
    cur = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    assert cur == BRANCH, f"harness pinned to {BRANCH!r} but on {cur!r} — edit the identity block for this phase"
    if not a.skip_train:
        train_all(a.seeds, a.max_epochs, a.num_envs)
    build_artifact(a.seeds)


if __name__ == "__main__":
    main()
