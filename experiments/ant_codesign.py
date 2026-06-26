"""Codesign report: train the unconditional morphology generator + combined PPO control on the full
ant, then aggregate the generator/control curves for the notebook (notebooks/ant_codesign.ipynb).

Gate: per-leg presence prob climbs base -> ~all-8, fast and stable (no leg/energy cost yet, so more
legs => more return). See ADR-0010 / temp/ppg_phase2_plan.md.

Runs are namespaced `report_s{seed}` so this NEVER touches pre-existing `s{seed}` codesign dirs;
only `report_*` dirs are (re)created.

Outputs:
  data/ant_codesign/curves.npz   per-run scalar curves (step + value per tag)

Full run:  python experiments/ant_codesign.py
Dry run:   python experiments/ant_codesign.py --seeds 42 --max-epochs 3 --num-envs 256
"""
import os; os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

SCRIPT = "scripts/train_ant_codesign.py"
RUN = "runs/ant_codesign/codesign_transformer/{name}"   # name = report_s{seed}
_LEGS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]    # leg slots 1..8 (compass codes)
CURVE_TAGS = (
    [f"gen_p/{l}" for l in _LEGS]                          # per-leg built rate (gate)
    + [f"gen_marg/{l}" for l in _LEGS]                     # per-leg marginal value (headline)
    + ["built/mean_legcount", "gen/entropy", "gen/fraction",
       "gen/R_mean", "gen/value_R_corr", "gen/vloss",      # body quality R + v-vs-R fit
       "rewards/step",                                     # control skill (mean reward, all bodies)
       "v1/v1_s0", "v1/episode_return", "v1/calibration_gap"]   # R-unbiasedness calibration
)
DATA_DIR = _ROOT / "data" / "ant_codesign"
CONFIG = "configs/ppo_ant_codesign.yaml"
BASE_ENVS = 128             # eval envs for the base-morph rollout (averaged; reset-noise spread)
EVAL_SEED = 123


# ---- 1. training -----------------------------------------------------------------

def train_all(seeds, max_epochs, num_envs):
    for seed in seeds:
        name = f"report_s{seed}"
        run_dir = _ROOT / RUN.format(name=name)
        assert name.startswith("report_"), "safety: only ever delete report_* run dirs"
        if run_dir.exists():
            shutil.rmtree(run_dir)              # always retrain (only this exact report_ dir)
        cmd = [sys.executable, str(_ROOT / SCRIPT), "train",
               "--headless", "True", "--seed", str(seed), "--name", name]
        if max_epochs is not None:
            cmd += ["--max_epochs", str(max_epochs)]
        if num_envs is not None:
            cmd += ["--num_envs", str(num_envs)]
        print(f"\n[codesign] TRAIN {name}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True, cwd=_ROOT)


# ---- 2. curve aggregation (TB summaries) -----------------------------------------

def _event_file(run_dir: Path) -> Path | None:
    files = sorted((run_dir / "summaries").glob("events.out.tfevents.*"))
    return files[-1] if files else None


def aggregate_curves(seeds):
    out = {}
    for seed in seeds:
        name = f"report_s{seed}"
        ev = _event_file(_ROOT / RUN.format(name=name))
        if ev is None:
            print(f"[codesign] WARN no summaries for {name}"); continue
        ea = EventAccumulator(str(ev)); ea.Reload()
        have = set(ea.Tags()["scalars"])
        for tag in CURVE_TAGS:
            if tag not in have:
                continue
            sc = ea.Scalars(tag)
            key = f"s{seed}__{tag.replace('/', '_')}"
            out[key + "__step"] = np.array([s.step for s in sc], dtype=np.int64)
            out[key + "__val"] = np.array([s.value for s in sc], dtype=np.float32)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_DIR / "curves.npz",
                        seeds=np.array(seeds), legs=np.array(_LEGS), **out)
    print(f"[codesign] curves -> {DATA_DIR / 'curves.npz'} ({len(out) // 2} series)")


# ---- 2b. final value-vs-R scatter (per-env, dumped by the agent each window) ------
def aggregate_scatter(seeds):
    """Collect each run's final gen_scatter.npz (v(full) vs R, per env) for notebook panel 4."""
    out = {}
    for seed in seeds:
        f = _ROOT / RUN.format(name=f"report_s{seed}") / "gen_scatter.npz"
        if not f.exists():
            print(f"[codesign] WARN no gen_scatter for report_s{seed}"); continue
        s = np.load(f)
        out[f"s{seed}__v_full"] = s["v_full"]; out[f"s{seed}__R"] = s["R"]
    if out:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DATA_DIR / "gen_scatter.npz", seeds=np.array(seeds), **out)
        print(f"[codesign] gen_scatter -> {DATA_DIR / 'gen_scatter.npz'}")


# ---- 3. offline eval: control performance on the FIXED base morph over checkpoints --------

def _ckpts_by_epoch(run_dir: Path):
    """All `*_ep_{N}.pth` checkpoints, deduped by epoch, sorted -> [(epoch, path), ...]."""
    import re
    seen = {}
    for p in (run_dir / "nn").glob("*_ep_*.pth"):
        m = re.search(r"_ep_(\d+)", p.name)
        if m:
            seen.setdefault(int(m.group(1)), p)   # first match per epoch (e.g. last_*_ep_N)
    return sorted(seen.items())


def base_eval(seeds):
    """Per checkpoint, deterministic-mu rollout of the control policy on the fixed base morph;
    record mean+std raw episode return vs epoch. Isolates control skill on a fixed body."""
    import torch
    import yaml
    from experiments.ppg_parity import _load_policy, _rollout_return   # reuse proven machinery
    from envs.ant_envs.ant_multimorph import AntMultiMorphEnv
    from envs.ant_envs.build_vsim import Morphology

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    cfg = yaml.safe_load((_ROOT / CONFIG).read_text())
    base_legs = tuple(cfg.get("env", {}).get("base_legs", (1, 4, 6)))
    value_size = int(cfg.get("env", {}).get("value_size", 1))   # 2 => V1.0 head (+progress obs dim)
    net_params = cfg["params"]["network"]

    # One fixed-base env, reused across every checkpoint (1 group, BASE_ENVS envs).
    env = AntMultiMorphEnv(BASE_ENVS, device, morphologies=[Morphology.from_legs(base_legs)],
                           sample_morphs=False, rendering=False, raise_exception=False,
                           seed=EVAL_SEED, with_window=False, value_size=value_size)

    out = {}
    for seed in seeds:
        ckpts = _ckpts_by_epoch(_ROOT / RUN.format(name=f"report_s{seed}"))
        if not ckpts:
            print(f"[codesign] WARN no checkpoints for report_s{seed}"); continue
        eps, means, stds = [], [], []
        for epoch, ckpt in ckpts:
            net, obs_norm = _load_policy(ckpt, net_params, device, value_size=value_size)
            ret = _rollout_return(net, obs_norm, env, device)   # (BASE_ENVS,) raw episode return
            eps.append(epoch); means.append(float(ret.mean())); stds.append(float(ret.std()))
            print(f"[codesign] base-eval s{seed} ep {epoch}: return {means[-1]:.1f}", flush=True)
        out[f"s{seed}__epoch"] = np.array(eps, dtype=np.int64)
        out[f"s{seed}__ret_mean"] = np.array(means, dtype=np.float32)
        out[f"s{seed}__ret_std"] = np.array(stds, dtype=np.float32)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_DIR / "base_eval.npz",
                        seeds=np.array(seeds), base_legs=np.array(base_legs), **out)
    print(f"[codesign] base_eval -> {DATA_DIR / 'base_eval.npz'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--max-epochs", type=int, default=None, help="override per-run epochs (dry run)")
    ap.add_argument("--num-envs", type=int, default=None, help="override per-run num_actors (dry run)")
    ap.add_argument("--skip-train", action="store_true", help="aggregate existing report_ runs only")
    ap.add_argument("--skip-eval", action="store_true", help="skip the offline base-morph eval")
    a = ap.parse_args()
    if not a.skip_train:
        train_all(a.seeds, a.max_epochs, a.num_envs)
    aggregate_curves(a.seeds)
    aggregate_scatter(a.seeds)
    if not a.skip_eval:
        base_eval(a.seeds)


if __name__ == "__main__":
    main()
