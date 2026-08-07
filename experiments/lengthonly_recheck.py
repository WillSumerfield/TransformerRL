"""LENGTH-ONLY recheck of the phase-3 vs phase-5 commitment metrics.

The "skeleton" view (population_to_repr(..., collapse_subtypes=True)) is NOT phase-comparable: a
BARE cap emits no token while foot/pad/ball each emit a 'C', and collapsing 'C:foot'->'C' does not
delete that token. So the skeleton repr still carries a per-limb CAP-PRESENCE bit (~0.562 nats/limb
at uniform init) that phase 3's head cannot express at all.

This script recomputes the two headline metrics on the only view both phases can express: LENGTH
ONLY -- each limb is its module count, exactly diversity.counts_to_repr(counts). All type info
(eff_sub / cap_sub) is discarded.

The joint term is matched to the view: for phase 5, h_body = Rao-Blackwell over step_entropy_CAT
(the grow/stop decision -- the count axis, which is what a length-only repr encodes); step_entropy_
sub (which is where cap presence lives) is excluded. For phase 3, sample() has a single
step_entropy and that IS the grow/stop axis. Pairing the skeleton repr with h_cat -- what
commit_metrics.py does -- double-counts: cap-presence entropy is in the repr but not in h_body, so C
is inflated.

Estimators (limb_entropies / redundancy) are imported unchanged so they are bit-identical across
phases. On the phase-3 worktree they live in experiments/diversity.py; on phase-5, diversity_p5.py.

Usage:  uv run python experiments/lengthonly_recheck.py [--every 5] [--runs a,b,c]
"""
import os; os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import argparse
import re
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

import numpy as np

from experiments.diversity import counts_to_repr
try:                                                   # phase-5 tree
    from experiments.diversity_p5 import limb_entropies, rao_blackwell_h_body, redundancy
except ImportError:                                    # phase-3 worktree: same code, in diversity.py
    from experiments.diversity import limb_entropies, rao_blackwell_h_body, redundancy

RUN_DIR = _ROOT / "runs/ant_codesign/codesign_single_transformer"
OUT_DIR = _ROOT / "data/lengthonly_recheck"
N_SAMPLE = 4096

RUNS = ["phase5_s42", "phase5_s43", "phase5_s44", "21-09-23-06", "21-12-02-14",
        "phase3_s42", "phase3_s45"]


def _checkpoints(run_dir: Path, every: int):
    nn = run_dir / "nn"
    if not nn.exists():
        return []
    eps = sorted((int(m.group(1)), p) for p in nn.glob("*_ep_*.pth")
                 if (m := re.search(r"_ep_(\d+)", p.name)))
    return eps[::every]


def analyse(net):
    """Length-only metrics for one checkpoint."""
    import torch
    with torch.no_grad():
        out = net.net.sample(N_SAMPLE)
    counts = out["counts"].cpu().numpy().astype(int)
    active = out["active_step"].cpu().numpy()

    # count axis only: cat == grow/stop for P5, the single step entropy for P1/P3
    key = "step_entropy_cat" if "step_entropy_cat" in out else "step_entropy"
    h_len = rao_blackwell_h_body(out[key].cpu().numpy(), active)

    pop = [counts_to_repr(c) for c in counts]
    rec = redundancy(pop, h_len)
    rec["C_frac"] = rec["C_nats"] / rec["H_within_sum"] if rec["H_within_sum"] > 0 else 0.0
    rec["count_hist"] = np.bincount(counts.reshape(-1), minlength=8)[:8].astype(float)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--runs", type=str, default=",".join(RUNS))
    args = ap.parse_args()

    import torch
    import yaml
    from experiments.ppg_parity import _load_policy
    from task_envs.codesign_environment import _OBS_TOTAL as OBS_BASE, _N_DOFS_FULL as N_ACT

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.runs.split(","):
        run_dir = RUN_DIR / name
        ckpts = _checkpoints(run_dir, args.every)
        if not ckpts:
            print(f"[len] SKIP {name}: no checkpoints", flush=True)
            continue
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        net_params = cfg["params"]["network"]
        value_size = int(cfg.get("env", {}).get("value_size", 1))

        acc, epochs = {}, []
        for ep, ckpt in ckpts:
            for attempt in range(4):                   # another GPU job may be resident: retry OOM
                try:
                    net, _ = _load_policy(ckpt, net_params, device, value_size=value_size,
                                          obs_base=OBS_BASE, n_act=N_ACT)
                    rec = analyse(net)
                    break
                except torch.cuda.OutOfMemoryError:
                    net = None
                    torch.cuda.empty_cache()
                    print(f"[len] {name} ep{ep}: OOM, retry {attempt + 1}/4", flush=True)
                    time.sleep(20)
                except (RuntimeError, KeyError) as e:
                    if "out of memory" in str(e).lower():
                        net = None
                        torch.cuda.empty_cache()
                        print(f"[len] {name} ep{ep}: OOM, retry {attempt + 1}/4", flush=True)
                        time.sleep(20)
                        continue
                    print(f"[len] {name} ep{ep}: load failed ({type(e).__name__}: {e}) -- "
                          f"wrong branch for this run", flush=True)
                    rec = None
                    break
            else:
                print(f"[len] {name} ep{ep}: OOM after retries, skipping", flush=True)
                rec = None
            if rec is None:
                if net is None:
                    continue
                break
            epochs.append(ep)
            for k, val in rec.items():
                acc.setdefault(k, []).append(val)
            print(f"[len] {name} ep{ep}: N_limb={rec['N_limb_mean']:.3f} "
                  f"C/sumH={rec['C_frac']:.4f} C={rec['C_nats']:.3f} "
                  f"sumH={rec['H_within_sum']:.3f} N_body={rec['N_body']:.4g}", flush=True)
            del net
            torch.cuda.empty_cache()

        if epochs:
            np.savez(OUT_DIR / f"{name}.npz", epochs=np.array(epochs),
                     **{k: np.array(v) for k, v in acc.items()})
            print(f"[len] wrote {OUT_DIR / (name + '.npz')}", flush=True)


if __name__ == "__main__":
    main()
