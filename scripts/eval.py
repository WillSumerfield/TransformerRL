"""Headless checkpoint evaluation for single-network codesign runs.

Decouples (control policy) x (body source): for each run + epoch, install a FIXED body population,
roll out K episodes/body with the DETERMINISTIC control policy (mu), and report reward, robustness,
generator diversity/committance, and value calibration. The task build, policy load, body install and rollout
are `experiments/harness/evalpass.py`'s -- shared with the ladder and the specialization pass, so
"return" means one thing across the series. No rl_games runner, no train-script mode.

Body sources (all walk the SAME grammar-masked generation MDP; see net.sample):
  general  net.sample(N, 'stochastic')  -- the trained generator's distribution (in-distribution E)
  best     net.sample(N, 'greedy')      -- the generator's committed body (argmax => the 'best morph')
  random   net.sample(N, 'uniform')     -- a random policy on the MDP: the diversity reference (D)
                                           and the baseline for gen-advantage-over-random.

EPM == 1 for codesign (one body per env), so per-env stats ARE per-body stats; B = num_envs bodies.
Metrics come from two places: the rollout (reward/fall/ep_len/V0.98) and net.sample's own output
(diversity via experiments/harness, GenCrit/V1.0 body-quality = v_states[:, -1]).

Runs are compared SIDE-BY-SIDE: one wide CSV row per (run, epoch). Control uses mu always; the
generator draws are stochastic (except 'best' = greedy), seeded for reproducibility across runs.

Usage:
  python scripts/eval.py RUN [RUN ...] [--epochs best|N,N,...] [--episodes 32] [--top-k 10]
                         [--num-envs N] [--out evals/<stamp>.csv]
  RUN is a run-dir name under runs/ant_codesign/codesign_single_transformer/, or a full path.

NOTE: heavy -- B envs x 3 sources x K episodes, with a full sim rebuild per source. Lower
--num-envs / --episodes for a quick look.
"""
import os; os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

import numpy as np
import torch
import yaml

from experiments.harness.committance import population_to_repr, rao_blackwell_h_body, redundancy
from experiments.harness.diversity import within_run_metrics
from experiments.harness.evalpass import (
    EVAL_SEED, install, load_net, open_task, rollout, sample_bodies,
)

RUN_ROOT = _ROOT / "runs/ant_codesign/codesign_single_transformer"

# CSV column order (one wide row per run x epoch); console shows a subset.
FIELDS = ["run", "epoch",
          "gen_avg", "gen_topk_mean", "gen_top1", "gen_fall", "gen_ep_len",
          "val_calib_r", "gencrit_calib_r",
          "div_nmodes", "div_struct",
          "N_body_skel", "N_sub", "N_limb_mean", "rho",
          "best_avg", "best_fall", "best_n_unique",
          "random_avg", "gen_advantage"]


# ---- checkpoint / config resolution -----------------------------------------------

def _run_dir(name: str) -> Path:
    p = Path(name)
    return p if p.is_dir() else RUN_ROOT / name


def _resolve_ckpts(run_dir: Path, epochs_spec: str):
    """[(label, path)] for the requested epochs. 'best' = the bare <name>.pth (rl_games best-mean
    save); otherwise a comma list of epoch numbers matched against the *_ep_<N>_*.pth snapshots."""
    nn = run_dir / "nn"
    if not nn.exists():
        return []
    if epochs_spec == "best":
        best = [p for p in nn.glob("*.pth")
                if "_ep_" not in p.name and not p.name.startswith("last_")]
        return [("best", best[0])] if best else []
    want = {int(e) for e in epochs_spec.split(",")}
    out = []
    for p in nn.glob("*_ep_*.pth"):
        m = re.search(r"_ep_(\d+)", p.name)
        if m and int(m.group(1)) in want:
            out.append((int(m.group(1)), p))
    return sorted(out)


# ---- body sources + generator-intrinsic metrics -----------------------------------

def _typed_reps(out):
    counts = out["counts"].cpu().numpy().astype(int)
    eff, cap = out["eff_sub"].cpu().numpy(), out["cap_sub"].cpu().numpy()
    return population_to_repr(counts, eff, cap)


def _n_unique(out):
    reps = _typed_reps(out)
    return len({tuple(tuple(l) if l else None for l in b) for b in reps})


def _diversity(out):
    """Generator diversity + committance from net.sample's own output (no rollout). Committance is
    reported SPLIT skeleton vs subtype (the free_entropy finding: the skeleton commits while the
    subtype axis stays free). N_body_skel / N_sub are effective #designs -- lower == more committed."""
    active = out["active_step"].cpu().numpy()
    if "eff_sub" not in out:                                # phase-1/3 fallback (no subtype axis)
        from experiments.harness.diversity import counts_to_repr
        skel = [counts_to_repr(c) for c in out["counts"].cpu().numpy().astype(int)]
        h_skel = rao_blackwell_h_body(out["step_entropy"].cpu().numpy(), active)
        r = redundancy(skel, h_skel)
        wm = within_run_metrics(skel)                       # distance-clustered diversity (skeleton)
        return {"div_nmodes": wm["div_nmodes"], "div_struct": wm["div_struct"],
                "N_body_skel": r["N_body"], "N_sub": 1.0,
                "N_limb_mean": r["N_limb_mean"], "rho": r["rho"]}
    counts = out["counts"].cpu().numpy().astype(int)
    eff, cap = out["eff_sub"].cpu().numpy(), out["cap_sub"].cpu().numpy()
    skel = population_to_repr(counts, eff, cap, collapse_subtypes=True)
    h_skel = rao_blackwell_h_body(out["step_entropy_cat"].cpu().numpy(), active)
    h_sub = rao_blackwell_h_body(out["step_entropy_sub"].cpu().numpy(), active)
    r = redundancy(skel, h_skel)
    wm = within_run_metrics(skel)                           # diversity on subtype-collapsed skeleton:
    return {"div_nmodes": wm["div_nmodes"], "div_struct": wm["div_struct"],  # ignores free subtype flips
            "N_body_skel": r["N_body"], "N_sub": float(np.exp(h_sub)),
            "N_limb_mean": r["N_limb_mean"], "rho": r["rho"]}


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---- one (run, epoch) evaluation --------------------------------------------------

def evaluate(net, obs_norm, env, device, *, episodes, top_k):
    n = env.total_num_envs
    row = {}

    # --- general: in-distribution (metric E) + diversity/committance + calibration ---
    gen = sample_bodies(net, n, "stochastic")
    install(env, gen)
    g = rollout(net, obs_norm, env, device, episodes=episodes, label="gen")
    gret = g["return"]
    topk = np.sort(gret)[::-1][:top_k]
    v1 = gen["v_states"][:, -1].cpu().numpy()               # GenCrit/V1.0 predicted body quality
    row.update(gen_avg=float(gret.mean()), gen_topk_mean=float(topk.mean()),
               gen_top1=float(gret.max()), gen_fall=float(g["fall_rate"].mean()),
               gen_ep_len=float(g["ep_len"].mean()),
               val_calib_r=_pearson(g["v0"], gret), gencrit_calib_r=_pearson(v1, gret))
    row.update(_diversity(gen))

    # --- best morph: the generator's committed (greedy) body ---
    best = sample_bodies(net, n, "greedy")
    row["best_n_unique"] = _n_unique(best)
    install(env, best)
    b = rollout(net, obs_norm, env, device, episodes=episodes, label="best")
    row.update(best_avg=float(b["return"].mean()), best_fall=float(b["fall_rate"].mean()))

    # --- random source: diverse reference (metric D) + gen-advantage-over-random ---
    rnd = sample_bodies(net, n, "uniform")
    install(env, rnd)
    rr = rollout(net, obs_norm, env, device, episodes=episodes, label="rnd")
    row.update(random_avg=float(rr["return"].mean()),
               gen_advantage=float(gret.mean() - rr["return"].mean()))
    return row


# ---- output -----------------------------------------------------------------------

def _print_table(rows, top_k):
    cols = [("run", 18, "s"), ("epoch", 6, "s"), ("gen_avg", 9, ".1f"),
            (f"top{top_k}", 9, ".1f"), ("best_avg", 9, ".1f"), ("rand_avg", 9, ".1f"),
            ("gen_adv", 8, ".1f"), ("fall", 6, ".2f"), ("N_modes", 8, ".3g"),
            ("d_struct", 9, ".2f"), ("N_skel", 8, ".2g"),
            ("N_sub", 9, ".3g"), ("gencrit_r", 9, ".2f")]
    src = {"top%d" % top_k: "gen_topk_mean", "best_avg": "best_avg", "rand_avg": "random_avg",
           "gen_adv": "gen_advantage", "fall": "gen_fall", "N_modes": "div_nmodes",
           "d_struct": "div_struct", "N_skel": "N_body_skel",
           "N_sub": "N_sub", "gencrit_r": "gencrit_calib_r", "gen_avg": "gen_avg"}
    hdr = " ".join(f"{name:>{w}}" if name != "run" else f"{name:<{w}}" for name, w, _ in cols)
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = []
        for name, w, fmt in cols:
            key = src.get(name, name)
            v = r.get(key, "")
            if fmt == "s":
                cells.append(f"{str(v):<{w}}" if name == "run" else f"{str(v):>{w}}")
            else:
                cells.append(f"{v:>{w}{fmt}}" if isinstance(v, (int, float)) else f"{'':>{w}}")
        print(" ".join(cells))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run-dir name(s) under RUN_ROOT, or full path(s)")
    ap.add_argument("--epochs", default="best", help="'best' (default) or comma list of epoch numbers")
    ap.add_argument("--episodes", type=int, default=32, help="episodes per body (default 32)")
    ap.add_argument("--top-k", type=int, default=10, dest="top_k")
    ap.add_argument("--num-envs", type=int, default=None,
                    help="body population size B (default: the run's config num_actors)")
    ap.add_argument("--out", type=Path, default=None, help="CSV path (default evals/<stamp>.csv)")
    args = ap.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    jobs = []                                              # (run_name, run_dir, [(label, ckpt)], cfg)
    for name in args.runs:
        rd = _run_dir(name)
        ckpts = _resolve_ckpts(rd, args.epochs)
        if not ckpts:
            print(f"[eval] SKIP {name}: no matching checkpoints in {rd/'nn'}", flush=True)
            continue
        cfg = yaml.safe_load((rd / "config.yaml").read_text())
        jobs.append((Path(name).name, rd, ckpts, cfg))
    if not jobs:
        raise SystemExit("[eval] nothing to evaluate")

    n_envs = args.num_envs or int(jobs[0][3]["params"]["config"]["num_actors"])
    print(f"[eval] {n_envs} bodies x {args.episodes} episodes/body x 3 sources; seed={EVAL_SEED}",
          flush=True)
    # Task, library and sizing all come from the run's own stamp; every source below rebuilds onto
    # its own bodies.
    env, library, layout = open_task(jobs[0][3], n_envs, device=device)

    rows = []
    for run_name, rd, ckpts, cfg in jobs:
        for label, ckpt in ckpts:
            print(f"[eval] {run_name} @ {label}: {ckpt.name}", flush=True)
            net, obs_norm = load_net(ckpt, cfg, layout, device)
            row = {"run": run_name, "epoch": label}
            row.update(evaluate(net, obs_norm, env, device, episodes=args.episodes, top_k=args.top_k))
            rows.append(row)
            print(f"[eval]   gen={row['gen_avg']:.1f} best={row['best_avg']:.1f} "
                  f"rand={row['random_avg']:.1f} adv={row['gen_advantage']:+.1f} | "
                  f"N_modes={row['div_nmodes']:.3g} d_struct={row['div_struct']:.2f} "
                  f"N_skel={row['N_body_skel']:.2g} gencrit_r={row['gencrit_calib_r']:.2f}", flush=True)
            del net, obs_norm
            torch.cuda.empty_cache()

    _print_table(rows, args.top_k)

    out = args.out or _ROOT / "evals" / f"{datetime.now():%Y%m%d_%H%M%S}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"[eval] CSV -> {out}  ({len(rows)} row(s))", flush=True)


if __name__ == "__main__":
    main()
