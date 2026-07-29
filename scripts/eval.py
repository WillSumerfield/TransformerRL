"""Headless checkpoint evaluation for single-network codesign runs.

Decouples (control policy) x (body source): for each run + epoch, install a FIXED body population,
roll out K episodes/body with the DETERMINISTIC control policy (mu), and report reward, robustness,
generator diversity/committance, and value calibration. Reuses the lightweight experiments/ path
(ppg_parity._load_policy + a hand-driven rollout) -- no rl_games runner, no train-script mode.

Body sources (all walk the SAME grammar-masked generation MDP; see net.sample):
  general  net.sample(N, 'stochastic')  -- the trained generator's distribution (in-distribution E)
  best     net.sample(N, 'greedy')      -- the generator's committed body (argmax => the 'best morph')
  random   net.sample(N, 'uniform')     -- a random policy on the MDP: the diversity reference (D)
                                           and the baseline for gen-advantage-over-random.

EPM == 1 for codesign (one body per env), so per-env stats ARE per-body stats; B = num_envs bodies.
Metrics come from two places: the rollout (reward/fall/ep_len/V0.98) and net.sample's own output
(diversity via experiments/diversity_p5, GenCrit/V1.0 body-quality = v_states[:, -1]).

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

from experiments.ppg_parity import _load_policy
from experiments.diversity_p5 import population_to_repr, rao_blackwell_h_body, redundancy
from transformer_rl.models import _raw_tail
from envs.ant_envs.ant_codesign import AntCodesignEnv
from envs.ant_envs.ant_multimorph import _OBS_TOTAL, _N_DOFS_FULL

RUN_ROOT = _ROOT / "runs/ant_codesign/codesign_single_transformer"
EVAL_SEED = 123

# CSV column order (one wide row per run x epoch); console shows a subset.
FIELDS = ["run", "epoch",
          "gen_avg", "gen_topk_mean", "gen_top1", "gen_fall", "gen_ep_len",
          "val_calib_r", "gencrit_calib_r",
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


# ---- forward / rollout ------------------------------------------------------------

@torch.no_grad()
def _forward(net, obs_norm, obs):
    """Deterministic control step: normalized obs (raw {0,1} tail restored) -> (mu-clamped, V0.98)."""
    normed = obs_norm(obs).clone()
    t0, td = _raw_tail(net.net)                              # restore raw DOF-mask/type tail
    normed[..., t0:t0 + td] = obs[..., t0:t0 + td]
    mu, _, value, _ = net({"obs": normed})
    value = value.squeeze(-1) if value is not None else None
    return mu.clamp(-1.0, 1.0), value


@torch.no_grad()
def _rollout(net, obs_norm, env, device, *, episodes):
    """K episodes per env on the currently-installed (fixed) bodies. EPM==1 => per-env == per-body.
    Returns per-body arrays: return (mean over K), fall_rate (terminated vs truncated), ep_len, v0
    (mean V0.98 at episode starts). Bodies are held fixed -- resample() is NOT called here."""
    n, L = env.total_num_envs, env.max_episode_length
    z = lambda: torch.zeros(n, device=device)
    ret_sum, term_sum, len_sum = z(), z(), z()
    v0_sum, v0_n = z(), z()
    cur, curlen = z(), z()
    ep_done = torch.zeros(n, device=device)
    at_s0 = torch.ones(n, dtype=torch.bool, device=device)  # first obs of every env is an episode start

    obs, _ = env.reset()
    cap = (episodes + 2) * L
    step = 0
    while bool((ep_done < episodes).any()) and step < cap:
        mu, value = _forward(net, obs_norm, obs)
        collecting = ep_done < episodes
        if value is not None:
            take = at_s0 & collecting
            v0_sum += torch.where(take, value, torch.zeros_like(value))
            v0_n += take.float()
        obs, rew, term, trunc, _ = env.step(mu)
        rew = rew.squeeze(-1) if rew.ndim > 1 else rew
        done = term | trunc
        # VSim reports done one call late, after resetting the lane. Finish
        # from the already accumulated real transitions and exclude this
        # notification call's reset reward and extra step.
        finish = done & collecting
        ret_sum += torch.where(finish, cur, torch.zeros_like(cur))
        len_sum += torch.where(finish, curlen, torch.zeros_like(curlen))
        term_sum += (term & finish).float()
        ep_done += finish.float()
        retained = collecting & ~done
        cur += torch.where(retained, rew, torch.zeros_like(rew))
        curlen += retained.float()
        cur = torch.where(done, torch.zeros_like(cur), cur)
        curlen = torch.where(done, torch.zeros_like(curlen), curlen)
        at_s0 = done
        step += 1

    e = float(episodes)
    return {"return": (ret_sum / e).cpu().numpy(),
            "fall_rate": (term_sum / e).cpu().numpy(),
            "ep_len": (len_sum / e).cpu().numpy(),
            "v0": (v0_sum / v0_n.clamp(min=1)).cpu().numpy()}


# ---- body sources + generator-intrinsic metrics -----------------------------------

def _sample(net, n, mode):
    torch.manual_seed(EVAL_SEED)                            # reproducible draws across runs/epochs
    return net.net.sample(n, mode=mode)


def _install(env, out):
    env.set_next(out["counts"].long(), out["eff_sub"], out["cap_sub"])
    env.resample()


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
        from experiments.diversity import counts_to_repr
        skel = [counts_to_repr(c) for c in out["counts"].cpu().numpy().astype(int)]
        h_skel = rao_blackwell_h_body(out["step_entropy"].cpu().numpy(), active)
        r = redundancy(skel, h_skel)
        return {"N_body_skel": r["N_body"], "N_sub": 1.0,
                "N_limb_mean": r["N_limb_mean"], "rho": r["rho"]}
    counts = out["counts"].cpu().numpy().astype(int)
    eff, cap = out["eff_sub"].cpu().numpy(), out["cap_sub"].cpu().numpy()
    skel = population_to_repr(counts, eff, cap, collapse_subtypes=True)
    h_skel = rao_blackwell_h_body(out["step_entropy_cat"].cpu().numpy(), active)
    h_sub = rao_blackwell_h_body(out["step_entropy_sub"].cpu().numpy(), active)
    r = redundancy(skel, h_skel)
    return {"N_body_skel": r["N_body"], "N_sub": float(np.exp(h_sub)),
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
    gen = _sample(net, n, "stochastic")
    _install(env, gen)
    g = _rollout(net, obs_norm, env, device, episodes=episodes)
    gret = g["return"]
    topk = np.sort(gret)[::-1][:top_k]
    v1 = gen["v_states"][:, -1].cpu().numpy()               # GenCrit/V1.0 predicted body quality
    row.update(gen_avg=float(gret.mean()), gen_topk_mean=float(topk.mean()),
               gen_top1=float(gret.max()), gen_fall=float(g["fall_rate"].mean()),
               gen_ep_len=float(g["ep_len"].mean()),
               val_calib_r=_pearson(g["v0"], gret), gencrit_calib_r=_pearson(v1, gret))
    row.update(_diversity(gen))

    # --- best morph: the generator's committed (greedy) body ---
    best = _sample(net, n, "greedy")
    row["best_n_unique"] = _n_unique(best)
    _install(env, best)
    b = _rollout(net, obs_norm, env, device, episodes=episodes)
    row.update(best_avg=float(b["return"].mean()), best_fall=float(b["fall_rate"].mean()))

    # --- random source: diverse reference (metric D) + gen-advantage-over-random ---
    rnd = _sample(net, n, "uniform")
    _install(env, rnd)
    rr = _rollout(net, obs_norm, env, device, episodes=episodes)
    row.update(random_avg=float(rr["return"].mean()),
               gen_advantage=float(gret.mean() - rr["return"].mean()))
    return row


# ---- output -----------------------------------------------------------------------

def _print_table(rows, top_k):
    cols = [("run", 18, "s"), ("epoch", 6, "s"), ("gen_avg", 9, ".1f"),
            (f"top{top_k}", 9, ".1f"), ("best_avg", 9, ".1f"), ("rand_avg", 9, ".1f"),
            ("gen_adv", 8, ".1f"), ("fall", 6, ".2f"), ("N_skel", 8, ".2g"),
            ("N_sub", 9, ".3g"), ("gencrit_r", 9, ".2f")]
    src = {"top%d" % top_k: "gen_topk_mean", "best_avg": "best_avg", "rand_avg": "random_avg",
           "gen_adv": "gen_advantage", "fall": "gen_fall", "N_skel": "N_body_skel",
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
    env = AntCodesignEnv(n_envs, device, sample_morphs=True, rendering=False,
                         raise_exception=False, seed=EVAL_SEED, with_window=False)

    rows = []
    for run_name, rd, ckpts, cfg in jobs:
        net_params = cfg["params"]["network"]
        value_size = int(cfg.get("env", {}).get("value_size", 1))
        for label, ckpt in ckpts:
            print(f"[eval] {run_name} @ {label}: {ckpt.name}", flush=True)
            net, obs_norm = _load_policy(ckpt, net_params, device, value_size=value_size,
                                         obs_base=_OBS_TOTAL, n_act=_N_DOFS_FULL)
            row = {"run": run_name, "epoch": label}
            row.update(evaluate(net, obs_norm, env, device, episodes=args.episodes, top_k=args.top_k))
            rows.append(row)
            print(f"[eval]   gen={row['gen_avg']:.1f} best={row['best_avg']:.1f} "
                  f"rand={row['random_avg']:.1f} adv={row['gen_advantage']:+.1f} | "
                  f"N_skel={row['N_body_skel']:.2g} N_sub={row['N_sub']:.3g} "
                  f"gencrit_r={row['gencrit_calib_r']:.2f}", flush=True)
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
