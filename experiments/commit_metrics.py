"""Generator-commitment metrics over training (Phase-5 investigation).

Answers "is the generator committing, and if not, in which axis" by sweeping saved checkpoints and
computing, per checkpoint:

  N_limb(n) = exp H(L_n)   within-limb  -- effective # limb designs at compass slot n
  rho       = exp(C)       cross-limb   -- total correlation; 1 == independent limb-lotteries
  H(B)                     joint        -- Rao-Blackwell, from the generator's own per-step entropy
  I(L_m;L_n)               8x8 MI       -- where the body-plan structure actually lives
  n_distinct               the OLD metric, kept only to demonstrate its ln(M) saturation

Identity: H(B) = sum_n H(L_n) - C.  See experiments/diversity.py for the estimators and why the
joint term may never be a plug-in count.

BRANCH-AGNOSTIC by feature detection on sample()'s output dict: phase-5 returns eff_sub/cap_sub
(typed bodies), phase-1/3 return counts only. Phase-3 checkpoints do NOT load into phase-5 code
(gen_head (2,d) vs gen_cat_head+gen_sub_head; embed_module in-dim 31 vs 36), so the phase-3 half is
produced by running this same script from a worktree at the phase-3 commit; the npz files merge.

Usage:  python experiments/commit_metrics.py [--every 5] [--curves] [--runs a,b,c]
"""
import os; os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import argparse
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

import numpy as np

from experiments.diversity import counts_to_repr
from experiments.diversity_p5 import (N_modes_curve, limb_entropies, pairwise_mi,
                                      population_to_repr, rao_blackwell_h_body, redundancy)
RUN_DIR = _ROOT / "runs/ant_codesign/codesign_single_transformer"
OUT_DIR = _ROOT / "data/commit_metrics"
N_SAMPLE = 4096
TAUS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 32]

RUNS = ["phase3_s42", "phase3_s43", "phase3_s44", "phase3_s45", "phase3_s46",
        "phase5_s42", "phase5_s43", "phase5_s44", "21-09-23-06", "21-12-02-14"]


def _checkpoints(run_dir: Path, every: int):
    """[(epoch, path)] for every `every`-th epoch checkpoint, ascending."""
    nn = run_dir / "nn"
    if not nn.exists():
        return []
    eps = sorted((int(m.group(1)), p) for p in nn.glob("*_ep_*.pth")
                 if (m := re.search(r"_ep_(\d+)", p.name)))
    return eps[::every]


def _n_distinct(bodies):
    return float(len({tuple(tuple(l) if l else None for l in b) for b in bodies}))


def analyse(net, curves=False):
    """Metrics for one checkpoint, reported on TWO body views so phases stay comparable:

      typed     -- full phase-5 vocabulary (skeleton + subtypes). Phase 1/3 have no subtype axis,
                   so for those runs the two views are identical.
      skeleton  -- subtypes collapsed ('E:knee'->'E', 'C:foot'->'C'). This is the ONLY view that is
                   apples-to-apples against phase 3, whose head cannot express a subtype at all.

    H(B) splits the same way, using the factored head's own additive entropy terms: h_cat IS the
    grow/stop decision phase 3 makes, h_sub is the axis phase 5 added. Phase 1/3 checkpoints have
    no split, so h_sub is 0 and skeleton == typed by construction."""
    import torch
    with torch.no_grad():
        out = net.net.sample(N_SAMPLE)
    counts = out["counts"].cpu().numpy().astype(int)
    active = out["active_step"].cpu().numpy()

    if "eff_sub" in out:                                   # phase 5: typed bodies
        eff, cap = out["eff_sub"].cpu().numpy(), out["cap_sub"].cpu().numpy()
        typed = population_to_repr(counts, eff, cap)
        skel = population_to_repr(counts, eff, cap, collapse_subtypes=True)
        h_skel = rao_blackwell_h_body(out["step_entropy_cat"].cpu().numpy(), active)
        h_sub = rao_blackwell_h_body(out["step_entropy_sub"].cpu().numpy(), active)
    else:                                                  # phase 1/3: no subtype axis exists
        typed = skel = [counts_to_repr(c) for c in counts]
        h_skel = rao_blackwell_h_body(out["step_entropy"].cpu().numpy(), active)
        h_sub = 0.0

    rec = {f"{k}_typed": v for k, v in redundancy(typed, h_skel + h_sub).items()}
    rec.update({f"{k}_skel": v for k, v in redundancy(skel, h_skel).items()})
    rec["H_sub"] = h_sub                                   # the phase-5-only axis, in nats
    rec["N_sub"] = float(np.exp(h_sub))
    rec["mi_skel"] = pairwise_mi(skel)
    rec["n_distinct_typed"] = _n_distinct(typed)           # the saturating metric, for contrast
    rec["n_distinct_skel"] = _n_distinct(skel)
    if curves:
        for tag, pop in (("typed", typed), ("skel", skel)):
            c = N_modes_curve(pop, TAUS)
            rec[f"nk_{tag}"] = np.array([c[t] for t in TAUS])
        rec["taus"] = np.array(TAUS, dtype=float)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=5, help="use every Nth epoch checkpoint")
    ap.add_argument("--curves", action="store_true", help="also compute the N(k) dendrogram curves")
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
            print(f"[commit] SKIP {name}: no checkpoints (wrong branch?)", flush=True)
            continue
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())   # the run's OWN architecture
        net_params = cfg["params"]["network"]
        value_size = int(cfg.get("env", {}).get("value_size", 1))

        acc, epochs = {}, []
        for ep, ckpt in ckpts:
            try:
                net, _ = _load_policy(ckpt, net_params, device, value_size=value_size,
                                      obs_base=OBS_BASE, n_act=N_ACT)
            except (RuntimeError, KeyError) as e:
                print(f"[commit] {name} ep{ep}: load failed ({type(e).__name__}) -- "
                      f"architecture mismatch, wrong branch for this run", flush=True)
                break
            rec = analyse(net, args.curves)
            epochs.append(ep)
            for k, val in rec.items():
                acc.setdefault(k, []).append(val)
            print(f"[commit] {name} ep{ep}: skel N_body={rec['N_body_skel']:.3g} "
                  f"N_limb={rec['N_limb_mean_skel']:.2f} rho={rec['rho_skel']:.2f} | "
                  f"N_sub={rec['N_sub']:.3g} | typed N_body={rec['N_body_typed']:.3g} "
                  f"ndist={rec['n_distinct_typed']:.0f}", flush=True)
            del net
            torch.cuda.empty_cache()

        if epochs:
            np.savez(OUT_DIR / f"{name}.npz", epochs=np.array(epochs),
                     **{k: np.array(v) for k, v in acc.items()})
            print(f"[commit] wrote {OUT_DIR / (name + '.npz')}", flush=True)


if __name__ == "__main__":
    main()
