"""Q10: is the entropy bonus discharging into FREE entropy at grammar-forced cap steps?

The grammar (architectures._gen_masks) makes the deepest slot (depth >= max_effectors) CAP-ONLY.
There the category decision is degenerate -- h_cat == 0, logp 0 -- but the cap's subtype row is
UNMASKED (sub_cap == all-ones for depth > 0), so the subtype is still fully free. Hypothesis: the
entropy bonus buys its nats there, where they cost nothing in skeleton terms, which would explain a
committed skeleton sitting next to an N_sub ~ 1e9 subtype axis.

Test: sample N bodies from each run's final checkpoint, split ACTIVE steps into forced-cap vs
freely-chosen, and compare each group's share of steps against its share of subtype entropy.

Forced-cap steps are recovered EXACTLY, not by proxy: sample() returns slots/cat_actions/sub_actions,
and net.net._replay_states() re-walks the identical MDP to give depth_hist/active_hist per step, so
`active & (depth >= max_effectors)` is the same predicate _gen_masks used when the step was taken.

Usage:  uv run python experiments/free_entropy.py [--runs a,b,c] [--n 4096]
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

RUN_DIR = _ROOT / "runs/ant_codesign/codesign_single_transformer"
OUT = _ROOT / "temp/q10_free_entropy.md"
RUNS = ["phase5_s42", "phase5_s43", "phase5_s44", "21-09-23-06", "21-12-02-14"]


def _final_ckpt(run_dir: Path):
    """Highest-epoch *_ep_<N>.pth in the run's nn/ dir."""
    nn = run_dir / "nn"
    eps = [(int(m.group(1)), p) for p in nn.glob("*_ep_*.pth")
           if (m := re.search(r"_ep_(\d+)", p.name))] if nn.exists() else []
    return max(eps) if eps else (None, None)


def analyse(net, n_sample):
    import torch
    g = net.net
    with torch.no_grad():
        out = g.sample(n_sample)
        # Re-walk the same MDP teacher-forced to recover the per-step depth the mask was built from.
        _c, _cap, _e, active, depth, force_grow = g._replay_states(
            out["slots"], out["cat_actions"], out["sub_actions"])

    forced_cap = active & (depth >= g.max_effectors)     # cap-only row  => h_cat == 0 by construction
    free = active & ~forced_cap
    h_cat, h_sub = out["step_entropy_cat"], out["step_entropy_sub"]

    def grp(m):
        k = int(m.sum())
        sel = lambda x: (x[m].mean().item() if k else 0.0)
        return dict(n=k, h_cat=sel(h_cat), h_sub=sel(h_sub),
                    tot_cat=sel(h_cat) * k, tot_sub=sel(h_sub) * k)

    F, C, A = grp(forced_cap), grp(free), grp(active)
    r = dict(run=None, n_active=A["n"], frac_steps=F["n"] / max(A["n"], 1),
             f_hcat=F["h_cat"], f_hsub=F["h_sub"], c_hcat=C["h_cat"], c_hsub=C["h_sub"],
             f_totcat=F["tot_cat"], f_totsub=F["tot_sub"],
             c_totcat=C["tot_cat"], c_totsub=C["tot_sub"],
             frac_sub=F["tot_sub"] / max(A["tot_sub"], 1e-12),
             frac_cat=F["tot_cat"] / max(A["tot_cat"], 1e-12),
             # per-body nats (H(B) Rao-Blackwell split by group)
             hb_sub=A["tot_sub"] / n_sample, hb_cat=A["tot_cat"] / n_sample,
             n_forcegrow=int((active & force_grow).sum()))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=str, default=",".join(RUNS))
    ap.add_argument("--n", type=int, default=4096)
    args = ap.parse_args()

    import torch
    import yaml
    from experiments.ppg_parity import _load_policy
    from envs.ant_envs.ant_multimorph import _OBS_TOTAL as OBS_BASE, _N_DOFS_FULL as N_ACT

    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    rows = []
    for name in args.runs.split(","):
        run_dir = RUN_DIR / name
        ep, ckpt = _final_ckpt(run_dir)
        if ckpt is None:
            print(f"[q10] SKIP {name}: no checkpoints", flush=True)
            continue
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        net, _ = _load_policy(ckpt, cfg["params"]["network"], device,
                              value_size=int(cfg.get("env", {}).get("value_size", 1)),
                              obs_base=OBS_BASE, n_act=N_ACT)
        r = analyse(net, args.n)
        r["run"], r["ep"] = name, ep
        rows.append(r)
        print(f"[q10] {name} ep{ep}: forced {r['frac_steps']:.1%} of steps, "
              f"{r['frac_sub']:.1%} of subtype nats  (h_sub forced {r['f_hsub']:.3f} vs "
              f"chosen {r['c_hsub']:.3f}; forced h_cat {r['f_hcat']:.2e})", flush=True)
        del net
        torch.cuda.empty_cache()

    hdr = ("| run | ep | forced/active steps | H_sub forced | H_sub chosen | H_cat forced | "
           "H_cat chosen | nats_sub forced | nats_sub chosen | forced share of H_sub |\n"
           "|---|---|---|---|---|---|---|---|---|---|\n")
    body = "".join(
        f"| {r['run']} | {r['ep']} | {r['frac_steps']:.1%} ({int(r['frac_steps']*r['n_active'])}"
        f"/{r['n_active']}) | {r['f_hsub']:.3f} | {r['c_hsub']:.3f} | {r['f_hcat']:.2e} | "
        f"{r['c_hcat']:.3f} | {r['f_totsub']/args.n:.3f} | {r['c_totsub']/args.n:.3f} | "
        f"{r['frac_sub']:.1%} |\n" for r in rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "# Q10 -- free entropy at grammar-forced cap steps\n\n"
        f"Final checkpoint per run, {args.n} sampled bodies. Entropies in nats; `nats_sub` columns "
        "are per-body totals (mean x count / N), so they sum to the subtype half of H(B).\n"
        "Forced-cap step := active & depth >= max_effectors (cap-only mask, h_cat == 0).\n\n"
        + hdr + body + "\n## Verdict\n\n" + verdict(rows) + "\n")
    print(f"[q10] wrote {OUT}")
    print(verdict(rows))


def verdict(rows):
    if not rows:
        return "No checkpoints analysed."
    lift = np.mean([r["frac_sub"] / max(r["frac_steps"], 1e-9) for r in rows])
    fs = np.mean([r["frac_steps"] for r in rows])
    fh = np.mean([r["frac_sub"] for r in rows])
    tag = "CONFIRMED" if lift > 1.15 else "REFUTED"
    return (f"**{tag}.** Forced-cap steps are {fs:.1%} of all active steps and carry {fh:.1%} of "
            f"total subtype entropy (mean over runs), a concentration ratio of {lift:.2f}x. "
            + ("Forced steps buy subtype nats at zero skeleton cost, so the entropy bonus can be "
               "discharged there without touching the body plan."
               if tag == "CONFIRMED" else
               "Subtype entropy is NOT preferentially parked at the forced slots; the diffuse "
               "subtype axis is spread across freely-chosen steps too, so the grammar's free "
               "category slot does not explain the lack of subtype commitment."))


if __name__ == "__main__":
    main()
