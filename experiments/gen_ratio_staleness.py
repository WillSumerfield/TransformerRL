"""Q2: is the generator's PPO update still on-policy?

The agent draws the generator trace at window k-1's boundary and only USES it in
`_resample_update` at window k (`codesign_agent.py:768` runs the update BEFORE line 778 samples
the next trace). Between those two moments, ~63 epochs of control PPO update the SHARED TRUNK.
Nothing protects the generator from that drift -- the clone terms (beta*KL, lambda*MSE) protect
CONTROL from the generator's update, not the reverse, and there is no `gen/self_kl` logged.

So the generator's PPO ratio at optimisation step 0 is NOT 1. This measures how far from 1.

Method (fully offline -- no training run): load the checkpoint at epoch E, draw a trace (its
`old_logp` is exactly what training stores), then replay those SAME actions under the checkpoint at
epoch E+gap and recompute the log-prob. ratio = exp(new_logp - old_logp) over valid steps.

  gap=0    CONTROL. Must be exactly 1.0. This also empirically proves the OTHER half of Q2 --
           that sample()'s and gen_replay()'s grammar masks agree -- since any mask mismatch
           would show up here as a ratio != 1.
  gap=50   ~0.8 window of trunk drift (checkpoint cadence); a LOWER bound on the real staleness.
  gap=100  ~1.6 windows; brackets the true 63-epoch value from above.

Reported: median ratio, and the fraction of steps landing outside PPO's trust region
[1-e_clip, 1+e_clip]. Steps outside get their gradient clipped away -- if that fraction is large,
the generator is discarding most of its own learning signal every window, which would explain
"won't commit" far better than the entropy coef does.

Usage:  python experiments/gen_ratio_staleness.py [--runs a,b] [--gaps 0,50,100] [--n 4096]
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


def _ckpts(run_dir):
    return dict(sorted((int(m.group(1)), p) for p in (run_dir / "nn").glob("*_ep_*.pth")
                       if (m := re.search(r"_ep_(\d+)", p.name))))


def _replay_logp(net, trace, chunk=256):
    """Log-prob of the trace's OWN actions under `net`'s current weights. Phase-aware.

    Chunked over the population: gen_replay batches all L+1 prefixes in ONE trunk forward
    (N*(L+1) sequences), which OOMs at N=4096. Training avoids this via generator.minibatches."""
    import torch
    g = net.net
    lps, valids = [], []
    N = trace["slots"].shape[0]
    for i in range(0, N, chunk):
        sl = slice(i, i + chunk)
        if "cat_actions" in trace:                                # phase 5: factored head
            ca, sa = trace["cat_actions"][sl], trace["sub_actions"][sl]
            cat_lp, sub_lp, _v, valid = g.gen_replay(trace["slots"][sl], ca, sa)
            lp, _ = g.gen_logp_entropy(cat_lp, sub_lp, ca, sa)
        else:                                                      # phase 1/3: binary head
            a = trace["actions"][sl]
            logits, _v, valid = g.gen_replay(trace["slots"][sl], a)
            lp = logits.log_softmax(-1).gather(-1, a.unsqueeze(-1)).squeeze(-1)
        lps.append(lp); valids.append(valid)
    return torch.cat(lps), torch.cat(valids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="phase5_s42,21-12-02-14")
    ap.add_argument("--gaps", default="0,50,100")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--start", type=int, default=800, help="first epoch to probe (post-pretrain)")
    ap.add_argument("--stride", type=int, default=500)
    args = ap.parse_args()

    import torch
    import yaml
    from experiments.ppg_parity import _load_policy
    from task_envs.codesign_environment import _OBS_TOTAL as OBS_BASE, _N_DOFS_FULL as N_ACT

    device = torch.device("cuda:0")
    gaps = [int(g) for g in args.gaps.split(",")]

    for name in args.runs.split(","):
        run_dir = RUN_DIR / name
        ck = _ckpts(run_dir)
        if not ck:
            print(f"[q2] SKIP {name}: no checkpoints"); continue
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        e_clip = float(cfg["params"]["config"]["e_clip"])
        net_params = cfg["params"]["network"]
        value_size = int(cfg.get("env", {}).get("value_size", 1))
        load = lambda p: _load_policy(p, net_params, device, value_size=value_size,
                                      obs_base=OBS_BASE, n_act=N_ACT)[0]

        print(f"\n=== {name}  (e_clip={e_clip}, window=63 epochs) ===")
        print(f"{'epoch':>7}{'gap':>6}{'which':>8}{'med ratio':>9}{'p05':>9}{'p95':>9}"
              f"{'%outside clip':>15}{'mean|dlogp|':>13}")
        probes = [e for e in ck if e >= args.start][::max(1, args.stride // 50)]
        for E in probes:
            net_old = load(ck[E])
            with torch.no_grad():
                trace = net_old.net.sample(args.n)
                old_lp, valid = _replay_logp(net_old, trace)      # replay-consistent baseline
            for gap in gaps:
                if E + gap not in ck:
                    continue
                variants = {"all": load(ck[E + gap])} if gap else {"all": net_old}
                if gap:
                    # TRUNK-ONLY: new weights everywhere EXCEPT the generator's own heads, which are
                    # restored from the old checkpoint. Isolates drift the CONTROL updates induced in
                    # the shared trunk -- i.e. true staleness -- from the generator's own (legitimate)
                    # once-per-window PPO step, which a 50-epoch gap may also contain.
                    hyb = load(ck[E + gap])
                    old_sd = net_old.net.state_dict()
                    hyb.net.load_state_dict(
                        {k: (old_sd[k] if k.startswith(("gen_cat_head", "gen_sub_head", "gen_head"))
                             else v) for k, v in hyb.net.state_dict().items()})
                    variants["trunk"] = hyb
                for tag, net_new in variants.items():
                    with torch.no_grad():
                        new_lp, _ = _replay_logp(net_new, trace)
                        d = (new_lp - old_lp)[valid]
                        ratio = d.exp().float().cpu().numpy()
                    out = float(((ratio < 1 - e_clip) | (ratio > 1 + e_clip)).mean() * 100)
                    print(f"{E:>7}{gap:>6}  {tag:<6}{np.median(ratio):>9.4f}"
                          f"{np.percentile(ratio, 5):>9.3f}{np.percentile(ratio, 95):>9.3f}"
                          f"{out:>14.1f}%{np.abs(d.float().cpu().numpy()).mean():>13.4f}")
                if gap:
                    for v in variants.values():
                        del v
            del net_old
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
