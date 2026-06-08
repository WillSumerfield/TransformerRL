"""Per-leg-count performance comparison of seed-42 policy variants (old / v2 / nobern).

old    = pre-recast binary env            runs/ant_binary/binary_transformer_binaryold/s42
v2     = continuous-p, Bernoulli body     runs/ant_binary/binary_transformer/s42
nobern = continuous-p, body = center      runs/ant_binary/binary_transformer/s42_nobern

All three are evaluated on the SAME clean stable bodies (deterministic on_sets build, full-presence obs
on_p=1.0/off_p=0.0 so every leg reads "definitely present"), one deterministic-mu episode per env. We
report mean realized return per base leg-count -- the controlled apples-to-apples performance read the
training-mean reward can't give (v2's logged reward averages over degraded Bernoulli bodies). Each net
uses its OWN obs/value normalizers from its checkpoint. Run: python experiments/compare_seed42_variants.py
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

import numpy as np
import torch
import yaml

import experiments.value_grad_prop as vg
from envs.ant_envs.ant_binary_leg import AntBinaryLegEnv, HELDOUT
from envs.ant_envs.ant_multimorph import _stable_morphologies

K_PER_COUNT = 6      # stable bodies sampled per leg-count
ENVS_PER    = 128    # envs per body (averaged)
SEL_SEED    = 7      # fixed body selection
EVAL_SEED   = 123
SEEDS       = ["s42", "s43", "s44"]

# {variant: checkpoint path template (formatted with seed)}
VARIANT_TMPL = {
    "old":    "runs/ant_binary/binary_transformer_binaryold/{s}/nn/ant_binary_transformer.pth",
    "v2":     "runs/ant_binary/binary_transformer/{s}/nn/ant_binary_transformer.pth",
    "nobern": "runs/ant_binary/binary_transformer/{s}_nobern/nn/ant_binary_transformer.pth",
}


@torch.no_grad()
def _rollout_return(net, obs_norm, env, device):
    n = env.total_num_envs
    L = env.max_episode_length
    ep = torch.zeros(n, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    obs, _ = env.reset()
    t = 0
    while not bool(done.all()) and t < L + 5:
        normed = obs_norm(obs).clone()
        normed[..., -vg._MASK_DIM:] = obs[..., -vg._MASK_DIM:]   # mask tail raw
        mu, _, _, _ = net({"obs": normed})
        obs, rew, term, trunc, _ = env.step(mu.clamp(-1.0, 1.0))
        ep += rew * (~done).float()
        done |= (term | trunc)
        t += 1
    return ep.cpu().numpy()


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    cfg = yaml.safe_load(vg.CONFIG.read_text())
    off_scale = cfg.get("env", {}).get("off_scale", 0.05)

    # Same clean stable bodies for every variant: K per leg-count, held-out excluded.
    by_n: dict[int, list] = {}
    for s in _stable_morphologies():
        if frozenset(s) not in HELDOUT:
            by_n.setdefault(len(s), []).append(frozenset(s))
    counts = sorted(by_n)
    sel = np.random.default_rng(SEL_SEED)
    bodies, bcount = [], []
    for c in counts:
        pool = by_n[c]
        for i in sel.choice(len(pool), min(K_PER_COUNT, len(pool)), replace=False):
            bodies.append(pool[i]); bcount.append(c)
    bcount = np.array(bcount)

    env = AntBinaryLegEnv(len(bodies) * ENVS_PER, device, on_sets=bodies, on_p=1.0, off_p=0.0,
                          off_scale=off_scale, rendering=False, raise_exception=False,
                          seed=EVAL_SEED, with_window=False)
    EPM = env.envs_per_morph

    # per-body mean return for each variant, per seed -> results[name] = (n_seeds, n_bodies)
    results = {}
    for name, tmpl in VARIANT_TMPL.items():
        per_seed = []
        for s in SEEDS:
            ckpt = _ROOT / tmpl.format(s=s)
            if not ckpt.exists():
                print(f"[compare] SKIP {name}/{s}: missing {ckpt}"); continue
            net, obs_norm, _ = vg._load(ckpt, cfg, device)
            ep = _rollout_return(net, obs_norm, env, device)
            per_seed.append([ep[i * EPM:(i + 1) * EPM].mean() for i in range(len(bodies))])
        if per_seed:
            results[name] = np.array(per_seed)   # (n_seeds, n_bodies)

    # aggregate per leg-count: mean over seeds AND bodies-in-count
    names = list(results)
    print(f"\n[compare] mean realized return per leg-count  (seeds {SEEDS}, K={K_PER_COUNT} bodies/count, "
          f"{EPM} envs each; seed-averaged)")
    hdr = "  legs  " + "".join(f"{n:>12}" for n in names) + f"{'n_bodies':>10}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for c in counts:
        m = bcount == c
        row = f"  {c:>4}  " + "".join(f"{results[n][:, m].mean():>12.0f}" for n in names) + f"{m.sum():>10}"
        print(row)
    print("  " + "-" * (len(hdr) - 2))
    # overall row with seed spread (mean over bodies per seed, then mean +/- std over seeds)
    print("   all  " + "".join(f"{results[n].mean(1).mean():>12.0f}" for n in names) + f"{len(bodies):>10}")
    print("  +-sd  " + "".join(f"{results[n].mean(1).std():>12.0f}" for n in names))

    np.savez_compressed(_ROOT / "data" / "value_grad_prop" / "variant_compare.npz",
                        bodies_legcount=bcount, counts=np.array(counts), seeds=np.array(SEEDS),
                        **{f"ret_{n}": results[n] for n in names})

    # grouped bar chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = np.arange(len(counts)); w = 0.8 / max(len(names), 1)
        fig, ax = plt.subplots(figsize=(9, 5))
        for j, n in enumerate(names):
            means = [results[n][:, bcount == c].mean() for c in counts]
            ax.bar(x + (j - (len(names) - 1) / 2) * w, means, w, label=n)
        ax.set_xticks(x); ax.set_xticklabels(counts)
        ax.set_xlabel("base leg-count"); ax.set_ylabel("mean realized return")
        ax.set_title("policy performance per leg-count, seed-avg (clean bodies, full-presence obs)")
        ax.legend()
        out = _ROOT / "data" / "value_grad_prop" / "variant_compare.png"
        plt.tight_layout(); plt.savefig(out, dpi=120)
        print(f"\n[compare] chart -> {out}")
    except Exception as e:
        print(f"[compare] (plot skipped: {e})")


if __name__ == "__main__":
    main()
