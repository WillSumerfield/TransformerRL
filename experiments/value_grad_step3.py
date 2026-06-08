"""Value-gradient Step 3 — held-out 5-leg interpolation (docs/value_gradient_propagation.md).

The 5-on-leg builds are HELD OUT of training (the guard removes them + strips 5-leg bias centers), so
no 5-on *return* ever trains the critic and the 5-ish p-region is sparsely seen. This probes whether
V's gradient INTERPOLATES into that region: bypass the guard, use the 5-leg topologies as bias centers,
draw p the √U way around them (so p sweeps the interior), Bernoulli-build, and measure dV/dp across the
swept p. The notebook overlays this binned dV/dp-vs-p signature on the in-distribution Step-1 curve;
success = the held-out 5-leg curve coincides with in-distribution in sign and magnitude across p.

Same pipeline as Step 1 (value_grad_prop.py), only the env differs (pick_pool = 5-leg, guard=False).
Writes data/value_grad_prop/<run>/step3.npz (consumed by notebooks/value_grad_prop.ipynb).
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
from envs.ant_envs.ant_binary_leg import AntBinaryLegEnv, _all_topologies, _N_LEGS

RUN        = "s42_nobern"
SEED       = 124          # eval seed (distinct from Step 1's 123)
N_PATTERNS = 1024

# --- addrm mode (python value_grad_step3.py addrm): per-limb dV/dp on 3 held-out 5-leg bases, each
# vs a +1 (random added leg -> 6) and a -1 (random removed leg -> 4) variant, at interior obs-p. ---
SEEDS      = ["s42_nobern", "s43_nobern", "s44_nobern"]
ENVS_PER   = 256
ON_P, OFF_P = 0.75, 0.25  # interior obs-p (in-distribution); bodies still built exactly
SEL_SEED   = 0            # fixed -> same 3 bases + added/removed legs across all trained seeds
N_BASES    = 3


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    run = sys.argv[1] if len(sys.argv) > 1 else RUN
    checkpoint = vg._checkpoint(run)
    cfg = yaml.safe_load(vg.CONFIG.read_text())
    reward_scale = cfg["params"]["config"].get("reward_shaper", {}).get("scale_value", 1.0)
    off_scale = cfg.get("env", {}).get("off_scale", 0.05)
    net, obs_norm, val_norm = vg._load(checkpoint, cfg, device)

    five = _all_topologies(5, 5)                                   # all held-out 5-leg bias centers
    env = AntBinaryLegEnv(N_PATTERNS, device, pick_pool=five, guard=False, off_scale=off_scale,
                          rendering=False, raise_exception=False, seed=SEED, with_window=False)
    p = env._p.astype(np.float32)                                  # (N, 8) continuous p around 5-leg S
    on_mask = np.zeros((env.total_num_envs, _N_LEGS), dtype=np.int8)
    for e, on in enumerate(env._on_sets):
        for nleg in on:
            on_mask[e, nleg - 1] = 1
    leg_count = on_mask.sum(1).astype(np.int16)

    print(f"[step3] {env.total_num_envs} held-out 5-leg patterns  sampled leg-counts "
          f"{np.bincount(leg_count).tolist()}  ->  measuring dV/dp")
    rec = vg._rollout(net, obs_norm, val_norm, env, reward_scale, device)

    out_dir = _ROOT / "data" / "value_grad_prop" / checkpoint.resolve().parent.parent.name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "step3.npz", p=p, on_mask=on_mask, leg_count=leg_count,
                        reward_scale=reward_scale, seed=SEED, n_patterns=env.total_num_envs,
                        checkpoint=str(checkpoint), **rec)

    pf, gf = p.ravel(), rec["g_mean"].ravel()                     # binned dV/dp-vs-p (overlay headline)
    edges = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(pf, edges) - 1, 0, 9)
    print("[step3] held-out 5-leg dV/dp binned by p (should coincide with in-dist Step 1):")
    for b in range(10):
        m = idx == b
        if m.any():
            print(f"  p∈[{edges[b]:.1f},{edges[b+1]:.1f})  n={m.sum():6d}  "
                  f"mean dV/dp={gf[m].mean():+.4f}  frac>0={(gf[m]>0).mean():.3f}")
    print(f"[step3] wrote -> {out_dir / 'step3.npz'}")


def _mask(sets):
    m = np.zeros((len(sets), _N_LEGS), dtype=np.int8)
    for i, s in enumerate(sets):
        for n in s:
            m[i, n - 1] = 1
    return m


def main_addrm():
    """Per-limb dV/dp on 3 held-out 5-leg bases, each vs a +1 (random added leg) and -1 (random removed
    leg) variant, at interior obs-p, seed-averaged. Writes data/value_grad_prop/step3_addrm.npz."""
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    cfg = yaml.safe_load(vg.CONFIG.read_text())
    reward_scale = cfg["params"]["config"].get("reward_shaper", {}).get("scale_value", 1.0)
    off_scale = cfg.get("env", {}).get("off_scale", 0.05)

    sel = np.random.default_rng(SEL_SEED)                          # fixed -> identical across seeds
    five = _all_topologies(5, 5)
    bases = [frozenset(five[i]) for i in sel.choice(len(five), N_BASES, replace=False)]
    full = set(range(1, _N_LEGS + 1))
    add_leg, rm_leg, adds, rms = [], [], [], []
    for b in bases:
        a = int(sel.choice(sorted(full - b)))                     # random off leg -> add (6-leg)
        r = int(sel.choice(sorted(b)))                            # random on leg  -> remove (4-leg)
        add_leg.append(a); rm_leg.append(r)
        adds.append(frozenset(b | {a})); rms.append(frozenset(b - {r}))

    groups = [g for i in range(N_BASES) for g in (bases[i], adds[i], rms[i])]   # base,add,rm per base
    env = AntBinaryLegEnv(len(groups) * ENVS_PER, device, on_sets=groups, on_p=ON_P, off_p=OFF_P,
                          off_scale=off_scale, rendering=False, raise_exception=False,
                          seed=SEED, with_window=False)
    EPM = env.envs_per_morph

    seeds = [s for s in SEEDS if vg._checkpoint(s).exists()]
    g_base = np.zeros((len(seeds), N_BASES, _N_LEGS), np.float32)
    g_add, g_rm = np.zeros_like(g_base), np.zeros_like(g_base)
    for si, s in enumerate(seeds):
        net, obs_norm, val_norm = vg._load(vg._checkpoint(s), cfg, device)
        rec = vg._rollout(net, obs_norm, val_norm, env, reward_scale, device)
        for i in range(N_BASES):
            base_i = 3 * i
            for arr, gj in ((g_base, base_i), (g_add, base_i + 1), (g_rm, base_i + 2)):
                arr[si, i] = rec["g_mean"][gj * EPM:(gj + 1) * EPM].mean(0)

    out = _ROOT / "data" / "value_grad_prop" / "step3_addrm.npz"
    np.savez_compressed(out, seeds=np.array(seeds), on_p=ON_P, off_p=OFF_P,
                        base_mask=_mask(bases), add_mask=_mask(adds), rm_mask=_mask(rms),
                        added_leg=np.array(add_leg, np.int8), removed_leg=np.array(rm_leg, np.int8),
                        g_base=g_base, g_add=g_add, g_rm=g_rm)
    LEG = ['1F', '2FR', '3R', '4BR', '5B', '6BL', '7L', '8FL']
    print(f"[addrm] obs p on={ON_P} off={OFF_P}, seed-avg over {seeds}")
    for i, b in enumerate(bases):
        print(f"  base {sorted(b)}  +leg {add_leg[i]}({LEG[add_leg[i]-1]})  "
              f"-leg {rm_leg[i]}({LEG[rm_leg[i]-1]})")
    print(f"[addrm] wrote -> {out}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "addrm":
        main_addrm()
    else:
        main()
