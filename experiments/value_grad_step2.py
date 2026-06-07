"""Value-gradient Step 2 — curated held-out morphologies vs trained twins (docs/value_gradient_propagation.md).

Qualitative showcase on 3 hand-picked HELD-OUT on-sets, each paired with a TRAINED twin of (nearly) the
same shape (leg codes 1F 2FR 3R 4BR 5B 6BL 7L 8FL):
  redundancy       : held {2,5,6,7} vs twin {3,4,5,8}  (left-right MIRROR)  -> lone leg 2 high |g|,
                     adjacent trio 5,6,7 partly redundant (a flanked leg may go negative).
  critical_missing : held {5,6,7}   vs twin {3,4,5}    (left-right MIRROR)  -> off legs that restore
                     forward balance (front 1,2,8 / right 3,4) = largest positive 'add' grad.
  symmetric        : held {1,3,5,7} vs twin {2,4,6,8}  (ROTATION by 1)      -> ~uniform moderate on,
                     mild-positive off.

Built DETERMINISTICALLY (p in {0,1} = exactly the on-set; no Bernoulli, no guard). The mirror twins
preserve +x (ant is L/R symmetric) so they are task-equivalent -> expect near-exact agreement after
remapping legs through the transform; the rotation twin is similar-only. Per-leg dV/dp is averaged over
a deterministic-mu rollout, per group, per trained seed. The TRANSFORM (held-leg -> partner twin-leg)
is saved so the notebook can overlay aligned bars. Writes data/value_grad_prop/step2.npz.
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
from envs.ant_envs.ant_binary_leg import AntBinaryLegEnv

# Left-right mirror (reflect across +x): front 1 / back 5 fixed, swap 2<->8 3<->7 4<->6.
MIRROR = {1: 1, 2: 8, 3: 7, 4: 6, 5: 5, 6: 4, 7: 3, 8: 2}
ROT1   = {i: (i % 8) + 1 for i in range(1, 9)}          # rotate one slot: 1->2 ... 8->1

# (name, held-out on-set, trained twin on-set). The twin is the held body under a symmetry that gives
# the held-leg -> twin-leg correspondence (TRANSFORMS, aligned with PAIRS): redundancy & critical_missing
# use the left-right MIRROR (task-equivalent, preserves +x), symmetric uses ROT1 (similar-only).
PAIRS = [
    ("redundancy",       {2, 5, 6, 7}, {8, 3, 4, 5}),
    ("critical_missing", {5, 6, 7},    {3, 4, 5}),
    ("symmetric",        {1, 3, 5, 7}, {2, 4, 6, 8}),
]
TRANSFORMS = [MIRROR, MIRROR, ROT1]
ENVS_PER  = 256
SEEDS     = ["s42", "s43", "s44"]
EVAL_SEED = 123


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    # Obs-p reported for on/off legs (body is still built EXACTLY = on-set). Default = corners {1,0};
    # pass e.g. `0.75 0.25` to probe dV/dp off the p=0/1 corners (same bodies, interior obs-p).
    on_p  = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    off_p = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    cfg = yaml.safe_load(vg.CONFIG.read_text())
    reward_scale = cfg["params"]["config"].get("reward_shaper", {}).get("scale_value", 1.0)
    off_scale = cfg.get("env", {}).get("off_scale", 0.05)

    names = [p[0] for p in PAIRS]
    held = [frozenset(p[1]) for p in PAIRS]
    twin = [frozenset(p[2]) for p in PAIRS]
    # tmap[gi, L-1] = twin leg partnering held leg L (1-indexed), from the pair's symmetry; for aligning
    # twin bars onto the held frame. Verify the symmetry actually carries the held body onto the twin set.
    tmap = np.array([[t[l] for l in range(1, 9)] for t in TRANSFORMS], dtype=np.int8)
    for gi in range(len(PAIRS)):
        assert frozenset(TRANSFORMS[gi][l] for l in held[gi]) == twin[gi], \
            f"{names[gi]}: transform(held) {sorted(TRANSFORMS[gi][l] for l in held[gi])} != twin {sorted(twin[gi])}"

    groups = held + twin                                # 6 groups: 3 held, then 3 twins
    env = AntBinaryLegEnv(len(groups) * ENVS_PER, device, on_sets=groups, on_p=on_p, off_p=off_p,
                          off_scale=off_scale, rendering=False, raise_exception=False,
                          seed=EVAL_SEED, with_window=False)
    EPM = env.envs_per_morph
    G = len(PAIRS)

    def onmask(sets):
        m = np.zeros((G, 8), dtype=np.int8)
        for gi, s in enumerate(sets):
            for n in s:
                m[gi, n - 1] = 1
        return m

    seeds = [s for s in SEEDS if vg._checkpoint(s).exists()]
    shape = (len(seeds), G, 8)
    g_held, g_twin   = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    g0_held, g0_twin = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
    r_held = np.zeros((len(seeds), G), np.float32)
    r_twin = np.zeros((len(seeds), G), np.float32)
    for si, s in enumerate(seeds):
        net, obs_norm, val_norm = vg._load(vg._checkpoint(s), cfg, device)
        rec = vg._rollout(net, obs_norm, val_norm, env, reward_scale, device)
        for gi in range(G):
            hs = slice(gi * EPM, (gi + 1) * EPM)                # held group gi
            ts = slice((G + gi) * EPM, (G + gi + 1) * EPM)      # twin group gi
            g_held[si, gi],  g_twin[si, gi]  = rec["g_mean"][hs].mean(0), rec["g_mean"][ts].mean(0)
            g0_held[si, gi], g0_twin[si, gi] = rec["g0"][hs].mean(0),     rec["g0"][ts].mean(0)
            r_held[si, gi],  r_twin[si, gi]  = rec["ep_reward"][hs].mean(), rec["ep_reward"][ts].mean()

    tag = "" if (on_p, off_p) == (1.0, 0.0) else f"_p{on_p:g}-{off_p:g}"   # keep corner ref separate
    out = _ROOT / "data" / "value_grad_prop" / f"step2{tag}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, names=np.array(names), seeds=np.array(seeds), tmap=tmap,
                        held_mask=onmask(held), twin_mask=onmask(twin),
                        g_held=g_held, g_twin=g_twin, g0_held=g0_held, g0_twin=g0_twin,
                        r_held=r_held, r_twin=r_twin, envs_per=EPM)

    hm = onmask(held)
    gm = g_held.mean(0)                                  # seed-avg held (G, 8)
    gt = g_twin.mean(0)
    print(f"obs p: on={on_p} off={off_p}  (bodies built exactly = on-set)")
    print("seed-avg per-leg dV/dp (leg 1..8; *=on)   held reward / twin reward")
    for gi, name in enumerate(names):
        held_cells = " ".join(f"{gm[gi, l]:+6.1f}{'*' if hm[gi, l] else ' '}" for l in range(8))
        # twin aligned onto held frame for a direct partner comparison
        aligned = np.array([gt[gi, tmap[gi, l] - 1] for l in range(8)])
        twin_cells = " ".join(f"{aligned[l]:+6.1f}" for l in range(8))
        print(f"  {name:<16} held {held_cells}   {r_held[:, gi].mean():7.1f}")
        print(f"  {'':<16} twin {twin_cells}   {r_twin[:, gi].mean():7.1f}")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
