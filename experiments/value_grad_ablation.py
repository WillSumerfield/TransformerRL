"""Value-gradient grounding — does realized return track the critic's V and its ∂V/∂p? (docs/value_gradient_propagation.md)

The earlier steps judged ∂V/∂p against *intuition*; this grounds it against *realized return* using the
real leg-ablation idea. For B in-distribution base bodies S (deterministic builds at the trusted interior
obs-p 0.75/0.25), build S plus its 8 single-leg **toggles** (each leg flipped on↔off) as N+1 groups, and
for every body measure: realized episode return R (deterministic-mu rollout), the critic's value V (at
reset, predicts the return), and per-leg ∂V/∂p (rollout-averaged) at the base.

Two grounding tests (notebook does the scatter/correlation):
  1. **Value calibration** — V(body) vs R(body) over all bodies. Prerequisite: if V doesn't track R,
     no gradient of V can.
  2. **Gradient grounding** — at each base, ∂V/∂p_L vs the realized marginal value of leg L,
     ΔR_L = R(body with L on) − R(body with L off) = sign_L·(R(base) − R(toggle_L)), sign_L=+1 if L is
     on in S else −1. Does a more-positive gradient predict a more-positive realized leg value?

Same value/grad pipeline as value_grad_prop.py; bodies are deterministic (on_sets) like Step 2. One env,
all 3 trained seeds. Writes data/value_grad_prop/ablation.npz. Run `python value_grad_ablation.py [smoke]`.
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
from envs.ant_envs.ant_binary_leg import AntBinaryLegEnv, HELDOUT, _N_LEGS
from envs.ant_envs.ant_multimorph import _stable_morphologies

ON_P, OFF_P  = 0.75, 0.25     # trusted interior operating point (corners are OOD; see Step 2)
COUNTS       = (3, 4, 6, 7)   # base leg-counts (in-distribution; 5 is held out)
B_PER_COUNT  = 4
ENVS_PER     = 128
SEL_SEED     = 7              # fixed -> same bases across trained seeds
EVAL_SEED    = 123
SEEDS        = ["s42", "s43", "s44"]

if len(sys.argv) > 1 and sys.argv[1] == "smoke":
    COUNTS, B_PER_COUNT, ENVS_PER, SEEDS = (3, 4), 1, 16, ["s42"]   # fast path for a wiring check


def _grad_value_val(net, obs_norm, val_norm, obs, reward_scale):
    """vg._grad_value plus the per-env value V (raw-reward units) at the same obs."""
    x = obs.detach().clone().requires_grad_(True)
    normed = obs_norm(x).clone()
    normed[..., -vg._MASK_DIM:] = x[..., -vg._MASK_DIM:]
    mu, _, value, _ = net({"obs": normed})
    val = val_norm(value.view(-1, 1), denorm=True).squeeze(-1) / reward_scale
    val.sum().backward()
    g = x.grad[:, vg._LEN_OFF:vg._LEN_OFF + 16]
    per_leg = g[:, :_N_LEGS] + g[:, _N_LEGS:]
    return per_leg.detach(), mu.detach().clamp(-1.0, 1.0), val.detach()


def _rollout(net, obs_norm, val_norm, env, reward_scale, device):
    """One deterministic-mu episode per env; record V0 (value at reset), realized return, and g_mean."""
    n = env.total_num_envs
    L = env.max_episode_length
    gsum = torch.zeros((n, _N_LEGS), device=device)
    gcnt = torch.zeros((n, 1), device=device)
    v0   = torch.zeros(n, device=device)
    ep_reward = torch.zeros(n, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)

    obs, _ = env.reset()
    t = 0
    while not bool(done.all()) and t < L + 5:
        per_leg, mu, val = _grad_value_val(net, obs_norm, val_norm, obs, reward_scale)
        active = ~done
        if t == 0:
            v0[:] = val
        gsum += per_leg * active.unsqueeze(1).float()
        gcnt += active.unsqueeze(1).float()
        obs, rew, term, trunc, _ = env.step(mu)
        ep_reward += rew * active.float()
        done |= (term | trunc)
        t += 1
    return v0.cpu().numpy(), ep_reward.cpu().numpy(), (gsum / gcnt.clamp(min=1)).cpu().numpy()


def _mask(sets):
    m = np.zeros((len(sets), _N_LEGS), dtype=np.int8)
    for i, s in enumerate(sets):
        for nleg in s:
            m[i, nleg - 1] = 1
    return m


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    cfg = yaml.safe_load(vg.CONFIG.read_text())
    reward_scale = cfg["params"]["config"].get("reward_shaper", {}).get("scale_value", 1.0)
    off_scale = cfg.get("env", {}).get("off_scale", 0.05)

    # In-distribution base bodies: stable, not held out, B_PER_COUNT per leg-count.
    sel = np.random.default_rng(SEL_SEED)
    by_n: dict[int, list] = {}
    for s in _stable_morphologies():
        if frozenset(s) not in HELDOUT:
            by_n.setdefault(len(s), []).append(frozenset(s))
    bases = []
    for c in COUNTS:
        pool = by_n[c]
        bases += [pool[i] for i in sel.choice(len(pool), B_PER_COUNT, replace=False)]
    B = len(bases)
    full = set(range(1, _N_LEGS + 1))

    # 9 groups per base: slot 0 = base S, slot L = S with leg L toggled (on<->off).
    groups = []
    for S in bases:
        groups.append(S)
        for leg in range(1, _N_LEGS + 1):
            groups.append(frozenset(S - {leg}) if leg in S else frozenset(S | {leg}))
    GPB = _N_LEGS + 1
    env = AntBinaryLegEnv(len(groups) * ENVS_PER, device, on_sets=groups, on_p=ON_P, off_p=OFF_P,
                          off_scale=off_scale, rendering=False, raise_exception=False,
                          seed=EVAL_SEED, with_window=False)
    EPM = env.envs_per_morph

    seeds = [s for s in SEEDS if vg._checkpoint(s).exists()]
    V  = np.zeros((len(seeds), B, GPB), np.float32)   # value at reset, per body
    R  = np.zeros((len(seeds), B, GPB), np.float32)   # realized return, per body
    Gb = np.zeros((len(seeds), B, _N_LEGS), np.float32)  # per-leg dV/dp at the base body
    for si, s in enumerate(seeds):
        net, obs_norm, val_norm = vg._load(vg._checkpoint(s), cfg, device)
        v0, ret, gm = _rollout(net, obs_norm, val_norm, env, reward_scale, device)
        for bi in range(B):
            for slot in range(GPB):
                sl = slice((bi * GPB + slot) * EPM, (bi * GPB + slot + 1) * EPM)
                V[si, bi, slot] = v0[sl].mean()
                R[si, bi, slot] = ret[sl].mean()
            Gb[si, bi] = gm[(bi * GPB) * EPM:(bi * GPB + 1) * EPM].mean(0)   # base = slot 0

    out = _ROOT / "data" / "value_grad_prop" / "ablation.npz"
    np.savez_compressed(out, seeds=np.array(seeds), on_p=ON_P, off_p=OFF_P,
                        base_mask=_mask(bases), leg_count=np.array([len(s) for s in bases], np.int8),
                        V=V, R=R, g_base=Gb)

    # quick console grounding (seed-avg)
    Vm, Rm, Gm = V.mean(0), R.mean(0), Gb.mean(0)
    sign = np.where(_mask(bases) == 1, 1.0, -1.0)              # +1 if leg on in base else -1
    dR = sign * (Rm[:, [0]] - Rm[:, 1:])                       # realized marginal value of each leg
    vc = np.corrcoef(Vm.ravel(), Rm.ravel())[0, 1]
    gx, gy = Gm.ravel(), dR.ravel()
    gc = np.corrcoef(gx, gy)[0, 1]
    print(f"[ablation] {B} bases x {GPB} bodies @ {EPM} envs, seeds {seeds}")
    print(f"[ablation] value calibration   corr(V, R)        = {vc:+.3f}  (n={Vm.size})")
    print(f"[ablation] gradient grounding   corr(dV/dp, dR)   = {gc:+.3f}  (n={gx.size})")
    print(f"[ablation]                      sign-agree(dV/dp, dR) = {(np.sign(gx) == np.sign(gy)).mean():.3f}")
    print(f"[ablation] wrote -> {out}")


if __name__ == "__main__":
    main()
