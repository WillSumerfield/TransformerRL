"""Value-gradient propagation study — Phase 1 measurement (docs/value_gradient_propagation.md).

Loads a presence-probability checkpoint (AntBinaryLegEnv: all 8 legs active, presence carried by a
continuous probability p in BOTH length slots; body Bernoulli-sampled from p) and measures dV/dp per
leg over a population of sampled p-vectors. The gradient is the codesign training-signal proxy: its
SIGN should track the leg's DESIRED state (positive = should be on, negative = should be off),
independent of its current state.

Two metrics per leg per pattern:
  g0     : dV/dp at the t=0 reset state (canonical pose; the value net's static prior)
  g_mean : dV/dp averaged over a deterministic-mu rollout (the expected gradient a generator would see)

Value pipeline mirrors leg_ablation_value.py / rollout.py: obs RMS-normalized with the DOF-mask tail
restored raw (TransformerMaskedNorm), critic denormalized by value_mean_std, divided by the reward-shaper
scale -> raw-reward units. Unlike those, the forward pass is GRAD-ENABLED (we differentiate V w.r.t. the
raw length slots; both slots of a leg hold the same p, so per-leg gradient = hip-slot + ankle-slot = dV/dp).

Step 1 (this script): population of in-distribution p-vectors -> the myopia test, now read as dV/dp
binned by p (is dV/dp positive at low p?). Each env draws one p and one Bernoulli body; the population
+ p-binning recovers the expectation. Step 2/3's curated morphologies are added separately. Writes one
.npz per checkpoint to data/value_grad_prop/<run>/.
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
from tqdm import tqdm
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from transformer_rl.models import MultiMorphLegTransformerBuilder
from transformer_rl.tokenize import OBS_DIM_8, LEN_DIM_8, MASK_DIM_8
from envs.ant_envs.ant_binary_leg import AntBinaryLegEnv

_OBS_TOTAL = OBS_DIM_8 + LEN_DIM_8 + MASK_DIM_8   # 139
_LEN_OFF   = OBS_DIM_8                             # 107: start of the length block
_MASK_DIM  = MASK_DIM_8                            # 16 (DOF mask tail)
_N_ACT     = 16
_N_LEGS    = 8

# --- template constants: run name selects the checkpoint; override with argv[1] (e.g. s43) ------
RUN        = "s42"        # run dir under runs/ant_binary/binary_transformer/<RUN>/
CONFIG     = _ROOT / "configs/ppo_ant_binary.yaml"
SEED       = 123          # eval seed (distinct from training seed); draws the on/off population
N_PATTERNS = 1024         # distinct on/off patterns (one env each, epm=1)
# ----------------------------------------------------------------------------------------------


def _checkpoint(run: str) -> Path:
    return _ROOT / f"runs/ant_binary/binary_transformer/{run}/nn/ant_binary_transformer.pth"


def _load(checkpoint: Path, cfg: dict, device: torch.device):
    """Rebuild the network + obs/value normalizers from a checkpoint (no rl_games player)."""
    net = MultiMorphLegTransformerBuilder.Network(
        params=cfg["params"]["network"], actions_num=_N_ACT,
        input_shape=(_OBS_TOTAL,), num_seqs=1,
    )
    raw = torch.load(checkpoint, map_location=device, weights_only=False)["model"]
    state = {k.removeprefix("_orig_mod."): v for k, v in raw.items()}

    def _sub(prefix):
        return {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}

    net.load_state_dict(_sub("a2c_network."))
    obs_norm = RunningMeanStd(_OBS_TOTAL).to(device)
    obs_norm.load_state_dict(_sub("running_mean_std."))
    val_norm = RunningMeanStd((1,)).to(device)
    val_norm.load_state_dict(_sub("value_mean_std."))
    net.to(device).eval(); obs_norm.eval(); val_norm.eval()
    return net, obs_norm, val_norm


def _grad_value(net, obs_norm, val_norm, obs, reward_scale):
    """Per-leg dV/dp (raw-reward units) + deterministic action mu, at the given obs.

    Differentiates the raw-reward value w.r.t. the raw obs (the length slots are the leaf of interest).
    Both length slots of a leg hold the same presence prob p, so dV/dp = hip-slot grad + ankle-slot grad.
    """
    x = obs.detach().clone().requires_grad_(True)
    normed = obs_norm(x).clone()
    normed[..., -_MASK_DIM:] = x[..., -_MASK_DIM:]               # mask tail raw (TransformerMaskedNorm)
    mu, _, value, _ = net({"obs": normed})
    val = val_norm(value.view(-1, 1), denorm=True).squeeze(-1) / reward_scale
    val.sum().backward()
    g = x.grad[:, _LEN_OFF:_LEN_OFF + LEN_DIM_8]                 # (N, 16): 8 hip + 8 ankle slots
    per_leg = g[:, :_N_LEGS] + g[:, _N_LEGS:]                    # both slots = p -> dV/dp (N, 8)
    return per_leg.detach(), mu.detach().clamp(-1.0, 1.0)


def _rollout(net, obs_norm, val_norm, env, reward_scale, device):
    """One deterministic-mu episode per env; record g0 (t=0) and g_mean/g_std over the episode."""
    n = env.total_num_envs
    L = env.max_episode_length

    g0    = torch.zeros((n, _N_LEGS), device=device)
    gsum  = torch.zeros((n, _N_LEGS), device=device)
    gsq   = torch.zeros((n, _N_LEGS), device=device)
    gcnt  = torch.zeros((n, 1), device=device)
    ep_reward = torch.zeros(n, device=device)
    ep_len    = torch.zeros(n, dtype=torch.int16, device=device)
    ep_term   = torch.zeros(n, dtype=torch.bool, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)       # first episode finished

    obs, _ = env.reset()
    t = 0
    while not bool(done.all()) and t < L + 5:
        per_leg, mu = _grad_value(net, obs_norm, val_norm, obs, reward_scale)
        active = ~done
        if t == 0:
            g0[:] = per_leg
        am = active.unsqueeze(1).float()
        gsum += per_leg * am
        gsq  += (per_leg ** 2) * am
        gcnt += am

        obs, rew, term, trunc, _ = env.step(mu)
        fin = (term | trunc)
        ep_reward += rew * active.float()
        newly = active & fin
        ep_len[newly]  = t + 1
        ep_term[newly] = term[newly]
        done |= fin
        t += 1

    g_mean = gsum / gcnt.clamp(min=1)
    g_std  = (gsq / gcnt.clamp(min=1) - g_mean ** 2).clamp(min=0).sqrt()
    return dict(
        g0=g0.cpu().numpy(), g_mean=g_mean.cpu().numpy(), g_std=g_std.cpu().numpy(),
        ep_reward=ep_reward.cpu().numpy(), ep_len=ep_len.cpu().numpy(),
        ep_terminated=ep_term.cpu().numpy(),
    )


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    run = sys.argv[1] if len(sys.argv) > 1 else RUN
    checkpoint = _checkpoint(run)
    cfg = yaml.safe_load(CONFIG.read_text())
    reward_scale = cfg["params"]["config"].get("reward_shaper", {}).get("scale_value", 1.0)
    env_cfg = {k: v for k, v in cfg.get("env", {}).items() if k != "sample_morphs"}
    net, obs_norm, val_norm = _load(checkpoint, cfg, device)

    env = AntBinaryLegEnv(N_PATTERNS, device, rendering=False, raise_exception=False,
                          seed=SEED, with_window=False, **env_cfg)
    p = env._p.astype(np.float32)                                 # (N, 8) continuous presence prob
    on_mask = np.zeros((env.total_num_envs, _N_LEGS), dtype=np.int8)   # the Bernoulli-sampled body
    for e, on in enumerate(env._on_sets):
        for nleg in on:
            on_mask[e, nleg - 1] = 1
    leg_count = on_mask.sum(1).astype(np.int16)                   # sampled on-leg count

    print(f"[grad] {env.total_num_envs} patterns  sampled leg-counts {np.bincount(leg_count).tolist()}"
          f"  ->  measuring dV/dp")
    rec = _rollout(net, obs_norm, val_norm, env, reward_scale, device)

    stem = checkpoint.resolve().parent.parent.name
    out_dir = _ROOT / "data" / "value_grad_prop" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "grad.npz",
        p=p, on_mask=on_mask, leg_count=leg_count,
        reward_scale=reward_scale, seed=SEED, n_patterns=env.total_num_envs,
        checkpoint=str(checkpoint), **rec,
    )
    # Headline: the myopia test as dV/dp binned by p. Flatten (env, leg) pairs, bin by p, report mean
    # g_mean per bin — dV/dp should stay POSITIVE at low p (marginal value of adding a leg).
    pf, gf = p.ravel(), rec["g_mean"].ravel()
    edges = np.linspace(0.0, 1.0, 11)
    idx = np.clip(np.digitize(pf, edges) - 1, 0, 9)
    print("[grad] dV/dp binned by p (myopia = positive at low p):")
    for b in range(10):
        m = idx == b
        if m.any():
            print(f"  p∈[{edges[b]:.1f},{edges[b+1]:.1f})  n={m.sum():6d}  "
                  f"mean dV/dp={gf[m].mean():+.4f}  frac>0={ (gf[m]>0).mean():.3f}")
    print(f"[grad] wrote -> {out_dir / 'grad.npz'}")


if __name__ == "__main__":
    main()
