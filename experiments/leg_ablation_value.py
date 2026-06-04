"""Leg-ablation value study — does the critic's value estimate identify which legs a body needs?

For each leg count 3..8 we draw M random base morphs (topology + per-leg hip/ankle lengths). Each base
morph is rolled out alongside its single-leg ablations as N+1 EnvironmentGroups (group 0 = intact,
group i = leg-i removed), K episodes per env. We record per-step value (raw-reward units) + reward
traces and per-episode summaries, so leg importance can be compared as predicted V(intact)-V(ablate_i)
vs realized Return(intact)-Return(ablate_i), paired within a base morph. See experiments/CONTEXT.md.

One env build per base morph (removing a leg rebakes geometry); one .npz per base morph under
data/leg_ablation/<stem>/. K episodes/env is fixed so unstable bodies can't contribute extra short
episodes and bias the per-group statistics.

Value pipeline mirrors the morph-value sweep (rollout.py): obs RMS-normalized with the DOF-mask tail
restored raw (TransformerMaskedNorm), critic output denormalized by value_mean_std, then divided by
the reward-shaper scale so it is directly comparable to raw episode reward.
"""
import gc
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import random
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

import vlearn as v
from transformer_rl.models import MultiMorphLegTransformerBuilder
from transformer_rl.tokenize import OBS_DIM_8, LEN_DIM_8, MASK_DIM_8
from envs.ant_envs.ant_multimorph import AntMultiMorphEnv, _stable_morphologies
from envs.ant_envs.build_vsim import Morphology, HIP_RANGE, ANKLE_RANGE

_OBS_TOTAL = OBS_DIM_8 + LEN_DIM_8 + MASK_DIM_8   # 139
_MASK_DIM  = MASK_DIM_8                            # 16 (DOF mask tail of obs)
_N_ACT     = 16

# --- template constants: edit CHECKPOINT for your run -----------------------------------------
CHECKPOINT     = _ROOT / "runs/ant_full/full_transformer/LegAblation/nn/ant_full_transformer.pth"
CONFIG         = _ROOT / "configs/ppo_ant_full.yaml"
SEED           = 42
M              = 10        # base morphs sampled per leg count
K              = 5         # episodes recorded per env (fixed -> unbiased per-group stats)
ENVS_PER_GROUP = 455       # capped so the 8-leg worst case (9 groups) fits a 4096-env budget
MIN_LEGS, MAX_LEGS = 3, 8
# ----------------------------------------------------------------------------------------------


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


@torch.no_grad()
def _value_action(net, obs_norm, val_norm, obs, reward_scale):
    """Deterministic action (mu) + per-env value in raw-reward units, matching rollout.py units."""
    normed = obs_norm(obs).clone()
    normed[..., -_MASK_DIM:] = obs[..., -_MASK_DIM:]            # mask tail raw (TransformerMaskedNorm)
    mu, _, value, _ = net({"obs": normed})
    val = val_norm(value.view(-1, 1), denorm=True).squeeze(-1) / reward_scale
    return mu.clamp(-1.0, 1.0), val


def _sample_base_morphs(rng: random.Random) -> list[Morphology]:
    """M morphs per leg count: topology uniform within the count, hip/ankle lengths uniform per leg."""
    by_legs: dict[int, list[list[int]]] = {}
    for m in _stable_morphologies():
        by_legs.setdefault(len(m), []).append(sorted(m))
    out = []
    for legs in range(MIN_LEGS, MAX_LEGS + 1):
        for _ in range(M):
            topo = rng.choice(by_legs[legs])
            hip = {n: rng.uniform(*HIP_RANGE) for n in topo}
            ank = {n: rng.uniform(*ANKLE_RANGE) for n in topo}
            out.append(Morphology(frozenset(topo), hip, ank))
    return out


def _ablate(m: Morphology, leg: int) -> Morphology:
    """Same body with one leg (and its hip+ankle tokens / 2 DOFs) removed; others unchanged."""
    return Morphology(
        frozenset(m.legs - {leg}),
        {n: h for n, h in m.hip_lengths.items() if n != leg},
        {n: a for n, a in m.ankle_lengths.items() if n != leg},
    )


@torch.no_grad()
def _rollout_base(net, obs_norm, val_norm, env, reward_scale, device):
    """Roll K episodes per env; record per-step value/reward + per-episode summaries.

    Each env runs until it has completed exactly K episodes (auto-reset between them); once done it
    is masked out while slower envs finish, so episode counts are equal across groups. Per-step
    quantities are written into [env, episode, step] slots; episode reward/length/term accumulate.
    """
    n = env.total_num_envs
    L = env.max_episode_length
    vs = torch.full((n, K, L), float("nan"), dtype=torch.float16, device=device)
    rs = torch.full((n, K, L), float("nan"), dtype=torch.float16, device=device)
    ep_reward = torch.zeros((n, K), dtype=torch.float32, device=device)
    ep_len    = torch.zeros((n, K), dtype=torch.int16, device=device)
    ep_term   = torch.zeros((n, K), dtype=torch.bool, device=device)

    cur_ep = torch.zeros(n, dtype=torch.long, device=device)   # episodes completed so far
    cur_t  = torch.zeros(n, dtype=torch.long, device=device)   # step within current episode
    ar     = torch.arange(n, device=device)

    obs, _ = env.reset()
    cap = K * L + 5
    step = 0
    active = cur_ep < K
    while bool(active.any()) and step < cap:
        action, val = _value_action(net, obs_norm, val_norm, obs, reward_scale)
        obs, rew, term, trunc, _ = env.step(action)
        done = term | trunc

        valid = active & (cur_t < L)
        flat = (ar * K + cur_ep) * L + cur_t          # [env, ep, step] -> flat
        vs.view(-1)[flat[valid]] = val[valid].half()
        rs.view(-1)[flat[valid]] = rew[valid].half()

        ep_flat = ar * K + cur_ep                      # [env, ep] -> flat
        ep_reward.view(-1)[ep_flat[valid]] += rew[valid]
        nd = done & active                             # episodes ending this step
        ep_len.view(-1)[ep_flat[nd]]  = (cur_t[nd] + 1).to(torch.int16)
        ep_term.view(-1)[ep_flat[nd]] = term[nd]

        cur_ep = cur_ep + nd.long()
        cur_t  = torch.where(done, torch.zeros_like(cur_t), cur_t + 1)
        active = cur_ep < K
        step += 1

    return dict(
        value_steps=vs.cpu().numpy(), reward_steps=rs.cpu().numpy(),
        ep_reward=ep_reward.cpu().numpy(), ep_len=ep_len.cpu().numpy(),
        ep_terminated=ep_term.cpu().numpy(),
    )


def _teardown(env):
    """Drop gym-backed refs then free the gym, so the next base morph builds cleanly (cf. _rebuild)."""
    env._get_cmd_array = env._set_cmd_array = None
    env.all_motor_cmd_array = env.all_sensor_cmd_array = None
    env.groups = []
    env.env_groups = []
    env.gym = env.gym_render = None
    gc.collect()
    v.delete_gym()


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    cfg = yaml.safe_load(CONFIG.read_text())
    reward_scale = cfg["params"]["config"].get("reward_shaper", {}).get("scale_value", 1.0)
    net, obs_norm, val_norm = _load(CHECKPOINT, cfg, device)

    bases = _sample_base_morphs(random.Random(SEED))
    stem = CHECKPOINT.resolve().parent.parent.name
    out_dir = _ROOT / "data" / "leg_ablation" / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ablation] {len(bases)} base morphs ({M}/leg-count) x up-to-{MAX_LEGS + 1} groups "
          f"@ {ENVS_PER_GROUP} envs/group, K={K}  ->  {out_dir}")

    for bi, base in enumerate(tqdm(bases, desc="base morphs", unit="morph")):
        legs = sorted(base.legs)
        morphs = [base] + [_ablate(base, leg) for leg in legs]   # group 0 intact, then per-leg ablation
        removed_leg = np.array([0] + legs, dtype=np.int16)       # 0 = intact reference group
        num_envs = len(morphs) * ENVS_PER_GROUP

        env = AntMultiMorphEnv(num_envs, device, morphologies=morphs,
                               rendering=False, raise_exception=False, seed=SEED + bi)
        rec = _rollout_base(net, obs_norm, val_norm, env, reward_scale, device)

        np.savez_compressed(
            out_dir / f"base_{bi:03d}.npz",
            leg_count=len(legs),
            base_legs=np.array(legs, dtype=np.int16),
            base_hip=np.array([base.hip_lengths[n] for n in legs], dtype=np.float32),
            base_ankle=np.array([base.ankle_lengths[n] for n in legs], dtype=np.float32),
            removed_leg=removed_leg, envs_per_group=ENVS_PER_GROUP,
            reward_scale=reward_scale, max_episode_length=env.max_episode_length,
            checkpoint=str(CHECKPOINT), **rec,
        )
        _teardown(env)
        del env, rec

    print(f"[ablation] wrote {len(bases)} base morphs -> {out_dir}")


if __name__ == "__main__":
    main()
