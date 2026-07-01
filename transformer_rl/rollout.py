"""Own-the-loop test rollout for the limb transformer (ADR-0007).

`test` mode reuses the rl_games player only to *load* the checkpoint (weights + obs/value
normalizers); this module owns the rollout so it can read the critic's per-step value estimate
(which the player discards) and call ``env.resample()`` between Samples. Two data-types:

  - ``summary``: per-morph episode-score table (CSV/PNG/MD), parity with the pre-ADR-0007 path.
  - ``full``:    per-step value/reward traces (.npz) + per-env summary (CSV) for the morph-value
                 sweep — one Sample = one 4096-morph draw, first episode per env only.

Value is stored in **raw-reward units**: the model already denormalizes (value_mean_std) in eval
forward; we additionally divide by the reward-shaper scale so it is comparable to raw episode reward.
"""
from __future__ import annotations

from pathlib import Path

import torch
from rl_games.algos_torch.players import rescale_actions

# obs layout (see envs/ant_envs/ant_multimorph.py): [107:123] = 8 hip + 8 ankle lengths,
# [123:139] = 16-bit DOF mask (2 per limb).
_LEN_LO, _LEN_HI = 107, 123
_MASK_LO, _MASK_HI = 123, 139


@torch.no_grad()
def _forward(player, obs):
    """One model forward on batched obs -> (deterministic action, denormalized value [N])."""
    obs = player._preproc_obs(obs)
    res = player.model({"is_train": False, "prev_actions": None, "obs": obs, "rnn_states": None})
    action = res["mus"]
    if player.clip_actions:
        action = rescale_actions(
            player.actions_low, player.actions_high, torch.clamp(action, -1.0, 1.0)
        )
    return action, res["values"].squeeze(-1)


def run_test_rollout(player, env, *, data_type, num_episodes, num_samples,
                     reward_scale, checkpoint, out_stem, split_labels):
    if data_type == "full":
        _run_full(player, env, num_samples=num_samples, reward_scale=reward_scale,
                  checkpoint=checkpoint, out_stem=out_stem)
    else:
        _run_summary(player, env, num_episodes=num_episodes, checkpoint=checkpoint,
                     split_labels=split_labels)


def _run_summary(player, env, *, num_episodes, checkpoint, split_labels):
    """Per-morph episode scores over a fixed morph set (no resample). Mirrors the old _TestTracker."""
    from tqdm import tqdm
    from .train_utils import _print_and_save_test_results

    n = env.total_num_envs
    epm = env.envs_per_morph
    scores: list[list[float]] = [[] for _ in range(len(env.groups))]
    ep_counts = torch.zeros(n, dtype=torch.long)
    cur_rew = torch.zeros(n)

    bar = tqdm(total=n * num_episodes, unit="ep", desc="testing")
    obs, _ = env.reset()
    cap = (num_episodes + 1) * env.max_episode_length + 5
    step = 0
    while not bool((ep_counts >= num_episodes).all()) and step < cap:
        action, _ = _forward(player, obs)
        obs, rew, term, trunc, _ = env.step(action)
        done = (term | trunc).cpu()
        cur_rew += rew.cpu()
        for i in torch.nonzero(done).flatten().tolist():
            if ep_counts[i] < num_episodes:
                scores[i // epm].append(float(cur_rew[i]))
                bar.update(1)
            ep_counts[i] += 1
            cur_rew[i] = 0.0
        step += 1
    bar.close()
    _print_and_save_test_results(scores, env.groups, checkpoint, split_labels=split_labels)


def _run_full(player, env, *, num_samples, reward_scale, checkpoint, out_stem):
    """Per-step value/reward + per-env morph features over `num_samples` fresh morph draws."""
    import numpy as np
    from tqdm import tqdm

    dev = env.device
    n = env.total_num_envs
    cap = env.max_episode_length + 5

    # per-env summary (one row per env per sample) + ragged per-step traces (padded later)
    s_sample, s_limbs, s_lengths, s_rew, s_len, s_term, s_vt0 = ([] for _ in range(7))
    val_traces: list[torch.Tensor] = []   # each [n, len_s] half, cpu
    rew_traces: list[torch.Tensor] = []

    for s in range(num_samples):
        if s > 0:
            print(f"[test] resampling morphologies for Sample {s} ...")
            env.resample()
        obs, _ = env.reset()

        lengths = obs[:, _LEN_LO:_LEN_HI].clone()                              # [n,16] hip*8 ankle*8
        limb_count = (obs[:, _MASK_LO:_MASK_HI] > 0).sum(dim=1) // 2            # [n]

        active = torch.ones(n, dtype=torch.bool, device=dev)   # still in first episode
        cur_rew = torch.zeros(n, device=dev)
        ep_len = torch.zeros(n, dtype=torch.long, device=dev)
        termed = torch.zeros(n, dtype=torch.bool, device=dev)  # ended by termination (vs truncation)
        v_t0 = None
        step_vals, step_rews = [], []

        bar = tqdm(total=n, unit="env", desc=f"sample {s}")
        step = 0
        while bool(active.any()) and step < cap:
            action, val = _forward(player, obs)
            val = val / reward_scale                                          # raw-reward units
            if step == 0:
                v_t0 = val.clone()
            obs, rew, term, trunc, _ = env.step(action)
            done = term | trunc
            nan = torch.tensor(float("nan"), device=dev)
            step_vals.append(torch.where(active, val, nan).half().cpu())
            step_rews.append(torch.where(active, rew, nan).half().cpu())
            cur_rew += torch.where(active, rew, torch.zeros_like(rew))
            ep_len += active.long()
            newly = done & active
            termed |= term & newly
            bar.update(int(newly.sum().item()))
            active &= ~done
            step += 1
        bar.close()

        s_sample.append(torch.full((n,), s, dtype=torch.long))
        s_limbs.append(limb_count.cpu())
        s_lengths.append(lengths.cpu())
        s_rew.append(cur_rew.cpu())
        s_len.append(ep_len.cpu())
        s_term.append(termed.cpu())
        s_vt0.append(v_t0.cpu())
        val_traces.append(torch.stack(step_vals, dim=1))   # [n, len_s]
        rew_traces.append(torch.stack(step_rews, dim=1))

    # --- pad ragged traces to a common length -> [n_sample, n, max_len] ---
    max_len = max(t.shape[1] for t in val_traces)

    def _pad_stack(traces):
        out = torch.full((len(traces), n, max_len), float("nan"), dtype=torch.float16)
        for s, t in enumerate(traces):
            out[s, :, : t.shape[1]] = t
        return out.numpy()

    limbs = torch.stack(s_limbs).numpy()
    lengths = torch.stack(s_lengths).numpy()
    rew = torch.stack(s_rew).numpy()
    ep_len = torch.stack(s_len).numpy()
    termed = torch.stack(s_term).numpy()
    v_t0 = torch.stack(s_vt0).numpy()

    out_dir = Path(__file__).resolve().parent.parent / "data" / "morph_value_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-step traces + self-contained per-env metadata. Keyed by out_stem (run-dir name or
    # --name) so distinct checkpoints don't collide on the shared checkpoint filename.
    npz_path = out_dir / f"{out_stem}_traces.npz"
    np.savez_compressed(
        npz_path,
        value=_pad_stack(val_traces),      # [S, n, max_len] raw-reward units, nan = post-episode pad
        reward=_pad_stack(rew_traces),     # [S, n, max_len] per-step raw reward
        limb_count=limbs, lengths=lengths, episode_reward=rew, episode_len=ep_len,
        terminated=termed, value_t0=v_t0,
        reward_scale=reward_scale, max_episode_length=env.max_episode_length,
        checkpoint=str(checkpoint),
    )
    print(f"[test] trace -> {npz_path}  ({n} envs x {num_samples} samples x {max_len} steps)")
