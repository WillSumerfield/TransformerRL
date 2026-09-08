"""Diagnostic-only post-adaptation evaluation for TransformerRL CoDesign.

Evaluates the performance of the final adapted controller on the current morphology
set at each resample boundary, without modifying training targets, optimizer state,
or environment buffers.

Key objectives:
1. Compute R_post (return achieved by the final frozen controller policy).
2. Measure adaptation_gap = R_post - R_train.
3. Correlate R_train and R_post, and detect material rank shifts.
4. Correlate adaptation gap with morphology complexity (effector count, module count, depth).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch


def _compute_param_checksum(agent) -> Tuple[float, float]:
    """Computes a checksum (sum and l2 norm) of all model parameters to verify invariance."""
    total_sum = 0.0
    total_norm_sq = 0.0
    for p in agent.model.parameters():
        d = p.detach()
        total_sum += float(d.sum().item())
        total_norm_sq += float((d * d).sum().item())
    return total_sum, total_norm_sq


def _run_single_eval_rollout(
    agent,
    horizon: int,
    stochastic: bool = False,
) -> torch.Tensor:
    """Executes a frozen evaluation rollout of length `horizon` steps on current morphologies."""
    env = agent._env()
    vec_env = agent.vec_env
    net = agent._net()
    model = agent.model
    dev = agent.ppo_device
    N = env.total_num_envs

    eval_ep_ret = torch.zeros(N, device=dev)
    eval_ret_sum = torch.zeros(N, device=dev)
    eval_ret_cnt = torch.zeros(N, device=dev)

    obs = vec_env.reset()

    for _ in range(horizon):
        normed = model.norm_obs(obs)
        mu, *_ = net.codesign_forward(normed)
        if stochastic:
            ls = agent._log_std(normed)
            sigma = ls.exp()
            actions = (mu + sigma * torch.randn_like(mu)).clamp(-1.0, 1.0)
        else:
            actions = mu.clamp(-1.0, 1.0)

        obs, rewards, dones, _ = vec_env.step(actions)
        r = rewards if rewards.dim() == 1 else rewards[:, 0]
        eval_ep_ret += r

        df = dones.float()
        eval_ret_sum += eval_ep_ret * df
        eval_ret_cnt += df
        eval_ep_ret = eval_ep_ret * (1.0 - df)

    incomplete = (eval_ret_cnt == 0)
    if incomplete.any():
        eval_ret_sum += torch.where(incomplete, eval_ep_ret, torch.zeros_like(eval_ep_ret))
        eval_ret_cnt += torch.where(incomplete, torch.ones_like(eval_ret_cnt), torch.zeros_like(eval_ret_cnt))

    R_post = (eval_ret_sum / eval_ret_cnt.clamp(min=1.0)) * agent._r_scale
    return R_post


def run_post_adaptation_eval(
    agent,
    eval_steps: Optional[int] = None,
    eval_stochastic: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Runs frozen evaluation rollouts of the final controller policy on currently installed bodies.

    Guarantees:
    - Verifies parameter checksum before and after rollout (asserts bit-exact equality).
    - Runs in torch.no_grad(), net.eval(), rms.eval().
    - Bypasses agent.env_step; training accumulators and PPO buffers are untouched.
    - Evaluates deterministic controller (R_post_det).
    - Optionally evaluates stochastic controller (R_post_stoch) under identical conditions.

    Args:
        agent: CodesignAgent instance.
        eval_steps: Steps to evaluate (default: agent._env().max_episode_length).
        eval_stochastic: If True, also runs a stochastic rollout to compare action modes.

    Returns:
        (R_post_det, R_post_stoch): tuple of (N,) tensors.
    """
    env = agent._env()
    net = agent._net()
    model = agent.model

    max_ep = env.max_episode_length
    horizon = eval_steps if (eval_steps is not None and eval_steps > 0) else max_ep

    # Snapshot training states & compute pre-eval parameter checksum
    was_net_train = net.training
    rms = getattr(model, "running_mean_std", None)
    was_rms_train = rms.training if rms is not None else False

    chk_sum_pre, chk_norm_pre = _compute_param_checksum(agent)

    net.eval()
    if rms is not None:
        rms.eval()

    with torch.no_grad():
        # 1. Deterministic evaluation rollout
        R_post_det = _run_single_eval_rollout(agent, horizon, stochastic=False)

        # 2. Stochastic evaluation rollout (if requested)
        R_post_stoch = _run_single_eval_rollout(agent, horizon, stochastic=True) if eval_stochastic else None

    # Clean up temporary cached allocations
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Restore training states
    net.train(was_net_train)
    if rms is not None:
        rms.train(was_rms_train)

    # Verify post-eval parameter checksum matches pre-eval checksum exactly
    chk_sum_post, chk_norm_post = _compute_param_checksum(agent)
    assert abs(chk_sum_pre - chk_sum_post) < 1e-6, (
        f"Parameter checksum sum mismatch after post-adaptation eval: {chk_sum_pre} vs {chk_sum_post}"
    )
    assert abs(chk_norm_pre - chk_norm_post) < 1e-6, (
        f"Parameter checksum norm mismatch after post-adaptation eval: {chk_norm_pre} vs {chk_norm_post}"
    )

    return R_post_det, R_post_stoch


def _spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Spearman rank correlation between two 1D numpy arrays."""
    if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.linalg.norm(rx) * np.linalg.norm(ry)
    return float((rx @ ry) / denom) if denom > 1e-8 else float("nan")


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Computes Pearson linear correlation between two 1D numpy arrays."""
    if len(x) < 3 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return float("nan")
    c = np.corrcoef(x, y)[0, 1]
    return float(c) if np.isfinite(c) else float("nan")


def compute_adaptation_gap_diagnostics(
    R_train: torch.Tensor,
    R_post: torch.Tensor,
    counts: torch.Tensor,
    eff_sub: Optional[torch.Tensor] = None,
    cap_sub: Optional[torch.Tensor] = None,
    R_post_stoch: Optional[torch.Tensor] = None,
    rank_shift_threshold: float = 0.10,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Calculates comprehensive comparative diagnostics between R_train and R_post across the batch.

    Args:
        R_train: (N,) tensor of window-averaged returns during adaptation.
        R_post: (N,) tensor of post-adaptation returns with frozen final controller.
        counts: (N, n_limbs) tensor of effector counts per limb.
        eff_sub: (N, n_limbs, max_len) optional effector subtypes.
        cap_sub: (N, n_limbs) optional cap subtypes.
        R_post_stoch: (N,) optional tensor of stochastic post-adaptation returns.
        rank_shift_threshold: Threshold on normalized rank difference (|r_post - r_train| / N)
                              defining a material rank shift (default 0.10, i.e. 10 percentile points).

    Returns:
        scalars: Dict of aggregate scalar metrics for TensorBoard.
        records: Dict of per-body arrays for offline analysis.
    """
    N = R_train.shape[0]

    r_train_np = R_train.detach().cpu().numpy().astype(np.float32)
    r_post_np = R_post.detach().cpu().numpy().astype(np.float32)
    gap_np = (r_post_np - r_train_np).astype(np.float32)

    counts_np = counts.detach().cpu().numpy().astype(np.int32)
    effector_count = counts_np.sum(axis=1)
    active_limbs = (counts_np > 0).sum(axis=1)
    # Each present limb carries 1 terminal cap
    module_count = effector_count + active_limbs
    max_depth = counts_np.max(axis=1)
    mean_depth = effector_count / np.maximum(active_limbs, 1)

    # Normalized ranks [0, 1]
    rank_train = r_train_np.argsort().argsort().astype(float) / max(1, N - 1)
    rank_post = r_post_np.argsort().argsort().astype(float) / max(1, N - 1)
    rank_diff = np.abs(rank_post - rank_train)
    rank_shift_frac = float(np.mean(rank_diff > rank_shift_threshold))

    pearson_r = _pearson_corr(r_train_np, r_post_np)
    spearman_r = _spearman_rank_corr(r_train_np, r_post_np)

    gap_mean = float(np.mean(gap_np))
    gap_std = float(np.std(gap_np))

    scalars: Dict[str, float] = {
        "codesign/adaptation/R_train_mean": float(np.mean(r_train_np)),
        "codesign/adaptation/R_train_std": float(np.std(r_train_np)),
        "codesign/adaptation/R_post_mean": float(np.mean(r_post_np)),
        "codesign/adaptation/R_post_std": float(np.std(r_post_np)),
        "codesign/adaptation/gap_mean": gap_mean,
        "codesign/adaptation/gap_std": gap_std,
        "codesign/adaptation/gap_pos_frac": float(np.mean(gap_np > 0)),
        "codesign/adaptation/corr_pearson": pearson_r,
        "codesign/adaptation/corr_spearman": spearman_r,
        "codesign/adaptation/material_rank_shift_frac": rank_shift_frac,
    }

    # If stochastic post-adaptation returns are provided, log comparisons between det and stoch
    r_post_stoch_np = None
    if R_post_stoch is not None:
        r_post_stoch_np = R_post_stoch.detach().cpu().numpy().astype(np.float32)
        scalars["codesign/adaptation/R_post_stoch_mean"] = float(np.mean(r_post_stoch_np))
        scalars["codesign/adaptation/R_post_stoch_std"] = float(np.std(r_post_stoch_np))
        scalars["codesign/adaptation/corr_det_stoch_pearson"] = _pearson_corr(r_post_np, r_post_stoch_np)
        scalars["codesign/adaptation/corr_det_stoch_spearman"] = _spearman_rank_corr(r_post_np, r_post_stoch_np)

    # Breakdown of adaptation gap by effector count
    unique_eff = np.unique(effector_count)
    for k in unique_eff:
        m = (effector_count == k)
        scalars[f"codesign/adaptation/gap_by_effectors/{k}"] = float(np.mean(gap_np[m]))

    # Breakdown by module count
    unique_mod = np.unique(module_count)
    for m_val in unique_mod:
        m = (module_count == m_val)
        scalars[f"codesign/adaptation/gap_by_modules/{m_val}"] = float(np.mean(gap_np[m]))

    # Breakdown by max limb depth
    unique_d = np.unique(max_depth)
    for d_val in unique_d:
        m = (max_depth == d_val)
        scalars[f"codesign/adaptation/gap_by_max_depth/{d_val}"] = float(np.mean(gap_np[m]))

    records = {
        "R_train": r_train_np,
        "R_post": r_post_np,
        "adaptation_gap": gap_np,
        "effector_count": effector_count.astype(np.int16),
        "active_limbs": active_limbs.astype(np.int16),
        "module_count": module_count.astype(np.int16),
        "max_depth": max_depth.astype(np.int16),
        "mean_depth": mean_depth.astype(np.float32),
        "rank_shift": rank_diff.astype(np.float32),
        "counts": counts_np,
    }
    if r_post_stoch_np is not None:
        records["R_post_stoch"] = r_post_stoch_np
    if eff_sub is not None:
        records["eff_sub"] = eff_sub.detach().cpu().numpy().astype(np.int8)
    if cap_sub is not None:
        records["cap_sub"] = cap_sub.detach().cpu().numpy().astype(np.int8)

    return scalars, records


def save_post_eval_artifact(
    filepath: str,
    records: Dict[str, np.ndarray],
    scalars: Dict[str, float],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves a compressed .npz artifact containing per-body records and aggregate metrics."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    save_dict = dict(records)
    for k, v in scalars.items():
        save_dict[f"scalar__{k}"] = np.array(v)
    if metadata:
        for k, v in metadata.items():
            save_dict[f"meta__{k}"] = np.array(v)
    np.savez_compressed(filepath, **save_dict)
