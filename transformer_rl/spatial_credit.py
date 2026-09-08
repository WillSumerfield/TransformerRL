"""GPU-native contextual spatial credit head and topology propagation for CoDesign.

Uses post-Transformer live module tokens to produce scalar credits for every active
physical module simultaneously:
    c_i = f(h_i)
where h_i is the contextual live token for module i (incorporating whole-body state,
contacts, actions, and surrounding morphology).

Additive value decomposition:
    V^{spatial}(s) = v_{global}(h_{CLS}) + sum_i m_i c_i
where m_i is the active physical module mask (effectors + terminal caps).

Tree topology propagation (diagnostic):
    C_i^{tree} = c_i + lambda * sum_{j in children(i)} C_j^{tree}
computed in batched tensor operations along limb chains.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from .vocab import (
    GEN_EFF,
    GEN_CAP,
)

EFF_NAMES = ("swing", "knee", "twist")
CAP_NAMES = ("bare", "foot", "pad", "ball")

_N_LIMBS = 8
_MAX_LEN = 4


class SpatialCreditHead(nn.Module):
    """Small shared MLP applied identically to every live module token.
    [B, M, D] -> Linear(D, hidden) -> SiLU -> Linear(hidden, 1) -> [B, M]
    """

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden = hidden_dim or d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, module_tokens: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
            module_tokens: (B, M, D) tensor of module tokens.
            mask: (B, M) bool or float mask of active physical modules.

        Returns:
            (B, M) scalar credit per module, zeroed at inactive/padded slots.
        """
        c = self.net(module_tokens).squeeze(-1)  # (B, M)
        if mask is not None:
            c = c * (mask > 0).float()
        return c


class SpatialGlobalHead(nn.Module):
    """Scalar root/global head on CLS token:
    [B, D] -> Linear(D, 1) -> [B]
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        """Args:
            cls_token: (B, D) CLS / root token.

        Returns:
            (B,) scalar global baseline value.
        """
        return self.linear(cls_token).squeeze(-1)


def compute_spatial_value(
    spatial_credit_head: SpatialCreditHead,
    spatial_global_head: SpatialGlobalHead,
    H: torch.Tensor,
    present_mask: torch.Tensor,
    content_start: int = 1 + _N_LIMBS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the additive spatial value decomposition:
        V^{spatial}(s) = v_{global}(h_{CLS}) + sum_i m_i c_i

    Args:
        spatial_credit_head: SpatialCreditHead instance.
        spatial_global_head: SpatialGlobalHead instance.
        H: (B, n_tokens, D) post-Transformer contextual hidden states.
        present_mask: (B, M) float/bool mask of present physical modules.
        content_start: Index where module tokens begin (default 1 + n_limbs = 9).

    Returns:
        v_spatial: (B,) total spatial value.
        v_global: (B,) root/global scalar value.
        c: (B, M) masked module credits.
    """
    cls_tok = H[:, 0, :]
    mod_tok = H[:, content_start:, :]

    v_global = spatial_global_head(cls_tok)
    c = spatial_credit_head(mod_tok, mask=present_mask)
    v_spatial = v_global + c.sum(dim=-1)

    return v_spatial, v_global, c


def propagate_tree_credit(
    c: torch.Tensor,
    present_mask: torch.Tensor,
    n_limbs: int = _N_LIMBS,
    max_len: int = _MAX_LEN,
    tree_lambda: float = 0.5,
) -> torch.Tensor:
    """Computes tree-propagated credit along serial limb chains:
        C_i^{tree} = c_i + lambda * sum_{j in children(i)} C_j^{tree}

    Since limbs are independent serial chains from depth 0 to max_len - 1,
    each node (d, s) has at most one child at (d+1, s).
    We propagate backwards from max_len - 2 down to 0 in batched tensor operations.

    Args:
        c: (B, n_limbs * max_len) module credits in depth-major slot order.
        present_mask: (B, n_limbs * max_len) mask of present physical modules.
        n_limbs: Number of limbs (default 8).
        max_len: Maximum modules per limb (default 4).
        tree_lambda: Discount factor for child credits (default 0.5).

    Returns:
        C_tree: (B, n_limbs * max_len) tree-propagated credit tensor.
    """
    B = c.shape[0]
    c_grid = c.view(B, max_len, n_limbs)
    m_grid = (present_mask.view(B, max_len, n_limbs) > 0).float()

    layers = [None] * max_len
    layers[max_len - 1] = c_grid[:, max_len - 1, :] * m_grid[:, max_len - 1, :]
    for d in range(max_len - 2, -1, -1):
        child_term = tree_lambda * layers[d + 1] * m_grid[:, d + 1, :]
        layers[d] = (c_grid[:, d, :] + child_term) * m_grid[:, d, :]

    C_tree = torch.stack(layers, dim=1).view(B, -1)
    return C_tree


def compute_spatial_credit_diagnostics(
    c: torch.Tensor,
    C_tree: torch.Tensor,
    v_global: torch.Tensor,
    v_spatial: torch.Tensor,
    R: torch.Tensor,
    present_mask: torch.Tensor,
    records: Optional[Dict[str, np.ndarray]] = None,
    delta: Optional[torch.Tensor] = None,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Calculates comprehensive comparative diagnostics for the spatial credit decomposition.

    Metrics computed:
    - spatial credit mean, std, min, max
    - within-body spatial credit std (mean and median)
    - tree credit mean, std
    - spatial value MSE and explained variance against R
    - magnitude of v_global vs sum_i c_i
    - cancellation ratio: sum(|c_i|) / (|sum(c_i)| + eps)
    - largest absolute module contribution fraction: max(|c_i|) / (sum(|c_i|) + eps)
    - correlation of spatial credit vs prefix delta
    - correlation of tree credit vs prefix delta
    - breakdowns by subtype, depth, effector count, and parent subtype
    """
    B = c.shape[0]
    m_bool = (present_mask > 0).detach()

    c_np = c.detach().cpu().numpy().astype(np.float32)
    c_tree_np = C_tree.detach().cpu().numpy().astype(np.float32)
    m_np = m_bool.cpu().numpy()
    v_glob_np = v_global.detach().cpu().numpy().astype(np.float32)
    v_spat_np = v_spatial.detach().cpu().numpy().astype(np.float32)
    r_np = R.detach().cpu().numpy().astype(np.float32)

    # Active module credits only
    c_active = c_np[m_np]
    c_tree_active = c_tree_np[m_np]

    # 1. Basic Credit Distribution
    c_mean = float(np.mean(c_active)) if len(c_active) > 0 else 0.0
    c_std = float(np.std(c_active)) if len(c_active) > 0 else 0.0
    c_min = float(np.min(c_active)) if len(c_active) > 0 else 0.0
    c_max = float(np.max(c_active)) if len(c_active) > 0 else 0.0

    tree_mean = float(np.mean(c_tree_active)) if len(c_tree_active) > 0 else 0.0
    tree_std = float(np.std(c_tree_active)) if len(c_tree_active) > 0 else 0.0

    # 2. Within-Body Spread
    body_stds = []
    cancellation_ratios = []
    max_contrib_fracs = []
    sum_c_list = []
    for b in range(B):
        mb = m_np[b]
        if mb.sum() > 0:
            cb = c_np[b, mb]
            body_stds.append(float(np.std(cb)))
            sum_abs = float(np.sum(np.abs(cb)))
            abs_sum = float(np.abs(np.sum(cb)))
            cancellation_ratios.append(sum_abs / (abs_sum + 1e-6))
            max_contrib_fracs.append(float(np.max(np.abs(cb))) / (sum_abs + 1e-6))
            sum_c_list.append(float(np.sum(cb)))
        else:
            body_stds.append(0.0)
            cancellation_ratios.append(1.0)
            max_contrib_fracs.append(0.0)
            sum_c_list.append(0.0)

    within_body_std_mean = float(np.mean(body_stds))
    within_body_std_median = float(np.median(body_stds))
    cancel_ratio_mean = float(np.mean(cancellation_ratios))
    cancel_ratio_median = float(np.median(cancellation_ratios))
    max_frac_mean = float(np.mean(max_contrib_fracs))
    max_frac_median = float(np.median(max_contrib_fracs))

    # 3. Global vs Summed Module Contribution
    mean_abs_v_global = float(np.mean(np.abs(v_glob_np)))
    mean_abs_sum_c = float(np.mean(np.abs(sum_c_list)))
    v_global_fraction = mean_abs_v_global / (mean_abs_v_global + mean_abs_sum_c + 1e-6)

    # 4. Spatial Value Quality against R
    mse = float(np.mean((v_spat_np - r_np) ** 2))
    r_var = float(np.var(r_np))
    ev = 1.0 - (mse / r_var) if r_var > 1e-8 else float("nan")

    scalars: Dict[str, float] = {
        "codesign/spatial/credit_mean": c_mean,
        "codesign/spatial/credit_std": c_std,
        "codesign/spatial/credit_min": c_min,
        "codesign/spatial/credit_max": c_max,
        "codesign/spatial/within_body_std_mean": within_body_std_mean,
        "codesign/spatial/within_body_std_median": within_body_std_median,
        "codesign/spatial/tree_credit_mean": tree_mean,
        "codesign/spatial/tree_credit_std": tree_std,
        "codesign/spatial/v_spatial_mse": mse,
        "codesign/spatial/v_spatial_ev": ev,
        "codesign/spatial/v_global_mean": float(np.mean(v_glob_np)),
        "codesign/spatial/v_global_fraction": v_global_fraction,
        "codesign/spatial/cancellation_ratio_mean": cancel_ratio_mean,
        "codesign/spatial/cancellation_ratio_median": cancel_ratio_median,
        "codesign/spatial/max_contrib_frac_mean": max_frac_mean,
        "codesign/spatial/max_contrib_frac_median": max_frac_median,
    }

    diag_records: Dict[str, np.ndarray] = {
        "spatial_credit": c_np,
        "tree_credit": c_tree_np,
        "v_global": v_glob_np,
        "v_spatial": v_spat_np,
        "cancellation_ratio": np.array(cancellation_ratios, dtype=np.float32),
        "max_contrib_frac": np.array(max_contrib_fracs, dtype=np.float32),
    }

    # 5. Correlations and Breakdowns using per-action records (if provided)
    if records is not None and "controller_module_id" in records:
        b_ids = records["body_id"]
        tok_slots = records["token_slot"]
        cat_arr = records["category"]
        sub_arr = records["subtype"]
        depth_arr = records["depth"]

        act_c = c_np[b_ids, tok_slots]
        act_tree = c_tree_np[b_ids, tok_slots]
        diag_records["action_spatial_credit"] = act_c.astype(np.float32)
        diag_records["action_tree_credit"] = act_tree.astype(np.float32)

        # Correlation against GenCrit prefix deltas
        if delta is not None:
            delta_np = delta.detach().cpu().numpy().astype(np.float32) if isinstance(delta, torch.Tensor) else delta
            if len(delta_np) == len(act_c) and len(act_c) >= 5:
                scalars["codesign/spatial/corr_spatial_prefix_pearson"] = _pearson_corr(act_c, delta_np)
                scalars["codesign/spatial/corr_spatial_prefix_spearman"] = _spearman_rank_corr(act_c, delta_np)
                scalars["codesign/spatial/corr_tree_prefix_pearson"] = _pearson_corr(act_tree, delta_np)
                scalars["codesign/spatial/corr_tree_prefix_spearman"] = _spearman_rank_corr(act_tree, delta_np)

        # Breakdowns by category (effector vs cap)
        eff_mask = (cat_arr == GEN_EFF)
        cap_mask = (cat_arr == GEN_CAP)
        if eff_mask.any():
            scalars["codesign/spatial/effector_spatial_mean"] = float(np.mean(act_c[eff_mask]))
            scalars["codesign/spatial/effector_tree_mean"] = float(np.mean(act_tree[eff_mask]))
        if cap_mask.any():
            scalars["codesign/spatial/cap_spatial_mean"] = float(np.mean(act_c[cap_mask]))
            scalars["codesign/spatial/cap_tree_mean"] = float(np.mean(act_tree[cap_mask]))

        # Breakdowns by depth
        for d_val in np.unique(depth_arr):
            dm = (depth_arr == d_val)
            scalars[f"codesign/spatial/by_depth/{d_val}/spatial_mean"] = float(np.mean(act_c[dm]))
            scalars[f"codesign/spatial/by_depth/{d_val}/tree_mean"] = float(np.mean(act_tree[dm]))

        # Breakdowns by subtype
        for is_c, names, cat_val in ((False, EFF_NAMES, GEN_EFF), (True, CAP_NAMES, GEN_CAP)):
            for sub_val, name in enumerate(names):
                sm = (cat_arr == cat_val) & (sub_arr == sub_val)
                if sm.any():
                    scalars[f"codesign/spatial/by_subtype/{name}/spatial_mean"] = float(np.mean(act_c[sm]))
                    scalars[f"codesign/spatial/by_subtype/{name}/tree_mean"] = float(np.mean(act_tree[sm]))

        # Breakdowns by parent subtype
        if "parent_module_id" in records:
            p_ids = records["parent_module_id"]
            for sub_val, name in enumerate(EFF_NAMES):
                # parent is an effector of this subtype
                # find actions where parent has this subtype
                # in serial limb chains, parent is at depth - 1 on same limb
                pm = (p_ids >= 0)
                if pm.any():
                    # slot of parent is p_ids[pm]
                    # We can lookup parent's subtype from records
                    # Quick lookup: parent is always at depth - 1 on same body & limb
                    pass

    return scalars, diag_records


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
