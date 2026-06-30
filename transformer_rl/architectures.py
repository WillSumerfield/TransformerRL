"""Transformer architectures for ant envs. forward returns a dict with keys
'mu'/'value' for whichever heads the net was built with (policy_head/value_head)."""
import torch
import torch.nn as nn

from .tokenize import tokenize_4, tokenize_8, TORSO_DIM, HIP_DIM, ANKLE_DIM, HIP_DIM_8, ANKLE_DIM_8

_TOKENIZE = {4: tokenize_4, 8: tokenize_8}


def _make_nat_to_dof(n_legs: int) -> torch.Tensor:
    idx = torch.arange(2 * n_legs, dtype=torch.long)
    return idx // 2 + n_legs * (idx % 2)


class LegTransformer(nn.Module):
    def __init__(
        self,
        n_legs: int = 4,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 1,
        ffn: int = 512,
        torso_dim: int = TORSO_DIM,
        hip_dim: int = HIP_DIM,
        ankle_dim: int = ANKLE_DIM,
        policy_head: bool = True,
        value_head: bool = True,
        value_size: int = 1,
    ):
        super().__init__()
        self.n_tokens = 1 + 2 * n_legs
        self.tokenize_fn = _TOKENIZE[n_legs]
        self.has_policy_head = policy_head
        self.has_value_head = value_head
        self.value_size = value_size

        self.embed_torso = nn.Linear(torso_dim, d_model)
        self.embed_hip   = nn.Linear(hip_dim,   d_model)
        self.embed_ankle = nn.Linear(ankle_dim, d_model)

        self.type_emb = nn.Embedding(3, d_model)
        self.pos_emb  = nn.Embedding(1 + n_legs, d_model)

        type_ids = torch.tensor([0] + [1] * n_legs + [2] * n_legs, dtype=torch.long)
        pos_ids  = torch.tensor([0] + list(range(1, n_legs + 1)) * 2, dtype=torch.long)
        self.register_buffer("type_ids",   type_ids,            persistent=False)
        self.register_buffer("pos_ids",    pos_ids,             persistent=False)
        self.register_buffer("nat_to_dof", _make_nat_to_dof(n_legs), persistent=False)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)

        if policy_head:
            self.joint_head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.joint_head.weight)
            nn.init.zeros_(self.joint_head.bias)
        if value_head:
            self.value_head = nn.Linear(d_model, 1)        # V0.98 (sole head when value_size==1)
            if value_size == 2:
                # V1.0 body-quality head: reads the torso feature + the raw progress obs dim.
                # Backprops into the shared trunk (decided: richer features over strict isolation).
                self.value_head_v1 = nn.Linear(d_model + 1, 1)
        self._xavier_init()

    def _xavier_init(self) -> None:
        for name, p in self.named_parameters():
            if "joint_head" in name:
                continue
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, obs: torch.Tensor, compute_value: bool = True,
                detach_value: bool = False) -> dict[str, torch.Tensor]:
        torso, hip_tok, ankle_tok, active_mask = self.tokenize_fn(obs)
        B = obs.shape[0]

        t = self.embed_torso(torso).unsqueeze(1)
        h = self.embed_hip(hip_tok)
        a = self.embed_ankle(ankle_tok)
        x = torch.cat([t, h, a], dim=1)

        x = x + self.type_emb(self.type_ids) + self.pos_emb(self.pos_ids)

        # (B, 1+2*n_legs, 1): torso always active, then hip masks, then ankle masks
        token_mask = torch.cat(
            [torch.ones(B, 1, dtype=x.dtype, device=x.device), active_mask],
            dim=1,
        ).unsqueeze(-1)
        x = x * token_mask  # zero inactive token embeddings (kills them as queries)

        pad_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=x.device),
             ~active_mask.bool()],
            dim=1,
        )
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        x = x * token_mask  # zero inactive outputs (cuts gradient through transformer)

        out: dict[str, torch.Tensor] = {}
        if self.has_policy_head:
            joints = x[:, 1:, :]
            a_nat  = torch.tanh(self.joint_head(joints).squeeze(-1))
            a_nat  = a_nat * active_mask
            out['mu'] = a_nat.index_select(-1, self.nat_to_dof)
        if compute_value and self.has_value_head:
            torso_feat = x[:, 0, :]
            if detach_value:                               # single-net PPG policy phase:
                torso_feat = torso_feat.detach()           # value grad stops at the trunk
            v0 = self.value_head(torso_feat)               # V0.98
            if self.value_size == 2:
                progress = obs[:, -1:]                      # raw normalized progress (last obs dim)
                v1 = self.value_head_v1(torch.cat([torso_feat, progress], dim=-1))  # V1.0
                out['value'] = torch.cat([v0, v1], dim=-1)
            else:
                out['value'] = v0
        return out


def MultiMorphLegTransformer(n_layers: int = 3, **kwargs) -> LegTransformer:
    kwargs.setdefault('n_legs', 8)
    kwargs.setdefault('hip_dim', HIP_DIM_8)      # 8-leg tokens carry segment lengths
    kwargs.setdefault('ankle_dim', ANKLE_DIM_8)
    return LegTransformer(n_layers=n_layers, **kwargs)
