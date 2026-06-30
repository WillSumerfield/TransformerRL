"""Transformer architectures for ant envs. forward returns a dict with keys
'mu'/'value' for whichever heads the net was built with (policy_head/value_head)."""
import math

import torch
import torch.nn as nn

from .tokenize import tokenize_4, tokenize_8, TORSO_DIM, HIP_DIM, ANKLE_DIM, HIP_DIM_8, ANKLE_DIM_8

_TOKENIZE = {4: tokenize_4, 8: tokenize_8}

# codesign token type / mode ids (see CONTEXT.md "Codesign tokens")
_T_TORSO, _T_HIP, _T_ANKLE, _T_START = 0, 1, 2, 3
_MODE_LIVE, _MODE_COMMITTED, _MODE_STOP = 0, 1, 2


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
        codesign_tokens: bool = False,
    ):
        super().__init__()
        self.n_legs = n_legs
        self.tokenize_fn = _TOKENIZE[n_legs]
        self.has_policy_head = policy_head
        self.has_value_head = value_head
        self.value_size = value_size
        self.codesign_tokens = codesign_tokens

        self.embed_torso = nn.Linear(torso_dim, d_model)
        self.embed_hip   = nn.Linear(hip_dim,   d_model)
        self.embed_ankle = nn.Linear(ankle_dim, d_model)

        self.type_emb = nn.Embedding(4 if codesign_tokens else 3, d_model)
        self.pos_emb  = nn.Embedding(1 + n_legs, d_model)

        if codesign_tokens:
            # fixed 25-token layout: [CLS, start*n, hip*n, ankle*n]; off-legs are stop tokens,
            # never masked. content (hip/ankle) carries a live/committed/stop mode embedding;
            # start tokens are persistent per-slot anchors (type+pos+angle).
            self.mode_emb = nn.Embedding(3, d_model)             # LIVE / COMMITTED / STOP
            self.angle_proj = nn.Linear(2, d_model)
            angle_enc = torch.tensor(
                [[math.sin(i * math.pi / 4), math.cos(i * math.pi / 4)] for i in range(n_legs)],
                dtype=torch.float32)                             # matches tokenize._LEG_ENC_8
            self.register_buffer("angle_enc", angle_enc, persistent=False)
            type_ids = torch.tensor(
                [_T_TORSO] + [_T_START] * n_legs + [_T_HIP] * n_legs + [_T_ANKLE] * n_legs,
                dtype=torch.long)
            pos_ids  = torch.tensor([0] + list(range(1, n_legs + 1)) * 3, dtype=torch.long)
            self._content_start = 1 + n_legs                     # content tokens begin after starts
        else:
            type_ids = torch.tensor([0] + [1] * n_legs + [2] * n_legs, dtype=torch.long)
            pos_ids  = torch.tensor([0] + list(range(1, n_legs + 1)) * 2, dtype=torch.long)
            self._content_start = 1
        self.n_tokens = type_ids.numel()
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

    def _encode_legacy(self, torso, hip_tok, ankle_tok, active_mask, B):
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
            [torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~active_mask.bool()],
            dim=1,
        )
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return x * token_mask  # zero inactive outputs (cuts gradient through transformer)

    def _encode_codesign(self, torso, hip_tok, ankle_tok, active_mask, B):
        """Fixed 25-token live-mode pass: off-legs become stop tokens (never masked), content
        carries a live/stop mode embedding, plus persistent start anchors. See CONTEXT.md."""
        n = self.n_legs
        leg_active = active_mask[:, :n]                          # (B,n) hip slots == per-leg presence
        h = self.embed_hip(hip_tok)   * leg_active.unsqueeze(-1)  # zero stop legs' state embeds
        a = self.embed_ankle(ankle_tok) * leg_active.unsqueeze(-1)
        cls = self.embed_torso(torso).unsqueeze(1)
        start = self.angle_proj(self.angle_enc).unsqueeze(0).expand(B, -1, -1)  # (B,n,d) angle only
        x = torch.cat([cls, start, h, a], dim=1)                 # (B, 1+3n, d)
        x = x + self.type_emb(self.type_ids) + self.pos_emb(self.pos_ids)

        content_active = active_mask                            # (B,2n) hip then ankle
        mode_ids = torch.where(content_active > 0, content_active.new_full((), _MODE_LIVE),
                               content_active.new_full((), _MODE_STOP)).long()
        mode = torch.cat([x.new_zeros(B, 1 + n, x.shape[-1]), self.mode_emb(mode_ids)], dim=1)
        return self.encoder(x + mode)                          # all 25 tokens real -> no padding

    def forward(self, obs: torch.Tensor, compute_value: bool = True,
                detach_value: bool = False) -> dict[str, torch.Tensor]:
        torso, hip_tok, ankle_tok, active_mask = self.tokenize_fn(obs)
        B = obs.shape[0]
        if self.codesign_tokens:
            x = self._encode_codesign(torso, hip_tok, ankle_tok, active_mask, B)
        else:
            x = self._encode_legacy(torso, hip_tok, ankle_tok, active_mask, B)

        out: dict[str, torch.Tensor] = {}
        if self.has_policy_head:
            joints = x[:, self._content_start:, :]
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
