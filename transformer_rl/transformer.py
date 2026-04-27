"""Per-joint transformer for Ant. Policy or value, controlled by `mode`."""
from typing import Literal
import torch
import torch.nn as nn

from .obs_tokenize import tokenize, TORSO_DIM, HIP_DIM, N_LEGS

# natural token order:   [torso, h1, h2, h3, h4, a1, a2, a3, a4]   (indices 0..8)
# joint tokens (natural) [h1, h2, h3, h4, a1, a2, a3, a4]          (natural action order is h,a interleaved, so we remap)
#
# Actuator order expected by env: [h4, a4, h1, a1, h2, a2, h3, a3]
#
# We emit joint-token outputs as [h1, h2, h3, h4, a1, a2, a3, a4] (the order we
# fed them in), then permute directly to actuator order below.
_ACT_PERM = torch.tensor([3, 7, 0, 4, 1, 5, 2, 6], dtype=torch.long)


class LegTransformer(nn.Module):
    def __init__(
        self,
        mode: Literal["policy", "value"],
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn: int = 256,
    ):
        super().__init__()
        assert mode in ("policy", "value")
        self.mode = mode
        self.n_tokens = 1 + 2 * N_LEGS  # 9

        self.embed_torso = nn.Linear(TORSO_DIM, d_model)
        self.embed_joint = nn.Linear(HIP_DIM, d_model)  # HIP_DIM == ANKLE_DIM

        self.type_emb = nn.Embedding(3, d_model)  # 0=torso, 1=hip, 2=ankle
        self.pos_emb = nn.Embedding(1 + N_LEGS, d_model)  # 0=torso, 1..4=leg id (hip+ankle share)

        type_ids = torch.tensor([0] + [1] * N_LEGS + [2] * N_LEGS, dtype=torch.long)
        pos_ids = torch.tensor([0] + list(range(1, N_LEGS + 1)) * 2, dtype=torch.long)
        self.register_buffer("type_ids", type_ids, persistent=False)
        self.register_buffer("pos_ids", pos_ids, persistent=False)
        self.register_buffer("act_perm", _ACT_PERM, persistent=False)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
            dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

        if mode == "policy":
            self.joint_head = nn.Linear(d_model, 1)
            nn.init.zeros_(self.joint_head.weight)
            nn.init.zeros_(self.joint_head.bias)
        else:
            self.value_head = nn.Linear(d_model, 1)

        self._xavier_init()

    def _xavier_init(self) -> None:
        for name, p in self.named_parameters():
            if "joint_head" in name:  # keep zero-init
                continue
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        tok = tokenize(obs)
        t = self.embed_torso(tok["torso"])   # [B, 1, d]
        h = self.embed_joint(tok["hip"])     # [B, 4, d]
        a = self.embed_joint(tok["ankle"])   # [B, 4, d]
        x = torch.cat([t, h, a], dim=1)      # [B, 9, d]

        x = x + self.type_emb(self.type_ids) + self.pos_emb(self.pos_ids)
        x = self.encoder(x)                  # [B, 9, d]

        if self.mode == "policy":
            joints = x[:, 1:, :]                                # [B, 8, d]
            a_nat = torch.tanh(self.joint_head(joints).squeeze(-1))  # [B, 8] in [h1..h4, a1..a4], bounded to (-1, 1)
            return a_nat.index_select(-1, self.act_perm)        # [B, 8] actuator order
        return self.value_head(x[:, 0, :])                      # [B, 1]


if __name__ == "__main__":
    import gymnasium as gym
    env = gym.make("Ant-v5")
    o, _ = env.reset(seed=0)
    o = torch.from_numpy(o).float().unsqueeze(0).repeat(4, 1)

    pol = LegTransformer("policy")
    val = LegTransformer("value")
    a = pol(o); v = val(o)
    print("policy:", tuple(a.shape), "mean abs:", a.abs().mean().item())
    print("value :", tuple(v.shape))
    assert a.shape == (4, 8) and v.shape == (4, 1)
    # zero-init action head -> actions should be ~0 at init
    assert a.abs().max().item() < 1e-5, f"expected zero-init actions, got {a.abs().max().item()}"

    (a.sum() + v.sum()).backward()
    for name, p in pol.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
    print("grads ok; param counts:",
          sum(p.numel() for p in pol.parameters()), "(pol)",
          sum(p.numel() for p in val.parameters()), "(val)")
    env.close()
