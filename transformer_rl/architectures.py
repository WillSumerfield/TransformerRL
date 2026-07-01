"""Transformer architectures for ant envs. forward returns a dict with keys
'mu'/'value' for whichever heads the net was built with (policy_head/value_head)."""
import math

import torch
import torch.nn as nn

from .tokenize import tokenize_4, tokenize_8, ROOT_DIM, EFF0_DIM, EFF1_DIM, EFF0_DIM_8, EFF1_DIM_8

_TOKENIZE = {4: tokenize_4, 8: tokenize_8}

# codesign token type / mode ids (see CONTEXT.md "Codesign tokens")
_T_ROOT, _T_EFF0, _T_EFF1, _T_START = 0, 1, 2, 3
_MODE_LIVE, _MODE_COMMITTED, _MODE_STOP = 0, 1, 2
_GEN_ON, _GEN_STOP = 0, 1                          # GenAct categorical action ids {on, stop}


def _make_nat_to_dof(n_limbs: int) -> torch.Tensor:
    idx = torch.arange(2 * n_limbs, dtype=torch.long)
    return idx // 2 + n_limbs * (idx % 2)


class LimbTransformer(nn.Module):
    def __init__(
        self,
        n_limbs: int = 4,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 1,
        ffn: int = 512,
        root_dim: int = ROOT_DIM,
        eff0_dim: int = EFF0_DIM,
        eff1_dim: int = EFF1_DIM,
        policy_head: bool = True,
        value_head: bool = True,
        codesign_tokens: bool = False,
    ):
        super().__init__()
        self.n_limbs = n_limbs
        self.tokenize_fn = _TOKENIZE[n_limbs]
        self.has_policy_head = policy_head
        self.has_value_head = value_head
        self.codesign_tokens = codesign_tokens

        self.embed_root = nn.Linear(root_dim, d_model)
        self.embed_eff0   = nn.Linear(eff0_dim,   d_model)
        self.embed_eff1 = nn.Linear(eff1_dim, d_model)

        self.type_emb = nn.Embedding(4 if codesign_tokens else 3, d_model)
        self.pos_emb  = nn.Embedding(1 + n_limbs, d_model)

        if codesign_tokens:
            # fixed 25-token layout: [CLS, start*n, eff0*n, eff1*n]; off-limbs are stop tokens,
            # never masked. content (eff0/eff1) carries a live/committed/stop mode embedding;
            # start tokens are persistent per-slot anchors (type+pos+angle).
            self.mode_emb = nn.Embedding(3, d_model)             # LIVE / COMMITTED / STOP
            self.angle_proj = nn.Linear(2, d_model)
            angle_enc = torch.tensor(
                [[math.sin(i * math.pi / 4), math.cos(i * math.pi / 4)] for i in range(n_limbs)],
                dtype=torch.float32)                             # matches tokenize._LIMB_ENC_8
            self.register_buffer("angle_enc", angle_enc, persistent=False)
            type_ids = torch.tensor(
                [_T_ROOT] + [_T_START] * n_limbs + [_T_EFF0] * n_limbs + [_T_EFF1] * n_limbs,
                dtype=torch.long)
            pos_ids  = torch.tensor([0] + list(range(1, n_limbs + 1)) * 3, dtype=torch.long)
            self._content_start = 1 + n_limbs                     # content tokens begin after starts
        else:
            type_ids = torch.tensor([0] + [1] * n_limbs + [2] * n_limbs, dtype=torch.long)
            pos_ids  = torch.tensor([0] + list(range(1, n_limbs + 1)) * 2, dtype=torch.long)
            self._content_start = 1
        self.n_tokens = type_ids.numel()
        self.register_buffer("type_ids",   type_ids,            persistent=False)
        self.register_buffer("pos_ids",    pos_ids,             persistent=False)
        self.register_buffer("nat_to_dof", _make_nat_to_dof(n_limbs), persistent=False)

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
            self.value_head = nn.Linear(d_model, 1)        # V0.98 (sole control critic)

        if codesign_tokens:
            # generator heads on the shared trunk (single-network codesign):
            #   GenAct    = on/stop logits, read from each slot's START token (design mode).
            #   GenCrit   = V1.0 body-quality value, read from CLS; NO time feature, evaluable on
            #               both live full-state tokens and partial designed prefixes (same weights).
            #   cls_design = learned CLS content used in design mode (generation has no root state).
            self.gen_head = nn.Linear(d_model, 2)
            nn.init.zeros_(self.gen_head.weight)
            nn.init.zeros_(self.gen_head.bias)             # p_on = 0.5 at init
            self.gencrit_head = nn.Linear(d_model, 1)
            self.cls_design = nn.Parameter(torch.zeros(d_model))
        self._xavier_init()

    def _xavier_init(self) -> None:
        for name, p in self.named_parameters():
            if "joint_head" in name or "gen_head" in name:  # zero-init heads, leave them be
                continue
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def _encode_legacy(self, root, eff0_tok, eff1_tok, active_mask, B):
        t = self.embed_root(root).unsqueeze(1)
        h = self.embed_eff0(eff0_tok)
        a = self.embed_eff1(eff1_tok)
        x = torch.cat([t, h, a], dim=1)
        x = x + self.type_emb(self.type_ids) + self.pos_emb(self.pos_ids)

        # (B, 1+2*n_limbs, 1): root always active, then eff0 masks, then eff1 masks
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

    def _encode_codesign(self, root, eff0_tok, eff1_tok, active_mask, B):
        """Fixed 25-token live-mode pass: off-limbs become stop tokens (never masked), content
        carries a live/stop mode embedding, plus persistent start anchors. See CONTEXT.md."""
        n = self.n_limbs
        limb_active = active_mask[:, :n]                          # (B,n) eff0 slots == per-limb presence
        h = self.embed_eff0(eff0_tok)   * limb_active.unsqueeze(-1)  # zero stop limbs' state embeds
        a = self.embed_eff1(eff1_tok) * limb_active.unsqueeze(-1)
        cls = self.embed_root(root).unsqueeze(1)
        start = self.angle_proj(self.angle_enc).unsqueeze(0).expand(B, -1, -1)  # (B,n,d) angle only
        x = torch.cat([cls, start, h, a], dim=1)                 # (B, 1+3n, d)
        x = x + self.type_emb(self.type_ids) + self.pos_emb(self.pos_ids)

        content_active = active_mask                            # (B,2n) eff0 then eff1
        mode_ids = torch.where(content_active > 0, content_active.new_full((), _MODE_LIVE),
                               content_active.new_full((), _MODE_STOP)).long()
        mode = torch.cat([x.new_zeros(B, 1 + n, x.shape[-1]), self.mode_emb(mode_ids)], dim=1)
        return self.encoder(x + mode)                          # all 25 tokens real -> no padding

    # ---- design mode: morphology-only generation pass (no physical state) -----------------------
    def _encode_design(self, committed_on, committed_stop):
        """Encode a designed prefix over the fixed 25-token layout. committed_on/committed_stop are
        (M,n) bool per-slot decisions; pending slots (neither) have their content tokens MASKED.
        CLS uses the learned design content; content tokens carry no physical state (type+pos+mode
        only). Returns H (M, n_tokens, d): CLS at 0, start tokens 1..n, content after."""
        M, n = committed_on.shape
        d = self.cls_design.shape[0]
        decided = committed_on | committed_stop                 # (M,n) present iff decided
        cls = self.cls_design.view(1, 1, -1).expand(M, 1, -1)
        start = self.angle_proj(self.angle_enc).unsqueeze(0).expand(M, -1, -1)   # (M,n,d) angle only
        content = cls.new_zeros(M, 2 * n, d)                    # eff0+eff1: no state in design mode
        x = torch.cat([cls, start, content], dim=1)            # (M,1+3n,d)
        x = x + self.type_emb(self.type_ids) + self.pos_emb(self.pos_ids)

        # NB: build the long mode ids by arithmetic, NOT new_full(committed_on) -- committed_on is
        # bool, so new_full would cast both _MODE_* (1 and 2) to True and erase committed-vs-stop.
        mode_slot = committed_on.long() * _MODE_COMMITTED + committed_stop.long() * _MODE_STOP
        mode_ids = mode_slot.repeat(1, 2)                      # eff0 then eff1 share slot mode (M,2n)
        mode = torch.cat([x.new_zeros(M, 1 + n, d), self.mode_emb(mode_ids)], dim=1)
        pad = torch.cat([x.new_zeros(M, 1 + n, dtype=torch.bool),
                         ~decided.repeat(1, 2)], dim=1)        # mask pending content tokens (M,1+3n)
        return self.encoder(x + mode, src_key_padding_mask=pad)

    def codesign_forward(self, obs: torch.Tensor):
        """Live pass returning ContAct mu, ContCrit V0.98, and GenCrit/V1.0 in ONE trunk encode
        (resample-update path, grad-enabled). obs is model-normalized; the global log_std lives on
        the builder Network and is applied by the caller."""
        root, eff0_tok, eff1_tok, active_mask = self.tokenize_fn(obs)
        H = self._encode_codesign(root, eff0_tok, eff1_tok, active_mask, obs.shape[0])
        joints = H[:, self._content_start:, :]
        mu = torch.tanh(self.joint_head(joints).squeeze(-1)) * active_mask
        mu = mu.index_select(-1, self.nat_to_dof)
        return mu, self.value_head(H[:, 0]), self.gencrit_head(H[:, 0])

    @torch.no_grad()
    def sample(self, n: int) -> dict[str, torch.Tensor]:
        """Unroll the generation MDP for n envs. Per env a random slot order; each step encode the
        current designed prefix, read v(prefix) from CLS, emit on/stop on the step's START token,
        commit. >=1-limb guard forces ON on the final slot if nothing is on yet."""
        dev, L = self.type_ids.device, self.n_limbs
        order = torch.argsort(torch.rand(n, L, device=dev), dim=1)   # (n,L) per-env permutation
        on   = torch.zeros(n, L, dtype=torch.bool, device=dev)
        stop = torch.zeros(n, L, dtype=torch.bool, device=dev)
        actions  = torch.empty(n, L, dtype=torch.long, device=dev)
        old_logp = torch.empty(n, L, device=dev)
        v_states = torch.empty(n, L + 1, device=dev)                 # v(prefix) at 0..L decided
        arange = torch.arange(n, device=dev)

        for t in range(L):
            H = self._encode_design(on, stop)
            v_states[:, t] = self.gencrit_head(H[:, 0]).squeeze(-1)
            slot = order[:, t]
            logits = self.gen_head(H[arange, 1 + slot])              # (n,2) from start token
            if t == L - 1:                                           # >=1-limb guard on forced slot
                logits[~on.any(1), _GEN_STOP] = float('-inf')
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            old_logp[:, t] = dist.log_prob(a)
            is_on = a == _GEN_ON
            on[arange, slot]   = is_on
            stop[arange, slot] = ~is_on
            actions[:, t] = a

        H = self._encode_design(on, stop)                           # v at the full body
        v_states[:, L] = self.gencrit_head(H[:, 0]).squeeze(-1)
        return {"order": order, "slots": order, "actions": actions,
                "old_logp": old_logp, "v_states": v_states, "presence": on.float()}

    def gen_replay(self, slots: torch.Tensor, actions: torch.Tensor):
        """Teacher-forced replay WITH grad, batched over all L+1 prefixes in one trunk forward
        (stack B*(L+1) designed prefixes, mask pending content per row). Returns the per-step GenAct
        logits at the chosen slot (B,L,2) and v(prefix) at every prefix (B,L+1)."""
        B, L = slots.shape
        n = self.n_limbs
        inv = torch.argsort(slots, dim=1)                          # (B,L) slot -> step decided
        act_by_slot = torch.gather(actions, 1, inv)               # (B,L) slot -> action
        steps = torch.arange(L + 1, device=slots.device).view(1, L + 1, 1)
        decided = inv.unsqueeze(1) < steps                        # (B,L+1,n) decided by prefix t
        on   = decided & (act_by_slot.unsqueeze(1) == _GEN_ON)
        stop = decided & ~on
        M = B * (L + 1)
        H = self._encode_design(on.reshape(M, n), stop.reshape(M, n))   # (M,n_tokens,d)
        v = self.gencrit_head(H[:, 0]).reshape(B, L + 1)          # v(prefix_t)

        start_out = H[:, 1:1 + n].reshape(B, L + 1, n, -1)        # start-token outputs per prefix
        d = start_out.shape[-1]
        idx = slots.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, d)
        chosen = torch.gather(start_out[:, :L], 2, idx).squeeze(2)    # (B,L,d) prefix_t's slot t
        logits = self.gen_head(chosen)                           # (B,L,2)

        on_steps = (actions == _GEN_ON).float()
        on_before = on_steps.cumsum(1) - on_steps                # exclusive: #on strictly before t
        last = torch.arange(L, device=slots.device) == L - 1
        guard = last.unsqueeze(0) & (on_before == 0)             # only final step, nothing on yet
        logits[..., _GEN_STOP] = torch.where(guard, logits.new_full((), float('-inf')),
                                             logits[..., _GEN_STOP])
        return logits, v

    def forward(self, obs: torch.Tensor, compute_value: bool = True,
                detach_value: bool = False) -> dict[str, torch.Tensor]:
        root, eff0_tok, eff1_tok, active_mask = self.tokenize_fn(obs)
        B = obs.shape[0]
        if self.codesign_tokens:
            x = self._encode_codesign(root, eff0_tok, eff1_tok, active_mask, B)
        else:
            x = self._encode_legacy(root, eff0_tok, eff1_tok, active_mask, B)

        out: dict[str, torch.Tensor] = {}
        if self.has_policy_head:
            joints = x[:, self._content_start:, :]
            a_nat  = torch.tanh(self.joint_head(joints).squeeze(-1))
            a_nat  = a_nat * active_mask
            out['mu'] = a_nat.index_select(-1, self.nat_to_dof)
        if compute_value and self.has_value_head:
            root_feat = x[:, 0, :]
            if detach_value:                               # single-net PPG policy phase:
                root_feat = root_feat.detach()           # value grad stops at the trunk
            out['value'] = self.value_head(root_feat)     # V0.98
        return out


def MultiMorphLimbTransformer(n_layers: int = 3, **kwargs) -> LimbTransformer:
    kwargs.setdefault('n_limbs', 8)
    kwargs.setdefault('eff0_dim', EFF0_DIM_8)      # 8-limb tokens carry segment lengths
    kwargs.setdefault('eff1_dim', EFF1_DIM_8)
    return LimbTransformer(n_layers=n_layers, **kwargs)
