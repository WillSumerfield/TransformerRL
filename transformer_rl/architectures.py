"""Transformer architectures for ant envs. forward returns a dict with keys
'mu'/'value' for whichever heads the net was built with (policy_head/value_head)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenize import (tokenize_4, tokenize_8, tokenize_modules, token_dims, limb_enc,
                       ROOT_DIM, EFF0_DIM, EFF1_DIM, EFF0_DIM_8, EFF1_DIM_8, MODULE_DIM)

_TOKENIZE = {4: tokenize_4, 8: tokenize_8}

# codesign token type / mode ids (see CONTEXT.md "Codesign tokens")
_T_ROOT, _T_START, _T_MODULE = 0, 1, 2            # uniform module token (Phase 1) — no eff0/eff1 split
_MODE_LIVE, _MODE_COMMITTED, _MODE_STOP = 0, 1, 2
_GEN_ON, _GEN_STOP = 0, 1                          # GenAct categorical action ids {continue, stop}


def _make_nat_to_dof(n_limbs: int) -> torch.Tensor:
    idx = torch.arange(2 * n_limbs, dtype=torch.long)
    return idx // 2 + n_limbs * (idx % 2)


class MatmulEmbedding(nn.Embedding):
    """Drop-in `nn.Embedding` whose forward is `one_hot(ids) @ weight` instead of a row gather.

    Forward output is identical (bit-for-bit). The only difference is the BACKWARD. A gather's
    backward is a scatter-accumulate into the weight rows; `torch.use_deterministic_algorithms(True)`
    (which we enable whenever `--seed` is passed) forces that scatter onto a serialized deterministic
    kernel. That is catastrophically slow when a *tiny* table is indexed by a *huge* batch -- our
    3-row mode table hit by B*n_dof tokens made its backward ~60% of GPU time and seeded codesign
    ~2.4x slower. The matmul form's backward is `one_hotᵀ @ grad`, a GEMM: deterministic AND fast.

    Use ONLY for small categorical markers (a handful of rows). For real vocabularies keep
    `nn.Embedding` -- the one-hot would waste memory. Basic lookup only (no padding_idx / max_norm).
    Full rationale: docs/deterministic_embedding.md.
    """
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.one_hot(ids, self.num_embeddings).to(self.weight.dtype) @ self.weight


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
        max_limb_length: int = 1,
    ):
        super().__init__()
        self.n_limbs = n_limbs
        self.has_policy_head = policy_head
        self.has_value_head = value_head
        self.codesign_tokens = codesign_tokens
        self.max_limb_length = max_limb_length
        self._d_model = d_model

        self.embed_root = nn.Linear(root_dim, d_model)

        if codesign_tokens:
            # Phase 1 uniform module-token codesign. A limb is a chain of up to max_limb_length
            # modules; each module is ONE 12-D token. Token layout (n_tokens = 1 + n + n*max_len):
            #   [CLS] [start x n] [module x (n*max_len)]  -- modules in depth-major slot order
            #   slot(n,d) = (d-1)*n + (n-1)  (== ant_multimorph._slot, == env action order).
            # tdims is the single source of truth for the derived obs layout (must match the env's
            # _OBS_* constants, which derive from the SAME (n_limbs, max_limb_length)).
            self.tdims = token_dims(n_limbs, max_limb_length)
            self.n_module_tokens = self.tdims["n_module_tokens"]        # n*max_len
            self.embed_module = nn.Linear(MODULE_DIM, d_model)
            self.register_buffer("_enc", limb_enc(n_limbs), persistent=False)  # (n,2) sin/cos

            # additive learned embeddings summed onto the projected base tokens:
            #   type {root,start,module}; pos = limb slot; depth = swing(0)/knee(1..); mode.
            self.type_emb  = nn.Embedding(3, d_model)
            self.pos_emb   = nn.Embedding(1 + n_limbs, d_model)
            self.depth_emb = nn.Embedding(max_limb_length, d_model)     # per within-limb depth
            # LIVE / COMMITTED / STOP mode. MatmulEmbedding (not nn.Embedding): a 3-row table indexed
            # by B*n_dof tokens has a scatter backward ~2.4x slower under deterministic algorithms
            # (--seed) + torch.compile. See MatmulEmbedding / docs/deterministic_embedding.md.
            self.mode_emb   = MatmulEmbedding(3, d_model)
            self.angle_proj = nn.Linear(2, d_model)
            angle_enc = torch.tensor(
                [[math.sin(i * math.pi / 4), math.cos(i * math.pi / 4)] for i in range(n_limbs)],
                dtype=torch.float32)                                    # matches tokenize.limb_enc
            self.register_buffer("angle_enc", angle_enc, persistent=False)

            type_ids = torch.tensor(
                [_T_ROOT] + [_T_START] * n_limbs + [_T_MODULE] * self.n_module_tokens,
                dtype=torch.long)
            pos_ids  = torch.tensor(
                [0] + list(range(1, n_limbs + 1)) + list(range(1, n_limbs + 1)) * max_limb_length,
                dtype=torch.long)
            # depth id per module token (depth-major): [0]*n, [1]*n, ... encodes swing vs knee.
            module_depth_ids = torch.arange(max_limb_length).repeat_interleave(n_limbs)
            self.register_buffer("module_depth_ids", module_depth_ids, persistent=False)
            self._content_start = 1 + n_limbs                          # module tokens begin after starts
        else:
            self.embed_eff0 = nn.Linear(eff0_dim, d_model)
            self.embed_eff1 = nn.Linear(eff1_dim, d_model)
            self.type_emb = nn.Embedding(3, d_model)
            self.pos_emb  = nn.Embedding(1 + n_limbs, d_model)
            type_ids = torch.tensor([0] + [1] * n_limbs + [2] * n_limbs, dtype=torch.long)
            pos_ids  = torch.tensor([0] + list(range(1, n_limbs + 1)) * 2, dtype=torch.long)
            self._content_start = 1
            self.register_buffer("nat_to_dof", _make_nat_to_dof(n_limbs), persistent=False)

        self.n_tokens = type_ids.numel()
        self.register_buffer("type_ids", type_ids, persistent=False)
        self.register_buffer("pos_ids",  pos_ids,  persistent=False)

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
            #   GenAct    = continue/stop logits, read from each limb's START token (design mode).
            #   GenCrit   = V1.0 body-quality value, read from CLS; NO time feature, evaluable on
            #               both live full-state tokens and partial designed prefixes (same weights).
            #   cls_design = learned CLS content used in design mode (generation has no root state).
            self.gen_head = nn.Linear(d_model, 2)
            nn.init.zeros_(self.gen_head.weight)
            nn.init.zeros_(self.gen_head.bias)             # p_continue = 0.5 at init
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

    # ---- shared additive: depth embedding onto the module block only ----------------------------
    def _depth_add(self) -> torch.Tensor:
        """(1, n_tokens, d) additive: depth_emb on the module block, zeros for CLS + start tokens."""
        dep = self.depth_emb(self.module_depth_ids)                    # (n_dof, d)
        pad = dep.new_zeros(1 + self.n_limbs, dep.shape[-1])
        return torch.cat([pad, dep], dim=0).unsqueeze(0)

    def _tokenize_modules(self, obs):
        return tokenize_modules(obs, self.n_limbs, self.max_limb_length, self.angle_enc)

    # ---- classic (non-codesign) legacy path -----------------------------------------------------
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

    # ---- live control pass (uniform module tokens) ----------------------------------------------
    def _encode_codesign(self, root, module_tok, active_mask, B):
        """Fixed (1+n+n*max_len)-token live-mode pass: inactive module slots become STOP tokens
        (never masked, state embed zeroed), real modules carry LIVE mode; plus persistent start
        anchors + CLS. Depth/type/pos are additive. See CONTEXT.md."""
        n, d = self.n_limbs, self._d_model
        m = self.embed_module(module_tok) * active_mask.unsqueeze(-1)   # zero inactive module state
        cls = self.embed_root(root).unsqueeze(1)
        start = self.angle_proj(self.angle_enc).unsqueeze(0).expand(B, -1, -1)  # (B,n,d) angle anchor
        x = torch.cat([cls, start, m], dim=1)                          # (B, 1+n+n_dof, d)
        x = x + self.type_emb(self.type_ids) + self.pos_emb(self.pos_ids) + self._depth_add()

        mode_ids = torch.where(active_mask > 0, active_mask.new_full((), _MODE_LIVE),
                               active_mask.new_full((), _MODE_STOP)).long()      # (B,n_dof)
        mode = torch.cat([x.new_zeros(B, 1 + n, d), self.mode_emb(mode_ids)], dim=1)
        return self.encoder(x + mode)                                  # all tokens real -> no padding

    def codesign_forward(self, obs: torch.Tensor):
        """Live pass returning ContAct mu, ContCrit V0.98, and GenCrit/V1.0 in ONE trunk encode
        (resample-update path, grad-enabled). obs is model-normalized; the global log_std lives on
        the builder Network and is applied by the caller. Module tokens are already in canonical
        depth-major slot order == env action order, so NO nat_to_dof remap."""
        root, module_tok, active_mask = self._tokenize_modules(obs)
        H = self._encode_codesign(root, module_tok, active_mask, obs.shape[0])
        modules = H[:, self._content_start:, :]
        mu = torch.tanh(self.joint_head(modules).squeeze(-1)) * active_mask
        return mu, self.value_head(H[:, 0]), self.gencrit_head(H[:, 0])

    # ---- design mode: morphology-only generation pass (no physical state) ------------------------
    def _encode_design(self, count, stopped):
        """Encode a designed prefix. count (M,n) long = committed modules per limb; stopped (M,n)
        bool = limb finalized (explicit stop or reached max_len). Per module slot (n,d):
          COMMITTED if d<=count; STOP marker if d==count+1 and stopped; else PENDING (masked).
        CLS uses the learned design content; module tokens carry no physical state (type+pos+depth
        +mode only). Returns H (M, n_tokens, d)."""
        M, n = count.shape
        d, max_len = self._d_model, self.max_limb_length
        dev = count.device
        depth1 = torch.arange(1, max_len + 1, device=dev).view(1, max_len, 1)   # (1,max_len,1)
        cnt = count.unsqueeze(1)                                        # (M,1,n)
        committed = depth1 <= cnt                                       # (M,max_len,n)
        is_stop   = stopped.unsqueeze(1) & (depth1 == cnt + 1)          # (M,max_len,n)
        mode_slot = committed.long() * _MODE_COMMITTED + is_stop.long() * _MODE_STOP
        pad_slot  = ~(committed | is_stop)                             # True = pending -> mask
        mode_ids = mode_slot.reshape(M, self.n_module_tokens)          # depth-major slot order
        pad_mod  = pad_slot.reshape(M, self.n_module_tokens)

        cls = self.cls_design.view(1, 1, -1).expand(M, 1, -1)
        start = self.angle_proj(self.angle_enc).unsqueeze(0).expand(M, -1, -1)  # (M,n,d) angle only
        content = cls.new_zeros(M, self.n_module_tokens, d)            # no state in design mode
        x = torch.cat([cls, start, content], dim=1)
        x = x + self.type_emb(self.type_ids) + self.pos_emb(self.pos_ids) + self._depth_add()
        mode = torch.cat([x.new_zeros(M, 1 + n, d), self.mode_emb(mode_ids)], dim=1)
        pad = torch.cat([x.new_zeros(M, 1 + n, dtype=torch.bool), pad_mod], dim=1)
        return self.encoder(x + mode, src_key_padding_mask=pad)

    @torch.no_grad()
    def sample(self, N: int) -> dict[str, torch.Tensor]:
        """Unroll the frontier generation MDP for N envs (fixed L = n*max_len max steps). Each step:
        encode the current designed prefix, read v(prefix) from CLS, pick a RANDOM still-growable
        limb, read continue/stop on its START token, commit. A limb stops on `stop` or at max_len.
        >=1-limb guard: force continue when the body is still empty and only one growable limb is
        left. Steps where an env has no growable limb are no-ops (active_step=False, masked later)."""
        dev = self.type_ids.device
        n, max_len = self.n_limbs, self.max_limb_length
        L = n * max_len
        count   = torch.zeros(N, n, dtype=torch.long, device=dev)
        stopped = torch.zeros(N, n, dtype=torch.bool, device=dev)
        actions = torch.zeros(N, L, dtype=torch.long, device=dev)
        slots   = torch.zeros(N, L, dtype=torch.long, device=dev)
        old_logp    = torch.zeros(N, L, device=dev)
        active_step = torch.zeros(N, L, dtype=torch.bool, device=dev)
        v_states = torch.zeros(N, L + 1, device=dev)
        arange = torch.arange(N, device=dev)

        for t in range(L):
            H = self._encode_design(count, stopped)
            v_states[:, t] = self.gencrit_head(H[:, 0]).squeeze(-1)
            growable = ~stopped                                        # count==max_len already stopped
            active = growable.any(1)                                   # (N,) still deciding
            r = torch.rand(N, n, device=dev)
            slot = torch.where(growable, r, r.new_full((), -1.0)).argmax(1)   # random growable limb
            logits = self.gen_head(H[arange, 1 + slot])               # (N,2) from START token
            force = active & (count.sum(1) == 0) & (growable.sum(1) == 1)     # >=1-limb guard
            logits[force, _GEN_STOP] = float('-inf')
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            old_logp[:, t] = dist.log_prob(a)
            cont = (a == _GEN_ON) & active
            count[arange, slot] += cont.long()
            reached = count[arange, slot] >= max_len
            stopped[arange, slot] |= ((a == _GEN_STOP) & active) | (cont & reached)
            actions[:, t] = a
            slots[:, t] = slot
            active_step[:, t] = active

        H = self._encode_design(count, stopped)                       # v at the full body
        v_states[:, L] = self.gencrit_head(H[:, 0]).squeeze(-1)
        return {"slots": slots, "actions": actions, "old_logp": old_logp,
                "v_states": v_states, "active_step": active_step,
                "counts": count.float(), "presence": (count > 0).float()}

    def _replay_states(self, slots, actions):
        """Teacher-forced scan (no encode) reconstructing the frontier state at every prefix. Returns
        counts_hist (B,L+1,n), stopped_hist (B,L+1,n), active_hist (B,L), force_hist (B,L)."""
        B, L = slots.shape
        n, max_len = self.n_limbs, self.max_limb_length
        dev = slots.device
        arange = torch.arange(B, device=dev)
        count   = torch.zeros(B, n, dtype=torch.long, device=dev)
        stopped = torch.zeros(B, n, dtype=torch.bool, device=dev)
        counts_hist  = torch.zeros(B, L + 1, n, dtype=torch.long, device=dev)
        stopped_hist = torch.zeros(B, L + 1, n, dtype=torch.bool, device=dev)
        active_hist  = torch.zeros(B, L, dtype=torch.bool, device=dev)
        force_hist   = torch.zeros(B, L, dtype=torch.bool, device=dev)
        for t in range(L):
            counts_hist[:, t] = count
            stopped_hist[:, t] = stopped
            growable = ~stopped
            active = growable.any(1)
            active_hist[:, t] = active
            force_hist[:, t] = active & (count.sum(1) == 0) & (growable.sum(1) == 1)
            slot, a = slots[:, t], actions[:, t]
            cont = (a == _GEN_ON) & active
            count[arange, slot] += cont.long()
            reached = count[arange, slot] >= max_len
            stopped[arange, slot] |= ((a == _GEN_STOP) & active) | (cont & reached)
        counts_hist[:, L] = count
        stopped_hist[:, L] = stopped
        return counts_hist, stopped_hist, active_hist, force_hist

    def gen_replay(self, slots: torch.Tensor, actions: torch.Tensor):
        """Teacher-forced replay WITH grad, batched over all L+1 prefixes in one trunk forward.
        Returns per-step GenAct logits at the chosen slot's START token (B,L,2), v(prefix) at every
        prefix (B,L+1), and a valid-step mask (B,L) (False on no-op frontier steps)."""
        B, L = slots.shape
        n = self.n_limbs
        counts_hist, stopped_hist, active_hist, force_hist = self._replay_states(slots, actions)
        M = B * (L + 1)
        H = self._encode_design(counts_hist.reshape(M, n), stopped_hist.reshape(M, n))
        v = self.gencrit_head(H[:, 0]).reshape(B, L + 1)

        start_out = H[:, 1:1 + n].reshape(B, L + 1, n, -1)            # start-token outputs per prefix
        d = start_out.shape[-1]
        idx = slots.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, d)  # (B,L,1,d)
        chosen = torch.gather(start_out[:, :L], 2, idx).squeeze(2)    # (B,L,d) prefix_t's slot t
        logits = self.gen_head(chosen)                               # (B,L,2)
        logits[..., _GEN_STOP] = torch.where(force_hist, logits.new_full((), float('-inf')),
                                             logits[..., _GEN_STOP])
        return logits, v, active_hist

    def forward(self, obs: torch.Tensor, compute_value: bool = True,
                detach_value: bool = False) -> dict[str, torch.Tensor]:
        B = obs.shape[0]
        out: dict[str, torch.Tensor] = {}
        if self.codesign_tokens:
            root, module_tok, active_mask = self._tokenize_modules(obs)
            x = self._encode_codesign(root, module_tok, active_mask, B)
            if self.has_policy_head:
                modules = x[:, self._content_start:, :]
                out['mu'] = torch.tanh(self.joint_head(modules).squeeze(-1)) * active_mask
        else:
            root, eff0_tok, eff1_tok, active_mask = self.tokenize_fn(obs)
            x = self._encode_legacy(root, eff0_tok, eff1_tok, active_mask, B)
            if self.has_policy_head:
                joints = x[:, self._content_start:, :]
                a_nat  = torch.tanh(self.joint_head(joints).squeeze(-1)) * active_mask
                out['mu'] = a_nat.index_select(-1, self.nat_to_dof)
        if compute_value and self.has_value_head:
            root_feat = x[:, 0, :]
            if detach_value:                               # single-net PPG policy phase:
                root_feat = root_feat.detach()           # value grad stops at the trunk
            out['value'] = self.value_head(root_feat)     # V0.98
        return out

    @property
    def tokenize_fn(self):
        return _TOKENIZE[self.n_limbs]


def MultiMorphLimbTransformer(n_layers: int = 3, **kwargs) -> LimbTransformer:
    kwargs.setdefault('n_limbs', 8)
    kwargs.setdefault('eff0_dim', EFF0_DIM_8)      # 8-limb legacy tokens carry segment lengths
    kwargs.setdefault('eff1_dim', EFF1_DIM_8)
    return LimbTransformer(n_layers=n_layers, **kwargs)
