"""Transformer architectures for ant envs. forward returns a dict with keys
'mu'/'value' for whichever heads the net was built with (policy_head/value_head)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenize import (tokenize_4, tokenize_8, tokenize_modules, token_dims, limb_enc,
                       ROOT_DIM, EFF0_DIM, EFF1_DIM, EFF0_DIM_8, EFF1_DIM_8, MODULE_DIM, ROOT_DIM_P2)

_TOKENIZE = {4: tokenize_4, 8: tokenize_8}

# codesign token type / mode ids (see CONTEXT.md "Codesign tokens")
_T_ROOT, _T_START, _T_MODULE = 0, 1, 2            # uniform module token (Phase 1) — no eff0/eff1 split
_MODE_LIVE, _MODE_COMMITTED, _MODE_STOP = 0, 1, 2
_N_TYPE, _N_MODE = 3, 3     # type / mode vocab sizes — one-hot CONCATENATED into token content (2a)
_GEN_ON, _GEN_STOP = 0, 1                          # GenAct categorical action ids {continue, stop}


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
            # 2a: type + mode one-hots are CONCATENATED into each token's content (not additive), so
            # each content projection is widened by the one-hot dims. type disambiguates token kind
            # (constant per projection today; the discriminator once module SUBTYPES share embed_module
            # at Phase 5). mode (LIVE/COMMITTED/STOP) rides embed_module only.
            self.embed_root   = nn.Linear(ROOT_DIM_P2 + _N_TYPE, d_model)           # override root dim
            self.embed_module = nn.Linear(MODULE_DIM + _N_TYPE + _N_MODE, d_model)
            self.angle_proj   = nn.Linear(2 + _N_TYPE, d_model)

            # additive learned embeddings (still summed on top): pos = limb slot (SHARED across start +
            # module), depth = swing(0)/knee(1..). type/mode are NOT additive anymore (see above).
            self.pos_emb   = nn.Embedding(1 + n_limbs, d_model)
            self.depth_emb = nn.Embedding(max_limb_length, d_model)     # per within-limb depth
            angle_enc = torch.tensor(
                [[math.sin(i * math.pi / 4), math.cos(i * math.pi / 4)] for i in range(n_limbs)],
                dtype=torch.float32)                                    # matches tokenize.limb_enc
            self.register_buffer("angle_enc", angle_enc, persistent=False)

            type_ids = torch.tensor(
                [_T_ROOT] + [_T_START] * n_limbs + [_T_MODULE] * self.n_module_tokens,
                dtype=torch.long)
            # constant per-token type one-hot, concatenated into content per projection (sliced below).
            self.register_buffer("type_oh", F.one_hot(type_ids, _N_TYPE).float(), persistent=False)
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
            # JEPA: shared learned [MASK] latent (swapped in pre-additive at masked positions) +
            # BYOL-style 2-layer predictor (LayerNorm inner norm: batch-independent, safe for the
            # variable-size masked-token index). Inert unless jepa_loss is called (agent config gate).
            self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
            self.jepa_predictor = nn.Sequential(
                nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
                nn.GELU(), nn.Linear(d_model, d_model),
            )
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
    def _encode_codesign(self, root, module_tok, active_mask, B, mask_pos=None):
        """Fixed (1+n+n*max_len)-token live-mode pass: inactive module slots become STOP tokens
        (never masked, state embed zeroed), real modules carry LIVE mode; plus persistent start
        anchors + CLS. type + mode are CONCATENATED into content (2a); pos/depth stay additive.
        mask_pos (B, n_tokens) bool (JEPA): swap the post-embed latent for the learned [MASK] at
        masked positions BEFORE additive pos/depth (which still disambiguate the slot)."""
        toh = self.type_oh                                             # (n_tokens, _N_TYPE)
        mode_ids = torch.where(active_mask > 0, active_mask.new_full((), _MODE_LIVE),
                               active_mask.new_full((), _MODE_STOP)).long()      # (B, n_dof)
        mode_oh = F.one_hot(mode_ids, _N_MODE).to(module_tok.dtype)             # (B, n_dof, _N_MODE)
        module_in = torch.cat(                                         # [physical, type_MODULE, mode]
            [module_tok, toh[self._content_start:].expand(B, -1, -1), mode_oh], dim=-1)
        m = self.embed_module(module_in) * active_mask.unsqueeze(-1)   # zero inactive module state
        cls = self.embed_root(torch.cat([root, toh[0:1].expand(B, -1)], dim=-1)).unsqueeze(1)
        start_in = torch.cat([self.angle_enc, toh[1:1 + self.n_limbs]], dim=-1)  # (n, 2+_N_TYPE)
        start = self.angle_proj(start_in).unsqueeze(0).expand(B, -1, -1)         # (B, n, d) anchor
        x = torch.cat([cls, start, m], dim=1)                          # (B, 1+n+n_dof, d)
        if mask_pos is not None:
            x = torch.where(mask_pos.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        x = x + self.pos_emb(self.pos_ids) + self._depth_add()
        return self.encoder(x)                                         # all tokens real -> no padding

    def codesign_forward(self, obs: torch.Tensor, return_hidden: bool = False):
        """Live pass returning ContAct mu, ContCrit V0.98, and GenCrit/V1.0 in ONE trunk encode
        (resample-update path, grad-enabled). obs is model-normalized; the global log_std lives on
        the builder Network and is applied by the caller. Module tokens are already in canonical
        depth-major slot order == env action order, so NO nat_to_dof remap.
        return_hidden=True also returns the post-trunk hidden states H (B, n_tokens, d) for JEPA
        target / repr-anchor use."""
        root, module_tok, active_mask = self._tokenize_modules(obs)
        H = self._encode_codesign(root, module_tok, active_mask, obs.shape[0])
        modules = H[:, self._content_start:, :]
        mu = torch.tanh(self.joint_head(modules).squeeze(-1)) * active_mask
        out = (mu, self.value_head(H[:, 0]), self.gencrit_head(H[:, 0]))
        return out + (H,) if return_hidden else out

    def _sample_jepa_mask(self, active_mask: torch.Tensor, mask_prob: float) -> torch.Tensor:
        """(B, n_tokens) bool JEPA mask. Maskable = CLS + active modules (never start/inactive-STOP).
        Bernoulli(mask_prob) per maskable token, then per-sample guards force >=1 masked AND >=1
        unmasked among the maskable set (always satisfiable: >=1 module + CLS => >=2 maskable)."""
        B, T, dev = active_mask.shape[0], self.n_tokens, active_mask.device
        maskable = torch.zeros(B, T, dtype=torch.bool, device=dev)
        maskable[:, 0] = True                                          # CLS
        maskable[:, self._content_start:] = active_mask.bool()         # active modules
        mask_pos = maskable & (torch.rand(B, T, device=dev) < mask_prob)
        ar = torch.arange(B, device=dev)
        neg = torch.full((B, T), -1.0, device=dev)
        need_mask = mask_pos.sum(1) == 0                               # force one maskable -> masked
        r = torch.where(maskable, torch.rand(B, T, device=dev), neg)
        mask_pos[ar[need_mask], r.argmax(1)[need_mask]] = True
        need_unmask = mask_pos.sum(1) == maskable.sum(1)              # force one masked -> unmasked
        r2 = torch.where(mask_pos, torch.rand(B, T, device=dev), neg)
        mask_pos[ar[need_unmask], r2.argmax(1)[need_unmask]] = False
        return mask_pos

    def jepa_loss(self, obs: torch.Tensor, mask_prob: float):
        """Same-step I-JEPA on the control trunk: mask a random subset of tokens (CLS + active
        modules) and predict their post-trunk latent from the unmasked context. Target =
        stop-grad post-trunk H_full (unmasked pass); pred = predictor(H_masked[mask]). BYOL cosine
        loss (2-2cos). Grad flows into embed_*/trunk (via unmasked tokens) + mask_token + predictor.
        obs is model-normalized."""
        root, module_tok, active_mask = self._tokenize_modules(obs)
        B = obs.shape[0]
        with torch.no_grad():                                          # target: unmasked context
            H_full = self._encode_codesign(root, module_tok, active_mask, B)
        mask_pos = self._sample_jepa_mask(active_mask, mask_prob)
        H_masked = self._encode_codesign(root, module_tok, active_mask, B, mask_pos=mask_pos)
        pred = self.jepa_predictor(H_masked[mask_pos])                 # (n_masked, d)
        tgt = H_full[mask_pos]                                         # already no-grad
        loss = (2 - 2 * (F.normalize(pred, dim=-1) * F.normalize(tgt, dim=-1)).sum(-1)).mean()
        return loss

    # ---- design mode: morphology-only generation pass (no physical state) ------------------------
    def _encode_design(self, count, stopped):
        """Encode a designed prefix. count (M,n) long = committed modules per limb; stopped (M,n)
        bool = limb finalized (explicit stop or reached max_len). Per module slot (n,d):
          COMMITTED if d<=count; STOP marker if d==count+1 and stopped; else PENDING (masked).
        CLS uses the learned design content; module tokens carry NO physical state — their content is
        zeros(MODULE_DIM) with the type + mode one-hots concatenated (same embed_module path as live
        mode), so type/mode enter via content while pos/depth stay additive. Returns H (M,n_tokens,d)."""
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

        toh = self.type_oh
        mode_oh = F.one_hot(mode_ids, _N_MODE).float()                 # (M, n_dof, _N_MODE)
        phys = mode_oh.new_zeros(M, self.n_module_tokens, MODULE_DIM)  # no physical state in design
        module_in = torch.cat([phys, toh[self._content_start:].expand(M, -1, -1), mode_oh], dim=-1)
        m = self.embed_module(module_in)
        cls = self.cls_design.view(1, 1, -1).expand(M, 1, -1)
        start_in = torch.cat([self.angle_enc, toh[1:1 + n]], dim=-1)
        start = self.angle_proj(start_in).unsqueeze(0).expand(M, -1, -1)  # (M,n,d) angle anchor
        x = torch.cat([cls, start, m], dim=1)
        x = x + self.pos_emb(self.pos_ids) + self._depth_add()
        pad = torch.cat([x.new_zeros(M, 1 + n, dtype=torch.bool), pad_mod], dim=1)
        return self.encoder(x, src_key_padding_mask=pad)

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
