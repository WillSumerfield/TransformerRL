"""Transformer architectures for ant envs. forward returns a dict with keys
'mu'/'value' for whichever heads the net was built with (policy_head/value_head)."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenize import (tokenize_4, tokenize_8, tokenize_modules, token_counts, limb_enc,
                       contact_mask, global_span,
                       ROOT_DIM, EFF0_DIM, EFF1_DIM, EFF0_DIM_8, EFF1_DIM_8, MODULE_DIM)
from .vocab import (CAT_ROOT, CAT_START, CAT_EFFECTOR, CAT_CAP, N_CAT,
                    GEN_EFF, GEN_CAP, N_GEN_CAT)
from codesigner.interfaces import ModuleType

from . import runtime

_TOKENIZE = {4: tokenize_4, 8: tokenize_8}

# codesign token category / mode ids (see CONTEXT.md "Type embedding" / "Codesign tokens").
# Phase 5 splits phase-1's single `module` category into effector/cap and adds a SEPARATE subtype
# one-hot; both are CONCATENATED into token content (2a convention). Pad slots (deeper than a limb's
# cap) carry an all-zero category AND subtype one-hot.
# mode = STATE AVAILABILITY, not token kind (the kind is the category one-hot since Phase 5):
#   LIVE      present, physical state attached (live control pass)
#   COMMITTED present, no physical state (design pass over a token prefix)
#   PAD       slot deeper than this limb's cap -- absent in both passes
_MODE_LIVE, _MODE_COMMITTED, _MODE_PAD = 0, 1, 2
_N_MODE = 3
_FD_MODULE_DIM = 21         # FD raw module target: relpos(3)+relrot6d(6)+relvel(6)+cfrc(6) (2b)
_FK_MODULE_DIM = 15         # FK torso-frame target: pos(3)+rot6d(6)+vel(6, lin+ang) (2b)
# The limb visit order a greedy draw walks. Arbitrary but FIXED, which is the whole point: it is
# what makes "the committed design" one body rather than a family (LimbTransformer.sample).
_GREEDY_ORDER_SEED = 0


def _sixd_to_R(x6: torch.Tensor) -> torch.Tensor:
    """6D rotation (..., 6) = two columns -> orthonormal R (..., 3, 3) via Gram-Schmidt (Zhou 2019)."""
    a1, a2 = x6[..., 0:3], x6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)                       # columns


def _R_to_sixd(R: torch.Tensor) -> torch.Tensor:
    """R (..., 3, 3) -> 6D = first two columns concatenated (matches 2a rel-rot layout)."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def _oh(ids: torch.Tensor, n: int, dtype: torch.dtype) -> torch.Tensor:
    """One-hot with a NULL row: ids < 0 -> an all-zero vector (pad slots, root/start subtype)."""
    return F.one_hot(ids.clamp(min=0), n).to(dtype) * (ids >= 0).unsqueeze(-1).to(dtype)


def _make_nat_to_dof(n_limbs: int) -> torch.Tensor:
    idx = torch.arange(2 * n_limbs, dtype=torch.long)
    return idx // 2 + n_limbs * (idx % 2)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


class _CustomEncoderLayer(nn.Module):
    """Pre-norm block: manual QKV + SDPA, so both depth-only RoPE (3a) and attention-weight dropout
    (3b AttentionDrop) can hook in — stock nn.MultiheadAttention only exposes ONE shared dropout
    float, which can't be split from the post-block dropout (3b Dropout). Mirrors
    nn.TransformerEncoderLayer(norm_first=True, activation='gelu') otherwise. attn_dropout/
    block_dropout are mutually exclusive in practice (LimbTransformer's reg_mode gate) but the layer
    itself doesn't assume that."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 attn_dropout: float = 0.0, block_dropout: float = 0.0):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.attn_dropout = attn_dropout
        self.block_dropout = block_dropout
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Linear(dim_feedforward, d_model))

    def forward(self, x, cos, sin, attn_mask=None):
        B, T, D = x.shape
        h = self.norm1(x)
        qkv = self.qkv_proj(h).view(B, T, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                          # (B, nhead, T, head_dim)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        p = self.attn_dropout if self.training else 0.0
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=p)
        attn = attn.transpose(1, 2).reshape(B, T, D)
        attn = F.dropout(self.out_proj(attn), self.block_dropout, self.training)
        x = x + attn
        ffn_out = F.dropout(self.ffn(self.norm2(x)), self.block_dropout, self.training)
        x = x + ffn_out
        return x


class _CustomEncoder(nn.Module):
    """Drop-in stand-in for nn.TransformerEncoder when LimbTransformer needs RoPE and/or split
    dropout (use_rope=True or reg_mode != 'none'): same call signature
    (`forward(x, src_key_padding_mask=None)`) so call sites don't branch. depth_ids=None -> identity
    rotation (cos=1/sin=0, broadcasts over T) for the reg-only/no-RoPE case; otherwise cos/sin are
    precomputed once from the FIXED per-token depth id (CLS/start = phase 0, matching depth_emb's
    zero-pad convention) since the token layout is constant for a given (n_limbs, max_limb_length)."""
    def __init__(self, d_model: int, nhead: int, n_layers: int, dim_feedforward: int,
                 depth_ids: torch.Tensor = None, attn_dropout: float = 0.0,
                 block_dropout: float = 0.0, base: float = 10000.0):
        super().__init__()
        head_dim = d_model // nhead
        if depth_ids is not None:
            assert head_dim % 2 == 0, "RoPE needs an even head_dim (d_model // n_heads)"
            inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
            angles = depth_ids.float()[:, None] * inv_freq[None, :]        # (T, head_dim/2)
            cos = torch.cat([angles.cos(), angles.cos()], dim=-1)          # (T, head_dim)
            sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
            cos, sin = cos.view(1, 1, -1, head_dim), sin.view(1, 1, -1, head_dim)
        else:
            cos, sin = torch.ones(1, 1, 1, head_dim), torch.zeros(1, 1, 1, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.layers = nn.ModuleList(
            [_CustomEncoderLayer(d_model, nhead, dim_feedforward, attn_dropout, block_dropout)
             for _ in range(n_layers)])

    def forward(self, x, src_key_padding_mask=None):
        attn_mask = None
        if src_key_padding_mask is not None:
            attn_mask = ~src_key_padding_mask[:, None, None, :]        # (B,1,1,T) True=allowed
        for layer in self.layers:
            x = layer(x, self.cos, self.sin, attn_mask=attn_mask)
        return x


class LimbTransformer(nn.Module):
    def __init__(
        self,
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
        fd_variant: str = 'raw',
        use_rope: bool = False,
        reg_mode: str = 'none',
        dropout: float = 0.1,
    ):
        super().__init__()
        if use_rope and not codesign_tokens:
            raise ValueError("use_rope needs codesign_tokens (depth-only rotary needs limb-chain depth)")
        if reg_mode not in ('none', 'dropout', 'attention_drop'):
            raise ValueError(f"reg_mode must be 'none'|'dropout'|'attention_drop', got {reg_mode!r}")
        # The run's ONE library (D14) -- the same instance the Task was set up with, not a second
        # one built from a name here. Slot count and chain depth are its facts, so they are read off
        # it rather than configured: a config that could disagree with the library is a config that
        # eventually does.
        ml = runtime.library()
        # generator/tokenizer subtype vocabulary, derived from the public modules API (not hardcoded).
        # "bare" is OUR choice of the constrained decoder's default/no-cap type (matches the literal
        # vocabulary used everywhere else, e.g. transformer_rl.morphology.CANONICAL_CAP), not
        # library data.
        self.n_eff = len(ml.names(ModuleType.EFFECTOR))
        self.n_cap = len(ml.names(ModuleType.CAP))
        self.cap_bare = ml.names(ModuleType.CAP).index("bare")
        # Subtype one-hot width, == obs_layout()["n_sub"]. The SHARED width is the larger of the two
        # per-type vocabularies, so n_eff and n_cap are each <= n_sub and neither may be assumed to
        # fill it -- the generator's masks are built from the per-type counts, never from n_sub.
        self.n_sub = ml.subtype_width
        n_limbs = ml.n_slots
        max_limb_length = ml.max_depth
        self.n_limbs = n_limbs
        self.has_policy_head = policy_head
        self.has_value_head = value_head
        self.codesign_tokens = codesign_tokens
        self.max_limb_length = max_limb_length
        self.fd_variant = fd_variant                # 'raw' (21-D obs-space) | 'latent' (content embed)
        self.use_rope = use_rope                    # depth-only RoPE vs additive depth_emb (3a)
        self.reg_mode = reg_mode                    # 'none'|'dropout'|'attention_drop' (3b, mutex)
        self._d_model = d_model

        self.embed_root = nn.Linear(root_dim, d_model)

        if codesign_tokens:
            # Phase 1 uniform module-token codesign. A limb is a chain of up to max_limb_length
            # modules; each module is ONE 12-D token. Token layout (n_tokens = 1 + n + n*max_len):
            #   [CLS] [start x n] [module x (n*max_len)]  -- modules in depth-major slot order
            #   slot(n,d) = (d-1)*n + (n-1)  (== the Task's slot order, == env action order).
            # tdims is the Task's OWN obs_layout() (D23) -- where each block starts is the package's
            # fact, published once and read here, not re-derived from (n_limbs, max_limb_length) on
            # this side of the boundary and hand-kept in step.
            if max_limb_length < 2:
                raise ValueError("Phase-5 grammar forces the deepest slot to a cap -> the library's "
                                 f"max_depth must be >= 2 (got {max_limb_length} from "
                                 f"{ml.name!r}); max effectors/limb = it - 1")
            self.tdims = runtime.obs_layout()
            self.n_module_tokens = token_counts(self.tdims)["n_module_tokens"]   # n*max_len
            self.max_effectors = max_limb_length - 1   # deepest slot is grammar-forced to a cap (5a)
            # 2a: category + subtype + mode one-hots are CONCATENATED into each token's content (not
            # additive), so each content projection is widened by the one-hot dims. category
            # disambiguates token kind {root, start, effector, cap}; subtype (library-sized index)
            # picks the concrete kind within it; mode (LIVE/COMMITTED/STOP) rides embed_module only.
            # The root token's content is the WHOLE global region (ADR-0019): the robot's root state,
            # wider for a world-mounted task, plus whatever the Task declared about its objective.
            # One width, read off the layout -- there is no separate "extra" tail to add on, because
            # the region is contiguous and the policy has no reason to tell the two apart.
            self.n_root_axes = self.tdims["n_root_axes"]
            g_start, g_stop = global_span(self.tdims)
            self.root_dim = g_stop - g_start
            self.embed_root = nn.Linear(self.root_dim + N_CAT, d_model)    # override root dim
            self.embed_module = nn.Linear(MODULE_DIM + N_CAT + self.n_sub + _N_MODE, d_model)
            self.angle_proj   = nn.Linear(2 + N_CAT, d_model)

            # additive learned embeddings (still summed on top): pos = limb slot (SHARED across start +
            # module), depth = swing(0)/knee(1..). type/mode are NOT additive anymore (see above).
            self.pos_emb   = nn.Embedding(1 + n_limbs, d_model)
            self.depth_emb = nn.Embedding(max_limb_length, d_model)     # per within-limb depth
            angle_enc = torch.tensor(
                [[math.sin(i * math.pi / 4), math.cos(i * math.pi / 4)] for i in range(n_limbs)],
                dtype=torch.float32)                                    # matches tokenize.limb_enc
            self.register_buffer("angle_enc", angle_enc, persistent=False)

            type_ids = torch.tensor(
                [CAT_ROOT] + [CAT_START] * n_limbs + [CAT_EFFECTOR] * self.n_module_tokens,
                dtype=torch.long)
            # Constant CLS/start category one-hots (rows 0 .. n_limbs), concatenated into content per
            # projection. Module rows are NOT constant anymore -- each module slot's category
            # (effector / cap / pad-null) and subtype are per-sample, built by _module_onehots.
            self.register_buffer("type_oh", F.one_hot(type_ids, N_CAT).float(), persistent=False)
            pos_ids  = torch.tensor(
                [0] + list(range(1, n_limbs + 1)) + list(range(1, n_limbs + 1)) * max_limb_length,
                dtype=torch.long)
            # depth id per module token (depth-major): [0]*n, [1]*n, ... encodes swing vs knee.
            module_depth_ids = torch.arange(max_limb_length).repeat_interleave(n_limbs)
            self.register_buffer("module_depth_ids", module_depth_ids, persistent=False)
            self._content_start = 1 + n_limbs                          # module tokens begin after starts
        else:
            self.n_root_axes = 0      # legacy fixed-4-limb ant: free-floating, no task fields
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

        if use_rope or reg_mode != 'none':
            depth_ids_full = None
            if use_rope:
                # full-token depth id: CLS + start tokens get phase 0 (identity rotation), matching
                # depth_emb's existing zero-pad convention; module tokens get module_depth_ids.
                depth_ids_full = torch.cat(
                    [torch.zeros(1 + n_limbs, dtype=torch.long), module_depth_ids])
            attn_dropout = dropout if reg_mode == 'attention_drop' else 0.0
            block_dropout = dropout if reg_mode == 'dropout' else 0.0
            self.encoder = _CustomEncoder(d_model, n_heads, n_layers, ffn, depth_ids_full,
                                          attn_dropout=attn_dropout, block_dropout=block_dropout)
        else:
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
            # A task that mounts its root on actuated axes acts on those too. They come off the CLS
            # token as ONE multi-output head rather than one token per axis, because a root axis is
            # fixed by the Task and never designed -- a token for it would need special-casing out of
            # the frontier, the prefix stack, design mode and the category grammar (ADR-0019).
            # NOT CONSTRUCTED at zero width: Ant must stay parameter-identical.
            if codesign_tokens and self.n_root_axes:
                self.root_axis_head = nn.Linear(d_model, self.n_root_axes)
                nn.init.zeros_(self.root_axis_head.weight)
                nn.init.zeros_(self.root_axis_head.bias)
        if value_head:
            self.value_head = nn.Linear(d_model, 1)        # V0.98 (sole control critic)

        if codesign_tokens:
            # generator heads on the shared trunk (single-network codesign):
            #   GenAct    = FACTORED module-type emission, read from each limb's START token (design
            #               mode): a CATEGORY head {effector, cap} (positionally grammar-masked) and
            #               a SUBTYPE head (shared index, width 4) masked to the sampled category.
            #               logp = logp(cat) + logp(sub | cat).
            #   GenCrit   = V1.0 body-quality value, read from CLS; NO time feature, evaluable on
            #               both live full-state tokens and partial designed prefixes (same weights).
            #   cls_design = learned CLS content used in design mode (generation has no root state).
            self.gen_cat_head = nn.Linear(d_model, N_GEN_CAT)
            self.gen_sub_head = nn.Linear(d_model, self.n_sub)
            for h in (self.gen_cat_head, self.gen_sub_head):
                nn.init.zeros_(h.weight)
                nn.init.zeros_(h.bias)   # uniform over the VALID set at init; p(effector)=0.5 == p1
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
            # Forward Dynamics (2b, raw variant): per-ACTIVE-MODULE next-step prediction from post-trunk
            # H[t] + OWN sampled action (1 dim, concat at head). No distal/CLS aggregation — module
            # targets are parent-relative so own joint action is the first-order driver (grill 2026-07-09;
            # CLS/root DROPPED, world-absolute target would need whole-body action). -> next parent-
            # relative relpos(3)+relrot(6)+relvel(6)+cfrc(6)=21. Inert unless fd_loss_raw is called.
            # latent variant: head instead predicts the content-only embed of next_obs (d_model), a
            # JEPA target anchored by embed_module (RL-pinned -> no collapse); cosine loss (grill 2026-07-12).
            _fd_out = d_model if fd_variant == 'latent' else _FD_MODULE_DIM
            self.fd_module_head = nn.Sequential(
                nn.Linear(d_model + 1, d_model), nn.GELU(), nn.Linear(d_model, _fd_out))
            self._fd_armed = False           # agent arms this per PPO minibatch (fused aux loss)
            # _enabled: fixed per-run compile-time gate (agent sets from config before torch.compile);
            # forward runs the head iff enabled, so a feature-off run never compiles it in (baseline-
            # identical). _armed is the per-minibatch runtime toggle -> only touches get_aux_loss.
            self._fd_enabled = self._fk_enabled = False
            self._fd_pred = self._fd_active = None
            # Forward Kinematics (2b, same-timestep): each active module token predicts its OWN pose
            # FULLY in the torso frame (pos(3)+rot6D(6)+vel(6)=15) from post-trunk H[t] alone (no action,
            # no mask — self-prediction leaks nothing so full-attention H is fine; grill 2026-07-10).
            # Target = pure limb-chain composition of rel-pos/rel-rot/rel-vel (root terms cancel), built
            # agent-side from RAW obs, per-DEPTH normalized. Inert unless fk_arm'd.
            self.fk_module_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, _FK_MODULE_DIM))
            self._fk_armed = False
            self._fk_pred = None
            self._fk_tgt = self._fk_active = None
            self.register_buffer('fk_mean', torch.zeros(max_limb_length, _FK_MODULE_DIM))
            self.register_buffer('fk_var', torch.ones(max_limb_length, _FK_MODULE_DIM))
            self.register_buffer('fk_count', torch.full((max_limb_length,), 1e-4))
            # Compile the aux loss math (head-forward already rides forward's compile). default mode
            # fuses the launch-bound elementwise+reduction kernels; dynamic=False -> one static graph.
            self._fd_loss_c = torch.compile(self._fd_loss_impl, dynamic=False)
            self._fk_loss_c = torch.compile(self._fk_loss_impl, dynamic=False)
        self._xavier_init()

    def _xavier_init(self) -> None:
        for name, p in self.named_parameters():
            if "joint_head" in name or "gen_cat_head" in name or "gen_sub_head" in name:
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
        return tokenize_modules(obs, self.tdims, self.angle_enc)

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
    def _encode_codesign(self, root, module_tok, active_mask, cap_mask, sub_oh, B, mask_pos=None):
        """Fixed (1+n+n*max_len)-token live-mode pass. Each module slot is an EFFECTOR (has a DOF ->
        active_mask=1), this limb's CAP (present but actionless -> active_mask=0, cap_mask=1, carries
        its subtype one-hot + the limb's contact force), or a PAD slot deeper than the cap (all-zero
        category/subtype, MODE_PAD; never attention-masked, matching phase 1). Plus persistent start
        anchors + CLS. category/subtype/mode are CONCATENATED into content (2a); pos/depth stay
        additive. mask_pos (B, n_tokens) bool (JEPA): swap the post-embed latent for the learned
        [MASK] at masked positions BEFORE additive pos/depth (which still disambiguate the slot).

        NOTE: unlike phase 1 there is no post-projection `* active_mask` — that would erase the cap's
        type + contact. The PHYSICAL content is already zero on non-effector slots (the env masks
        every dynamic obs block by the DOF mask, and tokenize routes cfrc only to the contact slot),
        so the type/mode one-hots are all that survives there. This also makes the live pass agree
        with _encode_design, which never masked its module embeddings."""
        dt = module_tok.dtype
        toh = self.type_oh                                             # (n_tokens, N_CAT)
        cat_oh = module_tok.new_zeros(B, self.n_module_tokens, N_CAT)   # pad slots stay all-zero
        cat_oh[..., CAT_EFFECTOR] = active_mask
        cat_oh[..., CAT_CAP] = cap_mask                                # mutually exclusive by build
        present = active_mask + cap_mask
        mode_ids = torch.where(present > 0, present.new_full((), _MODE_LIVE),
                               present.new_full((), _MODE_PAD)).long()          # (B, n_dof)
        mode_oh = F.one_hot(mode_ids, _N_MODE).to(dt)                           # (B, n_dof, _N_MODE)
        module_in = torch.cat(                              # [physical, category, subtype, mode]
            [module_tok, cat_oh, sub_oh.to(dt), mode_oh], dim=-1)
        m = self.embed_module(module_in)
        cls = self.embed_root(torch.cat([root, toh[0:1].expand(B, -1)], dim=-1)).unsqueeze(1)
        start_in = torch.cat([self.angle_enc, toh[1:1 + self.n_limbs]], dim=-1)  # (n, 2+N_CAT)
        start = self.angle_proj(start_in).unsqueeze(0).expand(B, -1, -1)         # (B, n, d) anchor
        x = torch.cat([cls, start, m], dim=1)                          # (B, 1+n+n_dof, d)
        if mask_pos is not None:
            x = torch.where(mask_pos.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        x = x + self.pos_emb(self.pos_ids)                             # limb slot: always additive
        if not self.use_rope:
            x = x + self._depth_add()                                 # depth: additive unless RoPE
        return self.encoder(x)                                         # all tokens real -> no padding

    def codesign_forward(self, obs: torch.Tensor, return_hidden: bool = False):
        """Live pass returning ContAct mu, ContCrit V0.98, and GenCrit/V1.0 in ONE trunk encode
        (resample-update path, grad-enabled). obs is model-normalized; the global log_std lives on
        the builder Network and is applied by the caller. Module tokens are already in canonical
        depth-major slot order == env action order, so NO nat_to_dof remap.
        return_hidden=True also returns the post-trunk hidden states H (B, n_tokens, d) for JEPA
        target / repr-anchor use."""
        root, module_tok, active_mask, cap_mask, sub_oh = self._tokenize_modules(obs)
        H = self._encode_codesign(root, module_tok, active_mask, cap_mask, sub_oh, obs.shape[0])
        mu = self._action_mu(H, active_mask)
        out = (mu, self.value_head(H[:, 0]), self.gencrit_head(H[:, 0]))
        return out + (H,) if return_hidden else out

    def _action_mu(self, H, active_mask):
        """The full action vector: masked module actions, then the task's root-axis actions.

        Order is POSITIONAL and matches the env's DOF buffer, `[padded module DOFs] ++ [root axes]`
        (`modular.py`'s `_n_dofs_ext`). Reorder either side and the policy drives the wrong joints,
        silently -- there is no name on either end to catch it.

        Root-axis outputs are deliberately UNMASKED: root axes are fixed per env and always active,
        unlike limb slots, and the env's own DOF mask excludes them for the same reason. `tanh` suits
        them because the action space is Box(-1, 1) over the full extended width.
        """
        mu = torch.tanh(self.joint_head(H[:, self._content_start:, :]).squeeze(-1)) * active_mask
        if self.n_root_axes:
            mu = torch.cat([mu, torch.tanh(self.root_axis_head(H[:, 0]))], dim=-1)
        return mu

    def _sample_jepa_mask(self, present_mask: torch.Tensor, mask_prob: float) -> torch.Tensor:
        """(B, n_tokens) bool JEPA mask. Maskable = CLS + PRESENT modules (effectors AND caps; never
        start tokens or pad slots). Bernoulli(mask_prob) per maskable token, then per-sample guards
        force >=1 masked AND >=1 unmasked among the maskable set (always satisfiable: >=1 module +
        CLS => >=2 maskable)."""
        B, T, dev = present_mask.shape[0], self.n_tokens, present_mask.device
        maskable = torch.zeros(B, T, dtype=torch.bool, device=dev)
        maskable[:, 0] = True                                          # CLS
        maskable[:, self._content_start:] = present_mask.bool()        # effectors + caps
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
        root, module_tok, active_mask, cap_mask, sub_oh = self._tokenize_modules(obs)
        B = obs.shape[0]
        enc = lambda mp: self._encode_codesign(root, module_tok, active_mask, cap_mask, sub_oh,
                                               B, mask_pos=mp)
        with torch.no_grad():                                          # target: unmasked context
            H_full = enc(None)
        mask_pos = self._sample_jepa_mask(active_mask + cap_mask, mask_prob)
        H_masked = enc(mask_pos)
        pred = self.jepa_predictor(H_masked[mask_pos])                 # (n_masked, d)
        tgt = H_full[mask_pos]                                         # already no-grad
        loss = (2 - 2 * (F.normalize(pred, dim=-1) * F.normalize(tgt, dim=-1)).sum(-1)).mean()
        return loss

    # ---- Forward Dynamics (2b): per-active-module next-step prediction (raw variant) -------------
    def fd_predict(self, H, actions):
        """module_pred (B, n_dof, 21) from post-trunk H[t] + OWN sampled action. `actions` aligns 1:1
        with the module tokens H[:, content_start:] (both depth-major slot order; mu is one tanh
        action per module token). Own action only — no aggregation.

        `actions` arrives at the FULL action width, so its root-axis tail is sliced off here: those
        entries belong to the CLS token, not to any module, and are not part of any module's own
        next-step dynamics. The slice is a no-op on a free-floating task."""
        n_mod = H.shape[1] - self._content_start
        mod_in = torch.cat([H[:, self._content_start:, :], actions[:, :n_mod].unsqueeze(-1)], dim=-1)
        return self.fd_module_head(mod_in)

    def _fd_loss_impl(self, mod_pred, next_obs, active_mask, valid):
        """Raw FD MSE in NORMALIZED space over ACTIVE modules. mod_pred (B, n_dof, 21) = the head output
        computed in forward (fused PPO pass, over obs[t]'s H + own action). next_obs = model-normalized
        obs[t+1]. Target order = relpos(3)+relrot(6)+relvel(6)+cfrc(6). geom (15) supervised on active
        modules; cfrc (6) on TERMINAL modules only. Masks + terminal computed from obs[t] active_mask
        (morphology constant within episode; next_obs's normalized mask tail is unusable). valid (B,)
        masks last-horizon-step + `done`. torch.compiled -> fuses the target-derivation + masked MSE."""
        B = next_obs.shape[0]
        # geometry target: straight slices (mask-independent) from tokenized normalized next_obs
        _, mod_t, *_ = tokenize_modules(next_obs, self.tdims)
        if self.fd_variant == 'latent':
            # JEPA target = content-only embed of next physical state: next_phys @ W_phys.T (drop the
            # constant type/mode one-hot cols + bias -> the fixed offset c the head could trivially fit).
            # stop-grad (embed_module is RL-pinned, can't collapse -> no EMA). cosine 2-2cos, active*valid.
            tgt = (mod_t @ self.embed_module.weight[:, :MODULE_DIM].t()).detach()   # (B, n_dof, d_model)
            mm = (active_mask * valid.float().unsqueeze(-1)).unsqueeze(-1)          # (B, n_dof, 1)
            cos = F.cosine_similarity(mod_pred, tgt, dim=-1, eps=1e-6).unsqueeze(-1)  # (B, n_dof, 1)
            return ((2.0 - 2.0 * cos) * mm).sum() / mm.sum().clamp(min=1)
        geom_tgt = mod_t[..., 10:25]                                    # relpos3+relrot6+relvel6 (15)
        # cfrc target + mask both come from the slots that actually carry a sensor, per the layout's
        # `has_sensor` field -- so the module asked to predict contact is the one that observes it.
        # This used to re-derive the placement rule (cap if a real body, else terminal effector);
        # the library states it now, and a modlib placing sensors elsewhere stays correct here.
        cfrc_tgt = mod_t[..., 4:10]                                     # (B, n_dof, 6)
        cont = contact_mask(next_obs, self.tdims)                        # (B, n_dof)
        v = valid.float().unsqueeze(-1)
        geom_mm, cfrc_mm = active_mask * v, cont * v                    # (B, n_dof) each
        geom_mse = (((mod_pred[..., :15] - geom_tgt) ** 2).mean(-1) * geom_mm).sum() / geom_mm.sum().clamp(min=1)
        cfrc_mse = (((mod_pred[..., 15:] - cfrc_tgt) ** 2).mean(-1) * cfrc_mm).sum() / cfrc_mm.sum().clamp(min=1)
        return geom_mse + cfrc_mse

    def fd_arm(self, next_obs, valid, coef):
        """Agent arms the loss targets right before the PPO forward (the head reads its own action from
        forward's prev_actions, so no action plumbing here); get_aux_loss() consumes it after."""
        self._fd_armed, self._fd_coef = True, coef
        self._fd_next_obs, self._fd_valid = next_obs, valid

    def fd_disarm(self):
        self._fd_armed = False
        self._fd_pred = self._fd_active = None
        self._fd_next_obs = self._fd_valid = None

    # ---- Forward Kinematics (2b): per-active-module OWN torso-frame pose (same-timestep) ---------
    def fk_compose_target(self, obs):
        """(B, n_dof, 15) torso-frame FK target + (B, n_dof) active_mask, composed from RAW obs
        geometry. Pure limb-chain composition (NO root term -> CLS inert): with P_d = prod of the
        chain's rel-rots up to depth d, pos_d = pos_{d-1} + P_{d-1}@relpos_d, vel likewise, rot_d =
        P_d (as 6D). Depth-major slot order == module tokens. Grill 2026-07-10."""
        _, mod, active, *_ = self._tokenize_modules(obs)                # mod (B, n_dof, 25)
        B, n, D = obs.shape[0], self.n_limbs, self.max_limb_length
        md = mod.view(B, D, n, MODULE_DIM)                             # depth-major -> (B, D, n, 25)
        P = torch.eye(3, device=obs.device, dtype=mod.dtype).expand(B, n, 3, 3).contiguous()
        pos = mod.new_zeros(B, n, 3); vlin = mod.new_zeros(B, n, 3); vang = mod.new_zeros(B, n, 3)
        out = mod.new_zeros(B, D, n, _FK_MODULE_DIM)
        for d in range(D):
            rp, r6 = md[:, d, :, 10:13], md[:, d, :, 13:19]
            rvl, rva = md[:, d, :, 19:22], md[:, d, :, 22:25]
            dpos = torch.einsum('bnij,bnj->bni', P, rp)               # P = P_{d-1}: torso-frame step
            # env rel_lin carries -w_p x dp (rotating parent frame, ant_multimorph.py:607); add it back
            # so vlin telescopes to the clean world-velocity R_root^-1(v_d - v_root) (still pure chain:
            # uses composed vang_{d-1} + dpos, no root token). vang/pos are plain difference chains.
            vlin = vlin + torch.einsum('bnij,bnj->bni', P, rvl) + torch.cross(vang, dpos, dim=-1)
            pos = pos + dpos
            vang = vang + torch.einsum('bnij,bnj->bni', P, rva)
            P = P @ _sixd_to_R(r6)                                     # now P_d
            out[:, d] = torch.cat([pos, _R_to_sixd(P), vlin, vang], dim=-1)
        return out.reshape(B, D * n, _FK_MODULE_DIM), active

    def fk_update_stats(self, tgt, active):
        """Per-DEPTH parallel (Welford) update of the FK target normalizer over ACTIVE modules only.
        Called once per rollout window (agent prepare_dataset); NOT touched in the loss."""
        M, _, Fd = tgt.shape
        D, n = self.max_limb_length, self.n_limbs
        t, a = tgt.view(M, D, n, Fd), active.view(M, D, n)
        for d in range(D):
            x = t[:, d].reshape(-1, Fd)[a[:, d].reshape(-1) > 0]       # (k, 15) active at depth d
            k = x.shape[0]
            if k == 0:
                continue
            bm, bv = x.mean(0), x.var(0, unbiased=False)
            c = self.fk_count[d]; nk = c + k; delta = bm - self.fk_mean[d]
            self.fk_mean[d] = self.fk_mean[d] + delta * (k / nk)
            self.fk_var[d] = (self.fk_var[d] * c + bv * k + delta ** 2 * (c * k / nk)) / nk
            self.fk_count[d] = nk

    def fk_normalize(self, tgt):
        """(M, n_dof, 15) raw target -> per-depth standardized (eval-only; reads buffers, no update)."""
        M = tgt.shape[0]
        t = tgt.view(M, self.max_limb_length, self.n_limbs, -1)
        t = (t - self.fk_mean[None, :, None, :]) / torch.sqrt(self.fk_var[None, :, None, :] + 1e-5)
        return t.reshape(M, -1, tgt.shape[-1])

    def fk_arm(self, tgt, active, coef):
        self._fk_armed, self._fk_coef = True, coef
        self._fk_tgt, self._fk_active = tgt, active

    def fk_disarm(self):
        self._fk_armed = False
        self._fk_pred = None
        self._fk_tgt = self._fk_active = None

    def _fk_loss_impl(self, pred, tgt, active):
        """Masked FK MSE over active modules. pred (B, n_dof, 15) = head output from forward. torch.
        compiled -> fuses the elementwise diff + masked reduction."""
        return (((pred - tgt) ** 2).mean(-1) * active).sum() / active.sum().clamp(min=1)

    def get_aux_loss(self):
        """rl_games hook (a2c_continuous.py:194): each term summed into the single PPO loss, inside
        autocast. Eager dispatcher -- reads the head preds stashed by forward + armed targets, calls
        the compiled loss fns. No head armed -> None -> bit-identical to the phase-1 baseline."""
        aux = {}
        if getattr(self, '_fd_armed', False):
            l = self._fd_loss_c(self._fd_pred, self._fd_next_obs, self._fd_active, self._fd_valid)
            self._fd_last = l.detach().float(); aux['fd'] = self._fd_coef * l
        if getattr(self, '_fk_armed', False):
            l = self._fk_loss_c(self._fk_pred, self._fk_tgt, self._fk_active)
            self._fk_last = l.detach().float(); aux['fk'] = self._fk_coef * l
        return aux or None

    # ---- design mode: morphology-only generation pass (no physical state) ------------------------
    def _encode_design(self, count, cap_sub, eff_sub):
        """Encode a designed prefix (M prefixes batched):
          count   (M,n)         long = committed EFFECTORS per limb
          cap_sub (M,n)         long = the limb's cap subtype, or -1 while the limb is still growable
          eff_sub (M,n,max_len) long = per-depth effector subtype (meaningful where depth < count)
        Per module slot (limb n, 0-based depth d):
          EFFECTOR if d < count; CAP if d == count and capped; else PAD (attention-masked).
        Module tokens carry NO physical state — content is zeros(MODULE_DIM) with the category +
        subtype + mode one-hots concatenated (same embed_module path as live mode), so type enters
        via content while pos/depth stay additive. Returns H (M, n_tokens, d)."""
        M, n = count.shape
        max_len, dev = self.max_limb_length, count.device
        depth0 = torch.arange(max_len, device=dev).view(1, max_len, 1)   # (1,max_len,1)
        cnt = count.unsqueeze(1)                                         # (M,1,n)
        committed = depth0 < cnt                                         # (M,max_len,n)
        is_cap    = (cap_sub >= 0).unsqueeze(1) & (depth0 == cnt)
        pad_slot  = ~(committed | is_cap)                                # True = pending -> mask

        eff_d = eff_sub.permute(0, 2, 1)                                 # (M,max_len,n) depth-major
        cap_d = cap_sub.unsqueeze(1).expand(M, max_len, n)
        null  = count.new_full((), -1)
        cat_slot  = torch.where(committed, count.new_full((), CAT_EFFECTOR),
                                torch.where(is_cap, count.new_full((), CAT_CAP), null))
        sub_slot  = torch.where(committed, eff_d, torch.where(is_cap, cap_d, null))
        mode_slot = torch.where(pad_slot, count.new_full((), _MODE_PAD),
                                count.new_full((), _MODE_COMMITTED))

        nd = self.n_module_tokens
        pad_mod = pad_slot.reshape(M, nd)                                # depth-major slot order
        toh = self.type_oh
        dt = toh.dtype
        cat_oh  = _oh(cat_slot.reshape(M, nd), N_CAT, dt)
        sub_oh  = _oh(sub_slot.reshape(M, nd), self.n_sub, dt)
        mode_oh = F.one_hot(mode_slot.reshape(M, nd), _N_MODE).to(dt)
        phys = mode_oh.new_zeros(M, nd, MODULE_DIM)                      # no physical state in design
        m = self.embed_module(torch.cat([phys, cat_oh, sub_oh, mode_oh], dim=-1))
        cls = self.cls_design.view(1, 1, -1).expand(M, 1, -1)
        start_in = torch.cat([self.angle_enc, toh[1:1 + n]], dim=-1)
        start = self.angle_proj(start_in).unsqueeze(0).expand(M, -1, -1)  # (M,n,d) angle anchor
        x = torch.cat([cls, start, m], dim=1)
        x = x + self.pos_emb(self.pos_ids)                               # limb slot: always additive
        if not self.use_rope:
            x = x + self._depth_add()                                    # depth: additive unless RoPE
        pad = torch.cat([x.new_zeros(M, 1 + n, dtype=torch.bool), pad_mod], dim=1)
        return self.encoder(x, src_key_padding_mask=pad)

    # ---- constrained decoder (5a): purely positional valid-next masks ----------------------------
    def _gen_masks(self, depth, force_grow):
        """depth (N,) long = 0-based depth of the slot about to be filled; force_grow (N,) bool =
        the >=1-limb guard. Returns cat_mask (N, N_GEN_CAT) and sub_mask (N, N_GEN_CAT, n_sub),
        both bool, True = allowed. The rules are positional only (the prev-type mechanism the
        general decoder supports is unused until Stage-3 connectors):
          depth 0             -> effectors + the BARE cap only. A morphology cap needs a limb to sit
                                 on, and `bare cap at depth 0` IS how the generator says "no limb".
                                 force_grow additionally removes the whole cap category.
          1 .. max_len-2      -> every effector + every cap the library actually defines
          max_len-1 (deepest) -> caps only  =>  at most max_len-1 effectors per limb
        Applied IDENTICALLY here and in gen_replay, so the PPO ratio is over the same distribution
        that produced the trace."""
        N, dev = depth.shape[0], depth.device
        eff_ok = depth < self.max_effectors                     # deepest slot is cap-only
        # If the grammar leaves no effector, the cap category must stay open regardless of the guard
        # (unreachable with max_limb_length >= 2, but keeps the row from being fully masked).
        cap_ok = (~force_grow) | (~eff_ok)
        cat_mask = torch.stack([eff_ok, cap_ok], dim=-1)                 # (N, N_GEN_CAT)
        sub_idx = torch.arange(self.n_sub, device=dev)
        sub_eff = (sub_idx < self.n_eff).expand(N, self.n_sub)
        # Each row is masked to its OWN type's vocabulary. n_cap == n_sub in `simple`, so a row of
        # ones was right there by coincidence; a library with fewer caps than effectors (`basic`: 1
        # cap, n_sub 2) would otherwise let the decoder sample a cap subtype that does not exist,
        # and indexing names(CAP) with it raises only once the body reaches the Task.
        cap_bare = F.one_hot(torch.full((N,), self.cap_bare, device=dev), self.n_sub).bool()
        sub_cap = torch.where((depth == 0).unsqueeze(-1), cap_bare,
                              (sub_idx < self.n_cap).expand(N, self.n_sub))
        return cat_mask, torch.stack([sub_eff, sub_cap], dim=-2)         # (N, N_GEN_CAT, n_sub)

    @staticmethod
    def gen_dist(cat_logits, sub_logits, cat_mask, sub_mask):
        """Masked log-probs for the FACTORED GenAct action. Returns
          cat_logp (..., N_GEN_CAT)
          sub_logp (..., N_GEN_CAT, n_sub)  -- row c = the subtype distribution CONDITIONED on
                                               category c (the same n_sub logits under c's mask).
        Masking uses finfo.min rather than -inf so a masked entry contributes p*logp == 0 to the
        entropy instead of 0 * -inf == NaN."""
        neg = torch.finfo(cat_logits.dtype).min
        cat_logp = F.log_softmax(cat_logits.masked_fill(~cat_mask, neg), dim=-1)
        sub_logp = F.log_softmax(
            sub_logits.unsqueeze(-2).expand_as(sub_mask).masked_fill(~sub_mask, neg), dim=-1)
        return cat_logp, sub_logp

    @staticmethod
    def gen_logp_entropy(cat_logp, sub_logp, cat_a, sub_a):
        """Joint log-prob of the taken (category, subtype) pair and the EXACT joint entropy
        H(cat) + sum_c p(c) H(sub | c). Shapes broadcast over any leading dims."""
        lp = cat_logp.gather(-1, cat_a.unsqueeze(-1)).squeeze(-1)
        row = cat_a.unsqueeze(-1).unsqueeze(-1).expand(*cat_a.shape, 1, sub_logp.shape[-1])
        sub_c = sub_logp.gather(-2, row).squeeze(-2)                     # (..., n_sub) under cat_a
        lp = lp + sub_c.gather(-1, sub_a.unsqueeze(-1)).squeeze(-1)
        p_cat = cat_logp.exp()
        h_cat = -(p_cat * cat_logp).sum(-1)
        h_sub = -(sub_logp.exp() * sub_logp).sum(-1)                     # (..., N_GEN_CAT)
        return lp, h_cat + (p_cat * h_sub).sum(-1)

    @staticmethod
    def gen_entropy_split(cat_logp, sub_logp):
        """The two ADDITIVE terms of the joint entropy that gen_logp_entropy sums:
          h_cat      = H(category)                -- the SKELETON decision (effector vs cap), which
                                                     is EXACTLY the phase-1/3 binary grow/stop head
          h_sub_cond = sum_c p(c) H(sub | c)      -- the subtype axis phase 5 ADDED
        Split so Rao-Blackwell H(B) can be attributed to skeleton vs subtype: comparing phase-5's
        total against phase-3's overconstrains the comparison, since phase 3 cannot express a
        subtype difference at all. Analysis-only -- the PPO path still uses gen_logp_entropy."""
        p_cat = cat_logp.exp()
        h_cat = -(p_cat * cat_logp).sum(-1)
        h_sub = -(sub_logp.exp() * sub_logp).sum(-1)                 # (..., N_GEN_CAT)
        return h_cat, (p_cat * h_sub).sum(-1)

    @staticmethod
    def commit(count, cap_sub, eff_sub, arange, slot, depth, cat_a, sub_a, active):
        """Apply one emitted token to the frontier state, IN PLACE. Emitting an effector appends it
        at `depth`; emitting a cap finalizes the limb. Inactive envs (no growable limb left) are
        no-ops. Shared by sample(), the replay scan, and the agent's scripted teacher so all three
        walk one identical MDP."""
        is_eff = (cat_a == GEN_EFF) & active
        is_cap = (cat_a == GEN_CAP) & active
        eff_sub[arange, slot, depth] = torch.where(is_eff, sub_a, eff_sub[arange, slot, depth])
        count[arange, slot] += is_eff.long()
        cap_sub[arange, slot] = torch.where(is_cap, sub_a, cap_sub[arange, slot])

    @torch.no_grad()
    def sample(self, N: int, mode: str = "stochastic",
               beta: float | None = None) -> dict[str, torch.Tensor]:
        """Unroll the frontier generation MDP for N envs (fixed L = n*max_len steps). Each step:
        encode the current designed prefix, read v(prefix) from CLS, pick a RANDOM still-growable
        limb (one with no cap yet), read the FACTORED GenAct on its START token under the positional
        grammar mask, and commit the emitted module.

        `mode` selects how the (category, subtype) pair is drawn -- ALL three walk the identical
        grammar-masked MDP, differing only in the pick:
          stochastic  the trained GenAct distribution (default; training + in-distribution eval)
          greedy      argmax of the masked GenAct -- the generator's committed 'best morph'
          uniform     uniform over the grammar-VALID set (zero logits into gen_dist) -- a random
                      policy on the same MDP; the diversity-reference / random-generator body source.

        `beta` is the same three modes as ONE continuous knob: an inverse temperature multiplying the
        raw GenAct logits before the grammar mask. beta=1 is `stochastic`, beta=0 zeroes the logits
        and so IS `uniform`, and beta=inf takes the argmax and so IS `greedy` -- the named modes are
        exactly these three values, not approximations of them (argmax is invariant to a positive
        scale, and masking after scaling leaves the valid set untouched at every beta). Passing it
        overrides `mode`. What the package's **spread control** is expressed in: the interpolation
        between a committed design and the grammar has to be continuous to be bisected on, and the
        internal knob is deliberately never the axis reported.

        NOTE `old_logp` and `step_entropy*` come back under the SAMPLED distribution, so a trace
        drawn at beta != 1 is not a valid PPO trace. Sampling-only paths only.

        Emitting a cap finalizes the limb, and the grammar forces a cap at the deepest slot, so
        every limb ends with exactly one cap and costs at most (max_len-1 effectors + 1 cap) =
        max_len steps — L = n*max_len steps therefore always suffice.
        >=1-limb guard: force an EFFECTOR when the body is still empty and only one growable limb is
        left. Steps where an env has no growable limb are no-ops (active_step=False, masked later)."""
        if beta is None:
            beta = {"stochastic": 1.0, "uniform": 0.0, "greedy": float("inf")}[mode]
        assert beta >= 0.0, f"beta must be non-negative, got {beta}"
        dev = self.type_ids.device
        # A greedy draw has to be THE committed design: the same body on every row and the same
        # body on the next call. Argmax alone does not give that -- the limb VISIT ORDER is random,
        # and order changes the conditioning, so draws differ (which is why `evalpass.modal_design`
        # exists). Under greedy the order is therefore drawn from a fixed seed and shared across
        # rows. `None` leaves every other beta on the global stream, bit-for-bit as before.
        order_rng = None
        if beta == float("inf"):
            order_rng = torch.Generator(device=dev)
            order_rng.manual_seed(_GREEDY_ORDER_SEED)
        n, max_len = self.n_limbs, self.max_limb_length
        L = n * max_len
        count   = torch.zeros(N, n, dtype=torch.long, device=dev)
        cap_sub = torch.full((N, n), -1, dtype=torch.long, device=dev)
        eff_sub = torch.full((N, n, max_len), -1, dtype=torch.long, device=dev)
        cat_a   = torch.zeros(N, L, dtype=torch.long, device=dev)
        sub_a   = torch.zeros(N, L, dtype=torch.long, device=dev)
        slots   = torch.zeros(N, L, dtype=torch.long, device=dev)
        old_logp    = torch.zeros(N, L, device=dev)
        step_ent    = torch.zeros(N, L, device=dev)     # analytic per-step H -> Rao-Blackwell H(B)
        step_ent_cat = torch.zeros(N, L, device=dev)    # skeleton term (== the phase-1/3 head)
        step_ent_sub = torch.zeros(N, L, device=dev)    # subtype term (new in phase 5)
        active_step = torch.zeros(N, L, dtype=torch.bool, device=dev)
        v_states = torch.zeros(N, L + 1, device=dev)
        arange = torch.arange(N, device=dev)

        for t in range(L):
            H = self._encode_design(count, cap_sub, eff_sub)
            v_states[:, t] = self.gencrit_head(H[:, 0]).squeeze(-1)
            growable = cap_sub < 0                                 # uncapped == still growable
            active = growable.any(1)                               # (N,) still deciding
            r = (torch.rand(1, n, device=dev, generator=order_rng).expand(N, n)
                 if order_rng is not None else torch.rand(N, n, device=dev))
            slot = torch.where(growable, r, r.new_full((), -1.0)).argmax(1)   # random growable limb
            depth = count[arange, slot]
            force = active & (count.sum(1) == 0) & (growable.sum(1) == 1)     # >=1-limb guard
            cat_mask, sub_mask = self._gen_masks(depth, force)
            h = H[arange, 1 + slot]                                # (N,d) from the START token
            cat_in, sub_in = self.gen_cat_head(h), self.gen_sub_head(h)
            if beta == 0.0:                                        # random policy: uniform over valid
                cat_in, sub_in = torch.zeros_like(cat_in), torch.zeros_like(sub_in)
            elif beta != 1.0 and beta != float("inf"):             # argmax ignores a positive scale
                cat_in, sub_in = cat_in * beta, sub_in * beta
            cat_logp, sub_logp = self.gen_dist(cat_in, sub_in, cat_mask, sub_mask)
            if beta == float("inf"):
                c = cat_logp.argmax(-1)
                s = sub_logp[arange, c].argmax(-1)
            else:
                c = torch.distributions.Categorical(logits=cat_logp).sample()
                s = torch.distributions.Categorical(logits=sub_logp[arange, c]).sample()
            old_logp[:, t], step_ent[:, t] = self.gen_logp_entropy(cat_logp, sub_logp, c, s)
            step_ent_cat[:, t], step_ent_sub[:, t] = self.gen_entropy_split(cat_logp, sub_logp)
            self.commit(count, cap_sub, eff_sub, arange, slot, depth, c, s, active)
            cat_a[:, t], sub_a[:, t] = c, s
            slots[:, t] = slot
            active_step[:, t] = active

        H = self._encode_design(count, cap_sub, eff_sub)           # v at the full body
        v_states[:, L] = self.gencrit_head(H[:, 0]).squeeze(-1)
        return {"slots": slots, "cat_actions": cat_a, "sub_actions": sub_a, "old_logp": old_logp,
                "v_states": v_states, "active_step": active_step, "step_entropy": step_ent,
                "step_entropy_cat": step_ent_cat, "step_entropy_sub": step_ent_sub,
                "counts": count.float(), "presence": (count > 0).float(),
                "eff_sub": eff_sub, "cap_sub": cap_sub}

    def _replay_states(self, slots, cat_a, sub_a):
        """Teacher-forced scan (no encode) reconstructing the frontier state at every prefix.
        Returns count_hist (B,L+1,n), cap_hist (B,L+1,n), eff_hist (B,L+1,n,max_len),
        active_hist (B,L), depth_hist (B,L), force_hist (B,L)."""
        B, L = slots.shape
        n, max_len, dev = self.n_limbs, self.max_limb_length, slots.device
        arange = torch.arange(B, device=dev)
        count   = torch.zeros(B, n, dtype=torch.long, device=dev)
        cap_sub = torch.full((B, n), -1, dtype=torch.long, device=dev)
        eff_sub = torch.full((B, n, max_len), -1, dtype=torch.long, device=dev)
        count_hist = torch.zeros(B, L + 1, n, dtype=torch.long, device=dev)
        cap_hist   = torch.zeros(B, L + 1, n, dtype=torch.long, device=dev)
        eff_hist   = torch.zeros(B, L + 1, n, max_len, dtype=torch.long, device=dev)
        active_hist = torch.zeros(B, L, dtype=torch.bool, device=dev)
        depth_hist  = torch.zeros(B, L, dtype=torch.long, device=dev)
        force_hist  = torch.zeros(B, L, dtype=torch.bool, device=dev)
        for t in range(L):
            count_hist[:, t], cap_hist[:, t], eff_hist[:, t] = count, cap_sub, eff_sub
            growable = cap_sub < 0
            active = growable.any(1)
            slot = slots[:, t]
            depth = count[arange, slot]
            active_hist[:, t] = active
            depth_hist[:, t] = depth
            force_hist[:, t] = active & (count.sum(1) == 0) & (growable.sum(1) == 1)
            self.commit(count, cap_sub, eff_sub, arange, slot, depth,
                        cat_a[:, t], sub_a[:, t], active)
        count_hist[:, L], cap_hist[:, L], eff_hist[:, L] = count, cap_sub, eff_sub
        return count_hist, cap_hist, eff_hist, active_hist, depth_hist, force_hist

    def gen_replay(self, slots: torch.Tensor, cat_a: torch.Tensor, sub_a: torch.Tensor):
        """Teacher-forced replay WITH grad, batched over all L+1 prefixes in one trunk forward.
        Returns (cat_logp (B,L,N_GEN_CAT), sub_logp (B,L,N_GEN_CAT,n_sub), v (B,L+1), valid (B,L)).
        The grammar mask is rebuilt from the replayed state EXACTLY as in sample() — same positional
        rule, same >=1-limb guard — so the PPO ratio is over the same constrained distribution."""
        B, L = slots.shape
        n, max_len = self.n_limbs, self.max_limb_length
        c_h, cap_h, eff_h, active_h, depth_h, force_h = self._replay_states(slots, cat_a, sub_a)
        M = B * (L + 1)
        H = self._encode_design(c_h.reshape(M, n), cap_h.reshape(M, n),
                                eff_h.reshape(M, n, max_len))
        v = self.gencrit_head(H[:, 0]).reshape(B, L + 1)

        start_out = H[:, 1:1 + n].reshape(B, L + 1, n, -1)            # start-token outputs per prefix
        d = start_out.shape[-1]
        idx = slots.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, d)  # (B,L,1,d)
        chosen = torch.gather(start_out[:, :L], 2, idx).squeeze(2)    # (B,L,d) prefix_t's slot t
        cat_mask, sub_mask = self._gen_masks(depth_h.reshape(-1), force_h.reshape(-1))
        cat_logp, sub_logp = self.gen_dist(
            self.gen_cat_head(chosen), self.gen_sub_head(chosen),
            cat_mask.view(B, L, N_GEN_CAT), sub_mask.view(B, L, N_GEN_CAT, self.n_sub))
        return cat_logp, sub_logp, v, active_h


    def forward(self, obs: torch.Tensor, compute_value: bool = True,
                detach_value: bool = False, actions: torch.Tensor = None) -> dict[str, torch.Tensor]:
        B = obs.shape[0]
        out: dict[str, torch.Tensor] = {}
        if self.codesign_tokens:
            root, module_tok, active_mask, cap_mask, sub_oh = self._tokenize_modules(obs)
            x = self._encode_codesign(root, module_tok, active_mask, cap_mask, sub_oh, B)
            # Fused FD/FK aux (2b): run the heads here so they ride this pass's H[t] + the forward
            # compile (0 extra trunk passes). Fixed _enabled gate -> off-run never compiles them in.
            # FD also needs the own action (from prev_actions); absent in rollout -> skipped there.
            if self._fk_enabled and self.training:
                self._fk_pred = self.fk_module_head(x[:, self._content_start:, :])
            if self._fd_enabled and actions is not None:
                self._fd_pred, self._fd_active = self.fd_predict(x, actions), active_mask
            if self.has_policy_head:
                out['mu'] = self._action_mu(x, active_mask)
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
    # Slot count is the library's now (D14), so there is nothing to default here.
    kwargs.setdefault('eff0_dim', EFF0_DIM_8)      # 8-limb legacy tokens carry segment lengths
    kwargs.setdefault('eff1_dim', EFF1_DIM_8)
    return LimbTransformer(n_layers=n_layers, **kwargs)
