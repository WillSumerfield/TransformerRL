"""SeqGenerator: the sequential per-token morphology generator (ADR-0012, supersedes the ADR-0010
Bernoulli bandit in generator.py). Append-only autoregressive design: the token set starts as
[CLS] + 8 finger-start tokens; each step picks a start token (per-env random order), applies a 2-way
{on,stop} policy head to its encoded output, and APPENDS a committed token. The CLS token feeds the
value head v(prefix). Full bidirectional attention over the current set (1 layer) gives the
sequential conditioning; append-only + immutable + 1-layer make a KV cache an exact drop-in later
(deferred). Standalone / rl_games-agnostic: the agent calls sample() at window start and the update
(Stage 3) at window end. This (max limb length 1) is the degenerate warmup for general autoregressive
design: here we only generate-from start tokens (each once -> 8 steps); longer-limb envs lift that
mask and generate-from an on-token to extend a limb."""
import math
import torch
import torch.nn as nn

ON, STOP = 0, 1                                   # policy action ids (categorical over {on, stop})
_T_START, _T_ON, _T_STOP, _T_CLS = 0, 1, 2, 3     # token type-embedding ids


class SeqGenerator(nn.Module):
    def __init__(self, n_legs=8, d_model=128, n_heads=8, n_layers=1, ffn=512,
                 lr=1e-3, clip=0.2, entropy_coef=0.01, critic_coef=0.5,
                 epochs=4, minibatches=4, n_pretrain=8, device="cuda:0"):
        super().__init__()
        self.n_legs, self.device = n_legs, device
        self.clip, self.entropy_coef, self.critic_coef = clip, entropy_coef, critic_coef
        self.epochs, self.minibatches, self.n_pretrain = epochs, minibatches, n_pretrain
        self.window = 0

        self.type_emb = nn.Embedding(4, d_model)              # START / ON / STOP / CLS
        self.pos_emb = nn.Embedding(n_legs, d_model)          # limb index 0..7
        self.angle_proj = nn.Linear(2, d_model)               # sin/cos leg angle -> d_model
        self.cls_emb = nn.Parameter(torch.zeros(d_model))     # learned CLS content

        angle_enc = torch.tensor(
            [[math.sin(i * math.pi / 4), math.cos(i * math.pi / 4)] for i in range(n_legs)],
            dtype=torch.float32)                              # (n_legs, 2), matches tokenize._LEG_ENC_8
        self.register_buffer("angle_enc", angle_enc, persistent=False)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)

        self.policy_head = nn.Linear(d_model, 2)              # {on, stop}
        self.value_head = nn.Linear(d_model, 1)               # v(prefix) on CLS
        nn.init.zeros_(self.policy_head.weight); nn.init.zeros_(self.policy_head.bias)  # p_on=0.5 start

        self.to(device)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    # ---- token embeddings (recomputed each forward; cheap, keeps grads for Stage-3 update) -------
    def _cls_embed(self):                                     # (d,)
        return self.cls_emb + self.type_emb.weight[_T_CLS]

    def _start_embeds(self):                                  # (n_legs, d)
        idx = torch.arange(self.n_legs, device=self.device)
        stype = torch.full((self.n_legs,), _T_START, dtype=torch.long, device=self.device)
        return self.type_emb(stype) + self.pos_emb(idx) + self.angle_proj(self.angle_enc)

    def _committed_embed(self, slot, action):                # (B,d) for one appended token per env
        tid = torch.where(action == ON, torch.full_like(action, _T_ON),
                          torch.full_like(action, _T_STOP))
        return self.type_emb(tid) + self.pos_emb(slot) + self.angle_proj(self.angle_enc[slot])

    def _encode(self, cls, start, committed):                # full re-encode; CLS at index 0, starts 1..8
        seq = torch.cat([cls, start] + committed, dim=1)     # (B, 1+8+t, d)
        return self.encoder(seq)

    @torch.no_grad()
    def sample(self, n):
        """Unroll the generation MDP for n envs. Per env: random slot order; each step encode the
        current set, read v(prefix) from CLS, emit on/stop on the chosen start token, append it.
        >=1-leg guard masks stop on the final step when no on yet. Returns the trace Stage 3 needs."""
        dev, L = self.device, self.n_legs
        order = torch.argsort(torch.rand(n, L, device=dev), dim=1)   # (n,L) per-env permutation
        presence = torch.zeros(n, L, device=dev)
        slots = torch.empty(n, L, dtype=torch.long, device=dev)
        actions = torch.empty(n, L, dtype=torch.long, device=dev)
        old_logp = torch.empty(n, L, device=dev)
        v_states = torch.empty(n, L + 1, device=dev)                 # v(prefix) at 0..L committed

        cls = self._cls_embed().view(1, 1, -1).expand(n, 1, -1)
        start = self._start_embeds().unsqueeze(0).expand(n, -1, -1)
        committed, arange = [], torch.arange(n, device=dev)

        for t in range(L):
            H = self._encode(cls, start, committed)
            v_states[:, t] = self.value_head(H[:, 0]).squeeze(-1)
            slot = order[:, t]
            logits = self.policy_head(H[arange, 1 + slot])           # (n,2); starts at seq idx 1..8
            if t == L - 1:                                           # >=1-leg guard on forced last slot
                logits[presence.sum(1) == 0, STOP] = float('-inf')
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            old_logp[:, t] = dist.log_prob(a)
            presence[arange, slot] = (a == ON).float()
            slots[:, t], actions[:, t] = slot, a
            committed.append(self._committed_embed(slot, a).unsqueeze(1))

        H = self._encode(cls, start, committed)                      # v at the full body
        v_states[:, L] = self.value_head(H[:, 0]).squeeze(-1)
        return {"order": order, "slots": slots, "actions": actions,
                "old_logp": old_logp, "v_states": v_states, "presence": presence}

    # ---- teacher-forced replay (shared by RL + BC): re-encode each prefix WITH grad -------------
    def _replay(self, slots, actions):
        """Reconstruct the trajectory's prefixes from committed (slots, actions) and return the
        per-step policy logits (guard-consistent) at the chosen slots + v(prefix) at all 9 states.
        slots/actions: (B, L) in generation order. Returns logits (B, L, 2), v (B, L+1)."""
        B, L = slots.shape
        cls = self._cls_embed().view(1, 1, -1).expand(B, 1, -1)
        start = self._start_embeds().unsqueeze(0).expand(B, -1, -1)
        committed, arange = [], torch.arange(B, device=self.device)
        logits = slots.new_zeros(B, L, 2, dtype=torch.float32)
        v = slots.new_zeros(B, L + 1, dtype=torch.float32)
        on_so_far = slots.new_zeros(B, dtype=torch.float32)

        for t in range(L):
            H = self._encode(cls, start, committed)
            v[:, t] = self.value_head(H[:, 0]).squeeze(-1)
            slot = slots[:, t]
            lg = self.policy_head(H[arange, 1 + slot])               # (B,2)
            if t == L - 1:                                           # replicate the >=1-leg guard
                lg = lg.masked_fill(
                    ((on_so_far == 0).unsqueeze(1) & torch.tensor([[False, True]], device=self.device)),
                    float('-inf'))
            logits[:, t] = lg
            a = actions[:, t]
            on_so_far = on_so_far + (a == ON).float()
            committed.append(self._committed_embed(slot, a).unsqueeze(1))

        H = self._encode(cls, start, committed)
        v[:, L] = self.value_head(H[:, 0]).squeeze(-1)
        return logits, v

    def _mb(self, n):
        idx = torch.randperm(n, device=self.device); m = max(1, n // self.minibatches)
        return [idx[s:s + m] for s in range(0, n, m)]

    def rl_update(self, trace, returns):
        """PPO-clip over the per-token decisions + value distillation. Method B advantage: telescoping
        v_snap differences (terminal bootstraps v(full)); every prefix state regresses to R_i."""
        R = returns.to(self.device).float()
        slots, actions = trace['slots'], trace['actions']
        old_logp, v_snap = trace['old_logp'], trace['v_states']       # (n,L), (n,L+1)
        adv = v_snap[:, 1:] - v_snap[:, :-1]                          # (n,L) telescoping
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)                 # joint-normalize over n*L
        n = slots.shape[0]
        for _ in range(self.epochs):
            for mb in self._mb(n):
                logits, v = self._replay(slots[mb], actions[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(actions[mb])                 # (m,L)
                ratio = torch.exp(new_logp - old_logp[mb])
                a_mb = adv[mb]
                pg = -torch.min(ratio * a_mb,
                                ratio.clamp(1 - self.clip, 1 + self.clip) * a_mb).mean()
                vloss = (v - R[mb].unsqueeze(1)).pow(2).mean()        # all L+1 states -> R_i
                ent = dist.entropy().mean()
                self.opt.zero_grad()
                (pg + self.critic_coef * vloss - self.entropy_coef * ent).backward()
                self.opt.step()
        self.window += 1
        return self._log(pg, vloss, ent, adv)

    def pretrain_update(self, target_presence, returns):
        """Teacher-forced per-step BC toward supplied bodies (CE = -logp) + value distillation."""
        R = returns.to(self.device).float()
        tgt = target_presence.to(self.device).float()                # (n,8) 1=on
        n = tgt.shape[0]
        order = torch.argsort(torch.rand(n, self.n_legs, device=self.device), dim=1)
        act = torch.where(tgt > 0, torch.full_like(tgt, ON), torch.full_like(tgt, STOP)).long()
        act_steps = torch.gather(act, 1, order)                      # target action at each step
        for _ in range(self.epochs):
            for mb in self._mb(n):
                logits, v = self._replay(order[mb], act_steps[mb])
                dist = torch.distributions.Categorical(logits=logits)
                bc = -dist.log_prob(act_steps[mb]).mean()
                vloss = (v - R[mb].unsqueeze(1)).pow(2).mean()
                ent = dist.entropy().mean()
                self.opt.zero_grad()
                (bc + self.critic_coef * vloss - self.entropy_coef * ent).backward()
                self.opt.step()
        self.window += 1
        return self._log(bc, vloss, ent, None)

    def in_pretrain(self):  return self.window < self.n_pretrain

    def gen_fraction(self):
        return 1.0 if self.window >= self.n_pretrain else self.window / max(1, self.n_pretrain)

    def _log(self, loss_a, vloss, ent, adv):
        d = {"loss_a": loss_a.item(), "vloss": vloss.item(), "ent": ent.item()}
        if adv is not None:
            d["adv_mean"], d["adv_std"] = adv.mean().item(), adv.std().item()
        return d
