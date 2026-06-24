"""Generator: the unconditional morphology generator -- an 8-arm independent-Bernoulli presence
bandit + scalar return baseline. No trunk, no observation input (body-agnostic baseline by
construction; see ADR-0010). Standalone, rl_games-agnostic: the agent calls sample() at window
start and pretrain_update()/rl_update() at window end."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, n_legs=8, lr=1e-2, clip=0.2, entropy_coef=0.01, critic_coef=0.5,
                 epochs=4, minibatches=4, n_pretrain=8, device="cuda:0"):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_legs, device=device))  # per-leg presence (p0=0.5)
        self.v      = nn.Parameter(torch.zeros((), device=device))      # scalar return baseline
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.n_legs, self.clip = n_legs, clip
        self.entropy_coef, self.critic_coef = entropy_coef, critic_coef
        self.epochs, self.minibatches, self.n_pretrain = epochs, minibatches, n_pretrain
        self.device, self.window = device, 0

    @torch.no_grad()
    def sample(self, n):
        """n per-env bodies. Returns (presence (n,8){0,1}, old_logits (8,) snapshot)."""
        p = torch.sigmoid(self.logits)
        presence = torch.bernoulli(p.expand(n, self.n_legs))
        zero = presence.sum(1) == 0
        while zero.any():                                       # rejection-resample all-zero rows
            presence[zero] = torch.bernoulli(p.expand(int(zero.sum()), self.n_legs))
            zero = presence.sum(1) == 0
        return presence, self.logits.detach().clone()

    def _logp(self, presence, logits):                          # sum per-leg Bernoulli log-prob (N,)
        return -F.binary_cross_entropy_with_logits(
            logits.expand_as(presence), presence, reduction="none").sum(1)

    def _entropy(self, logits):                                 # summed Bernoulli entropy (scalar)
        p = torch.sigmoid(logits)
        return -(p * F.logsigmoid(logits) + (1 - p) * F.logsigmoid(-logits)).sum()

    def _mb(self, N):
        idx = torch.randperm(N, device=self.device); m = max(1, N // self.minibatches)
        return [idx[s:s + m] for s in range(0, N, m)]

    def rl_update(self, presence, old_logits, returns):
        """PPO-clip PG + value MSE + entropy bonus over the window's (presence, return) samples."""
        presence, returns = presence.to(self.device).float(), returns.to(self.device).float()
        adv = returns - self.v.detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        old_logp = self._logp(presence, old_logits)
        for _ in range(self.epochs):
            for mb in self._mb(presence.shape[0]):
                ratio = torch.exp(self._logp(presence[mb], self.logits) - old_logp[mb])
                pg = -torch.min(ratio * adv[mb],
                                ratio.clamp(1 - self.clip, 1 + self.clip) * adv[mb]).mean()
                vloss, ent = (self.v - returns[mb]).pow(2).mean(), self._entropy(self.logits)
                self.opt.zero_grad()
                (pg + self.critic_coef * vloss - self.entropy_coef * ent).backward()
                self.opt.step()
        self.window += 1
        return self._log(pg, vloss, ent)

    def pretrain_update(self, supplied_presence, returns):
        """BC logits->supplied bodies + value MSE + entropy (no return-driven PG)."""
        sup, returns = supplied_presence.to(self.device).float(), returns.to(self.device).float()
        for _ in range(self.epochs):
            for mb in self._mb(sup.shape[0]):
                bc = F.binary_cross_entropy_with_logits(self.logits.expand_as(sup[mb]), sup[mb])
                vloss, ent = (self.v - returns[mb]).pow(2).mean(), self._entropy(self.logits)
                self.opt.zero_grad()
                (bc + self.critic_coef * vloss - self.entropy_coef * ent).backward()
                self.opt.step()
        self.window += 1
        return self._log(bc, vloss, ent)

    def in_pretrain(self):  return self.window < self.n_pretrain

    def gen_fraction(self):                                      # ramp 0->~1 over pretrain, then 1.0
        return 1.0 if self.window >= self.n_pretrain else self.window / max(1, self.n_pretrain)

    def _log(self, l, vloss, ent):
        return {"loss_a": l.item(), "vloss": vloss.item(), "ent": ent.item(),
                "v": self.v.item(), "p": torch.sigmoid(self.logits).detach().cpu()}
