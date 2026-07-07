"""CodesignAgent: single-network codesign. Control (ContAct + ContCrit/V0.98) and the morphology
generator (GenAct + GenCrit/V1.0) share ONE LimbTransformer trunk (codesign_tokens=True) under one
optimizer. See temp/codesign_single_network_plan.md and CONTEXT.md "Codesign heads/tokens".

Two training regimes, both on self.optimizer:
- per step (window, body fixed): plain combined PPO on control (ContAct + V0.98) -- the stock
  LoggingA2CAgent path, value_size==1, trunk free. The generator heads get no gradient here.
- at each resample (window boundary): _resample_update -- snapshot control, then jointly
    * fit GenCrit/V1.0 on rollout states -> R AND on designed prefixes -> R,
    * GenAct PPO-clip (marginal-Shapley advantage) [or BC toward the built body in pretrain],
    * CLONE control so the shared-trunk update doesn't drift it:
      beta * KL[ContAct_old, ContAct] + lam * MSE(ContCrit, ContCrit_old).
  Then sample the next body, ramp, full gym rebuild.

R_i = the body's true mean completed-episode return over the window (gamma=1), accumulated in
env_step. Requires codesign_tokens (the gen/critic heads). Supersedes the SeqGenerator path."""
import os

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from .architectures import _GEN_ON, _GEN_STOP
from .logging_agent import LoggingA2CAgent, _LIMB_CODE

_N_LIMBS = 8                                 # obs mask offset / n_dof derive from the net's tdims


def _gauss_kl(mu_old, ls_old, mu_new, ls_new):
    """KL[old || new] for a diagonal Gaussian policy, summed over action dims, mean over batch.
    Inactive dims have ls==0 (sigma=1) for both -> they contribute exactly 0 (no masking needed)."""
    var_new = (2.0 * ls_new).exp()
    kl = (ls_new - ls_old) + ((2.0 * ls_old).exp() + (mu_old - mu_new) ** 2) / (2.0 * var_new) - 0.5
    return kl.sum(-1).mean()


class CodesignAgent(LoggingA2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        net = self.model.a2c_network.net
        assert getattr(net, 'codesign_tokens', False), \
            "single-network codesign requires transformer.codesign_tokens=true"
        cd = self.config.get('generator', {})
        dev = self.ppo_device
        N = self.num_actors * self.num_agents

        # obs layout + action dim from the net's tdims (single source of truth). Phase-1: 32 DOF,
        # mask at obs[187:219]; a limb is a chain of up to _max_len modules.
        self._n_dof = net.tdims['n_dof']
        self._mask_off = net.tdims['obs_base'] + net.tdims['len_dim']
        self._max_len = net.max_limb_length

        env = self._env()                                  # base_legs from the ENV -> window-0 match
        self._base_legs = tuple(sorted(getattr(env, '_base_legs', cd.get('base_legs', (1, 4, 6)))))
        self._flip = float(cd.get('desirable_flip_prob', 0.10))    # per-token continue/stop flip prob
        bset = set(self._base_legs)
        # base morph = base_legs @ count-2 (1 swing + 1 knee). Per-token flip noise around this target
        # length gives P(limb differs by +-1 token)=flip, +-2=flip^2, ... (presence + length unified).
        self._base_target = torch.tensor(
            [2 if (i + 1) in bset else 0 for i in range(_N_LIMBS)], dtype=torch.long, device=dev)
        self._base_counts = self._base_target.clone()      # window-0 realized body == base target

        # generator hyperparameters (shared optimizer; the heads live on self.model)
        self._n_pretrain = cd.get('n_pretrain', 8)
        self._gen_epochs = cd.get('epochs', 4)
        self._gen_minibatches = cd.get('minibatches', 4)
        # gen_replay encodes M = mb_size*(L+1) designed prefixes in ONE grad forward; L grew from 8
        # (presence) to n_dof=32 (variable length), so a fixed N/minibatches fraction OOMs. Cap the
        # minibatch by a prefix budget so that forward stays near the old ~9k-prefix footprint.
        self._gen_max_prefixes = cd.get('max_prefixes', 6000)
        self._gen_clip = cd.get('clip', 0.2)
        self._gen_ent = cd.get('entropy_coef', 0.01)
        self._gencrit_coef = cd.get('gencrit_coef', 0.5)   # weight on the V1.0 fit (prefixes+rollout)
        self._beta = cd.get('beta', 1.0)                   # control-actor KL clone
        self._lam = cd.get('lam', 1.0)                     # control-critic MSE clone
        # GenCrit/V1.0 regresses to R; scale R by the reward shaper (same O(1) scale the control
        # critic fits) so the value fit + marginal advantage aren't swamped by raw-return magnitude.
        _shaper = self.config.get('reward_shaper', None)    # rl_games swaps the dict for the shaper obj
        self._r_scale = float(_shaper['scale_value'] if isinstance(_shaper, dict)
                              else getattr(_shaper, 'scale_value', 1.0))

        # window state: window 0 is the env's base build, so _cur_counts = base everywhere.
        self._cur_counts = self._base_counts.view(1, _N_LIMBS).expand(N, _N_LIMBS).clone()
        self._cur_trace = None                             # last sample() trace (RL update input)
        self._base_draw = None                             # last base+-flip ramp draws (counts, pretrain)
        self._gen_window = 0
        self._gen_log = None
        self._last_R = None

        # JEPA aux (Phase 2): same-step masked-latent prediction on the shared control trunk.
        # Disabled -> zero extra forwards, behaviour == phase-1 baseline (A/B control).
        jc = self.config.get('jepa', {})
        self._jepa_enabled = bool(jc.get('enabled', False))
        self._jepa_coef = float(jc.get('coef', 1.0))
        self._jepa_mask_prob = float(jc.get('mask_prob', 0.25))
        self._jepa_chunk = int(jc.get('chunk_states', 8192))   # 0 -> whole minibatch (memory cap)
        self._jepa_losses = []

        # R_i accumulator: true completed-episode return per env over the window (gamma=1).
        self._ep_ret = torch.zeros(N, device=dev)
        self._win_ret_sum = torch.zeros(N, device=dev)
        self._win_ret_cnt = torch.zeros(N, device=dev)

    def _env(self):
        return getattr(getattr(self.vec_env, 'envs', None), 'env', None)

    def _net(self):
        return self.model.a2c_network.net

    def calc_gradients(self, input_dict):
        # PPO control step (+ saturation logging) first, untouched -> control stays clean.
        super().calc_gradients(input_dict)
        if not self._jepa_enabled:
            return
        # Separate JEPA forward+backward+step on the SAME rollout obs (own step; PPO already stepped).
        # Plain optimizer path (no scaler), matching the _resample_update aux convention.
        # The masked forward+backward over the full minibatch OOMs a 15GB card on top of PPO's
        # resident memory, so grad-accumulate over chunks (peak = one chunk; ALL states used).
        net = self._net()
        obs = self.model.norm_obs(input_dict['obs'])
        B = obs.shape[0]
        chunk = self._jepa_chunk if self._jepa_chunk > 0 else B
        self.optimizer.zero_grad()
        acc = 0.0
        for s in range(0, B, chunk):
            ob = obs[s:s + chunk]
            jl = net.jepa_loss(ob, self._jepa_mask_prob)
            (self._jepa_coef * jl * (ob.shape[0] / B)).backward()   # batch-mean over chunks
            acc += jl.detach() * ob.shape[0]
        if self.truncate_grads:
            clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.optimizer.step()
        self._jepa_losses.append(acc / B)

    def _log_std(self, obs):
        mask = (obs[..., self._mask_off:self._mask_off + self._n_dof] > 0).float()
        return mask * self.model.a2c_network.log_std_param

    # ---- scripted frontier rollout: target-length teacher, optional per-token flip noise ---------
    @torch.no_grad()
    def _frontier_rollout(self, target, flip):
        """Walk the SAME frontier MDP as net.sample but with a scripted policy: at a limb of current
        length c, canonical action = continue if c < target[limb] else stop, flipped with prob `flip`.
        Random growable tip each step, >=1-limb guard, cap _max_len. Returns
        (slots, actions, active_step, counts) over L=n*max_len steps. Used to (a) draw base+-flip
        bodies (flip=p, target=base) and (b) reconstruct a valid token sequence for a target body
        (flip=0, target=counts) for BC."""
        dev = target.device
        N, n = target.shape
        max_len = self._max_len
        L = n * max_len
        count = torch.zeros(N, n, dtype=torch.long, device=dev)
        stopped = torch.zeros(N, n, dtype=torch.bool, device=dev)
        slots = torch.zeros(N, L, dtype=torch.long, device=dev)
        actions = torch.zeros(N, L, dtype=torch.long, device=dev)
        active_step = torch.zeros(N, L, dtype=torch.bool, device=dev)
        arange = torch.arange(N, device=dev)
        for t in range(L):
            growable = ~stopped
            active = growable.any(1)
            r = torch.rand(N, n, device=dev)
            slot = torch.where(growable, r, r.new_full((), -1.0)).argmax(1)
            canon_cont = count[arange, slot] < target[arange, slot]
            if flip > 0:
                canon_cont = canon_cont ^ (torch.rand(N, device=dev) < flip)
            force = active & (count.sum(1) == 0) & (growable.sum(1) == 1)   # >=1-limb guard
            do_cont = canon_cont | force
            cont = do_cont & active
            count[arange, slot] += cont.long()
            reached = count[arange, slot] >= max_len
            stopped[arange, slot] |= ((~do_cont) & active) | (cont & reached)
            slots[:, t] = slot
            actions[:, t] = torch.where(do_cont, do_cont.new_full((), _GEN_ON),
                                        do_cont.new_full((), _GEN_STOP)).long()
            active_step[:, t] = active
        return slots, actions, active_step, count

    @torch.no_grad()
    def _draw_base_counts(self, N):
        """Base +- per-token flip noise -> body counts (N, n)."""
        return self._frontier_rollout(self._base_target.view(1, _N_LIMBS).expand(N, _N_LIMBS),
                                      self._flip)[3]

    # ---- accumulate true episode returns (R_i, gamma=1) over the window ----------------
    def env_step(self, actions):
        obs, rewards, dones, infos = super().env_step(actions)
        r = rewards if rewards.dim() == 1 else rewards[:, 0]   # raw per-env reward (value_size==1)
        self._ep_ret += r
        # flush completed episodes fully vectorized -- NO `if d.any()` host sync per rollout step.
        df = dones.float()
        self._win_ret_sum += self._ep_ret * df                # add ep return for done envs
        self._win_ret_cnt += df
        self._ep_ret = self._ep_ret * (1.0 - df)              # reset done envs
        return obs, rewards, dones, infos

    @torch.no_grad()
    def _window_Ri(self):
        return self._win_ret_sum / self._win_ret_cnt.clamp(min=1.0)   # (N,) true mean return

    def _in_pretrain(self):
        return self._gen_window < self._n_pretrain

    def _gen_fraction(self):
        return 1.0 if self._gen_window >= self._n_pretrain else self._gen_window / max(1, self._n_pretrain)

    # ---- ramp: replace (1 - gen_fraction) of envs with base+-flip bodies -----------
    @torch.no_grad()
    def _apply_ramp(self, counts):
        """counts: (N, n) long generator-sampled module counts. Mix in base+-flip draws by fraction."""
        frac = self._gen_fraction()
        if frac >= 1.0:
            self._base_draw = None                         # RL phase: pure gen samples, no base draws
            return counts
        N = counts.shape[0]
        base = self._draw_base_counts(N)                   # (N,n) counts (>=1-module guaranteed)
        self._base_draw = base                             # the around-base samples (build/*_base)
        use_gen = (torch.rand(N, device=counts.device) < frac).unsqueeze(1)
        return torch.where(use_gen, counts, base)

    # ---- the resample-boundary joint update (one optimizer) -------------------------
    def _resample_update(self, R, obses):
        """R: (N,) body returns. obses: (H,N,obs) the window's last rollout (rollout-state sample).
        Jointly: GenCrit fit (prefixes->R + rollout states->R), GenAct PPO/BC, control clone."""
        net = self._net()
        dev = R.device
        N = obses.shape[1]
        obs_flat = obses.reshape(-1, obses.shape[-1])      # (H*N, obs)
        HN = obs_flat.shape[0]
        R_roll = R[torch.arange(HN, device=dev) % N]       # rollout-state target (env's R)

        # snapshot control_old on the rollout states (current net, no grad). Chunk the forward:
        # H*N can be ~65k states -> a single trunk pass OOMs at scale (storing the result is cheap).
        with torch.no_grad():
            ls_old = self._log_std(obs_flat)
            mu_old = obs_flat.new_empty(HN, self._n_dof)
            v098_old = obs_flat.new_empty(HN, 1)
            for s in range(0, HN, self.minibatch_size):
                m, v, _ = net.codesign_forward(self.model.norm_obs(obs_flat[s:s + self.minibatch_size]))
                mu_old[s:s + self.minibatch_size] = m
                v098_old[s:s + self.minibatch_size] = v

        pretrain = self._in_pretrain()
        if pretrain:                                       # BC toward the built body (frontier tokens)
            # reconstruct a valid token sequence that yields _cur_counts (random tip order, no flip)
            slots, actions, valid, _ = self._frontier_rollout(self._cur_counts, 0.0)
            old_logp, adv, raw_adv = None, None, None
        else:                                              # RL: PPO over the sampled trace
            tr = self._cur_trace
            slots, actions, old_logp = tr['slots'], tr['actions'], tr['old_logp']
            valid = tr['active_step']                                     # mask no-op frontier steps
            raw_adv = tr['v_states'][:, 1:] - tr['v_states'][:, :-1]      # telescoping (Shapley)
            sel = raw_adv[valid]
            adv = torch.zeros_like(raw_adv)
            adv[valid] = (sel - sel.mean()) / (sel.std() + 1e-8)

        L1 = self._n_dof + 1                               # (L+1) prefixes gen_replay stacks per env
        mb_size = max(1, min(N // self._gen_minibatches, self._gen_max_prefixes // L1))
        logs = {k: [] for k in ('gen_pg', 'ent', 'v_prefix', 'v_roll', 'kl', 'crit', 'gn')}
        for _ in range(self._gen_epochs):
            perm = torch.randperm(N, device=dev)
            for s in range(0, N, mb_size):
                mb = perm[s:s + mb_size]
                # --- generator: GenAct (PPO or BC) + GenCrit prefix fit ---
                # gen_replay re-derives the valid (non-no-op) step mask from (slots,actions); average
                # GenAct losses over valid steps only so no-op frontier steps don't dilute the signal.
                logits, v, vm = net.gen_replay(slots[mb], actions[mb])   # (m,L,2), (m,L+1), (m,L)
                vf = vm.float()
                nval = vf.sum().clamp(min=1.0)
                dist = torch.distributions.Categorical(logits=logits)
                if pretrain:
                    gen_pg = -(dist.log_prob(actions[mb]) * vf).sum() / nval
                else:
                    ratio = (dist.log_prob(actions[mb]) - old_logp[mb]).exp()
                    a = adv[mb]
                    per = torch.min(ratio * a, ratio.clamp(1 - self._gen_clip, 1 + self._gen_clip) * a)
                    gen_pg = -(per * vf).sum() / nval
                ent = (dist.entropy() * vf).sum() / nval
                v_prefix = (v - R[mb].unsqueeze(1)).pow(2).mean()        # all L+1 prefixes -> R

                # --- control clone + GenCrit rollout-state fit (sampled rollout minibatch) ---
                ridx = torch.randint(0, HN, (N,), device=dev)
                ob = obs_flat[ridx]
                mu_n, v098_n, v1_n = net.codesign_forward(self.model.norm_obs(ob))
                kl = _gauss_kl(mu_old[ridx], ls_old[ridx], mu_n, self._log_std(ob))
                crit = (v098_n - v098_old[ridx]).pow(2).mean()
                v_roll = (v1_n.squeeze(-1) - R_roll[ridx]).pow(2).mean()

                loss = (gen_pg - self._gen_ent * ent
                        + self._gencrit_coef * (v_prefix + v_roll)
                        + self._beta * kl + self._lam * crit)
                self.optimizer.zero_grad()
                loss.backward()
                logs['gn'].append(clip_grad_norm_(self.model.parameters(), self.grad_norm))
                self.optimizer.step()
                for k, val in (('gen_pg', gen_pg), ('ent', ent), ('v_prefix', v_prefix),
                               ('v_roll', v_roll), ('kl', kl), ('crit', crit)):
                    logs[k].append(val.detach())

        self._gen_log = {k: torch.stack(v).mean().item() for k, v in logs.items()}
        # body-quality outcome (the optimization target); R aligns with the current window's
        # realized bodies (_cur_counts), before _maybe_resample samples the next window.
        self._gen_log['R_var'] = R.var(unbiased=False).item()    # scale for the GenCrit vloss norm
        self._gen_log['R_mean'] = R.mean().item()
        self._gen_log['R_std'] = R.std().item()
        self._gen_log['by_limbcount'] = self._by_limbcount(R, self._cur_counts)
        if not pretrain:                                   # RL: built body == generated (ramp off)
            self._gen_log['marg'] = self._slot_marginal(raw_adv, slots, actions, valid)
            rank, ev, K = self._body_value_metrics(self._cur_trace['counts'],
                                                   self._cur_trace['v_states'][:, -1], R)
            self._gen_log['value_rank_corr'] = rank       # denoised Spearman (NaN if <5 bodies)
            self._gen_log['value_ev'] = ev                # denoised per-body explained variance
            self._gen_log['n_distinct_bodies'] = float(K)

    @torch.no_grad()
    def _slot_marginal(self, raw_adv, slots, actions, valid):
        """Per-limb marginal value of adding a module (continue token), over valid steps only."""
        on = (actions == _GEN_ON) & valid
        marg = torch.full((_N_LIMBS,), float('nan'), device=raw_adv.device)
        for k in range(_N_LIMBS):
            m = on & (slots == k)
            if m.any():
                marg[k] = raw_adv[m].mean()
        return marg

    @staticmethod
    @torch.no_grad()
    def _by_limbcount(R, counts):
        """Mean body-return R grouped by LIMB count (#limbs with >=1 module == (counts>0).sum(1)).
        Answers the codesign hypothesis: do more limbs earn more R?"""
        lc = (counts > 0).sum(1).long()
        return {int(k): R[lc == k].mean().item() for k in lc.unique()}

    @staticmethod
    @torch.no_grad()
    def _body_value_metrics(counts, v_full, R):
        """Denoised body-quality fit: group envs by distinct body (module-count vector), then compare
        the generator's v(full) to each body's MEAN R (removes reset noise). Returns (rank_corr, ev,
        n_bodies):
          rank_corr = Spearman over bodies (NaN if <5 -> unreliable); spread/scale-robust.
          ev        = 1 - Var(meanR - v)/Var(meanR) over bodies (NaN if <2).
        Valid only when built==generated (RL phase), where R matches the generated body."""
        dev = R.device
        base = int(counts.max().item()) + 1                       # radix over count values (>= max_len+1)
        body_id = (counts.long() * (base ** torch.arange(_N_LIMBS, device=dev))).sum(1)
        _, inv = body_id.unique(return_inverse=True)
        K = int(inv.max().item()) + 1
        cnt = torch.zeros(K, device=dev).index_add_(0, inv, torch.ones_like(R))
        meanR = torch.zeros(K, device=dev).index_add_(0, inv, R) / cnt
        meanv = torch.zeros(K, device=dev).index_add_(0, inv, v_full) / cnt
        ev = float('nan')
        if K >= 2 and meanR.var(unbiased=False) > 1e-8:
            ev = (1.0 - (meanR - meanv).var(unbiased=False) / meanR.var(unbiased=False)).item()
        # rank needs >=5 bodies AND real spread in both (a constant v -> meaningless tie-broken ranks)
        rank = (CodesignAgent._spearman(meanv, meanR)
                if K >= 5 and meanv.std() > 1e-8 and meanR.std() > 1e-8 else float('nan'))
        return rank, ev, K

    @staticmethod
    @torch.no_grad()
    def _spearman(x, y):
        rx = x.argsort().argsort().float(); ry = y.argsort().argsort().float()
        rx = rx - rx.mean(); ry = ry - ry.mean()
        denom = rx.norm() * ry.norm()
        return float('nan') if denom < 1e-8 else float((rx * ry).sum() / denom)

    # ---- window boundary: update generator, then sample + build next window ----------
    def _maybe_resample(self):
        interval = self.config.get('resample_interval', 0)
        if not interval:
            return
        env = self._env()
        if env is None or not getattr(env, '_sample_morphs', False):
            return
        self._steps_since_resample += self.horizon_length
        if self._steps_since_resample < interval * env.max_episode_length:
            return

        N = env.total_num_envs
        R = self._window_Ri() * self._r_scale               # true body return, scaled to control units
        obses = self.experience_buffer.tensor_dict['obses']  # (H,N,obs) rollout-state sample
        phase = 'pretrain' if self._in_pretrain() else 'rl'  # regime of the update just performed
        self._resample_update(R, obses)
        self._gen_window += 1
        self._last_R = R

        # final-scatter data: trace v(full) vs R (per env) for notebooks/ant_codesign.ipynb panel 4.
        if self._cur_trace is not None:
            np.savez(os.path.join(self.experiment_dir, 'gen_scatter.npz'),
                     v_full=self._cur_trace['v_states'][:, -1].cpu().numpy(),
                     R=R.cpu().numpy())

        trace = self._net().sample(N)
        counts = self._apply_ramp(trace['counts'].long())
        self._cur_trace, self._cur_counts = trace, counts
        env.set_next(counts)
        print(f"[resample #{self._gen_window} | {phase} | next_gen_frac={self._gen_fraction():.2f} | "
              f"epoch {self.epoch_num}] R_mean={R.mean().item():.3f} "
              f"limbcount={(counts > 0).sum(1).float().mean().item():.2f} "
              f"modules={counts.sum(1).float().mean().item():.2f}", flush=True)
        env.resample()
        self.obs = self.env_reset()
        self.current_rewards.zero_(); self.current_lengths.zero_()
        # reset R_i accumulators for the new window
        self._ep_ret.zero_(); self._win_ret_sum.zero_(); self._win_ret_cnt.zero_()
        self._morph_meta = None
        self._steps_since_resample = 0

    def _log_morph_stats(self, w, frame, epoch_num):
        return  # codesign reports body quality via quality/* (true R), not the base per-step morph_reward

    # ---- generator logging (sparse: only at window boundaries) ----------------------
    # Metrics are grouped by subsystem (see docs/codesign_metrics.md):
    #   build/    the body the generator produces      gen/     GenAct (generator actor) learning
    #   gencrit/  GenCrit/V1.0 value-head fit          quality/ body-quality outcome (the target)
    #   clone/    control preservation at resample
    # Note the half-step: build/* describe the NEXT window's bodies (just sampled), while the
    # learning + quality metrics describe the window that just ended (aligned with R).
    def write_stats(self, *args, **kwargs):
        super().write_stats(*args, **kwargs)
        w = self.writer
        if w is None:
            return
        frame = args[11] if len(args) > 11 else kwargs.get('frame')
        # JEPA loss logs every epoch (independent of the resample-boundary _gen_log).
        if self._jepa_losses:
            w.add_scalar('losses/jepa', torch.stack(self._jepa_losses).mean().item(), frame)
            self._jepa_losses = []
        if self._gen_log is None:
            return
        g = self._gen_log

        # --- build/: the body the generator produces (realized = built body, counts) ---
        rate = (self._cur_counts > 0).float().mean(0)      # per-limb realized presence rate
        for i in range(_N_LIMBS):
            w.add_scalar(f'build/p/{_LIMB_CODE[i + 1]}', rate[i].item(), frame)
        w.add_scalar('build/limbcount_realized', (self._cur_counts > 0).sum(1).float().mean().item(), frame)
        w.add_scalar('build/modulecount_realized', self._cur_counts.sum(1).float().mean().item(), frame)
        if self._base_draw is not None:                    # pretrain only: the around-base draws
            w.add_scalar('build/limbcount_base', (self._base_draw > 0).sum(1).float().mean().item(), frame)
            w.add_scalar('build/modulecount_base', self._base_draw.sum(1).float().mean().item(), frame)
        if self._cur_trace is not None:
            limbc = self._cur_trace['presence'].sum(1)      # per-env generated limb count (intent)
            modc = self._cur_trace['counts'].sum(1)         # per-env generated module count (intent)
            w.add_scalar('build/limbcount', limbc.mean().item(), frame)
            w.add_scalar('build/limbcount_var', limbc.var().item(), frame)  # diversity / collapse canary
            w.add_scalar('build/modulecount', modc.mean().item(), frame)
            w.add_scalar('build/modulecount_var', modc.var().item(), frame)
        if 'n_distinct_bodies' in g:
            w.add_scalar('build/n_distinct', g['n_distinct_bodies'], frame)

        # --- gen/: GenAct (generator actor) learning ---
        w.add_scalar('gen/actor_loss', g['gen_pg'], frame)
        w.add_scalar('gen/entropy', g['ent'], frame)
        w.add_scalar('gen/grad_norm', g['gn'], frame)
        w.add_scalar('gen/fraction', self._gen_fraction(), frame)
        if 'marg' in g:                                    # per-limb marginal value (RL phase only)
            for i in range(_N_LIMBS):
                m = g['marg'][i].item()
                if m == m:                                 # skip NaN slots (no `on` this window)
                    w.add_scalar(f'gen/marg/{_LIMB_CODE[i + 1]}', m, frame)

        # --- gencrit/: GenCrit/V1.0 value-head fit (scale-free: MSE / Var(R)) ---
        rvar = max(g['R_var'], 1e-8)
        w.add_scalar('gencrit/loss_prefix', g['v_prefix'] / rvar, frame)
        w.add_scalar('gencrit/loss_rollout', g['v_roll'] / rvar, frame)
        for key, tag in (('value_rank_corr', 'gencrit/value_rank_corr'),
                         ('value_ev', 'gencrit/value_ev')):
            if key in g and g[key] == g[key]:              # skip NaN (rank needs >=5 bodies)
                w.add_scalar(tag, g[key], frame)

        # --- quality/: body-quality outcome (the optimization target) ---
        w.add_scalar('quality/R_mean', g['R_mean'], frame)
        w.add_scalar('quality/R_std', g['R_std'], frame)
        for k, v in g['by_limbcount'].items():             # does more limbs earn more R?
            w.add_scalar(f'quality/by_limbcount/{k}', v, frame)

        # --- clone/: control preservation at resample ---
        w.add_scalar('clone/actor_kl', g['kl'], frame)
        w.add_scalar('clone/critic_mse', g['crit'], frame)

        self._gen_log = None

    # ---- checkpointing (gen heads live on self.model -> saved by base) --------------
    def get_full_state_weights(self):
        s = super().get_full_state_weights()
        s.update(gen_window=self._gen_window, cur_counts=self._cur_counts,
                 cur_trace=self._cur_trace, steps_since_resample=self._steps_since_resample)
        return s

    def set_full_state_weights(self, weights, set_epoch=True):
        super().set_full_state_weights(weights, set_epoch=set_epoch)
        if 'gen_window' not in weights:
            return
        self._gen_window = int(weights['gen_window'])
        self._cur_counts = weights['cur_counts'].to(self.ppo_device)
        tr = weights.get('cur_trace')
        self._cur_trace = {k: v.to(self.ppo_device) for k, v in tr.items()} if tr else None
        self._steps_since_resample = int(weights.get('steps_since_resample', 0))
