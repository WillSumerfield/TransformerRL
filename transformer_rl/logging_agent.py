"""A2CAgent subclass that logs extra PPO-health metrics.

Adds, per epoch, on top of rl_games' built-ins (enable via use_diagnostics=True,
which gives diagnostics/exp_var, diagnostics/clip_frac/*, diagnostics/rms_value/*):

    policy/sigma_{mean,min,max}  exploration std = exp(log_std); min is the log_std-collapse canary
    policy/action_sat            frac of *active* mean-action dims pinned at the tanh rail (|mu|>0.99)
    health/grad_norm             total grad norm BEFORE clipping (clip_grad_norm_ return)
    health/adv_{mean,std}        raw advantage (returns-values) BEFORE normalization -> true scale

Registered globally over 'a2c_continuous' in train_utils, so every continuous PPO
run (transformer or MLP) gets these. Metrics are generic; nothing transformer-specific.
"""
import time

import torch
from torch.nn.utils import clip_grad_norm_

from rl_games.common import a2c_common
from rl_games.common.common_losses import default_critic_loss
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent


def _explained_var(values, returns):
    """1 - Var(returns - values)/Var(returns); ->1 = critic explains the return variance."""
    return float(1.0 - (returns - values).var() / (returns.var() + 1e-8))

# Leg slot (1-8 at (n-1)*45 deg) -> compass code, relative to forward = +X (the reward axis),
# right = -Z, slot number increasing clockwise toward the right. See architectures/build_vsim.
_LEG_CODE = {1: "F", 2: "FR", 3: "R", 4: "BR", 5: "B", 6: "BL", 7: "L", 8: "FL"}
_LEADERBOARD_EVERY = 50  # epochs
_LEADERBOARD_K = 5       # top-k and bottom-k


def _morph_label(legs) -> str:
    """frozenset/list of leg slots -> compass-coded label, e.g. {2,4,6,8} -> 'FR.BR.BL.FL'."""
    return "·".join(_LEG_CODE[n] for n in sorted(legs))


class LoggingA2CAgent(A2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self._grad_norms: list[float] = []   # per-minibatch, flushed per epoch
        self._action_sats: list[float] = []  # per-minibatch, flushed per epoch
        self._adv_mean: float | None = None  # per-epoch (set in prepare_dataset)
        self._adv_std: float | None = None
        self._morph_meta = None  # None=undetected, False=single-morph, dict=multi-morph metadata
        self._steps_since_resample = 0  # env-steps since last morphology resample (full ant only)
        # --- codesign control V1.0 body-quality head (value_size==2, ADR-0012) ---------
        # value_size==2 => col0 = V0.98 (actor's critic, today's path), col1 = V1.0 (gamma=1,
        # trunc->0, time-aware). Per-dim gamma, dim-0-only actor advantage, split critic loss.
        self._v1 = self.value_size == 2
        if self._v1:
            self.gamma_v1 = self.config.get('gamma_v1', 1.0)
            self.v1_coef = self.config.get('v1_coef', 0.5 * self.critic_coef)
            self._gamma_vec = torch.tensor([self.gamma, self.gamma_v1], device=self.ppo_device)
        self._c_loss_v098: list = []  # per-minibatch, flushed per epoch
        self._c_loss_v1: list = []
        self._ev_v098 = None          # per-epoch (set in prepare_dataset)
        self._ev_v1 = None
        # opt-in synced phase timing (config.timing); shared with PPGAgent. Off -> stock path.
        self._timing = self.config.get('timing', False)
        self._timings = {}
        self._tics = {}

    # ---- opt-in phase timing (cuda.synchronize + perf_counter); shared w/ PPGAgent ----

    def _tic(self, key):
        if self._timing:
            torch.cuda.synchronize()
            self._tics[key] = time.perf_counter()

    def _toc(self, key):
        if self._timing:
            torch.cuda.synchronize()
            self._timings[key] = self._timings.get(key, 0.0) + (time.perf_counter() - self._tics[key])

    def _to_dev(self, t):
        """to(device) with optional transfer timing (perf/t_aux_transfer)."""
        if not self._timing:
            return t.to(self.ppo_device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = t.to(self.ppo_device)
        torch.cuda.synchronize()
        self._timings['perf/t_aux_transfer'] = \
            self._timings.get('perf/t_aux_transfer', 0.0) + (time.perf_counter() - t0)
        return r

    def train_epoch(self):
        if not self._timing:
            out = super().train_epoch()       # stock PPO path, untouched (baseline fidelity)
            self._maybe_resample()
            return out
        return self._train_epoch_timed()

    def _train_epoch_timed(self):
        # faithful copy of ContinuousA2CBase.train_epoch + synced t_rollout/t_update timers.
        a2c_common.A2CBase.train_epoch(self)
        self.set_eval()
        play_time_start = time.perf_counter()
        self._tic('perf/t_rollout')
        with torch.no_grad():
            batch_dict = self.play_steps_rnn() if self.is_rnn else self.play_steps()
        self._toc('perf/t_rollout')
        play_time_end = time.perf_counter()
        update_time_start = time.perf_counter()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        a_losses, c_losses, b_losses, entropies, kls = [], [], [], [], []
        self._tic('perf/t_update')
        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = \
                    self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)
                self.dataset.update_mu_sigma(cmu, csigma)
                if self.schedule_type == 'legacy':
                    self.last_lr, self.entropy_coef = self.scheduler.update(
                        self.last_lr, self.entropy_coef, self.epoch_num, 0, kl.item())
                    self.update_lr(self.last_lr)
            av_kls = torch_ext.mean_list(ep_kls)
            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = self.scheduler.update(
                    self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)
            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval()
        self._toc('perf/t_update')

        self._maybe_resample()

        update_time_end = time.perf_counter()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start
        return (batch_dict['step_time'], play_time, update_time, total_time,
                a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul)

    def _maybe_resample(self):
        """Every resample_interval episodes, draw a fresh morphology set (full gym rebuild) and
        refresh the agent's cached obs. No-op unless the env samples morphologies and the knob is set.
        See docs/morphology_resampling_cost.md."""
        interval = self.config.get('resample_interval', 0)  # episodes between resamples; 0 = off
        if not interval:
            return
        env = getattr(getattr(self.vec_env, 'envs', None), 'env', None)
        if env is None or not getattr(env, '_sample_morphs', False):
            return
        self._steps_since_resample += self.horizon_length
        if self._steps_since_resample < interval * env.max_episode_length:
            return
        print(f"[resample] new morphology set (every {interval} episodes)", flush=True)
        env.resample()
        self.obs = self.env_reset()             # rebuilt env -> refresh stale rollout-start obs
        self.current_rewards.zero_()            # the hard reset ends all episodes; drop partials
        self.current_lengths.zero_()
        self._morph_meta = None                 # morphs changed -> re-detect per-morph logging labels
        self._steps_since_resample = 0

    # ---- V1.0 head: per-dim gamma returns, reward duplication, split critic loss ------

    def env_step(self, actions):
        obs, rewards, dones, infos = super().env_step(actions)
        if self._v1:
            # rl_games unsqueezes scalar rewards only for value_size==1; for value_size==2 we
            # duplicate the single env reward into both value columns (both heads model the same R).
            if rewards.dim() == 1:
                rewards = rewards.unsqueeze(1)
            if rewards.shape[1] == 1:
                rewards = rewards.expand(-1, self.value_size)
        return obs, rewards, dones, infos

    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        if not self._v1:
            return super().discount_values(fdones, last_extrinsic_values, mb_fdones,
                                           mb_extrinsic_values, mb_rewards)
        # Per-dim gamma: col0=V0.98 (self.gamma), col1=V1.0 (gamma_v1=1). tau shared; trunc->0 via
        # the done mask (time_outs not exposed). gamma broadcasts over the value dim.
        gamma = self._gamma_vec
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t + 1]
                nextvalues = mb_extrinsic_values[t + 1]
            nextnonterminal = nextnonterminal.unsqueeze(1)
            delta = mb_rewards[t] + gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + gamma * self.tau * nextnonterminal * lastgaelam
        return mb_advs

    def prepare_dataset(self, batch_dict):
        if not self._v1:
            # Raw advantage scale, before rl_games normalizes it (mirrors a2c_common:1030,1038).
            with torch.no_grad():
                adv = (batch_dict['returns'] - batch_dict['values']).sum(dim=1)
                self._adv_mean = adv.mean().item()
                self._adv_std = adv.std().item()
            return super().prepare_dataset(batch_dict)

        # value_size==2: actor advantage from dim 0 (V0.98) ONLY; both return cols kept for the
        # value loss (V1.0 isolated from the actor's policy gradient). Mirrors a2c_common.prepare_dataset
        # (no rnn / central-value branches here) but slices the advantage to dim 0.
        returns, values = batch_dict['returns'], batch_dict['values']
        rnn_masks = batch_dict.get('rnn_masks', None)
        raw_adv = returns - values                                   # (B, 2), raw-return units
        with torch.no_grad():
            a0 = raw_adv[:, 0]
            self._adv_mean, self._adv_std = a0.mean().item(), a0.std().item()
            self._ev_v098 = _explained_var(values[:, 0], returns[:, 0])
            self._ev_v1 = _explained_var(values[:, 1], returns[:, 1])
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        advantages = raw_adv[:, 0]                                   # dim-0 only
        if self.normalize_advantage:
            if self.normalize_rms_advantage:
                advantages = self.advantage_mean_std(advantages)
            else:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        dataset_dict = {
            'old_values': values, 'old_logp_actions': batch_dict['neglogpacs'],
            'advantages': advantages, 'returns': returns, 'actions': batch_dict['actions'],
            'obs': batch_dict['obses'], 'dones': batch_dict['dones'],
            'rnn_states': batch_dict.get('rnn_states', None), 'rnn_masks': rnn_masks,
            'mu': batch_dict['mus'], 'sigma': batch_dict['sigmas'],
        }
        if self.use_action_masks:
            dataset_dict['action_masks'] = batch_dict['action_masks']
        self.dataset.update_values_dict(dataset_dict)

    def calc_losses(self, actor_loss_func, old_action_log_probs_batch, action_log_probs, advantage,
                    curr_e_clip, value_preds_batch, values, return_batch, mu, entropy, rnn_masks):
        if not self._v1:
            return super().calc_losses(actor_loss_func, old_action_log_probs_batch, action_log_probs,
                                       advantage, curr_e_clip, value_preds_batch, values, return_batch,
                                       mu, entropy, rnn_masks)
        # Split the critic loss per value dim so V1.0 gets its own (smaller) coef while shaping the
        # shared trunk. Mirrors a2c_continuous.calc_losses but with two weighted value terms.
        a_loss = actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage,
                                 self.ppo, curr_e_clip)
        cl = default_critic_loss(value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
        cl0, cl1 = cl[:, 0:1], cl[:, 1:2]                           # (B,1) each
        if self.bound_loss_type == 'regularisation':
            b_loss = self.reg_loss(mu)
        elif self.bound_loss_type == 'bound':
            b_loss = self.bound_loss(mu)
        else:
            b_loss = torch.zeros(1, device=self.ppo_device)
        losses, sum_mask = torch_ext.apply_masks(
            [a_loss.unsqueeze(1), cl0, cl1, entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
        a_loss, cl0, cl1, entropy, b_loss = losses[0], losses[1], losses[2], losses[3], losses[4]
        loss = (a_loss
                + 0.5 * self.critic_coef * cl0
                + 0.5 * self.v1_coef * cl1
                - entropy * self.entropy_coef
                + b_loss * self.bounds_loss_coef)
        self._c_loss_v098.append(cl0.detach())
        self._c_loss_v1.append(cl1.detach())
        c_loss = cl0 + cl1                                          # combined, for stock c_loss logging
        return loss, a_loss, c_loss, entropy, b_loss, sum_mask

    def calc_gradients(self, input_dict):
        super().calc_gradients(input_dict)
        # train_result = (a_loss, c_loss, entropy, kl, lr, lr_mul, mu, sigma, b_loss)
        with torch.no_grad():
            mu = self.train_result[6]
            saturated = (mu.abs() > 0.99).float().sum()
            # Inactive dims are masked to exactly 0 in our nets, so |mu|>eps selects active
            # dims (and is all dims for an MLP) -> saturation measured over active dims only.
            active = (mu.abs() > 1e-6).float().sum().clamp(min=1.0)
            self._action_sats.append((saturated / active).item())

    def trancate_gradients_and_step(self):
        if self.truncate_grads and not self.multi_gpu:
            self.scaler.unscale_(self.optimizer)
            grad_norm = clip_grad_norm_(self.model.parameters(), self.grad_norm)  # returns pre-clip norm
            self._grad_norms.append(float(grad_norm))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            super().trancate_gradients_and_step()

    def write_stats(self, *args, **kwargs):
        super().write_stats(*args, **kwargs)
        w = self.writer
        if w is None:
            return
        # frame is the global-step x-axis (12th positional, after total_time..lr_mul)
        frame = args[11] if len(args) > 11 else kwargs.get('frame')

        net = self.model.a2c_network
        logstd = getattr(net, 'log_std_param', None)
        if not isinstance(logstd, torch.nn.Parameter):
            logstd = getattr(net, 'sigma', None)  # rl_games MLP fixed_sigma stores log-std here
        if isinstance(logstd, torch.nn.Parameter):
            with torch.no_grad():
                sigma = torch.exp(logstd.detach())
            w.add_scalar('policy/sigma_mean', sigma.mean().item(), frame)
            w.add_scalar('policy/sigma_min', sigma.min().item(), frame)
            w.add_scalar('policy/sigma_max', sigma.max().item(), frame)

        if self._grad_norms:
            w.add_scalar('health/grad_norm', sum(self._grad_norms) / len(self._grad_norms), frame)
            self._grad_norms = []
        if self._action_sats:
            w.add_scalar('policy/action_sat', sum(self._action_sats) / len(self._action_sats), frame)
            self._action_sats = []
        if self._adv_mean is not None:
            w.add_scalar('health/adv_mean', self._adv_mean, frame)
            w.add_scalar('health/adv_std', self._adv_std, frame)
            self._adv_mean = self._adv_std = None

        if self._timings:
            for k, v in self._timings.items():
                w.add_scalar(k, v, frame)
            self._timings = {}

        if self._v1:
            self._log_v1_stats(w, frame)

        epoch_num = int(args[1]) if len(args) > 1 else kwargs.get('epoch_num', 0)
        self._log_morph_stats(w, frame, epoch_num)

    # ---- V1.0 body-quality head diagnostics (value_size==2) -------------------------

    @torch.no_grad()
    def _log_v1_stats(self, w, frame):
        """Stage-1 metrics 1-4: R trustworthiness, per-dim critic fit, gamma sanity, time-awareness.
        Values are buffered denorm (shaped-reward units); compared to shaped episode return."""
        eb = self.experience_buffer.tensor_dict
        obses, vals = eb['obses'], eb['values']        # (H,B,obs), (H,B,2)
        prog = obses[..., -1]                            # raw progress in [0,1)
        v098, v1 = vals[..., 0], vals[..., 1]

        # (3) gamma sanity: V1.0 (gamma=1) should sit above V0.98 (gamma=0.98) in magnitude.
        w.add_scalar('v1/v098_mean', v098.mean().item(), frame)
        w.add_scalar('v1/v1_mean', v1.mean().item(), frame)

        # (1) R trustworthiness: V1.0(s0) vs mean (shaped) episode return + calibration gap.
        s0 = prog == 0
        if s0.any():
            v1s0 = v1[s0].mean().item()
            ret = self.game_shaped_rewards.get_mean()
            ret = float(ret.mean()) if hasattr(ret, 'mean') else float(ret)
            w.add_scalar('v1/v1_s0', v1s0, frame)
            w.add_scalar('v1/episode_return', ret, frame)
            w.add_scalar('v1/calibration_gap', v1s0 - ret, frame)  # persistent +gap = trunc-bootstrap leak

        # (4) time-awareness: V1.0 should fall from early to late in the episode.
        first, last = v1[prog < 1.0 / 3], v1[prog > 2.0 / 3]
        if first.numel() and last.numel():
            w.add_scalar('v1/by_progress_first', first.mean().item(), frame)
            w.add_scalar('v1/by_progress_last', last.mean().item(), frame)

        # (2) per-dim critic fit: loss + explained variance.
        if self._c_loss_v098:
            w.add_scalar('v1/c_loss_v098', torch.stack(self._c_loss_v098).mean().item(), frame)
            w.add_scalar('v1/c_loss_v1', torch.stack(self._c_loss_v1).mean().item(), frame)
            self._c_loss_v098, self._c_loss_v1 = [], []
        if self._ev_v098 is not None:
            w.add_scalar('v1/explained_var_v098', self._ev_v098, frame)
            w.add_scalar('v1/explained_var_v1', self._ev_v1, frame)
            self._ev_v098 = self._ev_v1 = None

    # ---- per-morphology performance (multi-morph AntMultiMorphEnv only) -------------

    def _morph_metadata(self):
        """Detect a multi-morph env and cache per-morph layout/labels. Returns dict or False."""
        if self._morph_meta is not None:
            return self._morph_meta
        env = getattr(getattr(self.vec_env, 'envs', None), 'env', None)
        if env is None or not hasattr(env, 'envs_per_morph') or not hasattr(env, 'groups'):
            self._morph_meta = False
            return False
        morphs = [sorted(g['morph'].legs) for g in env.groups]
        by_legs: dict[int, list[int]] = {}
        for i, m in enumerate(morphs):
            by_legs.setdefault(len(m), []).append(i)
        self._morph_meta = {
            'epm': env.envs_per_morph,
            'n': len(morphs),
            'labels': [_morph_label(m) for m in morphs],
            'leg_counts': [len(m) for m in morphs],
            'by_legs': by_legs,
        }
        return self._morph_meta

    def _log_morph_stats(self, w, frame, epoch_num):
        meta = self._morph_metadata()
        if not meta:
            return
        n, epm = meta['n'], meta['epm']
        rewards = self.experience_buffer.tensor_dict['rewards']  # (horizon, N, 1)
        per_env = rewards.mean(dim=0).reshape(-1)                # (N,) mean shaped reward/step
        if per_env.numel() != n * epm:                          # layout guard
            return
        per_morph = per_env.view(n, epm).mean(dim=1)            # (n_morphs,)

        w.add_scalar('morph_reward/mean', per_morph.mean().item(), frame)
        w.add_scalar('morph_reward/median', per_morph.median().item(), frame)
        w.add_scalar('morph_reward/min', per_morph.min().item(), frame)
        w.add_scalar('morph_reward/max', per_morph.max().item(), frame)
        w.add_scalar('morph_reward/std', per_morph.std().item(), frame)

        pm = per_morph.detach().cpu()
        for k, idxs in meta['by_legs'].items():
            w.add_scalar(f'morph_reward_legs/{k}', pm[torch.tensor(idxs)].mean().item(), frame)

        if epoch_num and epoch_num % _LEADERBOARD_EVERY == 0:
            order = torch.argsort(pm, descending=True).tolist()
            k = min(_LEADERBOARD_K, n)
            rows = ['| rank | morph | legs | reward |', '|---:|:---|---:|---:|']
            for r in range(k):  # top
                i = order[r]
                rows.append(f'| {r + 1} | {meta["labels"][i]} | {meta["leg_counts"][i]} | {pm[i]:.3f} |')
            if n > 2 * k:
                rows.append('| … | … | … | … |')
            for r in range(max(k, n - k), n):  # bottom
                i = order[r]
                rows.append(f'| {r + 1} | {meta["labels"][i]} | {meta["leg_counts"][i]} | {pm[i]:.3f} |')
            w.add_text('morph_leaderboard', '\n'.join(rows), frame)
