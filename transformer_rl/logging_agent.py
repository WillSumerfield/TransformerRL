"""A2CAgent subclass that logs extra PPO-health metrics.

Adds, per epoch, on top of rl_games' built-ins (enable via use_diagnostics=True,
which gives diagnostics/exp_var, diagnostics/clip_frac/*, diagnostics/rms_value/*):

    control/sigma_{mean,min,max} exploration std = exp(log_std); min is the log_std-collapse canary
    control/action_sat           frac of *active* mean-action dims pinned at the tanh rail (|mu|>0.99)
    control/grad_norm            total grad norm BEFORE clipping (clip_grad_norm_ return)
    control/adv_{mean,std}       raw advantage (returns-values) BEFORE normalization -> true scale

Registered globally over 'a2c_continuous' in train_utils, so every continuous PPO
run (transformer or MLP) gets these. Metrics are generic; nothing transformer-specific.
"""
import time

import torch
from torch.nn.utils import clip_grad_norm_

from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent


# Limb slot (1-8 at (n-1)*45 deg) -> compass code, relative to forward = +X (the reward axis),
# right = -Z, slot number increasing clockwise toward the right. See architectures/build_vsim.
_LIMB_CODE = {1: "F", 2: "FR", 3: "R", 4: "BR", 5: "B", 6: "BL", 7: "L", 8: "FL"}
_LEADERBOARD_EVERY = 50  # epochs
_LEADERBOARD_K = 5       # top-k and bottom-k


def _morph_label(limbs) -> str:
    """frozenset/list of limb slots -> compass-coded label, e.g. {2,4,6,8} -> 'FR.BR.BL.FL'."""
    return "·".join(_LIMB_CODE[n] for n in sorted(limbs))


class LoggingA2CAgent(A2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self._grad_norms: list[float] = []   # per-minibatch, flushed per epoch
        self._action_sats: list[float] = []  # per-minibatch, flushed per epoch
        self._adv_mean: float | None = None  # per-epoch (set in prepare_dataset)
        self._adv_std: float | None = None
        self._morph_meta = None  # None=undetected, False=single-morph, dict=multi-morph metadata
        self._steps_since_resample = 0  # env-steps since last morphology resample (full ant only)
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

    # ---- raw advantage scale logging (health/adv_*) ---------------------------------

    def prepare_dataset(self, batch_dict):
        # Raw advantage scale, before rl_games normalizes it (mirrors a2c_common:1030,1038).
        with torch.no_grad():
            adv = (batch_dict['returns'] - batch_dict['values']).sum(dim=1)
            self._adv_mean = adv.mean().item()
            self._adv_std = adv.std().item()
        return super().prepare_dataset(batch_dict)

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
            w.add_scalar('control/sigma_mean', sigma.mean().item(), frame)
            w.add_scalar('control/sigma_min', sigma.min().item(), frame)
            w.add_scalar('control/sigma_max', sigma.max().item(), frame)

        if self._grad_norms:
            w.add_scalar('control/grad_norm', sum(self._grad_norms) / len(self._grad_norms), frame)
            self._grad_norms = []
        if self._action_sats:
            w.add_scalar('control/action_sat', sum(self._action_sats) / len(self._action_sats), frame)
            self._action_sats = []
        if self._adv_mean is not None:
            w.add_scalar('control/adv_mean', self._adv_mean, frame)
            w.add_scalar('control/adv_std', self._adv_std, frame)
            self._adv_mean = self._adv_std = None

        if self._timings:
            for k, v in self._timings.items():
                w.add_scalar(k, v, frame)
            self._timings = {}

        epoch_num = int(args[1]) if len(args) > 1 else kwargs.get('epoch_num', 0)
        self._log_morph_stats(w, frame, epoch_num)

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
        by_limbs: dict[int, list[int]] = {}
        for i, m in enumerate(morphs):
            by_limbs.setdefault(len(m), []).append(i)
        self._morph_meta = {
            'epm': env.envs_per_morph,
            'n': len(morphs),
            'labels': [_morph_label(m) for m in morphs],
            'limb_counts': [len(m) for m in morphs],
            'by_limbs': by_limbs,
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
        for k, idxs in meta['by_limbs'].items():
            w.add_scalar(f'morph_reward_limbs/{k}', pm[torch.tensor(idxs)].mean().item(), frame)

        if epoch_num and epoch_num % _LEADERBOARD_EVERY == 0:
            order = torch.argsort(pm, descending=True).tolist()
            k = min(_LEADERBOARD_K, n)
            rows = ['| rank | morph | limbs | reward |', '|---:|:---|---:|---:|']
            for r in range(k):  # top
                i = order[r]
                rows.append(f'| {r + 1} | {meta["labels"][i]} | {meta["limb_counts"][i]} | {pm[i]:.3f} |')
            if n > 2 * k:
                rows.append('| … | … | … | … |')
            for r in range(max(k, n - k), n):  # bottom
                i = order[r]
                rows.append(f'| {r + 1} | {meta["labels"][i]} | {meta["limb_counts"][i]} | {pm[i]:.3f} |')
            w.add_text('morph_leaderboard', '\n'.join(rows), frame)
