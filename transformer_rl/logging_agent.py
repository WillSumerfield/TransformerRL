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
import torch
from torch.nn.utils import clip_grad_norm_

from rl_games.algos_torch.a2c_continuous import A2CAgent

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

    def train_epoch(self):
        out = super().train_epoch()
        self._maybe_resample()
        return out

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

    def prepare_dataset(self, batch_dict):
        # Raw advantage scale, before rl_games normalizes it to ~N(0,1) (mirrors a2c_common:1030,1038).
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
