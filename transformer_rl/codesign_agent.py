"""CodesignAgent: combined classic-PPO control (LoggingA2CAgent) + the unconditional morphology
generator (Generator), wired at the resample-window boundary. The generator samples per-env bodies at
window start; the env builds them (full rebuild); raw completed-episode returns are accumulated per
env over the window (env_step hook); at the next boundary the generator updates (pretrain BC -> RL
PPO-clip) on (presence, return). See ADR-0010 / temp/ppg_phase2_plan.md."""
import torch

from .logging_agent import LoggingA2CAgent, _LEG_CODE
from .generator import Generator

_N_LEGS = 8


class CodesignAgent(LoggingA2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        cd = self.config.get('generator', {})
        dev = self.ppo_device
        N = self.num_actors * self.num_agents

        env = self._env()                                  # base_legs from the ENV -> window-0 match
        self._base_legs = tuple(sorted(getattr(env, '_base_legs', cd.get('base_legs', (1, 4, 6)))))
        flip = float(cd.get('desirable_flip_prob', 0.10))
        bset = set(self._base_legs)
        self._base_toggle_p = torch.tensor(
            [(1.0 - flip) if (i + 1) in bset else flip for i in range(_N_LEGS)], device=dev)
        self._base_row = torch.tensor(
            [1.0 if (i + 1) in bset else 0.0 for i in range(_N_LEGS)], device=dev)

        self.gen = Generator(
            n_legs=_N_LEGS, lr=cd.get('lr', 1e-2), clip=cd.get('clip', 0.2),
            entropy_coef=cd.get('entropy_coef', 0.01), critic_coef=cd.get('critic_coef', 0.5),
            epochs=cd.get('epochs', 4), minibatches=cd.get('minibatches', 4),
            n_pretrain=cd.get('n_pretrain', 8), device=dev)

        # window state: window 0 is the env's base build, so _cur_presence = base everywhere.
        self._cur_presence = self._base_row.expand(N, _N_LEGS).clone()
        self._cur_old_logits = None
        self._window = 0
        self._gen_log = None

        # per-env raw episode-return accumulators (gamma=1, mean completed-ep return over window).
        self._run_ret = torch.zeros(N, device=dev)   # in-progress ep return (carries across epochs)
        self._win_sum = torch.zeros(N, device=dev)   # sum of completed-ep returns this window
        self._win_cnt = torch.zeros(N, device=dev)   # # completed episodes this window

    def _env(self):
        return getattr(getattr(self.vec_env, 'envs', None), 'env', None)

    # ---- raw episode-return accumulation (the generator's reward) -------------------
    def env_step(self, actions):
        obs, rewards, dones, infos = super().env_step(actions)
        # value_size==2 duplicates reward into (N,2); col 0 = the single env reward.
        r_src = rewards[:, 0] if rewards.dim() > 1 else rewards
        r, d = r_src.reshape(-1), dones.float()            # (N,) raw reward; (N,) done
        self._run_ret += r
        self._win_sum += d * self._run_ret
        self._win_cnt += d
        self._run_ret *= (1.0 - d)
        return obs, rewards, dones, infos

    def _window_returns(self):                             # (N,) mean completed-ep return per env
        ret = self._win_sum / self._win_cnt.clamp(min=1.0)
        return torch.where(self._win_cnt == 0, self._run_ret, ret)   # no completion -> partial

    # ---- ramp: replace (1 - gen_fraction) of envs with base+-flip bodies -----------
    @torch.no_grad()
    def _apply_ramp(self, presence):
        frac = self.gen.gen_fraction()
        if frac >= 1.0:
            return presence                                # RL phase: pure gen samples
        N = presence.shape[0]
        base = torch.bernoulli(self._base_toggle_p.expand(N, _N_LEGS))
        base[base.sum(1) == 0] = self._base_row            # >=1-leg guard for base draws
        use_gen = (torch.rand(N, device=presence.device) < frac).unsqueeze(1)
        return torch.where(use_gen, presence, base)

    # ---- window boundary: update generator, then sample + build next window ---------
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

        returns = self._window_returns()
        if self.gen.in_pretrain():
            self._gen_log = self.gen.pretrain_update(self._cur_presence, returns)
        else:
            self._gen_log = self.gen.rl_update(self._cur_presence, self._cur_old_logits, returns)
        self._window += 1

        presence, old_logits = self.gen.sample(env.total_num_envs)
        presence = self._apply_ramp(presence)
        self._cur_presence, self._cur_old_logits = presence, old_logits
        env.set_next(presence)
        phase = 'pretrain' if self.gen.in_pretrain() else 'rl'
        print(f"[resample #{self._window} | {phase} | gen_frac={self.gen.gen_fraction():.2f} | "
              f"epoch {self.epoch_num}] mean_ret={returns.mean().item():.2f}", flush=True)
        env.resample()
        self.obs = self.env_reset()
        self.current_rewards.zero_(); self.current_lengths.zero_()
        self._run_ret.zero_(); self._win_sum.zero_(); self._win_cnt.zero_()
        self._morph_meta = None
        self._steps_since_resample = 0

    # ---- generator logging (sparse: only at window boundaries) ----------------------
    def write_stats(self, *args, **kwargs):
        super().write_stats(*args, **kwargs)
        w = self.writer
        if w is None or self._gen_log is None:
            return
        frame = args[11] if len(args) > 11 else kwargs.get('frame')
        p = self._gen_log['p']
        for i in range(_N_LEGS):
            w.add_scalar(f'gen_p/{_LEG_CODE[i + 1]}', p[i].item(), frame)
        w.add_scalar('gen/value', self._gen_log['v'], frame)
        w.add_scalar('gen/entropy', self._gen_log['ent'], frame)
        w.add_scalar('gen/fraction', self.gen.gen_fraction(), frame)
        w.add_scalar('gen/mean_legcount', p.sum().item(), frame)             # E[#legs] under p
        w.add_scalar('built/mean_legcount', self._cur_presence.sum(1).mean().item(), frame)
        self._gen_log = None

    # ---- checkpointing --------------------------------------------------------------
    def get_full_state_weights(self):
        s = super().get_full_state_weights()
        s.update(gen=self.gen.state_dict(), gen_opt=self.gen.opt.state_dict(),
                 gen_window=self.gen.window, cs_window=self._window,
                 cur_presence=self._cur_presence, cur_old_logits=self._cur_old_logits,
                 steps_since_resample=self._steps_since_resample)
        return s

    def set_full_state_weights(self, weights, set_epoch=True):
        super().set_full_state_weights(weights, set_epoch=set_epoch)
        if 'gen' not in weights:
            return
        self.gen.load_state_dict(weights['gen'])
        self.gen.opt.load_state_dict(weights['gen_opt'])
        self.gen.window = int(weights['gen_window'])
        self._window = int(weights['cs_window'])
        self._cur_presence = weights['cur_presence'].to(self.ppo_device)
        ol = weights['cur_old_logits']
        self._cur_old_logits = ol.to(self.ppo_device) if ol is not None else None
        self._steps_since_resample = int(weights.get('steps_since_resample', 0))
