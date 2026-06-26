"""CodesignAgent: combined classic-PPO control (LoggingA2CAgent) + the sequential per-token
morphology generator (SeqGenerator, ADR-0012), wired at the resample-window boundary.

The generator samples per-env bodies at window start (a full random-order trace); the env builds
them (full rebuild). The generator's reward is NOT the raw return -- it is the body quality
R_i = V1.0(s0_i): each env's reset observations (progress=0) are captured over the window, and at the
boundary the *updated* control V1.0 head is evaluated on them and averaged per env. At the boundary
the generator updates (pretrain BC -> RL PPO-clip marginal-value) on (trace, R_i), then samples and
builds the next window. Requires value_size==2 (the V1.0 head). Supersedes the ADR-0010 bandit
(transformer_rl/generator.py + the prior CodesignAgent, preserved in git history). See ADR-0012 /
temp/ppg_phase3_plan.md."""
import torch

from .logging_agent import LoggingA2CAgent, _LEG_CODE
from .seq_generator import SeqGenerator

_N_LEGS = 8


class CodesignAgent(LoggingA2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        assert self._v1, "sequential codesign requires value_size==2 (the control V1.0 head)"
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

        self.gen = SeqGenerator(
            n_legs=_N_LEGS, d_model=cd.get('d_model', 128), n_heads=cd.get('n_heads', 8),
            n_layers=cd.get('n_layers', 1), ffn=cd.get('ffn', 512),
            lr=cd.get('lr', 1e-3), clip=cd.get('clip', 0.2),
            entropy_coef=cd.get('entropy_coef', 0.01), critic_coef=cd.get('critic_coef', 0.5),
            epochs=cd.get('epochs', 4), minibatches=cd.get('minibatches', 4),
            n_pretrain=cd.get('n_pretrain', 8), device=dev)

        # window state: window 0 is the env's base build, so _cur_presence = base everywhere.
        self._cur_presence = self._base_row.expand(N, _N_LEGS).clone()
        self._cur_trace = None                             # last sample() trace (RL update input)
        self._window = 0
        self._gen_log = None
        self._last_R = None

        # R_i accumulator: reset observations (s0, progress=0) captured over the window, per env.
        self._s0_obs: list = []                            # list of (b_i, obs_dim) chunks
        self._s0_idx: list = []                            # list of (b_i,) env indices

    def _env(self):
        return getattr(getattr(self.vec_env, 'envs', None), 'env', None)

    # ---- capture reset obs (s0) for the R_i = V1.0(s0) average ----------------------
    def env_step(self, actions):
        obs, rewards, dones, infos = super().env_step(actions)  # rewards already (N,2) via LoggingA2CAgent
        d = dones.bool()
        if d.any():
            o = obs['obs'] if isinstance(obs, dict) else obs   # done envs' returned obs == reset s0
            self._s0_obs.append(o[d].clone())
            self._s0_idx.append(d.nonzero(as_tuple=False).squeeze(-1))
        return obs, rewards, dones, infos

    @torch.no_grad()
    def _window_Ri(self, N):
        """R_i = mean over the window's reset states of the UPDATED V1.0 head (col 1). Every env has
        the window-start seed, so cnt>=1."""
        if not self._s0_obs:                               # safety: no resets captured -> current obs
            return self.get_values(self.obs)[:, 1]
        obs = torch.cat(self._s0_obs, dim=0)
        idx = torch.cat(self._s0_idx, dim=0)
        v1 = self.get_values({'obs': obs})[:, 1]           # updated V1.0, denorm (shaped-reward units)
        sums = torch.zeros(N, device=self.ppo_device).index_add_(0, idx, v1)
        cnts = torch.zeros(N, device=self.ppo_device).index_add_(0, idx, torch.ones_like(v1))
        return sums / cnts.clamp(min=1.0)

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

    # ---- window boundary: R_i -> update generator, then sample + build next window ---
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
        R = self._window_Ri(N)                              # body quality from the updated V1.0 head
        if self.gen.in_pretrain():
            self._gen_log = self.gen.pretrain_update(self._cur_presence, R)   # BC toward built bodies
        else:
            self._gen_log = self.gen.rl_update(self._cur_trace, R)            # marginal-value PPO
        self._window += 1
        self._last_R = R

        trace = self.gen.sample(N)
        presence = self._apply_ramp(trace['presence'])
        self._cur_trace, self._cur_presence = trace, presence
        env.set_next(presence)
        phase = 'pretrain' if self.gen.in_pretrain() else 'rl'
        print(f"[resample #{self._window} | {phase} | gen_frac={self.gen.gen_fraction():.2f} | "
              f"epoch {self.epoch_num}] R_mean={R.mean().item():.3f} "
              f"legcount={presence.sum(1).mean().item():.2f}", flush=True)
        env.resample()
        self.obs = self.env_reset()
        self.current_rewards.zero_(); self.current_lengths.zero_()
        # reset + seed the s0 accumulator with the post-rebuild reset (every env, progress=0)
        self._s0_obs = [self.obs['obs'].clone()]
        self._s0_idx = [torch.arange(N, device=self.ppo_device)]
        self._morph_meta = None
        self._steps_since_resample = 0

    # ---- generator logging (sparse: only at window boundaries) ----------------------
    def write_stats(self, *args, **kwargs):
        super().write_stats(*args, **kwargs)
        w = self.writer
        if w is None or self._gen_log is None:
            return
        frame = args[11] if len(args) > 11 else kwargs.get('frame')
        rate = self._cur_presence.mean(0)                  # per-slot built on-rate
        for i in range(_N_LEGS):
            w.add_scalar(f'gen_p/{_LEG_CODE[i + 1]}', rate[i].item(), frame)
        w.add_scalar('gen/vloss', self._gen_log['vloss'], frame)
        w.add_scalar('gen/loss_a', self._gen_log['loss_a'], frame)
        w.add_scalar('gen/entropy', self._gen_log['ent'], frame)
        w.add_scalar('gen/fraction', self.gen.gen_fraction(), frame)
        w.add_scalar('built/mean_legcount', self._cur_presence.sum(1).mean().item(), frame)
        if 'adv_mean' in self._gen_log:
            w.add_scalar('gen/adv_mean', self._gen_log['adv_mean'], frame)
            w.add_scalar('gen/adv_std', self._gen_log['adv_std'], frame)
        if self._last_R is not None:
            w.add_scalar('gen/R_mean', self._last_R.mean().item(), frame)
        self._gen_log = None

    # ---- checkpointing --------------------------------------------------------------
    def get_full_state_weights(self):
        s = super().get_full_state_weights()
        s.update(gen=self.gen.state_dict(), gen_opt=self.gen.opt.state_dict(),
                 gen_window=self.gen.window, cs_window=self._window,
                 cur_presence=self._cur_presence, cur_trace=self._cur_trace,
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
        tr = weights.get('cur_trace')
        self._cur_trace = {k: v.to(self.ppo_device) for k, v in tr.items()} if tr else None
        self._steps_since_resample = int(weights.get('steps_since_resample', 0))
        # s0 accumulator is not persisted -> the resumed partial window seeds fresh on next boundary.
        self._s0_obs, self._s0_idx = [], []
