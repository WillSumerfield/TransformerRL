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
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from rl_games.algos_torch import torch_ext
from rl_games.common.a2c_common import swap_and_flatten01
from rl_games.common.schedulers import RLScheduler, AdaptiveScheduler

from codesigner.interfaces import ModuleType

from . import runtime
from .morphology import designs_from_arrays
from .vocab import GEN_EFF, GEN_CAP
from .logging_agent import LoggingA2CAgent, _LIMB_CODE
# perplexity diversity estimators for build/* logging (M1); pure-numpy, no transformer_rl dep
from experiments.diversity_p5 import population_to_repr, redundancy, rao_blackwell_h_body

# embedding-like matrices excluded from weight decay despite being nn.Linear (3b, grill 2026-07-15):
# pos_emb/depth_emb are nn.Embedding tables; embed_module/embed_root are content projections treated
# the same way (their weight rows behave like per-token/per-slot embeddings, not a generic FFN map).
_NO_DECAY_NAMES = ('pos_emb', 'depth_emb', 'embed_module', 'embed_root')


def _adamw_param_groups(model, weight_decay):
    """decay: weight matrices (Linear/attention/FFN, dim>=2, not embedding-like); no-decay: biases,
    LayerNorm weight/bias, bare vector Parameters (dim<2, catches mask_token/cls_design for free),
    and the embedding-like 2D matrices in _NO_DECAY_NAMES."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or any(n in name for n in _NO_DECAY_NAMES):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}]


class _WarmupThenAdaptiveScheduler(RLScheduler):
    """3c warmup: linear LR warmup 0->peak over `warmup_epochs`, then hand off to rl_games'
    AdaptiveScheduler (KL-reactive) for the rest of training, continuing from the LR warmup ended
    at (== peak by construction). No cosine annealing (considered, dropped). Span default was the
    n_pretrain BC-toward-teacher window (grill 2026-07-15) but that ran pretrain cold -> now
    decoupled via config `warmup_epochs` (grill 2026-07-17)."""
    def __init__(self, peak_lr: float, warmup_epochs: int, kl_threshold: float):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_epochs = max(1, warmup_epochs)
        self._adaptive = AdaptiveScheduler(kl_threshold)

    def update(self, current_lr, entropy_coef, epoch, frames, kl_dist, **kwargs):
        if epoch < self.warmup_epochs:
            return self.peak_lr * (epoch + 1) / self.warmup_epochs, entropy_coef
        return self._adaptive.update(current_lr, entropy_coef, epoch, frames, kl_dist, **kwargs)


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
        # 3b: AdamW w/ decoupled weight decay (rl_games' base builds plain Adam, one param group,
        # config['weight_decay'] fused into the gradient -- NOT decoupled). Two groups so decay only
        # touches real weight matrices; weight_decay=0 (rl_games default) reproduces base Adam exactly.
        self.optimizer = torch.optim.AdamW(
            _adamw_param_groups(self.model, self.weight_decay), float(self.last_lr),
            eps=1e-08, fused=True)
        cd = self.config.get('generator', {})
        dev = self.ppo_device
        N = self.num_actors * self.num_agents

        # obs layout + action dim from the net's tdims -- which IS the Task's obs_layout() (D23),
        # not a second derivation of it. Phase-1: 32 DOF, mask at obs[187:219]; a limb is a chain
        # of up to _max_len modules.
        self._n_dof = net.tdims['n_modules']
        self._mask_off = net.tdims['mask_off']
        self._n_limbs = net.n_limbs            # attachment slots, the library's (D14)
        self._n_sub = net.n_sub                # subtype index width, the library's (D14)
        self._max_len = net.max_limb_length
        self._max_eff = net.max_effectors      # max_len-1: the deepest slot is grammar-forced to a cap

        env = self._env()
        assert env is not None, "CodesignAgent requires a live codesign env (module_library/base body)"
        self._ml = env.module_library          # the run's ONE library, carried by the Task (D14)
        # Per-type subtype name tuples, derived once from the ModuleLibrary's public modules API
        # (self._ml.names) rather than hardcoded per-type constants -- cached here since
        # _frontier_rollout (hot path) needs the plain ints, not a name lookup per call. "swing"/
        # "knee"/"bare" are OUR choice of canonical/fallback body (matches the seed body's own
        # literal vocabulary, transformer_rl.morphology.CANONICAL_*), not something the library
        # hands out -- ModuleLibrary has no concept of "canonical", only the type vocabulary.
        self._eff_names = self._ml.names(ModuleType.EFFECTOR)
        self._cap_names = self._ml.names(ModuleType.CAP)
        self._eff_swing = self._eff_names.index("swing")
        self._eff_knee = self._eff_names.index("knee")
        self._cap_bare = self._cap_names.index("bare")
        self._flip = float(cd.get('desirable_flip_prob', 0.10))    # per-token grow/stop flip prob
        # Phase-5 teacher knob #2: per-token TYPE flip. On a flip the emitted subtype is redrawn
        # uniformly over the grammar-valid kinds for the category actually emitted. p = q = 0
        # reproduces the phase-1 canonical body (swing, then knees, bare cap) EXACTLY.
        self._type_flip = float(cd.get('type_flip_prob', 0.10))
        # Warmup teacher. 'flip' = base +- per-token noise (original). 'parts' = seed-relative
        # PARTS-COPY: per-token flip has the wrong geometry -- mass at edit-radius r decays like
        # flip^r, and the CHEAP edits are the degenerate ones (sprouting a limb costs 1 flip, a limb
        # of USEFUL length costs 3), so it only ever reaches base +- stubs. Parts-copy instead makes
        # each slot COPY a limb template from the seed, so a sprouted limb arrives full-length.
        self._teacher = str(cd.get('teacher', 'flip'))
        self._copy_prob = float(cd.get('copy_prob', 0.6))          # p: slot keeps its OWN base template
        self._len_keep = float(cd.get('len_keep_prob', 0.6))       # q: P(length offset == 0)
        self._prob_invalid = float(cd.get('prob_invalid', 0.1))    # keep an UNSTABLE body w.p. this
        # sigma solved from q = P(round(N(0,sigma)) == 0) = erf(0.5/(sigma*sqrt2))
        self._len_sigma = 0.5 / (math.sqrt(2) * torch.erfinv(torch.tensor(self._len_keep)).item())
        # The seed body is the algorithm's, not the Task's -- the Task takes bodies, it does not
        # keep a canonical one -- so it comes from the run record. Slots are 0-based on both sides.
        bset = set(runtime.base_morphology().occupied_slots)
        # seed morph = those slots @ count-2 (1 swing + 1 knee). Per-token flip noise around this
        # target length gives P(limb differs by +-1 token)=flip, +-2=flip^2, ... (presence + length
        # unified).
        self._base_target = torch.tensor(
            [2 if i in bset else 0 for i in range(self._n_limbs)], dtype=torch.long, device=dev)
        self._base_counts = self._base_target.clone()      # window-0 realized body == base target
        assert int(self._base_target.max()) <= self._max_eff, \
            f"base target exceeds max effectors/limb ({self._max_eff})"

        # generator hyperparameters (shared optimizer; the heads live on self.model)
        self._n_pretrain = cd.get('n_pretrain', 8)
        # 3c: LR warmup over the SAME n_pretrain window (not a new span). warmup_epochs is derived,
        # not swept directly -- one epoch == one _maybe_resample() call, so the epoch count spanning
        # n_pretrain resamples is fixed by (resample_interval * max_episode_length / horizon_length).
        # Off by default (config.lr_warmup) -> self.scheduler stays rl_games' plain AdaptiveScheduler.
        interval = self.config.get('resample_interval', 0)
        if bool(self.config.get('lr_warmup', False)) and interval and env is not None:
            epochs_per_window = max(1, math.ceil(interval * env.max_episode_length / self.horizon_length))
            # 3c sweep: DECOUPLE the LR-ramp span from pretrain. `warmup_epochs` (config) overrides;
            # unset (0) => n_pretrain*epochs_per_window (the old pretrain-coupled 504-epoch ramp that
            # ran the whole BC pretrain cold). A short ramp reaches peak within a window, then pretrain
            # runs hot. Peak stays == learning_rate; only the ramp DURATION changes.
            warmup_epochs = int(self.config.get('warmup_epochs', 0)) or self._n_pretrain * epochs_per_window
            self.scheduler = _WarmupThenAdaptiveScheduler(
                peak_lr=float(self.last_lr), warmup_epochs=warmup_epochs,
                kl_threshold=getattr(self, 'kl_threshold', self.config.get('kl_threshold', 0.008)))
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
        # window 0 is the env's canonical base build: base counts, canonical types, bare caps.
        self._cur_counts = self._base_counts.view(1, self._n_limbs).expand(N, self._n_limbs).clone()
        self._cur_eff = torch.full((N, self._n_limbs, self._max_len), -1, dtype=torch.long, device=dev)
        for d in range(self._max_len):
            self._cur_eff[:, :, d] = torch.where(
                self._cur_counts > d, self._eff_swing if d == 0 else self._eff_knee, -1)
        self._cur_cap = torch.where(self._cur_counts > 0, self._cap_bare, -1)
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
        self._jepa_anchor_coef = float(jc.get('anchor_coef', 1.0))         # repr-anchor in resample
        self._jepa_anchor_states = int(jc.get('anchor_snapshot_states', 16384))  # <=0 or >=HN -> full
        self._jepa_losses = []

        # Forward Dynamics aux (Phase 2b): per-active-module next-step prediction, FUSED into the PPO
        # step (0 extra trunk passes; reuses H[t]). Disabled -> behaviour == phase-1 baseline (A/B).
        # M1 (here): rollout plumbing only -- next_obs + valid injected into the shuffled dataset.
        fc = self.config.get('fd', {})
        self._fd_enabled = bool(fc.get('enabled', False))
        self._fd_coef = float(fc.get('coef', 1.0))
        self._fd_losses = []

        # Forward Kinematics aux (Phase 2b): each active module token predicts its OWN torso-frame
        # pose (pos+rot6D+vel, 15), same-timestep, FUSED into the PPO step (0 extra passes). Target
        # composed agent-side from RAW obs (no rollout injection -- 'obs' is already in the dataset);
        # per-depth normalizer updated once/window in prepare_dataset. Disabled -> baseline.
        kc = self.config.get('fk', {})
        self._fk_enabled = bool(kc.get('enabled', False))
        self._fk_coef = float(kc.get('coef', 1.0))
        self._fk_losses = []

        # Fixed compile-time gate on the net (set before torch_runner compiles the model): the aux
        # heads run in forward iff enabled, so an off-run never traces them in (baseline-identical).
        net._fd_enabled, net._fk_enabled = self._fd_enabled, self._fk_enabled

        # R_i accumulator: true completed-episode return per env over the window (gamma=1).
        self._ep_ret = torch.zeros(N, device=dev)
        self._win_ret_sum = torch.zeros(N, device=dev)
        self._win_ret_cnt = torch.zeros(N, device=dev)

    def _net(self):
        return self.model.a2c_network.net

    def prepare_dataset(self, batch_dict):
        # Build the shuffled minibatch dataset as usual, then (FD only) inject the per-sample
        # next_obs + valid so calc_gradients can supervise the (obs[t] -> obs[t+1]) transition.
        super().prepare_dataset(batch_dict)
        if self._fk_enabled:
            # FK target: compose ONCE/window over the whole rollout (raw obs), update per-depth
            # stats, normalize, and inject into the dataset (FD pattern) so calc_gradients just
            # reads it per-minibatch instead of recomposing ~5x/window (was fk_compose ~61ms/epoch).
            # Stored target is already normalized (eval space); fk_loss reads it directly.
            with self._prof('prep_fk'):
                net = self._net()
                obses = self.experience_buffer.tensor_dict['obses']     # (H, N, obs)
                H, N = obses.shape[0], obses.shape[1]
                with torch.no_grad():
                    tgt, act = net.fk_compose_target(obses.reshape(-1, obses.shape[-1]))
                    net.fk_update_stats(tgt, act)
                    tgt = net.fk_normalize(tgt)                          # normalize w/ updated stats
                vd = self.dataset.values_dict
                vd['fk_target'] = swap_and_flatten01(tgt.view(H, N, *tgt.shape[1:]))
                vd['fk_active'] = swap_and_flatten01(act.view(H, N, *act.shape[1:]))
        if not self._fd_enabled:
            return
        with self._prof('prep_fd'):
            obses = self.experience_buffer.tensor_dict['obses']          # (H, N, obs), raw
            dones = self.experience_buffer.tensor_dict['dones']          # (H, N[,1]) bool/byte
            if dones.dim() > 2:
                dones = dones.squeeze(-1)
            next_obs = torch.empty_like(obses)
            next_obs[:-1] = obses[1:]
            next_obs[-1] = obses[-1]                                 # last horizon step: dummy (valid=0)
            valid = torch.zeros(dones.shape, dtype=torch.bool, device=obses.device)   # (H, N)
            valid[:-1] = dones[1:] == 0        # transition t valid iff obs[t+1] is NOT a reset-initial obs
            vd = self.dataset.values_dict
            vd['fd_next_obs'] = swap_and_flatten01(next_obs)        # (H*N, obs), shuffled with the rest
            vd['fd_valid'] = swap_and_flatten01(valid)              # (H*N,)

    def _calc_gradients_prof(self, input_dict):
        """Region-instrumented copy of rl_games A2CAgent.calc_gradients: identical math, with
        self._prof around forward / losses / getaux / backward / opt. Assumes our config
        (non-RNN, single-GPU). Used only under timing/mem_profile; stock path otherwise."""
        assert not self.is_rnn and not self.multi_gpu, "prof path assumes non-RNN single-GPU"
        old_logp = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu, old_sigma = input_dict['mu'], input_dict['sigma']
        old_values, returns = input_dict['old_values'], input_dict['returns']
        actions = input_dict['actions']
        obs = self._preproc_obs(input_dict['obs'])
        curr_e_clip = self.e_clip
        batch_dict = {'is_train': True, 'prev_actions': actions, 'obs': obs}

        with torch.amp.autocast('cuda', enabled=self.mixed_precision, dtype=torch.bfloat16):
            with self._prof('forward'):
                res = self.model(batch_dict)
            with self._prof('losses'):
                loss, a_loss, c_loss, entropy, b_loss, _ = self.calc_losses(
                    self.actor_loss_func, old_logp, res['prev_neglogp'], advantage, curr_e_clip,
                    old_values, res['values'], returns, res['mus'], res['entropy'], None)
            with self._prof('getaux'):
                aux_loss = self.model.get_aux_loss()
                self.aux_loss_dict = {}
                if aux_loss is not None:
                    for k, v in aux_loss.items():
                        loss = loss + v
                        self.aux_loss_dict[k] = [v.detach()]
            for param in self.model.parameters():
                param.grad = None
        with self._prof('backward'):
            self.scaler.scale(loss).backward()
        with self._prof('opt'):
            self.trancate_gradients_and_step()

        with torch.no_grad():
            kl_dist = torch_ext.policy_kl(res['mus'].detach(), res['sigmas'].detach(),
                                          old_mu, old_sigma, True)
        self.diagnostics.mini_batch(self, {
            'values': old_values, 'returns': returns, 'new_neglogp': res['prev_neglogp'],
            'old_neglogp': old_logp, 'masks': None}, curr_e_clip, 0)
        self.train_result = (a_loss, c_loss, entropy, kl_dist, self.last_lr, 1.0,
                             res['mus'].detach(), res['sigmas'].detach(), b_loss)
        self._log_action_sat()

    def calc_gradients(self, input_dict):
        # FD (2b): arm the loss targets; the head runs in the PPO forward (rides its compile) and
        # rl_games' get_aux_loss() hook (a2c_continuous.py:194) adds fd_coef*fd_loss into the SAME
        # loss/backward -> 0 extra trunk passes. Disabled -> get_aux_loss() None -> baseline-identical.
        net = self._net()
        if self._fd_enabled:
            with self._prof('fd_target'):
                rms = getattr(self.model, 'running_mean_std', None)   # don't let target update stats
                was_train = rms.training if rms is not None else False
                if rms is not None:
                    rms.eval()
                with torch.no_grad():
                    next_obs = self.model.norm_obs(input_dict['fd_next_obs'])
                if rms is not None:
                    rms.train(was_train)
                net.fd_arm(next_obs, input_dict['fd_valid'], self._fd_coef)
        if self._fk_enabled:
            # FK target precomputed + normalized once/window (prepare_dataset), injected into the
            # dataset; just read + arm here (no per-minibatch recompose).
            with self._prof('fk_arm'):
                net.fk_arm(input_dict['fk_target'], input_dict['fk_active'], self._fk_coef)

        # PPO control step (+ saturation logging), untouched -> control stays clean. Profiling
        # runs take an instrumented copy that times/mem-probes forward/losses/getaux/backward/opt.
        if self._timing or self._mem_profile:
            self._calc_gradients_prof(input_dict)
        else:
            super().calc_gradients(input_dict)

        if self._fd_enabled:
            self._fd_losses.append(net._fd_last)
            net.fd_disarm()
        if self._fk_enabled:
            self._fk_losses.append(net._fk_last)
            net.fk_disarm()
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

    # ---- scripted frontier rollout: target-length teacher + per-token length/type flip noise -----
    @torch.no_grad()
    def _frontier_rollout(self, target, flip, type_flip=0.0, eff_types=None, cap_types=None):
        """Walk the SAME frontier MDP as net.sample (same random-tip order, same >=1-limb guard, same
        constrained decoder) but with a scripted policy. Two independent noise knobs:
          LENGTH: at a limb of current length c the canonical CATEGORY is effector if c <
                  target[limb] else cap, flipped with prob `flip`.
          TYPE:   the canonical SUBTYPE is the phase-1 chain (swing at depth 0, knee below) and the
                  bare cap; with prob `type_flip` it is redrawn UNIFORMLY over the grammar-valid
                  subtypes for the category actually emitted. flip = type_flip = 0 reproduces the
                  phase-1 body exactly.
        Pass eff_types/cap_types to instead RECONSTRUCT a specific body's token sequence (subtypes
        read off that body, type_flip ignored) -- used for BC / GenCrit's prefix fit on the body
        that actually ran.
        The grammar (not this policy) enforces the deepest-slot cap and the guard: whenever the
        scripted category is masked out, it flips to the only legal one.
        Returns (slots, cat_actions, sub_actions, active_step, counts, eff_sub, cap_sub)."""
        net = self._net()
        dev = target.device
        N, n = target.shape
        max_len = self._max_len
        L = n * max_len
        count   = torch.zeros(N, n, dtype=torch.long, device=dev)
        cap_sub = torch.full((N, n), -1, dtype=torch.long, device=dev)
        eff_sub = torch.full((N, n, max_len), -1, dtype=torch.long, device=dev)
        slots = torch.zeros(N, L, dtype=torch.long, device=dev)
        cat_a = torch.zeros(N, L, dtype=torch.long, device=dev)
        sub_a = torch.zeros(N, L, dtype=torch.long, device=dev)
        active_step = torch.zeros(N, L, dtype=torch.bool, device=dev)
        arange = torch.arange(N, device=dev)
        for t in range(L):
            growable = cap_sub < 0
            active = growable.any(1)
            r = torch.rand(N, n, device=dev)
            slot = torch.where(growable, r, r.new_full((), -1.0)).argmax(1)
            depth = count[arange, slot]
            force = active & (count.sum(1) == 0) & (growable.sum(1) == 1)   # >=1-limb guard
            cat_mask, sub_mask = net._gen_masks(depth, force)

            grow = depth < target[arange, slot]
            if flip > 0:
                grow = grow ^ (torch.rand(N, device=dev) < flip)
            c = torch.where(grow, grow.new_full((), GEN_EFF), grow.new_full((), GEN_CAP)).long()
            legal = cat_mask.gather(1, c.unsqueeze(1)).squeeze(1)
            c = torch.where(legal, c, 1 - c)                 # only 2 categories, >=1 always legal

            if eff_types is not None:                        # reconstruct a given body
                s_eff = eff_types[arange, slot, depth.clamp(max=max_len - 1)]
                s_cap = cap_types[arange, slot]
            else:                                            # canonical chain + bare cap
                s_eff = torch.where(depth == 0, depth.new_full((), self._eff_swing),
                                    depth.new_full((), self._eff_knee))
                s_cap = depth.new_full((N,), self._cap_bare)
            s = torch.where(c == GEN_EFF, s_eff, s_cap)
            sm = sub_mask[arange, c]                                          # (N, n_sub) valid set
            if type_flip > 0 and eff_types is None:
                u = torch.rand(N, self._n_sub, device=dev) * sm.float()       # invalid -> exactly 0
                s = torch.where(torch.rand(N, device=dev) < type_flip, u.argmax(1), s)
            ok = sm.gather(1, s.clamp(min=0).unsqueeze(1)).squeeze(1)
            s = torch.where(ok, s, sm.float().argmax(1))     # fall back to the first legal subtype

            net.commit(count, cap_sub, eff_sub, arange, slot, depth, c, s, active)
            slots[:, t] = slot
            cat_a[:, t] = c
            sub_a[:, t] = s
            active_step[:, t] = active
        return slots, cat_a, sub_a, active_step, count, eff_sub, cap_sub

    @torch.no_grad()
    def _is_stable(self, counts):
        """Stable = walkable: >=3 limbs, >=2 limbs of length >=2, no circular gap > 135deg (i.e. no
        run of >=3 empty slots on the 8-slot ring). Used ONLY to weight the warmup teacher's draws
        (see _draw_parts_counts); the generator itself is never masked by it."""
        pres = counts > 0
        ok = (pres.sum(1) >= 3) & ((counts >= 2).sum(1) >= 2)
        pad = torch.cat([pres, pres], 1)                              # wrap the ring
        for r in range(self._n_limbs):
            ok &= (pad[:, r] | pad[:, r + 1] | pad[:, r + 2])
        return ok

    @torch.no_grad()
    def _draw_parts_counts(self, N):
        """Seed-relative PARTS-COPY teacher -> body counts (N, n). Uses only the seed body + the
        token MDP (no morphology oracle beyond the stability WEIGHT below), so it survives Phase-5's
        tree vocabulary. Two stages:
          1. presence+template: each slot copies a limb template from the base -- its OWN w.p.
             copy_prob, else uniform over the other slots (absent templates included, so presence is
             inherited from the seed's density). A sprouted slot therefore arrives FULL-LENGTH.
          2. length: per-limb integer offset ~ round(N(0, sigma)), applied to PRESENT templates only
             (an absent slot never sprouts here). Offsets may delete a limb.
        Unstable draws are rejected and resampled, except kept w.p. prob_invalid -- so GenCrit still
        sees some bad bodies (that is how it learns 'bad' without an oracle)."""
        dev = self._base_target.device
        ar = torch.arange(self._n_limbs, device=dev)
        out = torch.zeros(N, self._n_limbs, dtype=torch.long, device=dev)
        filled = 0
        while filled < N:
            m = max(N, 2 * (N - filled))                              # over-draw; rejection is cheap
            own = torch.rand(m, self._n_limbs, device=dev) < self._copy_prob
            other = torch.randint(0, self._n_limbs - 1, (m, self._n_limbs), device=dev)
            other = other + (other >= ar).long()                      # uniform over the OTHER slots
            tmpl = self._base_target[torch.where(own, ar.expand(m, -1), other)]
            off = torch.round(torch.randn(m, self._n_limbs, device=dev) * self._len_sigma).long()
            c = torch.where(tmpl > 0, (tmpl + off).clamp(0, self._max_eff), torch.zeros_like(tmpl))
            c[c.sum(1) == 0, 0] = 2                                   # >=1-limb guard
            keep = c[self._is_stable(c) | (torch.rand(m, device=dev) < self._prob_invalid)]
            take = min(keep.shape[0], N - filled)
            out[filled:filled + take] = keep[:take]
            filled += take
        return out

    @torch.no_grad()
    def _teacher_rollout(self, N):
        """One scripted warmup-teacher draw. Returns the FULL frontier trace plus the realized body
        (slots, cat_actions, sub_actions, active_step, counts, eff_sub, cap_sub) -- BC reads the
        token trace, the ramp reads the body. See self._teacher for the length distribution; types
        always come from the type-flip knob."""
        if self._teacher == 'parts':
            return self._frontier_rollout(self._draw_parts_counts(N), 0.0, self._type_flip)
        return self._frontier_rollout(self._base_target.view(1, self._n_limbs).expand(N, self._n_limbs),
                                      self._flip, self._type_flip)

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
    def _apply_ramp(self, counts, eff_sub, cap_sub):
        """Generator-sampled DESIGN (counts, per-depth effector subtypes, cap subtype). Mix in
        teacher draws by fraction -- whole bodies, never a blend of two designs."""
        frac = self._gen_fraction()
        if frac >= 1.0:
            self._base_draw = None                         # RL phase: pure gen samples, no base draws
            return counts, eff_sub, cap_sub
        N = counts.shape[0]
        _, _, _, _, b_counts, b_eff, b_cap = self._teacher_rollout(N)
        self._base_draw = b_counts                         # the around-base samples (build/*_base)
        use = torch.rand(N, device=counts.device) < frac
        return (torch.where(use.unsqueeze(1), counts, b_counts),
                torch.where(use.view(N, 1, 1), eff_sub, b_eff),
                torch.where(use.unsqueeze(1), cap_sub, b_cap))

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

            # JEPA repr-anchor: snapshot H_full over a random SUBSET of rollout states (bf16; cosine
            # is scale-invariant so half-precision storage is fine). The generator update then pulls
            # H_full back toward this pre-update snapshot so it can't destroy the control rep.
            anchor_on = self._jepa_enabled and self._jepa_anchor_coef > 0
            if anchor_on:
                S = HN if (self._jepa_anchor_states <= 0 or self._jepa_anchor_states >= HN) \
                    else self._jepa_anchor_states
                sub = torch.randperm(HN, device=dev)[:S]     # subset global indices
                H_old = obs_flat.new_empty(S, net.n_tokens, net._d_model, dtype=torch.bfloat16)
                for s in range(0, S, self.minibatch_size):
                    idx = sub[s:s + self.minibatch_size]
                    *_, H = net.codesign_forward(self.model.norm_obs(obs_flat[idx]), return_hidden=True)
                    H_old[s:s + self.minibatch_size] = H.to(torch.bfloat16)

        pretrain = self._in_pretrain()
        # NOTE: (slots, actions) here is always the BUILT body -- GenCrit's prefix fit must stay on it
        # because R was measured on the body that actually ran. In pretrain, GenAct's BC target is a
        # separate, freshly-drawn TEACHER body (see the epoch loop).
        if pretrain:
            slots, cat_a, sub_a, valid, *_ = self._frontier_rollout(
                self._cur_counts, 0.0, eff_types=self._cur_eff, cap_types=self._cur_cap)
            old_logp, adv, raw_adv = None, None, None
        else:                                              # RL: PPO over the sampled trace
            tr = self._cur_trace
            slots, cat_a, sub_a = tr['slots'], tr['cat_actions'], tr['sub_actions']
            old_logp = tr['old_logp']
            valid = tr['active_step']                                     # mask no-op frontier steps
            raw_adv = tr['v_states'][:, 1:] - tr['v_states'][:, :-1]      # telescoping (Shapley)
            sel = raw_adv[valid]
            adv = torch.zeros_like(raw_adv)
            adv[valid] = (sel - sel.mean()) / (sel.std() + 1e-8)

        L1 = self._n_dof + 1                               # (L+1) prefixes gen_replay stacks per env
        mb_size = max(1, min(N // self._gen_minibatches, self._gen_max_prefixes // L1))
        logs = {k: [] for k in ('gen_pg', 'ent', 'v_prefix', 'v_roll', 'kl', 'crit', 'gn', 'anchor')}
        for _ in range(self._gen_epochs):
            if pretrain:
                # GenAct's BC target: a FRESH teacher draw (+ fresh random tip ordering) each epoch.
                # The teacher is a cheap GPU sampler, so BC fits the teacher DISTRIBUTION rather than
                # re-fitting one finite 4096-body sample of it. Deliberately NOT the built body: that
                # is (1-frac) teacher + frac generator-samples with frac -> ~1 across the ramp, so
                # cloning it makes the generator imitate ITSELF in the late windows -- a feedback loop
                # that compounds fit error instead of correcting it.
                bc_slots, bc_cat, bc_sub, *_ = self._teacher_rollout(N)
            perm = torch.randperm(N, device=dev)
            for s in range(0, N, mb_size):
                mb = perm[s:s + mb_size]
                # --- generator: GenAct (PPO or BC) + GenCrit prefix fit ---
                # gen_replay re-derives the valid (non-no-op) step mask from (slots,actions); average
                # GenAct losses over valid steps only so no-op frontier steps don't dilute the signal.
                # FACTORED GenAct: gen_replay returns masked log-probs for the category head and
                # the per-category subtype head; gen_logp_entropy folds them into the joint
                # logp = logp(cat) + logp(sub | cat) and the exact joint entropy.
                cat_lp, sub_lp, v, vm = net.gen_replay(slots[mb], cat_a[mb], sub_a[mb])
                vf = vm.float()
                v_prefix = (v - R[mb].unsqueeze(1)).pow(2).mean()        # all L+1 prefixes -> R

                if pretrain:                       # BC: separate replay on the teacher tokens
                    b_cat_lp, b_sub_lp, _, bm = net.gen_replay(bc_slots[mb], bc_cat[mb], bc_sub[mb])
                    bf = bm.float()
                    nval = bf.sum().clamp(min=1.0)
                    lp, ent_t = net.gen_logp_entropy(b_cat_lp, b_sub_lp, bc_cat[mb], bc_sub[mb])
                    gen_pg = -(lp * bf).sum() / nval
                    ent = (ent_t * bf).sum() / nval
                else:                              # RL: PPO-clip on the sampled trace
                    nval = vf.sum().clamp(min=1.0)
                    lp, ent_t = net.gen_logp_entropy(cat_lp, sub_lp, cat_a[mb], sub_a[mb])
                    ratio = (lp - old_logp[mb]).exp()
                    a = adv[mb]
                    per = torch.min(ratio * a, ratio.clamp(1 - self._gen_clip, 1 + self._gen_clip) * a)
                    gen_pg = -(per * vf).sum() / nval
                    ent = (ent_t * vf).sum() / nval

                # --- control clone + GenCrit rollout-state fit (sampled rollout minibatch) ---
                # Anchor on: sample from the snapshot SUBSET so H_full_new reuses this single forward
                # (no extra pass); the clone is a soft regularizer, so a subset sample is fine.
                if anchor_on:
                    j = torch.randint(0, S, (N,), device=dev)
                    ridx = sub[j]
                    ob = obs_flat[ridx]
                    mu_n, v098_n, v1_n, H_new = net.codesign_forward(self.model.norm_obs(ob),
                                                                     return_hidden=True)
                    anchor = (1 - F.cosine_similarity(H_new, H_old[j].float(), dim=-1)).mean()
                else:
                    ridx = torch.randint(0, HN, (N,), device=dev)
                    ob = obs_flat[ridx]
                    mu_n, v098_n, v1_n = net.codesign_forward(self.model.norm_obs(ob))
                    anchor = ob.new_zeros(())
                kl = _gauss_kl(mu_old[ridx], ls_old[ridx], mu_n, self._log_std(ob))
                crit = (v098_n - v098_old[ridx]).pow(2).mean()
                v_roll = (v1_n.squeeze(-1) - R_roll[ridx]).pow(2).mean()

                # entropy is an RL-only term: pretrain is supervised (gen_pg is plain NLL) and the
                # TEACHER is the entropy source, so a bonus here only fights the fit -- at 0.1 it
                # drove the token policy to 86% of ln2 (near coin-flip), i.e. geometric limb
                # lengths, and cut the parts teacher's 87% stable bodies down to 41%.
                ent_coef = 0.0 if pretrain else self._gen_ent
                loss = (gen_pg - ent_coef * ent
                        + self._gencrit_coef * (v_prefix + v_roll)
                        + self._beta * kl + self._lam * crit
                        + self._jepa_anchor_coef * anchor)
                self.optimizer.zero_grad()
                loss.backward()
                logs['gn'].append(clip_grad_norm_(self.model.parameters(), self.grad_norm))
                self.optimizer.step()
                for k, val in (('gen_pg', gen_pg), ('ent', ent), ('v_prefix', v_prefix),
                               ('v_roll', v_roll), ('kl', kl), ('crit', crit), ('anchor', anchor)):
                    logs[k].append(val.detach())

        self._gen_log = {k: torch.stack(v).mean().item() for k, v in logs.items()}
        # body-quality outcome (the optimization target); R aligns with the current window's
        # realized bodies (_cur_counts), before _maybe_resample samples the next window.
        self._gen_log['R_var'] = R.var(unbiased=False).item()    # scale for the GenCrit vloss norm
        self._gen_log['R_mean'] = R.mean().item()
        self._gen_log['R_std'] = R.std().item()
        self._gen_log['by_limbcount'] = self._by_limbcount(R, self._cur_counts)
        self._gen_log['types'] = self._type_usage()
        if not pretrain:                                   # RL: built body == generated (ramp off)
            self._gen_log['marg'] = self._slot_marginal(raw_adv, slots, cat_a, valid)
            body_id = self._body_key(self._cur_trace['counts'].long(), self._cur_trace['eff_sub'],
                                     self._cur_trace['cap_sub'])
            rank, ev, K = self._body_value_metrics(body_id,
                                                   self._cur_trace['v_states'][:, -1], R)
            self._gen_log['value_rank_corr'] = rank       # denoised Spearman (NaN if <5 bodies)
            self._gen_log['value_ev'] = ev                # denoised per-body explained variance
            self._gen_log['n_distinct_bodies'] = float(K)

    @torch.no_grad()
    def _slot_marginal(self, raw_adv, slots, cat_a, valid):
        """Per-limb marginal value of adding an EFFECTOR, over valid steps only."""
        on = (cat_a == GEN_EFF) & valid
        marg = torch.full((self._n_limbs,), float('nan'), device=raw_adv.device)
        for k in range(self._n_limbs):
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
    def _body_key(counts, eff_sub, cap_sub):
        """Distinct-body id INCLUDING types: counts alone now under-counts, since two bodies with the
        same limb lengths but different effector/cap kinds are different bodies. 64-bit polynomial
        hash over [counts | eff_sub+1 | cap_sub+1] (wraparound is fine, it is only a bucket id)."""
        B = counts.shape[0]
        flat = torch.cat([counts.reshape(B, -1), (eff_sub + 1).reshape(B, -1),
                          (cap_sub + 1).reshape(B, -1)], dim=1)
        h = torch.zeros(B, dtype=torch.long, device=counts.device)
        for j in range(flat.shape[1]):
            h = h * 1000003 + flat[:, j]
        return h

    @torch.no_grad()
    def _type_usage(self):
        """Realized type mix over the CURRENT window's built bodies: fraction of effectors of each
        kind and fraction of present limbs carrying each cap kind. The collapse canary for 5a --
        a generator that never leaves the canonical (swing, knee, bare) corner has learnt nothing
        about the new vocabulary."""
        eff, cap = self._cur_eff, self._cur_cap
        n_eff = (eff >= 0).sum().clamp(min=1)
        n_cap = (cap >= 0).sum().clamp(min=1)
        eff_names, cap_names = self._eff_names, self._cap_names
        out = {f'eff/{eff_names[t]}': ((eff == t).sum() / n_eff).item() for t in range(len(eff_names))}
        out.update({f'cap/{cap_names[t]}': ((cap == t).sum() / n_cap).item()
                    for t in range(len(cap_names))})
        return out

    @staticmethod
    @torch.no_grad()
    def _body_value_metrics(body_id, v_full, R):
        """Denoised body-quality fit: group envs by distinct body (see _body_key), then compare
        the generator's v(full) to each body's MEAN R (removes reset noise). Returns (rank_corr, ev,
        n_bodies):
          rank_corr = Spearman over bodies (NaN if <5 -> unreliable); spread/scale-robust.
          ev        = 1 - Var(meanR - v)/Var(meanR) over bodies (NaN if <2).
        Valid only when built==generated (RL phase), where R matches the generated body."""
        dev = R.device
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
        counts, eff_sub, cap_sub = self._apply_ramp(
            trace['counts'].long(), trace['eff_sub'], trace['cap_sub'])
        self._cur_trace = trace
        self._cur_counts, self._cur_eff, self._cur_cap = counts, eff_sub, cap_sub
        morphs = designs_from_arrays(self._ml, counts, eff_sub, cap_sub)
        print(f"[resample #{self._gen_window} | {phase} | next_gen_frac={self._gen_fraction():.2f} | "
              f"epoch {self.epoch_num}] R_mean={R.mean().item():.3f} "
              f"limbcount={(counts > 0).sum(1).float().mean().item():.2f} "
              f"modules={counts.sum(1).float().mean().item():.2f}", flush=True)
        env.resample(morphs)
        self.obs = self.env_reset()
        self.current_rewards.zero_(); self.current_lengths.zero_()
        # reset R_i accumulators for the new window
        self._ep_ret.zero_(); self._win_ret_sum.zero_(); self._win_ret_cnt.zero_()
        self._morph_meta = None
        self._steps_since_resample = 0

    def _log_morph_stats(self, w, frame, epoch_num):
        return  # codesign reports body quality via quality/* (true R), not the base per-step morph_reward

    # ---- generator logging (sparse: only at window boundaries) ----------------------
    # Metrics are grouped by subsystem (see docs/reference/codesign_metrics.md):
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
        if self._fd_losses:
            w.add_scalar('losses/fd', torch.stack(self._fd_losses).mean().item(), frame)
            self._fd_losses = []
        if self._fk_losses:
            w.add_scalar('losses/fk', torch.stack(self._fk_losses).mean().item(), frame)
            self._fk_losses = []
        if self._gen_log is None:
            return
        g = self._gen_log

        # --- build/: the body the generator produces (realized = built body, counts) ---
        rate = (self._cur_counts > 0).float().mean(0)      # per-limb realized presence rate
        for i in range(self._n_limbs):
            w.add_scalar(f'build/p/{_LIMB_CODE[i]}', rate[i].item(), frame)
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
            self._log_diversity(w, frame)                  # M1: perplexity metrics (both BC + RL)
        if 'n_distinct_bodies' in g:                       # kept as a saturating canary (M1)
            w.add_scalar('build/n_distinct', g['n_distinct_bodies'], frame)
        # realized TYPE mix (5a collapse canary: all-canonical => the vocabulary is unused)
        for k, val in g.get('types', {}).items():
            w.add_scalar(f'build/type/{k}', val, frame)

        # --- gen/: GenAct (generator actor) learning ---
        w.add_scalar('gen/actor_loss', g['gen_pg'], frame)
        w.add_scalar('gen/entropy', g['ent'], frame)
        w.add_scalar('gen/grad_norm', g['gn'], frame)
        w.add_scalar('gen/fraction', self._gen_fraction(), frame)
        if 'marg' in g:                                    # per-limb marginal value (RL phase only)
            for i in range(self._n_limbs):
                m = g['marg'][i].item()
                if m == m:                                 # skip NaN slots (no `on` this window)
                    w.add_scalar(f'gen/marg/{_LIMB_CODE[i]}', m, frame)

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
        w.add_scalar('clone/repr_anchor', g['anchor'], frame)

        self._gen_log = None

    @torch.no_grad()
    def _log_diversity(self, w, frame):
        """build/* perplexity diversity from the current window's generator trace (native/typed
        view). Replaces the saturating build/n_distinct (M1) -- a body count hard-capped at the
        sample size -- with perplexities that have no ln(M) rail:
          mean_limb_diversity = mean_n exp H(L_n)   within-limb commitment (effective # limb designs)
          limb_diversity/<n>  = exp H(L_n)          per compass slot
          body_diversity      = exp H(B)            effective # distinct bodies (Rao-Blackwell over
                                                    the generator's OWN step entropy, not a count)
          body_structure      = C / sum_n H(L_n)    scale-free cross-limb redundancy (0 = independent
                                                    limb-lotteries, >0 = correlated body plans)
        Typed reprs pair with the FULL step_entropy (== step_entropy_cat + step_entropy_sub), so the
        joint-entropy term is alphabet-matched to the repr and C is not inflated (M3b)."""
        tr = self._cur_trace
        reprs = population_to_repr(tr['counts'].detach().cpu().numpy().astype(int),
                                   tr['eff_sub'].detach().cpu().numpy().astype(int),
                                   tr['cap_sub'].detach().cpu().numpy().astype(int))
        h_body = rao_blackwell_h_body(tr['step_entropy'].detach().cpu().numpy(),
                                      tr['active_step'].detach().cpu().numpy())
        red = redundancy(reprs, h_body)                    # N_body, C_nats, N_limb, H_within_sum, ...
        w.add_scalar('build/mean_limb_diversity', red['N_limb_mean'], frame)
        w.add_scalar('build/body_diversity', red['N_body'], frame)
        sumH = red['H_within_sum']
        w.add_scalar('build/body_structure', red['C_nats'] / sumH if sumH > 1e-9 else 0.0, frame)
        for i in range(self._n_limbs):
            w.add_scalar(f'build/limb_diversity/{_LIMB_CODE[i]}', float(red['N_limb'][i]), frame)

    # ---- checkpointing (gen heads live on self.model -> saved by base) --------------
    def get_full_state_weights(self):
        s = super().get_full_state_weights()
        s.update(gen_window=self._gen_window, cur_counts=self._cur_counts,
                 cur_eff=self._cur_eff, cur_cap=self._cur_cap,
                 cur_trace=self._cur_trace, steps_since_resample=self._steps_since_resample)
        return s

    def set_full_state_weights(self, weights, set_epoch=True):
        super().set_full_state_weights(weights, set_epoch=set_epoch)
        if 'gen_window' not in weights:
            return
        self._gen_window = int(weights['gen_window'])
        self._cur_counts = weights['cur_counts'].to(self.ppo_device)
        if 'cur_eff' in weights:
            self._cur_eff = weights['cur_eff'].to(self.ppo_device)
            self._cur_cap = weights['cur_cap'].to(self.ppo_device)
        tr = weights.get('cur_trace')
        self._cur_trace = {k: v.to(self.ppo_device) for k, v in tr.items()} if tr else None
        self._steps_since_resample = int(weights.get('steps_since_resample', 0))
