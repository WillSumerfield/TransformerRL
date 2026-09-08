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
env_step. W_i is the same window's mean reward per env-STEP, episode structure ignored -- logged
beside it as quality/Window_Rew_Mean, never used as a training target. Requires codesign_tokens (the
gen/critic heads). Supersedes the SeqGenerator path."""
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
from .generator_credit import (
    build_action_module_mapping,
    compute_credit_diagnostics,
    save_credit_artifact,
)
from .post_adaptation_eval import (
    run_post_adaptation_eval,
    compute_adaptation_gap_diagnostics,
    save_post_eval_artifact,
)
from .spatial_credit import (
    propagate_tree_credit,
    compute_spatial_credit_diagnostics,
)
from .counterfactual_pairs import (
    encode_canonical_morphology,
    find_exact_matched_pairs,
    compute_pair_difference_loss,
    compute_pair_diagnostics,
)
# perplexity diversity estimators for build/* logging (M1); pure-numpy, no transformer_rl dep
from experiments.harness.committance import (population_to_repr, redundancy, rao_blackwell_h_body,
                                      modes_and_spread)

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
        # not a second derivation of it. A limb is a chain of up to _max_len modules; `mask` is a
        # per-module field, looked up by name in its group rather than by a remembered offset.
        self._n_dof = net.tdims['n_modules']
        self._mask_off = net.tdims['module']['mask']['off']
        # The ACTION is wider than the DOF mask on a world-mounted task: one scalar per module slot
        # plus one per root axis (ADR-0019). Equal on Ant, which has no root axes. Anything shaped
        # like an action (mu, log_std) uses _n_act; anything indexing the design MDP over module
        # slots stays on _n_dof, since a root axis is never designed.
        self._n_act = self._n_dof + net.tdims['n_root_axes']
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
        self._fixed = False                                # fixed-morphology phase (see _maybe_resample)
        # The bodies the sim currently HOLDS, and the ones the window that just closed was measured
        # on -- exactly the lists handed to setup/resample, not a re-decode of _cur_*. `_cur_*` is
        # the NEXT window's design by the time a window closes (the documented build/* half-step),
        # so reading it as "what the returns were earned on" attributes every window's reward to the
        # bodies that had not run yet. Set by the Algorithm at _start; swapped here.
        self._built_morphs = None
        self._ran_morphs = None
        # Only the package driver reads (and clears) the Task's Episode return. Under the training
        # SCRIPT nobody does, and carrying the accumulator across rebuilds there would freeze it at
        # the first episode forever instead of letting each rebuild reset it. Armed by
        # `CodesignAlgorithm._start`, so the script path stays exactly as it was.
        self._carry_ep_returns = False
        self._base_draw = None                             # last base+-flip ramp draws (counts, pretrain)
        self._gen_window = 0
        self._gen_log = None
        self._last_R = None
        self._post_eval_enabled = bool(cd.get('post_eval_enabled', True))
        self._post_eval_steps = int(cd.get('post_eval_steps', 0))   # 0 => full max_episode_length
        self._return_target = str(cd.get('return_target', 'train')).lower()
        assert self._return_target in ('train', 'post'), f"Unknown return_target: {self._return_target}"
        self._post_eval_action_mode = str(cd.get('post_eval_action_mode', 'deterministic')).lower()
        self._post_eval_log = None
        self._freeze_generator = bool(cd.get('freeze_generator', False))

        # Optional pre-specified initial morphologies (e.g. for common-controller evaluation)
        init_morphs_path = cd.get('initial_morphologies_npz', None)
        if init_morphs_path and os.path.exists(init_morphs_path):
            d_init = np.load(init_morphs_path, allow_pickle=True)
            self._cur_counts = torch.as_tensor(d_init['counts'], dtype=torch.long, device=dev)
            self._cur_eff = torch.as_tensor(d_init['eff_sub'], dtype=torch.long, device=dev)
            self._cur_cap = torch.as_tensor(d_init['cap_sub'], dtype=torch.long, device=dev)
            env = self._env()
            if env is not None:
                env.set_next(self._cur_counts, self._cur_eff, self._cur_cap)
                env.resample()
                print(f"[CodesignAgent] Loaded initial morphology set from {init_morphs_path}: "
                      f"{self._cur_counts.shape[0]} bodies active across envs!", flush=True)

        # Spatial credit head (additive contextual module credit)
        sc = cd.get('spatial_credit', {})
        self._spatial_credit_enabled = bool(sc.get('enabled', False))
        self._spatial_loss_coef = float(sc.get('loss_coef', 0.1))
        self._spatial_tree_lambda = float(sc.get('tree_lambda', 0.5))
        self._pair_supervision_enabled = bool(sc.get('pair_supervision', False))
        self._pair_loss_coef = float(sc.get('pair_loss_coef', 0.1))
        self._pair_batch_size = int(sc.get('pair_batch_size', 512))

        # Credit mode: none | aligned | shuffled | centered_aligned | centered_within_body_shuffled
        #              | body_mean | mean_plus_aligned_residual | mean_plus_shuffled_residual
        #              | direct_body_rpost
        raw_mode = str(sc.get('genact_credit_mode', '')).lower()
        if raw_mode:
            allowed_modes = (
                'none', 'aligned', 'shuffled', 'centered_aligned', 'centered_within_body_shuffled',
                'body_mean', 'mean_plus_aligned_residual', 'mean_plus_shuffled_residual',
                'direct_body_rpost'
            )
            assert raw_mode in allowed_modes, f"Unknown genact_credit_mode: {raw_mode}. Allowed: {allowed_modes}"
            self._genact_credit_mode = raw_mode
            self._use_for_genact = (raw_mode != 'none')
        else:
            self._use_for_genact = bool(sc.get('use_for_genact', False))
            self._genact_credit_mode = 'aligned' if self._use_for_genact else 'none'

        self._genact_beta = float(sc.get('genact_beta', 0.5))
        if self._use_for_genact and self._genact_credit_mode != 'direct_body_rpost':
            self._spatial_credit_enabled = True
        self._spatial_credit_log = None
        self._pair_credit_log = None
        self._oos_pair_credit_log = None
        self._genact_adv_log = None
        self._cur_R_post = None

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
        # W_i accumulator: mean raw reward per env-STEP over the window, episode structure ignored.
        # The companion to R_i, and the two answer different questions. R_i averages over COMPLETED
        # episodes, so it weights every episode equally and DISCARDS the one still in flight at the
        # boundary -- which, since episode length ramps from ~65 to ~850 steps across a window, is
        # systematically the best one. W_i weights every STEP equally and drops nothing.
        # Divided by the step count rather than left as a window TOTAL: the window is
        # ceil(interval * max_episode_length / horizon_length) * horizon_length steps == 1000/1008/
        # 1024 at horizon_length 8/16/32, so a total would hand h=32 a free 2.4% and make the metric
        # illegal to sweep h against. The count is accumulated, not derived, so it stays right on a
        # checkpoint resume that lands mid-window.
        self._win_r_sum = torch.zeros(N, device=dev)
        self._win_n_steps = 0
        # control/r_step accumulator: raw reward per env-step over the EPOCH. Kept on-device and
        # flushed once per epoch in write_stats, so it costs one sync per epoch and none per step.
        self._epoch_r_sum = torch.zeros((), device=dev)
        self._epoch_r_n = 0

    def _net(self):
        return self.model.a2c_network.net

    def _run_unit(self) -> int:
        """One `Algorithm.run()` == one generator window. `_gen_window` already counts exactly the
        resamples the base class's `_resample_count` would, and it is the counter every generator
        phase decision reads, so there is no second one to keep in step."""
        return self._gen_window

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
        """Must match `models.py`'s forward exactly -- this is the same quantity, recomputed for the
        generator's KL clone. Root axes are fixed per env and always active, so the env leaves them
        out of the obs mask; append ones so their sigma is learned rather than gated to 0."""
        mask = (obs[..., self._mask_off:self._mask_off + self._n_dof] > 0).float()
        if self._n_act > self._n_dof:
            mask = torch.cat([mask, mask.new_ones(*mask.shape[:-1], self._n_act - self._n_dof)],
                             dim=-1)
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
        self._win_r_sum += r                                  # -> quality/Window_Rew_Mean (W_i)
        self._win_n_steps += 1
        self._epoch_r_sum += r.mean()                         # -> control/r_step, per-epoch cadence
        self._epoch_r_n += 1
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

    # ---- the two window-boundary seams -----------------------------------------------
    # What a window DOES with its returns, and where the next window's bodies come from. Split out
    # so a baseline that has no generator can replace them without restating the boundary itself --
    # the accumulator resets, the Episode-return carry and the rebuild are the same for every arm
    # and belong in one place (experiments/CONTEXT.md, "Body source").

    def _window_update(self, R, obses):
        """Learn from the window that just ended. R: (N,) body returns, obses: (H,N,obs)."""
        self._resample_update(R, obses)

    @torch.no_grad()
    def _quality_log(self, R, W):
        """The `quality/*` and `build/type/*` half of `_gen_log` -- what the window MEASURED, as
        opposed to what it learnt.

        Separated because every arm measures and only some learn: `_resample_update` fills this in
        passing on its way to the generator's losses, while the fixed-body and random-design phases
        have no losses and would otherwise report nothing. `W` is the per-step window reward, in
        R's units -- diagnostic only, and never a learning target.

        Returned rather than assigned, so the generator path can merge it into the losses it has
        already collected instead of having them overwritten.
        """
        return {'R_var': R.var(unbiased=False).item(),
                'R_mean': R.mean().item(), 'R_std': R.std().item(),
                'W_mean': W.mean().item(), 'W_std': W.std().item(),
                'by_limbcount': self._by_limbcount(R, self._cur_counts),
                'types': self._type_usage()}

    def _next_population(self, N):
        """The next window's `N` bodies: `(trace, counts, eff_sub, cap_sub, morphologies)`.

        `trace` is the generator's sampling record and is what the per-window population dump and
        the intent-side `build/*` metrics are written from; a body source with no generator behind
        it returns `None` and those are legitimately absent rather than fabricated.
        """
        trace = self._net().sample(N)
        counts, eff_sub, cap_sub = self._apply_ramp(
            trace['counts'].long(), trace['eff_sub'], trace['cap_sub'])
        return (trace, counts, eff_sub, cap_sub,
                designs_from_arrays(self._ml, counts, eff_sub, cap_sub))

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
            mu_old = obs_flat.new_empty(HN, self._n_act)   # codesign_forward returns full action width
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

        # Pre-update frozen spatial credit forward pass & exact matched counterfactual pairs
        c_spat_pre, pres_pre, C_tree_pre = None, None, None
        matched_pairs = None
        if self._spatial_credit_enabled:
            try:
                with torch.no_grad():
                    ob_eval = self.model.norm_obs(obses[-1])
                    chunk_sz = 1024
                    cs_l, pr_l = [], []
                    for c_i in range(0, N, chunk_sz):
                        ob_c = ob_eval[c_i:c_i + chunk_sz]
                        _, _, cs_c, pr_c, _ = net.spatial_forward(ob_c)
                        cs_l.append(cs_c)
                        pr_l.append(pr_c)
                    c_spat_pre = torch.cat(cs_l, dim=0)
                    pres_pre = torch.cat(pr_l, dim=0)
                    C_tree_pre = propagate_tree_credit(
                        c_spat_pre, pres_pre, n_limbs=_N_LIMBS, max_len=self._max_len,
                        tree_lambda=self._spatial_tree_lambda,
                    )
                if self._pair_supervision_enabled:
                    morphs = encode_canonical_morphology(
                        self._cur_counts, self._cur_eff, self._cur_cap,
                        max_len=self._max_len, n_limbs=_N_LIMBS
                    )
                    matched_pairs = find_exact_matched_pairs(morphs, R, device=dev)
            except Exception as e_pre:
                print(f"[warning] pre-update spatial forward / pair discovery failed: {e_pre}", flush=True)

        pretrain = self._in_pretrain()
        # NOTE: (slots, actions) here is always the BUILT body -- GenCrit's prefix fit must stay on it
        # because R was measured on the body that actually ran. In pretrain, GenAct's BC target is a
        # separate, freshly-drawn TEACHER body (see the epoch loop).
        if pretrain or self._cur_trace is None:
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
            adv_pref_norm = (sel - sel.mean()) / (sel.std() + 1e-8)
            adv[valid] = adv_pref_norm

            if self._use_for_genact:
                if self._genact_credit_mode == 'direct_body_rpost':
                    R_source = getattr(self, '_cur_R_post', None)
                    if R_source is None:
                        R_source = R
                    assert R_source is not None, "direct_body_rpost requires R_post"

                    r_raw_mean = float(R_source.mean().item())
                    r_raw_std = float(R_source.std().item())

                    # 1. Standardize body signal across population once:
                    A_body = (R_source - R_source.mean()) / (R_source.std() + 1e-8)
                    assert A_body.shape[0] == tr['slots'].shape[0], f"A_body shape {A_body.shape} != N bodies {tr['slots'].shape[0]}"

                    # 2. Broadcast to valid generator actions of body b:
                    A_body_broadcast = A_body.unsqueeze(1).expand_as(adv)

                    # Every valid generator construction action receives beta * A_b^body
                    adv[valid] = adv[valid] + self._genact_beta * A_body_broadcast[valid]

                    # Diagnostics
                    if hasattr(self, '_cur_counts') and self._cur_counts is not None:
                        counts_tensor = torch.as_tensor(self._cur_counts, device=dev)
                        mod_count = counts_tensor.sum(dim=1).float()
                    else:
                        mod_count = valid.sum(dim=1).float()

                    actions_per_body = valid.sum(dim=1).float()
                    body_mass_abs = actions_per_body * (self._genact_beta * A_body.abs())

                    corr_mod_mass = 0.0
                    if mod_count.std() > 1e-8 and body_mass_abs.std() > 1e-8:
                        corr_mod_mass = float(torch.corrcoef(torch.stack([mod_count, body_mass_abs]))[0, 1].item())

                    corr_mod_actions = 0.0
                    if mod_count.std() > 1e-8 and actions_per_body.std() > 1e-8:
                        corr_mod_actions = float(torch.corrcoef(torch.stack([mod_count, actions_per_body]))[0, 1].item())

                    adv_body_corr = float(torch.corrcoef(torch.stack([A_body_broadcast[valid], adv[valid]]))[0, 1].item())

                    self._genact_adv_log = {
                        'raw_r_post_mean': r_raw_mean,
                        'raw_r_post_std': r_raw_std,
                        'body_adv_mean': float(A_body.mean().item()),
                        'body_adv_std': float(A_body.std().item()),
                        'corr_modcount_body_mass': corr_mod_mass,
                        'corr_modcount_actions': corr_mod_actions,
                        'adv_prefix_mean': adv_pref_norm.mean().item(),
                        'adv_prefix_std': adv_pref_norm.std().item(),
                        'adv_combined_mean': adv[valid].mean().item(),
                        'adv_combined_std': adv[valid].std().item(),
                        'adv_valid_corr': adv_body_corr,
                        'genact_beta': float(self._genact_beta),
                        'tree_valid_fraction': 1.0,
                    }
                    print(f"[resample #{self._gen_window} | GenAct Adv (direct_body_rpost)] "
                          f"R_post raw: mean={r_raw_mean:.4f}, std={r_raw_std:.4f} | "
                          f"A_body: mean={A_body.mean().item():.4f}, std={A_body.std().item():.4f} | "
                          f"Corr(N_mod, body_mass)={corr_mod_mass:+.4f}, Corr(N_mod, actions)={corr_mod_actions:+.4f} | "
                          f"Adv combined std={self._genact_adv_log['adv_combined_std']:.4f}, "
                          f"Adv-Body corr={adv_body_corr:+.4f}",
                          flush=True)
                elif C_tree_pre is not None:
                    mapping_data = build_action_module_mapping(tr)
                    depth_hist = torch.as_tensor(mapping_data['depth_hist'], device=slots.device, dtype=torch.long)
                    tok_slot = depth_hist * _N_LIMBS + slots
                    is_eff = (cat_a == GEN_EFF) & valid
                    act_pres = pres_pre.gather(1, tok_slot.clamp(0, 31))
                    valid_tree = is_eff & (act_pres > 0)

                    action_tree = C_tree_pre.gather(1, tok_slot.clamp(0, 31)).detach()
                    assert action_tree.requires_grad is False
                    if valid_tree.any():
                        mode = self._genact_credit_mode
                        is_decomp = mode in ('body_mean', 'mean_plus_aligned_residual', 'mean_plus_shuffled_residual')
                        is_centered = mode in ('centered_aligned', 'centered_within_body_shuffled')
                        is_within_body_shuffled = (mode == 'centered_within_body_shuffled')
                        is_global_shuffled = (mode == 'shuffled')

                        N_bodies, L_steps = valid_tree.shape
                        tree_vals = action_tree[valid_tree]

                        # Per-body tracking
                        body_raw_means = []
                        body_centered_means = []
                        body_centered_stds = []
                        body_residuals_all = []

                        if is_decomp:
                            # Exact decomposition: C_i^tree = mu_b + delta_i
                            S_action_tree = torch.zeros_like(action_tree)
                            base_seed = int(getattr(self, 'seed', 42)) + int(self._gen_window) * 10007
                            aligned_residuals = []
                            shuffled_residuals = []

                            for b in range(N_bodies):
                                b_mask = valid_tree[b]
                                n_b = b_mask.sum().item()
                                if n_b > 0:
                                    b_vals = action_tree[b, b_mask]
                                    mu_b = b_vals.mean()
                                    delta_b = b_vals - mu_b

                                    body_raw_means.append(mu_b.item())
                                    body_centered_means.append(delta_b.mean().item())
                                    body_centered_stds.append(delta_b.std().item() if n_b > 1 else 0.0)

                                    if mode == 'body_mean':
                                        S_action_tree[b, b_mask] = mu_b
                                    elif mode == 'mean_plus_aligned_residual':
                                        S_action_tree[b, b_mask] = mu_b + delta_b
                                    elif mode == 'mean_plus_shuffled_residual':
                                        if n_b > 1:
                                            rng_b = torch.Generator(device=dev)
                                            rng_b.manual_seed(base_seed + b * 31)
                                            perm_b = torch.randperm(n_b, generator=rng_b, device=dev)
                                            delta_shuf = delta_b[perm_b]
                                            S_action_tree[b, b_mask] = mu_b + delta_shuf

                                            aligned_residuals.append(delta_b)
                                            shuffled_residuals.append(delta_shuf)

                                            # Invariance check for residual within body
                                            assert torch.allclose(delta_shuf.mean(), delta_b.mean(), atol=1e-4)
                                            assert torch.allclose(delta_shuf.std(), delta_b.std(), atol=1e-4)
                                            assert torch.allclose(delta_shuf.min(), delta_b.min(), atol=1e-4)
                                            assert torch.allclose(delta_shuf.max(), delta_b.max(), atol=1e-4)
                                        else:
                                            S_action_tree[b, b_mask] = mu_b + delta_b

                            S_vals = S_action_tree[valid_tree]
                            # Verify sum_i delta_i approx 0 for every body
                            for b_mean in body_centered_means:
                                assert abs(b_mean) < 1e-4, f"Body residual mean not zero: {b_mean}"

                            if mode == 'mean_plus_aligned_residual':
                                # Numerically reproduces uncentred aligned implementation
                                assert torch.allclose(S_vals, tree_vals, atol=1e-5)
                                shuffle_corr = 1.0
                            elif mode == 'mean_plus_shuffled_residual' and aligned_residuals:
                                all_al = torch.cat(aligned_residuals)
                                all_sh = torch.cat(shuffled_residuals)
                                shuffle_corr = float(torch.corrcoef(torch.stack([all_al, all_sh]))[0, 1].item())
                            else:
                                shuffle_corr = 1.0

                            # Global standardization over valid generator actions
                            tree_norm = (S_vals - S_vals.mean()) / (S_vals.std() + 1e-8)
                            tree_signal = tree_norm

                        elif is_centered:
                            # Compute body-level mean mu_b over valid credited actions only
                            centered_action_tree = torch.zeros_like(action_tree)
                            shuffled_action_tree = torch.zeros_like(action_tree)
                            base_seed = int(getattr(self, 'seed', 42)) + int(self._gen_window) * 10007

                            for b in range(N_bodies):
                                b_mask = valid_tree[b]
                                n_b = b_mask.sum().item()
                                if n_b > 0:
                                    b_vals = action_tree[b, b_mask]
                                    mu_b = b_vals.mean()
                                    c_b = b_vals - mu_b
                                    centered_action_tree[b, b_mask] = c_b
                                    body_raw_means.append(mu_b.item())
                                    body_centered_means.append(c_b.mean().item())
                                    body_centered_stds.append(c_b.std().item() if n_b > 1 else 0.0)

                                    if is_within_body_shuffled:
                                        if n_b > 1:
                                            rng_b = torch.Generator(device=dev)
                                            rng_b.manual_seed(base_seed + b * 31)
                                            perm_b = torch.randperm(n_b, generator=rng_b, device=dev)
                                            s_b = c_b[perm_b]
                                            shuffled_action_tree[b, b_mask] = s_b

                                            # Invariance check within body
                                            assert torch.allclose(s_b.mean(), c_b.mean(), atol=1e-4)
                                            assert torch.allclose(s_b.std(), c_b.std(), atol=1e-4)
                                            assert torch.allclose(s_b.min(), c_b.min(), atol=1e-4)
                                            assert torch.allclose(s_b.max(), c_b.max(), atol=1e-4)
                                        else:
                                            shuffled_action_tree[b, b_mask] = c_b

                            base_credit_vals = centered_action_tree[valid_tree]
                            for b_mean in body_centered_means:
                                assert abs(b_mean) < 1e-4, f"Body centered mean not zero: {b_mean}"

                            if is_within_body_shuffled:
                                target_credit_vals = shuffled_action_tree[valid_tree]
                                assert torch.allclose(target_credit_vals.mean(), base_credit_vals.mean(), atol=1e-4)
                                assert torch.allclose(target_credit_vals.std(), base_credit_vals.std(), atol=1e-4)
                                assert torch.allclose(target_credit_vals.min(), base_credit_vals.min(), atol=1e-4)
                                assert torch.allclose(target_credit_vals.max(), base_credit_vals.max(), atol=1e-4)
                                assert target_credit_vals.numel() == base_credit_vals.numel()
                                shuffle_corr = float(torch.corrcoef(torch.stack([base_credit_vals, target_credit_vals]))[0, 1].item())
                            else:
                                target_credit_vals = base_credit_vals
                                shuffle_corr = 1.0

                            # Global standardization over valid generator actions
                            tree_norm = (target_credit_vals - target_credit_vals.mean()) / (target_credit_vals.std() + 1e-8)
                            tree_signal = tree_norm

                        elif is_global_shuffled:
                            # Global uncentered shuffle
                            tree_norm = (tree_vals - tree_vals.mean()) / (tree_vals.std() + 1e-8)
                            perm_seed = int(getattr(self, 'seed', 42)) + int(self._gen_window) * 10007
                            rng = torch.Generator(device=dev)
                            rng.manual_seed(perm_seed)
                            perm = torch.randperm(tree_norm.numel(), generator=rng, device=dev)
                            tree_signal = tree_norm[perm]

                            assert torch.allclose(tree_signal.mean(), tree_norm.mean(), atol=1e-4)
                            assert torch.allclose(tree_signal.std(), tree_norm.std(), atol=1e-4)
                            assert torch.allclose(tree_signal.min(), tree_norm.min(), atol=1e-4)
                            assert torch.allclose(tree_signal.max(), tree_norm.max(), atol=1e-4)
                            assert tree_signal.numel() == tree_norm.numel()
                            shuffle_corr = float(torch.corrcoef(torch.stack([tree_norm, tree_signal]))[0, 1].item())
                        else:
                            # Existing uncentered aligned
                            tree_norm = (tree_vals - tree_vals.mean()) / (tree_vals.std() + 1e-8)
                            tree_signal = tree_norm
                            shuffle_corr = 1.0

                        adv[valid_tree] = adv[valid_tree] + self._genact_beta * tree_signal

                        std_prefix = sel.std().item()
                        std_tree_raw = tree_vals.std().item()
                        raw_std_ratio = (self._genact_beta * std_tree_raw) / max(1e-8, std_prefix)

                        adv_valid_corr = float(torch.corrcoef(torch.stack([tree_signal, adv[valid_tree]]))[0, 1].item())

                        raw_mean_val = float(np.mean(body_raw_means)) if body_raw_means else tree_vals.mean().item()
                        cent_mean_val = float(np.mean(body_centered_means)) if body_centered_means else 0.0
                        cent_std_val = float(np.mean(body_centered_stds)) if body_centered_stds else 0.0
                        raw_std_val = float(np.std(body_raw_means)) if body_raw_means else 0.0
                        rel_ratio_val = raw_std_val / (cent_std_val + 1e-8)

                        self._genact_adv_log = {
                            'adv_prefix_mean': adv_pref_norm.mean().item(),
                            'adv_prefix_std': adv_pref_norm.std().item(),
                            'adv_tree_raw_mean': tree_vals.mean().item(),
                            'adv_tree_raw_std': std_tree_raw,
                            'adv_tree_norm_mean': tree_signal.mean().item(),
                            'adv_tree_norm_std': tree_signal.std().item(),
                            'adv_combined_mean': adv[valid].mean().item(),
                            'adv_combined_std': adv[valid].std().item(),
                            'tree_to_prefix_std_ratio_raw': raw_std_ratio,
                            'tree_to_prefix_std_ratio_norm': float(self._genact_beta),
                            'tree_valid_fraction': (valid_tree.sum().float() / valid.sum().clamp(min=1).float()).item(),
                            'shuffle_corr': shuffle_corr,
                            'adv_valid_corr': adv_valid_corr,
                            'mode_is_shuffled': 1.0 if ('shuffled' in self._genact_credit_mode) else 0.0,
                            'mode_is_centered': 1.0 if is_centered else 0.0,
                            'body_raw_mean': raw_mean_val,
                            'body_raw_std': raw_std_val,
                            'body_centered_mean': cent_mean_val,
                            'body_centered_std': cent_std_val,
                            'body_mu_to_delta_ratio': rel_ratio_val,
                        }
                        print(f"[resample #{self._gen_window} | GenAct Adv ({self._genact_credit_mode})] "
                              f"Prefix std={std_prefix:.4f}, Tree std={std_tree_raw:.4f}, "
                              f"Raw ratio={raw_std_ratio:.4f}, Valid frac={self._genact_adv_log['tree_valid_fraction']:.3f}, "
                              f"Shuffle corr={shuffle_corr:+.4f}, Adv-Tree corr={adv_valid_corr:+.4f}"
                              + (f", mu_b={raw_mean_val:.4f} (std={raw_std_val:.4f}), delta std={cent_std_val:.4f}, ratio={rel_ratio_val:.2f}" if (is_decomp or is_centered) else ""),
                              flush=True)

        # Out-of-sample pair evaluation (evaluated on new pairs before training on them)
        if (self._spatial_credit_enabled and matched_pairs is not None
                and matched_pairs['idx_A'].shape[0] > 0 and c_spat_pre is not None):
            try:
                pref_grid = None
                if not pretrain and self._cur_trace is not None:
                    raw_adv_eval = self._cur_trace['v_states'][:, 1:] - self._cur_trace['v_states'][:, :-1]
                    val_eval = self._cur_trace['active_step']
                    tr_slots = self._cur_trace['slots']
                    mapping_eval = build_action_module_mapping(self._cur_trace)
                    d_hist = torch.as_tensor(mapping_eval['depth_hist'], device=tr_slots.device, dtype=torch.long)
                    t_slot = d_hist * _N_LIMBS + tr_slots
                    pref_grid = torch.zeros_like(c_spat_pre)
                    pref_grid.scatter_(1, t_slot.clamp(0, 31), torch.where(val_eval, raw_adv_eval, torch.zeros_like(raw_adv_eval)))

                oos_diags = compute_pair_diagnostics(
                    c_spat_pre, C_tree_pre, matched_pairs, prefix_delta=pref_grid
                )
                self._oos_pair_credit_log = oos_diags
                t_r = oos_diags.get('tree_diff_pearson', 0.0)
                t_rho = oos_diags.get('tree_diff_spearman', 0.0)
                t_ev = oos_diags.get('tree_diff_ev', 0.0)
                p_r = oos_diags.get('prefix_diff_pearson', float('nan'))
                p_rho = oos_diags.get('prefix_diff_spearman', float('nan'))
                p_cnt = matched_pairs['idx_A'].shape[0]
                print(f"[resample #{self._gen_window} | OOS Pair Eval] "
                      f"Tree r={t_r:+.4f}, rho={t_rho:+.4f}, EV={t_ev:+.4f} | "
                      f"Prefix r={p_r:+.4f}, rho={p_rho:+.4f} | Pairs={p_cnt}", flush=True)
            except Exception as e_oos:
                print(f"[warning] out-of-sample pair evaluation failed: {e_oos}", flush=True)

        L1 = self._n_dof + 1                               # (L+1) prefixes gen_replay stacks per env
        mb_size = max(1, min(N // self._gen_minibatches, self._gen_max_prefixes // L1))
        logs = {k: [] for k in ('gen_pg', 'ent', 'v_prefix', 'v_roll', 'kl', 'crit', 'gn', 'anchor', 'v_spatial', 'pair_loss')}

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
                N_ctrl = min(1024, N)
                if anchor_on or self._spatial_credit_enabled:
                    j = torch.randint(0, S, (N_ctrl,), device=dev) if anchor_on else torch.randint(0, HN, (N_ctrl,), device=dev)
                    ridx = sub[j] if anchor_on else j
                    ob = obs_flat[ridx]
                    mu_n, v098_n, v1_n, H_new = net.codesign_forward(self.model.norm_obs(ob),
                                                                     return_hidden=True)
                    anchor = (1 - F.cosine_similarity(H_new, H_old[j].float(), dim=-1)).mean() if anchor_on else ob.new_zeros(())
                else:
                    ridx = torch.randint(0, HN, (N_ctrl,), device=dev)
                    ob = obs_flat[ridx]
                    mu_n, v098_n, v1_n = net.codesign_forward(self.model.norm_obs(ob))
                    anchor = ob.new_zeros(())
                    H_new = None
                kl = _gauss_kl(mu_old[ridx], ls_old[ridx], mu_n, self._log_std(ob))
                crit = (v098_n - v098_old[ridx]).pow(2).mean()
                v_roll = (v1_n.squeeze(-1) - R_roll[ridx]).pow(2).mean()

                if self._spatial_credit_enabled and H_new is not None:
                    v_spat, v_glob, c_spat, pres = net.spatial_from_hidden(H_new, ob)
                    v_spatial_loss = (v_spat - R_roll[ridx]).pow(2).mean()
                else:
                    v_spatial_loss = ob.new_zeros(())

                # Matched structural counterfactual supervision
                if (self._spatial_credit_enabled and self._pair_supervision_enabled
                        and matched_pairs is not None and matched_pairs['idx_A'].shape[0] > 0):
                    n_p = matched_pairs['idx_A'].shape[0]
                    p_sub = torch.randint(0, n_p, (min(self._pair_batch_size, n_p),), device=dev)
                    mb_p_A = matched_pairs['idx_A'][p_sub]
                    mb_p_B = matched_pairs['idx_B'][p_sub]
                    u_envs = torch.unique(torch.cat([mb_p_A, mb_p_B]))
                    ob_u = self.model.norm_obs(obses[-1, u_envs])
                    _, _, c_u, pres_u, _ = net.spatial_forward(ob_u)
                    C_tree_u = propagate_tree_credit(
                        c_u, pres_u, n_limbs=_N_LIMBS, max_len=self._max_len,
                        tree_lambda=self._spatial_tree_lambda,
                    )
                    lookup = torch.full((N,), -1, dtype=torch.long, device=dev)
                    lookup[u_envs] = torch.arange(len(u_envs), device=dev)
                    pair_batch_mapped = {
                        'idx_A': lookup[mb_p_A],
                        'idx_B': lookup[mb_p_B],
                        'slot': matched_pairs['slot'][p_sub],
                        'is_subtree': matched_pairs['is_subtree'][p_sub],
                        'delta_R': matched_pairs['delta_R'][p_sub],
                    }
                    pair_loss, _ = compute_pair_difference_loss(c_u, C_tree_u, pair_batch_mapped)
                else:
                    pair_loss = ob.new_zeros(())

                # entropy is an RL-only term: pretrain is supervised (gen_pg is plain NLL) and the
                # TEACHER is the entropy source, so a bonus here only fights the fit -- at 0.1 it
                # drove the token policy to 86% of ln2 (near coin-flip), i.e. geometric limb
                # lengths, and cut the parts teacher's 87% stable bodies down to 41%.
                ent_coef = 0.0 if pretrain else self._gen_ent
                loss = (gen_pg - ent_coef * ent
                        + self._gencrit_coef * (v_prefix + v_roll)
                        + self._beta * kl + self._lam * crit
                        + self._jepa_anchor_coef * anchor
                        + self._spatial_loss_coef * (v_spatial_loss + self._pair_loss_coef * pair_loss))
                self.optimizer.zero_grad()
                loss.backward()
                logs['gn'].append(clip_grad_norm_(self.model.parameters(), self.grad_norm))
                self.optimizer.step()
                for k, val in (('gen_pg', gen_pg), ('ent', ent), ('v_prefix', v_prefix),
                               ('v_roll', v_roll), ('kl', kl), ('crit', crit), ('anchor', anchor),
                               ('v_spatial', v_spatial_loss), ('pair_loss', pair_loss)):
                    logs[k].append(val.detach())

        self._gen_log = {k: torch.stack(v).mean().item() for k, v in logs.items()}
        # body-quality outcome (the optimization target); R aligns with the current window's
        # realized bodies (_cur_counts), before _maybe_resample samples the next window.
        self._gen_log['R_var'] = R.var(unbiased=False).item()    # scale for the GenCrit vloss norm
        self._gen_log['R_mean'] = R.mean().item()
        self._gen_log['R_std'] = R.std().item()
        R_sorted = torch.sort(R)[0]
        n_top10 = max(1, int(0.1 * len(R)))
        self._gen_log['R_max'] = R_sorted[-1].item()
        self._gen_log['R_top10_mean'] = R_sorted[-n_top10:].mean().item()
        if self._cur_trace is not None and 'cat_actions' in self._cur_trace:
            tr_cat = self._cur_trace['cat_actions']
            tr_val = self._cur_trace['active_step']
            val_cnt = tr_val.sum().clamp(min=1).float()
            self._gen_log['action_eff_prob'] = (((tr_cat == GEN_EFF) & tr_val).sum().float() / val_cnt).item()
            self._gen_log['action_cap_prob'] = (((tr_cat == GEN_CAP) & tr_val).sum().float() / val_cnt).item()

        # Complexity collapse diagnostics
        cur_mod_counts = self._cur_counts.sum(1)          # (N,)
        cur_limb_counts = (self._cur_counts > 0).sum(1)    # (N,)
        cur_mean_depth = cur_mod_counts.float() / cur_limb_counts.clamp(min=1).float()
        present_limbs = (self._cur_counts > 0)
        frac_limbs_max_depth = ((self._cur_counts == self._max_len) & present_limbs).sum().float() / present_limbs.sum().clamp(min=1).float()
        max_possible_modules = _N_LIMBS * self._max_len   # 32
        frac_bodies_max_mod = (cur_mod_counts == max_possible_modules).float().mean()
        frac_bodies_top10_mod = (cur_mod_counts >= int(0.9 * max_possible_modules)).float().mean()

        self._gen_log['mean_depth'] = cur_mean_depth.mean().item()
        self._gen_log['frac_limbs_max_depth'] = frac_limbs_max_depth.item()
        self._gen_log['frac_bodies_max_modules'] = frac_bodies_max_mod.item()
        self._gen_log['frac_bodies_top10_complexity'] = frac_bodies_top10_mod.item()

        # E[R | N_modules] and E[R | depth]
        R_by_mod = {}
        for mc in cur_mod_counts.unique():
            R_by_mod[int(mc.item())] = R[cur_mod_counts == mc].mean().item()
        self._gen_log['by_modcount'] = R_by_mod

        R_by_dep = {}
        rounded_depth = torch.round(cur_mean_depth).long().clamp(0, self._max_len)
        for dp in rounded_depth.unique():
            R_by_dep[int(dp.item())] = R[rounded_depth == dp].mean().item()
        self._gen_log['by_depth'] = R_by_dep

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

        if self._cur_trace is not None:
            try:
                mapping_data = build_action_module_mapping(self._cur_trace)
                credit_scalars, flat_records = compute_credit_diagnostics(
                    mapping_data, self._cur_trace, R, adv=adv, raw_adv=raw_adv
                )

                if self._spatial_credit_enabled:
                    try:
                        with torch.no_grad():
                            ob_eval = self.model.norm_obs(obses[-1])
                            chunk_sz = 1024
                            vs_l, vg_l, cs_l, pr_l = [], [], [], []
                            for c_i in range(0, ob_eval.shape[0], chunk_sz):
                                ob_c = ob_eval[c_i:c_i + chunk_sz]
                                vs_c, vg_c, cs_c, pr_c, _ = net.spatial_forward(ob_c)
                                vs_l.append(vs_c)
                                vg_l.append(vg_c)
                                cs_l.append(cs_c)
                                pr_l.append(pr_c)
                            v_spat = torch.cat(vs_l, dim=0)
                            v_glob = torch.cat(vg_l, dim=0)
                            c_spat = torch.cat(cs_l, dim=0)
                            pres = torch.cat(pr_l, dim=0)
                            C_tree = propagate_tree_credit(
                                c_spat, pres, n_limbs=_N_LIMBS, max_len=self._max_len,
                                tree_lambda=self._spatial_tree_lambda,
                            )
                            deltas = raw_adv[valid] if raw_adv is not None else None
                            spat_scalars, spat_records = compute_spatial_credit_diagnostics(
                                c_spat, C_tree, v_glob, v_spat, R, pres,
                                records=mapping_data['records'], delta=deltas,
                            )
                            self._spatial_credit_log = spat_scalars
                            if 'action_spatial_credit' in spat_records:
                                flat_records['spatial_credit'] = spat_records['action_spatial_credit']
                            if 'action_tree_credit' in spat_records:
                                flat_records['tree_credit'] = spat_records['action_tree_credit']

                            if self._pair_supervision_enabled and matched_pairs is not None:
                                pair_scalars = compute_pair_diagnostics(
                                    c_spat, C_tree, matched_pairs
                                )
                                self._pair_credit_log = pair_scalars
                                p_lim = min(8192, matched_pairs['idx_A'].shape[0])
                                flat_records['pair_idx_A'] = matched_pairs['idx_A'][:p_lim].cpu().numpy()
                                flat_records['pair_idx_B'] = matched_pairs['idx_B'][:p_lim].cpu().numpy()
                                flat_records['pair_slot'] = matched_pairs['slot'][:p_lim].cpu().numpy()
                                flat_records['pair_depth'] = matched_pairs['depth'][:p_lim].cpu().numpy()
                                flat_records['pair_limb'] = matched_pairs['limb'][:p_lim].cpu().numpy()
                                flat_records['pair_is_subtree'] = matched_pairs['is_subtree'][:p_lim].cpu().numpy()
                                flat_records['pair_delta_R'] = matched_pairs['delta_R'][:p_lim].cpu().numpy()
                    except Exception as e_spat:
                        print(f"[warning] spatial credit diagnostics failed: {e_spat}", flush=True)

                self._gen_log['credit_scalars'] = credit_scalars
                credit_dir = os.path.join(self.experiment_dir, 'credit')
                artifact_path = os.path.join(credit_dir, f'credit_window_{self._gen_window:04d}.npz')
                save_credit_artifact(
                    artifact_path,
                    flat_records,
                    mapping_data['controller_module_to_action'],
                    metadata={
                        'gen_window': self._gen_window,
                        'epoch': self.epoch_num,
                        'pretrain': pretrain,
                        'spatial_credit_enabled': self._spatial_credit_enabled,
                        'pair_supervision_enabled': self._pair_supervision_enabled,
                    },
                )
            except Exception as e:
                print(f"[warning] generator credit logging failed: {e}", flush=True)

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

        if self._fixed:
            # Fixed-morphology phase. The window boundary still TICKS -- it is what bounds one
            # `Algorithm.run()`, so a fixed run that stopped counting windows would never return to
            # the driver -- but nothing about the body may move: no draw, no joint update (it would
            # fit the designer to a one-body population), and no rebuild, the set it would rebuild
            # to being the set already standing. Episodes are deliberately left running; there is no
            # scene change to reset for, and truncating them every window is noise in the fine-tune.
            self._gen_window += 1
            self._ran_morphs = self._built_morphs      # unchanged: the same set ran and still stands
            # The body did not move, but it still earned a return, and quality/* is the tag every
            # other arm is read on -- a fixed-body baseline that logged nothing there would be the
            # one condition in the comparison with no curve on the axis it is the reference for.
            self._gen_log = self._quality_log(
                self._window_Ri() * self._r_scale,
                self._win_r_sum * (self._r_scale / max(1, self._win_n_steps)))
            self._win_ret_sum.zero_(); self._win_ret_cnt.zero_()
            self._win_r_sum.zero_(); self._win_n_steps = 0
            self._steps_since_resample = 0
            return

        N = env.total_num_envs
        R_train = self._window_Ri() * self._r_scale               # true body return, scaled to control units
        W = self._win_r_sum * (self._r_scale / max(1, self._win_n_steps))   # per-step, R's scale
        # Optional detached capture of the last training rollout, before evaluation/reset.
        probe_cfg = self.config.get('generator', {}).get('response_probe', {})
        if probe_cfg.get('enabled', False):
            from .response_probe import capture_response_probe
            capture_response_probe(
                self, os.path.join(self.experiment_dir, 'response_probe'),
                max_bodies=int(probe_cfg.get('max_bodies', 256)),
                samples=int(probe_cfg.get('samples', 8)))
        R = R_train
        R_post_det = None
        R_post_stoch = None
        R_post_primary = None

        # --- Diagnostic Post-Adaptation Evaluation ---
        if self._post_eval_enabled and self._cur_counts is not None:
            try:
                R_post_det, R_post_stoch = run_post_adaptation_eval(
                    self, eval_steps=self._post_eval_steps, eval_stochastic=True
                )
                R_post_primary = R_post_stoch if self._post_eval_action_mode == 'stochastic' else R_post_det
                self._cur_R_post = R_post_primary
                eval_scalars, eval_records = compute_adaptation_gap_diagnostics(
                    R_train, R_post_primary, self._cur_counts, self._cur_eff, self._cur_cap,
                    R_post_stoch=R_post_stoch
                )
                self._post_eval_log = eval_scalars
                eval_dir = os.path.join(self.experiment_dir, 'post_eval')
                artifact_path = os.path.join(eval_dir, f'post_eval_window_{self._gen_window:04d}.npz')
                save_post_eval_artifact(
                    artifact_path,
                    eval_records,
                    eval_scalars,
                    metadata={
                        'gen_window': self._gen_window,
                        'epoch': self.epoch_num,
                        'pretrain': self._in_pretrain(),
                        'return_target': self._return_target,
                        'action_mode': self._post_eval_action_mode,
                    },
                )
            except Exception as e:
                print(f"[warning] post-adaptation evaluation failed: {e}", flush=True)

        # Configurable GenCrit target: train | post
        if self._return_target == 'post' and R_post_primary is not None:
            R = R_post_primary
        if self._freeze_generator:
            print(f"[CodesignAgent] Generator is frozen (evaluation mode). Post-adaptation eval complete at epoch {self.epoch_num}.", flush=True)
            return
        obses = self.experience_buffer.tensor_dict['obses']  # (H,N,obs) rollout-state sample
        phase = 'pretrain' if self._in_pretrain() else 'rl'  # regime of the update just performed
        self._window_update(R, obses)
        # Diagnostic only -- deliberately NOT fed to GenCrit or the advantage. R is the training
        # target because the value heads regress a per-episode quantity; W is scored, not learned.
        self._gen_log['W_mean'] = W.mean().item()
        self._gen_log['W_std'] = W.std().item()
        self._gen_window += 1
        self._last_R = R

        # The window's generator population (ADR-0021), dumped here because this is the one point
        # where the closing window's trace and its R are both in hand -- which is what sidesteps the
        # half-step between build/* (next window) and the learning metrics (window that ended).
        # `_gen_window` was incremented just above, so the window that ENDED is `_gen_window - 1`.
        # Window 0 ran the seed body and never had a trace, so w0000.npz is legitimately absent and
        # the reader keys off the filename rather than position. Dumped from the trace -- the
        # generator's INTENT, before `_apply_ramp` mixes in teacher draws -- so travel and
        # build/n_modes describe the same object; R and v_full are per-env and pair with either.
        # int16 + compression is not fastidiousness: int64 uncompressed is ~50 MB/run against this
        # ADR's 5 MB budget. Supersedes gen_scatter.npz, whose two fields it contains.
        if self._cur_trace is not None:
            tr = self._cur_trace
            pop_dir = os.path.join(self.experiment_dir, 'gen_pop')
            os.makedirs(pop_dir, exist_ok=True)
            np.savez_compressed(
                os.path.join(pop_dir, f'w{self._gen_window - 1:04d}.npz'),
                counts=tr['counts'].detach().cpu().numpy().astype(np.int16),
                eff_sub=tr['eff_sub'].detach().cpu().numpy().astype(np.int16),
                cap_sub=tr['cap_sub'].detach().cpu().numpy().astype(np.int16),
                R=R.detach().cpu().numpy().astype(np.float32),
                v_full=tr['v_states'][:, -1].detach().cpu().numpy().astype(np.float32))

        trace, counts, eff_sub, cap_sub, morphs = self._next_population(N)
        self._cur_trace = trace
        self._cur_counts, self._cur_eff, self._cur_cap = counts, eff_sub, cap_sub
        # The phase and the ramp fraction are statements about a GENERATOR's schedule; a body
        # source with no generator has neither, and printing them would put a pretrain/RL phase on
        # a run that has no such phases.
        sched = (f"{phase} | next_gen_frac={self._gen_fraction():.2f} | "
                 if trace is not None else "")
        print(f"[resample #{self._gen_window} | {sched}"
              f"epoch {self.epoch_num}] R_mean={R.mean().item():.3f} "
              f"limbcount={(counts > 0).sum(1).float().mean().item():.2f} "
              f"modules={counts.sum(1).float().mean().item():.2f}", flush=True)
        self._ran_morphs = self._built_morphs          # what R was just measured on
        # The Task clears its Episode-return accumulator on rebuild, deliberately: a partial return
        # carried across a resample would score a new body with the old body's reward. But the
        # driver reads Episode return AFTER run() returns and our run boundary IS the rebuild, so
        # the window's FINISHED episodes -- already earned, already attributable -- would be wiped
        # one instant before the only place that reads them, leaving Training return unavailable on
        # every codesign run. Carried across by hand; `drive` takes and clears them immediately.
        carried = ((env._ep_return.clone(), env._ep_done.clone())
                   if self._carry_ep_returns else None)
        env.resample(morphs)
        if carried is not None:
            env._ep_return, env._ep_done = carried
        self._built_morphs = morphs
        self.obs = self.env_reset()
        self.current_rewards.zero_(); self.current_lengths.zero_()
        # reset the R_i / W_i accumulators for the new window
        self._ep_ret.zero_(); self._win_ret_sum.zero_(); self._win_ret_cnt.zero_()
        self._win_r_sum.zero_(); self._win_n_steps = 0
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

        # control/r_step -- mean RAW reward per env-step this epoch. The per-epoch performance
        # signal, and the only one the boundary-recovery trace can be folded on: rl_games'
        # rewards/iter is `game_rewards.get_mean()`, a ring buffer of the last 100 FINISHED episodes
        # pooled over all envs, so it moves only when episodes end. At resample_interval=1 every env
        # truncates at max_episode_length simultaneously -- the same instant as the resample -- so
        # that series' post-boundary shape is which episodes happened to finish (it folds to a 4x
        # excursion that tracks episode_lengths/iter exactly, and would appear identically with no
        # resampling at all). This is unlagged, unshaped, un-bootstrapped, and samples every
        # num_actors x horizon_length step of the epoch. ADR-0021.
        if self._epoch_r_n:
            w.add_scalar('control/r_step', (self._epoch_r_sum / self._epoch_r_n).item(), frame)
            self._epoch_r_sum.zero_()
            self._epoch_r_n = 0
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
            # M1 perplexity metrics log in BOTH phases; the M4 headline (build/n_modes) is RL-only,
            # keyed off the same 'n_distinct_bodies' presence that marks an RL window (set under
            # `if not pretrain` in _resample_update), so the two series always cover the same windows.
            self._log_diversity(w, frame, rl='n_distinct_bodies' in g)
        if 'n_distinct_bodies' in g:                       # kept as a saturating canary (M1)
            w.add_scalar('build/n_distinct', g['n_distinct_bodies'], frame)
        # realized TYPE mix (5a collapse canary: all-canonical => the vocabulary is unused)
        for k, val in g.get('types', {}).items():
            w.add_scalar(f'build/type/{k}', val, frame)

        # --- gen/: GenAct (generator actor) learning ---
        # Guarded on the update having happened at all. A body source with no generator (the
        # random-design baseline) closes windows and earns R like any other run, but has no actor
        # loss, no GenCrit fit and no clone -- and logging zeros for them would put three flat
        # curves in TensorBoard that read as a trained generator sitting at zero.
        if 'gen_pg' in g:
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
        # Mean reward per env-step over the window -- counts every step, including the episode still
        # running at the boundary, and divided by the step count so it does not carry the window
        # length in its units. Invariant to horizon_length, which is what makes it legal to sweep.
        w.add_scalar('quality/Window_Rew_Mean', g['W_mean'], frame)
        w.add_scalar('quality/Window_Rew_Std', g['W_std'], frame)
        if 'R_max' in g:
            w.add_scalar('quality/R_max', g['R_max'], frame)
        if 'R_top10_mean' in g:
            w.add_scalar('quality/R_top10_mean', g['R_top10_mean'], frame)
        if 'action_eff_prob' in g:
            w.add_scalar('gen/action_prob/eff', g['action_eff_prob'], frame)
            w.add_scalar('gen/action_prob/cap', g['action_cap_prob'], frame)
        for k, v in g['by_limbcount'].items():             # does more limbs earn more R?
            w.add_scalar(f'quality/by_limbcount/{k}', v, frame)
        if 'by_modcount' in g:
            for k, v in g['by_modcount'].items():
                w.add_scalar(f'quality/by_modcount/{k}', v, frame)
        if 'by_depth' in g:
            for k, v in g['by_depth'].items():
                w.add_scalar(f'quality/by_depth/{k}', v, frame)
        if 'mean_depth' in g:
            w.add_scalar('build/complexity/mean_depth', g['mean_depth'], frame)
            w.add_scalar('build/complexity/frac_limbs_max_depth', g['frac_limbs_max_depth'], frame)
            w.add_scalar('build/complexity/frac_bodies_max_modules', g['frac_bodies_max_modules'], frame)
            w.add_scalar('build/complexity/frac_bodies_top10_complexity', g['frac_bodies_top10_complexity'], frame)

        # --- codesign/genact/: combined advantage diagnostics ---
        if self._genact_adv_log:
            for k, val in self._genact_adv_log.items():
                if val == val and not np.isinf(val):       # skip NaN / inf
                    w.add_scalar(f'codesign/genact/{k}', val, frame)
            self._genact_adv_log = None

        # --- codesign/credit/: generator credit and advantage diagnostics ---
        if 'credit_scalars' in g:
            for k, val in g['credit_scalars'].items():
                if val == val and not np.isinf(val):       # skip NaN / inf
                    w.add_scalar(k, val, frame)

        # --- codesign/adaptation/: post-adaptation evaluation diagnostics ---
        if self._post_eval_log:
            for k, val in self._post_eval_log.items():
                if val == val and not np.isinf(val):       # skip NaN / inf
                    w.add_scalar(k, val, frame)
            self._post_eval_log = None

        # --- codesign/spatial/: spatial credit diagnostics ---
        if self._spatial_credit_log:
            for k, val in self._spatial_credit_log.items():
                if val == val and not np.isinf(val):       # skip NaN / inf
                    w.add_scalar(k, val, frame)
            self._spatial_credit_log = None

        if self._pair_credit_log:
            for k, val in self._pair_credit_log.items():
                if val == val and not np.isinf(val):       # skip NaN / inf
                    w.add_scalar(f'codesign/spatial/pair/{k}', val, frame)
            self._pair_credit_log = None

        # --- codesign/spatial/pair/oos_*: out-of-sample pair evaluation diagnostics ---
        if self._oos_pair_credit_log:
            for k, val in self._oos_pair_credit_log.items():
                if val == val and not np.isinf(val):       # skip NaN / inf
                    w.add_scalar(f'codesign/spatial/pair/oos_{k}', val, frame)
            self._oos_pair_credit_log = None

        w.add_scalar('codesign/return_target_is_post', 1.0 if self._return_target == 'post' else 0.0, frame)

        # --- clone/: control preservation at resample ---
        if 'kl' in g:
            w.add_scalar('clone/actor_kl', g['kl'], frame)
            w.add_scalar('clone/critic_mse', g['crit'], frame)
            w.add_scalar('clone/repr_anchor', g['anchor'], frame)

        self._gen_log = None

    @torch.no_grad()
    def _log_diversity(self, w, frame, rl):
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
        counts = tr['counts'].detach().cpu().numpy().astype(int)
        eff = tr['eff_sub'].detach().cpu().numpy().astype(int)
        cap = tr['cap_sub'].detach().cpu().numpy().astype(int)
        # M4 FIRST: the diversity HEADLINE (docs/reference/Metrics.md), on the SUBTYPE-COLLAPSED
        # skeleton.
        #   n_modes    = Hill(q=1) over single-linkage d_struct clusters at tau = 1 module
        #   div_struct = mean pairwise d_struct (threshold-free companion)
        # Collapsed because the free_entropy finding is that the skeleton commits while the subtype
        # axis stays FREE: every typed statistic (build/n_distinct, build/body_diversity) therefore
        # stays near its ceiling under total skeleton collapse and cannot detect it. n_modes == 1.0
        # is one design, by construction. This is what the tuner gates on -- ADR-0020.
        # RL-only, exactly matching build/n_distinct, so a tail over the whole series is precisely
        # "all RL windows" whatever n_pretrain is set to, with no arithmetic on the tuner side.
        #
        # ORDER MATTERS, and this block goes first BECAUSE the tuner gates on div_struct. It used to
        # run after the perplexity metrics below, so when _limb_keys crashed on a fully homogeneous
        # population the collapsed window logged no div_struct at all -- the gate could not fire on
        # total collapse, because the run died before reporting it, and "collapsed" and "crashed"
        # became the same observation. That crash is fixed, but the dependency was the real defect:
        # the gate's input must not sit downstream of a richer, more fragile statistic.
        if rl:
            n_modes, spread = modes_and_spread(
                population_to_repr(counts, eff, cap, collapse_subtypes=True))
            w.add_scalar('build/n_modes', n_modes, frame)
            w.add_scalar('build/div_struct', spread, frame)

        reprs = population_to_repr(counts, eff, cap)
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
        has_sub = 'cur_eff' in weights
        if has_sub:
            self._cur_eff = weights['cur_eff'].to(self.ppo_device)
            self._cur_cap = weights['cur_cap'].to(self.ppo_device)
        tr = weights.get('cur_trace')
        self._cur_trace = {k: v.to(self.ppo_device) for k, v in tr.items()} if tr else None
        self._steps_since_resample = int(weights.get('steps_since_resample', 0))

        # The sim we are restoring INTO was built on the seed body at setup(); the state above
        # describes a generated one. env.resample() is only ever reached from _maybe_resample, so
        # without this the run earns reward on the seed body while GenCrit is trained against a body
        # that never ran -- silently, until the next window boundary (up to ~62 epochs later). Skip
        # for window 0 (the seed build IS current) and for pre-subtype checkpoints, whose restored
        # counts would pair with base-derived subtypes and build a body matching neither.
        env = self._env()
        if env is not None and self._gen_window > 0 and has_sub:
            env.resample(designs_from_arrays(
                self._ml, self._cur_counts, self._cur_eff, self._cur_cap))
