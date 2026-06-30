"""PPG (Cobbe et al. 2020) agent: disjoint policy + value nets, two optimizers.

Phase-1 controller only (the aux phase is added in Stage 4). Subclasses
LoggingA2CAgent to reuse its diagnostics and the _maybe_resample seam. The policy
net carries an auxiliary value head trained only in the aux phase; all RL value
math uses the separate value net.
"""
import time

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from rl_games.common import a2c_common, common_losses, datasets
from rl_games.algos_torch import torch_ext

from .logging_agent import LoggingA2CAgent
from .models import MultiMorphValueBuilder, TransformerMaskedValue


class PPGAgent(LoggingA2CAgent):
    def __init__(self, base_name, params):
        # bypass A2CAgent.__init__ (single model/opt); build two nets ourselves
        a2c_common.ContinuousA2CBase.__init__(self, base_name, params)
        assert not self.has_central_value, "PPGAgent does not support central value"

        build_config = {
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'num_seqs': self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }

        self.shared_trunk = bool(self.config.get('ppg', {}).get('shared_trunk', False))

        # policy net (joint head + value head). Dual-net also builds a disjoint value net;
        # single-net (shared_trunk) keeps one net and detaches the value gradient at the trunk
        # during the policy phase (PPG paper sec 3.6), so its value head is the only value head.
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        if not self.shared_trunk:
            value_builder = MultiMorphValueBuilder()
            value_builder.load({'transformer': params['network'].get('transformer', {})})
            self.value_model = TransformerMaskedValue(value_builder).build(build_config)
            self.value_model.to(self.ppo_device)

        self.states = None
        self.init_rnn_from_model(self.model)
        self.last_lr = float(self.last_lr)
        self.bound_loss_type = self.config.get('bound_loss_type', 'bound')

        ppg = self.config.get('ppg', {})
        self.ppg_e_pi = ppg.get('e_pi', 1)
        self.ppg_e_v = ppg.get('e_v', 1)
        self.e_policy = ppg.get('e_policy', self.ppg_e_pi)   # fused policy+value epochs (single-net)
        self.ppg_n_pi = ppg.get('n_pi', 32)
        self.ppg_e_aux = ppg.get('e_aux', 6)
        self.ppg_beta_clone = float(ppg.get('beta_clone', 1.0))
        # aux chunk SIZE (not count) -> activations pinned to the known-good policy minibatch
        self.ppg_aux_mb_size = ppg.get('aux_minibatch_size', self.minibatch_size)
        self._aux_device = ppg.get('aux_buffer_device', 'cuda')
        self.value_lr = float(ppg.get('value_lr', self.config['learning_rate']))
        # PPG uses a FIXED policy LR. Adaptive KL-LR is incompatible with e_pi=1: KL is
        # measured pre-step on a near-unchanged policy -> ~0 -> scheduler ramps LR to its
        # ceiling -> entropy bonus blows up log_std. The paper uses fixed LR for this reason.
        self.policy_lr = float(ppg.get('policy_lr', self.config['learning_rate']))
        self.last_lr = self.policy_lr

        # timing state (helpers inherited from LoggingA2CAgent, whose __init__ we bypass)
        self._timing = self.config.get('timing', False)
        self._timings = {}
        self._tics = {}

        # policy: adaptive KL-LR (self.optimizer, retuned via update_lr). value: fixed lr.
        self.optimizer = optim.Adam(self.model.parameters(), self.last_lr,
                                    eps=1e-08, weight_decay=self.weight_decay, fused=True)
        if not self.shared_trunk:
            self.value_optimizer = optim.Adam(self.value_model.parameters(), self.value_lr,
                                              eps=1e-08, weight_decay=self.weight_decay, fused=True)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size,
                                           self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        # value-normalization stats: single-net uses the one model's; dual-net the value net's
        # (rollout values come from whichever net produces them).
        if self.normalize_value:
            self.value_mean_std = (self.model.value_mean_std if self.shared_trunk
                                   else self.value_model.value_mean_std)
        self.has_value_loss = self.use_experimental_cv or not self.has_central_value

        # LoggingA2CAgent instance state (we bypass its __init__)
        self._grad_norms = []
        self._value_grad_norms = []
        self._action_sats = []
        self._adv_mean = None
        self._adv_std = None
        self._morph_meta = None
        self._steps_since_resample = 0
        # V1.0 body-quality head state (mirrors LoggingA2CAgent.__init__; _v1=False for parity)
        self._v1 = self.value_size == 2
        if self._v1:
            self.gamma_v1 = self.config.get('gamma_v1', 1.0)
            self.v1_coef = self.config.get('v1_coef', 0.5 * self.critic_coef)
            self._gamma_vec = torch.tensor([self.gamma, self.gamma_v1], device=self.ppo_device)
        self._c_loss_v098: list = []
        self._c_loss_v1: list = []
        self._ev_v098 = None
        self._ev_v1 = None

        # PPG aux buffer: raw obs + raw V_targ across N_pi policy-phase iters
        n_total = self.ppg_n_pi * self.batch_size
        self._aux_obs = torch.empty((n_total, self.obs_shape[0]), device=self._aux_device)
        self._aux_vtarg = torch.empty((n_total, self.value_size), device=self._aux_device)
        self._aux_ptr = 0
        self._n_pi_count = 0
        self._aux_stats = None

        self.algo_observer.after_init(self)

    # ---- rollout: actions from policy net, value from value net -------------------

    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        if self.shared_trunk:                              # single net returns actions + values
            with torch.no_grad():
                return self.model({'is_train': False, 'prev_actions': None,
                                   'obs': processed_obs, 'rnn_states': self.rnn_states})
        self.value_model.eval()
        # SEPARATE dicts per model: model.forward normalizes obs in place (mutates the dict),
        # so a shared dict would double-normalize the value net's obs -> corrupt rollout values.
        with torch.no_grad():
            res_dict = self.model({'is_train': False, 'prev_actions': None,
                                   'obs': processed_obs, 'rnn_states': self.rnn_states})
            res_dict['values'] = self.value_model({'is_train': False, 'prev_actions': None,
                                                   'obs': processed_obs, 'rnn_states': self.rnn_states})['values']
        return res_dict

    def get_values(self, obs):
        with torch.no_grad():
            net = self.model if self.shared_trunk else self.value_model
            net.eval()
            processed_obs = self._preproc_obs(obs['obs'])
            input_dict = {'is_train': False, 'prev_actions': None,
                          'obs': processed_obs, 'rnn_states': self.rnn_states}
            return net(input_dict)['values']

    def set_eval(self):
        super().set_eval()
        if not self.shared_trunk:
            self.value_model.eval()

    def set_train(self):
        super().set_train()
        if not self.shared_trunk:
            self.value_model.train()

    def train(self):
        if torch.cuda.is_available() and not getattr(self, '_mem_reset', False):
            torch.cuda.reset_peak_memory_stats()       # peak_mem_mib measured from train start
            self._mem_reset = True
        # compile value_model after restore (torch_runner only compiles self.model)
        cc = self.config.get('torch_compile', True)
        if cc is not False and not self.shared_trunk and not getattr(self, '_value_compiled', False):
            mode = cc if isinstance(cc, str) else (cc.get('mode', 'default') if isinstance(cc, dict) else 'default')
            self.value_model = torch.compile(self.value_model, mode=mode)
            self._value_compiled = True
        return super().train()

    def get_full_state_weights(self):
        state = super().get_full_state_weights()  # policy model (+ its normalizers) + policy opt + scaler
        if not self.shared_trunk:
            state['value_model'] = self.value_model.state_dict()
            state['value_optimizer'] = self.value_optimizer.state_dict()
        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        super().set_full_state_weights(weights, set_epoch=set_epoch)
        if not self.shared_trunk:
            self.value_model.load_state_dict(weights['value_model'])
            self.value_optimizer.load_state_dict(weights['value_optimizer'])

    # ---- PPG policy phase (Stage 3: aux phase not yet implemented) ----------------

    def train_epoch(self):
        a2c_common.A2CBase.train_epoch(self)  # vec_env.set_train_info

        self.set_eval()
        play_time_start = time.perf_counter()
        self._tic('perf/t_rollout')
        with torch.no_grad():
            batch_dict = self.play_steps()
        self._toc('perf/t_rollout')
        play_time_end = time.perf_counter()
        update_time_start = time.perf_counter()

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        a_losses, b_losses, entropies, kls, c_losses = [], [], [], [], []
        if self.shared_trunk:
            # single-net policy phase: one fused pass per minibatch -- policy grad moves the
            # trunk+policy head, value grad (trunk detached) moves the value head only.
            self._tic('perf/t_policy')
            for _ in range(self.e_policy):
                ep_kls = []
                for i in range(len(self.dataset)):
                    a_loss, c_loss, entropy, b_loss, kl, cmu, csigma = self.calc_fused_grads(self.dataset[i])
                    a_losses.append(a_loss)
                    c_losses.append(c_loss)
                    entropies.append(entropy)
                    if self.bounds_loss_coef is not None:
                        b_losses.append(b_loss)
                    ep_kls.append(kl)
                    self.dataset.update_mu_sigma(cmu, csigma)
                kls.append(torch_ext.mean_list(ep_kls))
                if self.normalize_input:
                    self.model.running_mean_std.eval()
            self._toc('perf/t_policy')
        else:
            self._tic('perf/t_policy')
            for _ in range(self.ppg_e_pi):
                ep_kls = []
                for i in range(len(self.dataset)):
                    a_loss, entropy, b_loss, kl, cmu, csigma = self.calc_policy_grads(self.dataset[i])
                    a_losses.append(a_loss)
                    entropies.append(entropy)
                    if self.bounds_loss_coef is not None:
                        b_losses.append(b_loss)
                    ep_kls.append(kl)
                    self.dataset.update_mu_sigma(cmu, csigma)
                    # fixed policy LR: optimizer is NOT driven by the adaptive scheduler (see __init__).
                    # KL is kept as a diagnostic only.
                kls.append(torch_ext.mean_list(ep_kls))
                if self.normalize_input:
                    self.model.running_mean_std.eval()
            self._toc('perf/t_policy')

            self._tic('perf/t_value')
            for _ in range(self.ppg_e_v):
                for i in range(len(self.dataset)):
                    c_losses.append(self.calc_value_grads(self.dataset[i]))
                if self.normalize_input:
                    self.value_model.running_mean_std.eval()
            self._toc('perf/t_value')

        # accumulate this iter's (raw obs, raw V_targ); run the aux phase every N_pi
        self._aux_append(batch_dict['obses'], batch_dict['returns'])
        self._n_pi_count += 1
        if self._n_pi_count >= self.ppg_n_pi:
            self._tic('perf/t_aux')
            self._aux_phase()
            self._toc('perf/t_aux')
            self._n_pi_count = 0
            self._aux_ptr = 0

        self._maybe_resample()

        update_time_end = time.perf_counter()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start
        return (batch_dict['step_time'], play_time, update_time, total_time,
                a_losses, c_losses, b_losses, entropies, kls, self.last_lr, 1.0)

    def calc_policy_grads(self, input_dict):
        # compute_value=False -> policy-only pass (aux head gets no grad here regardless).
        # Only safe with is_train=True; the eval branch would denorm a None value.
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        actions_batch = input_dict['actions']
        obs_batch = self._preproc_obs(input_dict['obs'])
        curr_e_clip = self.e_clip

        batch_dict = {'is_train': True, 'prev_actions': actions_batch,
                      'obs': obs_batch, 'compute_value': False}

        with torch.amp.autocast('cuda', enabled=self.mixed_precision, dtype=torch.bfloat16):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs,
                                          advantage, self.ppo, curr_e_clip)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, _ = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), entropy.unsqueeze(1), b_loss.unsqueeze(1)], None)
            a_loss, entropy, b_loss = losses[0], losses[1], losses[2]
            loss = a_loss - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

            for param in self.model.parameters():
                param.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step(self.model, self.optimizer)

        with torch.no_grad():
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(),
                                          old_mu_batch, old_sigma_batch, True)
            saturated = (mu.abs() > 0.99).float().sum()
            active = (mu.abs() > 1e-6).float().sum().clamp(min=1.0)
            self._action_sats.append((saturated / active).item())

        self.diagnostics.mini_batch(self, {
            'values': input_dict['old_values'],
            'returns': input_dict['returns'],
            'new_neglogp': action_log_probs,
            'old_neglogp': old_action_log_probs_batch,
            'masks': None,
        }, curr_e_clip, 0)

        return a_loss, entropy, b_loss, kl_dist, mu.detach(), sigma.detach()

    def calc_value_grads(self, input_dict):
        value_preds_batch = input_dict['old_values']
        return_batch = input_dict['returns']
        obs_batch = self._preproc_obs(input_dict['obs'])
        curr_e_clip = self.e_clip

        batch_dict = {'is_train': True, 'prev_actions': None, 'obs': obs_batch}
        with torch.amp.autocast('cuda', enabled=self.mixed_precision, dtype=torch.bfloat16):
            values = self.value_model(batch_dict)['values']
            c_loss = common_losses.critic_loss(self.value_model, value_preds_batch, values,
                                               curr_e_clip, return_batch, self.clip_value)
            losses, _ = torch_ext.apply_masks([c_loss], None)
            c_loss = losses[0]
            loss = 0.5 * c_loss * self.critic_coef  # same value-loss scale as shared PPO

            for param in self.value_model.parameters():
                param.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step(self.value_model, self.value_optimizer, value=True)
        return c_loss

    def calc_fused_grads(self, input_dict):
        # single-net fused policy+value step. One forward computes both heads with the value
        # gradient detached at the trunk (detach_value=True): a_loss moves trunk+policy head,
        # c_loss moves the value head only. Joint grad-norm clip over the whole net, one step.
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        actions_batch = input_dict['actions']
        obs_batch = self._preproc_obs(input_dict['obs'])
        value_preds_batch = input_dict['old_values']
        return_batch = input_dict['returns']
        curr_e_clip = self.e_clip

        batch_dict = {'is_train': True, 'prev_actions': actions_batch, 'obs': obs_batch,
                      'compute_value': True, 'detach_value': True}

        with torch.amp.autocast('cuda', enabled=self.mixed_precision, dtype=torch.bfloat16):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values']
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            sigma = res_dict['sigmas']

            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs,
                                          advantage, self.ppo, curr_e_clip)
            c_loss = common_losses.critic_loss(self.model, value_preds_batch, values,
                                               curr_e_clip, return_batch, self.clip_value)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, _ = torch_ext.apply_masks(
                [a_loss.unsqueeze(1), c_loss.unsqueeze(1), entropy.unsqueeze(1), b_loss.unsqueeze(1)], None)
            a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3]
            loss = (a_loss + 0.5 * c_loss * self.critic_coef
                    - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef)

            for param in self.model.parameters():
                param.grad = None

        self.scaler.scale(loss).backward()
        self.trancate_gradients_and_step(self.model, self.optimizer)

        with torch.no_grad():
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(),
                                          old_mu_batch, old_sigma_batch, True)
            saturated = (mu.abs() > 0.99).float().sum()
            active = (mu.abs() > 1e-6).float().sum().clamp(min=1.0)
            self._action_sats.append((saturated / active).item())

        self.diagnostics.mini_batch(self, {
            'values': input_dict['old_values'],
            'returns': input_dict['returns'],
            'new_neglogp': action_log_probs,
            'old_neglogp': old_action_log_probs_batch,
            'masks': None,
        }, curr_e_clip, 0)

        return a_loss, c_loss, entropy, b_loss, kl_dist, mu.detach(), sigma.detach()

    # ---- PPG auxiliary phase (every N_pi iters) -----------------------------------

    def _aux_append(self, obses, returns):
        n = obses.shape[0]
        s = self._aux_ptr
        self._aux_obs[s:s + n] = obses.to(self._aux_device)
        self._aux_vtarg[s:s + n] = returns.to(self._aux_device)
        self._aux_ptr += n

    def _aux_phase(self):
        # eval() freezes input/value running stats; params still receive grads
        n = self._aux_ptr
        dev = self.ppo_device
        mb = min(n, self.ppg_aux_mb_size)
        self.model.eval()
        if not self.shared_trunk:
            self.value_model.eval()
        dummy = torch.zeros((mb, self.actions_num), device=dev)  # prev_actions (unused logits)

        # snapshot pi_old over the whole buffer, frozen through the E_aux epochs
        mu_old = torch.empty((n, self.actions_num), device=self._aux_device)
        sigma_old = torch.empty((n, self.actions_num), device=self._aux_device)
        self._tic('perf/t_aux_piold')
        with torch.no_grad():
            for s in range(0, n, mb):
                ob = self._to_dev(self._aux_obs[s:s + mb])
                res = self.model({'is_train': True, 'prev_actions': dummy[:ob.shape[0]],
                                  'obs': ob, 'compute_value': False})
                mu_old[s:s + mb] = res['mus'].to(self._aux_device)
                sigma_old[s:s + mb] = res['sigmas'].to(self._aux_device)
        self._toc('perf/t_aux_piold')

        aux_v, clone_kl, val_l = [], [], []
        self._tic('perf/t_aux_epochs')
        for _ in range(self.ppg_e_aux):
            perm = torch.randperm(n, device=self._aux_device)
            for s in range(0, n, mb):
                idx = perm[s:s + mb]
                ob = self._to_dev(self._aux_obs[idx])
                vt_norm = self.value_mean_std(self._to_dev(self._aux_vtarg[idx]))  # frozen -> shared target
                mo, so = self._to_dev(mu_old[idx]), self._to_dev(sigma_old[idx])
                dum = dummy[:ob.shape[0]]

                if self.shared_trunk:
                    # single-net: value grad flows through the whole net (no detach in aux);
                    # the clone KL holds the policy fixed. One step, no separate value net.
                    with torch.amp.autocast('cuda', enabled=self.mixed_precision, dtype=torch.bfloat16):
                        res = self.model({'is_train': True, 'prev_actions': dum,
                                          'obs': ob, 'compute_value': True})
                        l_aux = 0.5 * ((res['values'] - vt_norm) ** 2).mean()
                        kl = torch_ext.policy_kl(mo, so, res['mus'], res['sigmas'], True)
                        loss = l_aux + self.ppg_beta_clone * kl
                        for p in self.model.parameters():
                            p.grad = None
                    self.scaler.scale(loss).backward()
                    self.trancate_gradients_and_step(self.model, self.optimizer)
                    aux_v.append(l_aux.item()); clone_kl.append(kl.item()); val_l.append(l_aux.item())
                    continue

                # policy net: L_joint = L_aux + beta_clone * KL[pi_old, pi]
                with torch.amp.autocast('cuda', enabled=self.mixed_precision, dtype=torch.bfloat16):
                    res = self.model({'is_train': True, 'prev_actions': dum,
                                      'obs': ob, 'compute_value': True})
                    l_aux = 0.5 * ((res['values'] - vt_norm) ** 2).mean()
                    kl = torch_ext.policy_kl(mo, so, res['mus'], res['sigmas'], True)
                    loss_pi = l_aux + self.ppg_beta_clone * kl
                    for p in self.model.parameters():
                        p.grad = None
                self.scaler.scale(loss_pi).backward()
                self.trancate_gradients_and_step(self.model, self.optimizer)

                # value net: extra L_value on the same target
                with torch.amp.autocast('cuda', enabled=self.mixed_precision, dtype=torch.bfloat16):
                    v = self.value_model({'is_train': True, 'prev_actions': None, 'obs': ob})['values']
                    l_val = 0.5 * ((v - vt_norm) ** 2).mean()
                    for p in self.value_model.parameters():
                        p.grad = None
                self.scaler.scale(l_val).backward()
                self.trancate_gradients_and_step(self.value_model, self.value_optimizer, value=True)

                aux_v.append(l_aux.item()); clone_kl.append(kl.item()); val_l.append(l_val.item())
        self._toc('perf/t_aux_epochs')

        self._aux_stats = {
            'losses/aux_value': sum(aux_v) / len(aux_v),
            'losses/aux_clone_kl': sum(clone_kl) / len(clone_kl),
            'losses/aux_value_net': sum(val_l) / len(val_l),
        }

    def trancate_gradients_and_step(self, model=None, optimizer=None, value=False):
        model = model if model is not None else self.model
        optimizer = optimizer if optimizer is not None else self.optimizer
        if self.truncate_grads:
            self.scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), self.grad_norm)
            (self._value_grad_norms if value else self._grad_norms).append(float(grad_norm))
        self.scaler.step(optimizer)
        self.scaler.update()

    def write_stats(self, *args, **kwargs):
        super().write_stats(*args, **kwargs)  # also flushes self._timings (LoggingA2CAgent)
        if self.writer is None:
            return
        frame = args[11] if len(args) > 11 else kwargs.get('frame')
        if torch.cuda.is_available():                      # peak (monotonic) -> last value = run peak
            self.writer.add_scalar('perf/peak_mem_mib', torch.cuda.max_memory_allocated() / 1e6, frame)
        if self._value_grad_norms:
            self.writer.add_scalar('health/value_grad_norm',
                                   sum(self._value_grad_norms) / len(self._value_grad_norms), frame)
            self._value_grad_norms = []
        if self._aux_stats is not None:
            for k, v in self._aux_stats.items():
                self.writer.add_scalar(k, v, frame)
            self._aux_stats = None
