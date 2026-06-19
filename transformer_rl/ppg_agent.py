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

        # policy net (joint head + aux value head) and value net, disjoint weights
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
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
        self.value_lr = float(ppg.get('value_lr', self.config['learning_rate']))

        # policy: adaptive KL-LR (self.optimizer, retuned via update_lr). value: fixed lr.
        self.optimizer = optim.Adam(self.model.parameters(), self.last_lr,
                                    eps=1e-08, weight_decay=self.weight_decay, fused=True)
        self.value_optimizer = optim.Adam(self.value_model.parameters(), self.value_lr,
                                          eps=1e-08, weight_decay=self.weight_decay, fused=True)

        self.use_experimental_cv = self.config.get('use_experimental_cv', True)
        self.dataset = datasets.PPODataset(self.batch_size, self.minibatch_size,
                                           self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        # value net's stats drive all RL value normalization (rollout values come from it)
        if self.normalize_value:
            self.value_mean_std = self.value_model.value_mean_std
        self.has_value_loss = self.use_experimental_cv or not self.has_central_value

        # LoggingA2CAgent instance state (we bypass its __init__)
        self._grad_norms = []
        self._value_grad_norms = []
        self._action_sats = []
        self._adv_mean = None
        self._adv_std = None
        self._morph_meta = None
        self._steps_since_resample = 0

        self.algo_observer.after_init(self)

    # ---- rollout: actions from policy net, value from value net -------------------

    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        self.value_model.eval()
        input_dict = {'is_train': False, 'prev_actions': None,
                      'obs': processed_obs, 'rnn_states': self.rnn_states}
        with torch.no_grad():
            res_dict = self.model(input_dict)
            res_dict['values'] = self.value_model(input_dict)['values']
        return res_dict

    def get_values(self, obs):
        with torch.no_grad():
            self.value_model.eval()
            processed_obs = self._preproc_obs(obs['obs'])
            input_dict = {'is_train': False, 'prev_actions': None,
                          'obs': processed_obs, 'rnn_states': self.rnn_states}
            return self.value_model(input_dict)['values']

    def set_eval(self):
        super().set_eval()
        self.value_model.eval()

    def set_train(self):
        super().set_train()
        self.value_model.train()

    # ---- PPG policy phase (Stage 3: aux phase not yet implemented) ----------------

    def train_epoch(self):
        a2c_common.A2CBase.train_epoch(self)  # vec_env.set_train_info

        self.set_eval()
        play_time_start = time.perf_counter()
        with torch.no_grad():
            batch_dict = self.play_steps()
        play_time_end = time.perf_counter()
        update_time_start = time.perf_counter()

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()

        a_losses, b_losses, entropies, kls = [], [], [], []
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
            if self.normalize_input:
                self.model.running_mean_std.eval()

        c_losses = []
        for _ in range(self.ppg_e_v):
            for i in range(len(self.dataset)):
                c_losses.append(self.calc_value_grads(self.dataset[i]))
            if self.normalize_input:
                self.value_model.running_mean_std.eval()

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
        super().write_stats(*args, **kwargs)
        if self.writer is not None and self._value_grad_norms:
            frame = args[11] if len(args) > 11 else kwargs.get('frame')
            self.writer.add_scalar('health/value_grad_norm',
                                   sum(self._value_grad_norms) / len(self._value_grad_norms), frame)
            self._value_grad_norms = []
