"""A2CAgent subclass that logs extra PPO-health metrics.

Adds, per epoch, on top of rl_games' built-ins (enable via use_diagnostics=True,
which gives diagnostics/exp_var, diagnostics/clip_frac/*, diagnostics/rms_value/*):

    control/sigma_{mean,max}     exploration std = exp(log_std), usage-weighted over ACTIVE DOF dims
                                 only (mu!=0); dead dims -- depth > max_effectors, never an effector,
                                 pinned at sigma=1 -- are dropped so they no longer poison mean/max
    control/sigma_min            log_std-collapse canary (over all dims; active dims are the smallest)
    control/action_sat           frac of *active* mean-action dims pinned at the tanh rail (|mu|>0.99)
    control/grad_norm            total grad norm BEFORE clipping (clip_grad_norm_ return)
    control/adv_{mean,std}       raw advantage (returns-values) BEFORE normalization -> true scale

Registered globally over 'a2c_continuous' in train_utils, so every continuous PPO
run (transformer or MLP) gets these. Metrics are generic; nothing transformer-specific.
"""
import contextlib
import os
import random
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from rl_games.common import a2c_common
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.a2c_continuous import A2CAgent

from codesigner.progress import Progress

from .morphology import sample_bodies


# Attachment slot (0-7 at i*45 deg) -> compass code, relative to forward = +X (the reward axis),
# right = -Z, slot number increasing clockwise toward the right. 0-BASED, matching Morphology.
_LIMB_CODE = {0: "F", 1: "FR", 2: "R", 3: "BR", 4: "B", 5: "BL", 6: "L", 7: "FL"}
_LEADERBOARD_EVERY = 50  # epochs
_LEADERBOARD_K = 5       # top-k and bottom-k


def _morph_label(limbs) -> str:
    """frozenset/list of 0-based slots -> compass-coded label, e.g. {1,3,5,7} -> 'FR·BR·BL·FL'."""
    return "·".join(_LIMB_CODE[n] for n in sorted(limbs))


class LoggingA2CAgent(A2CAgent):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self._grad_norms: list[float] = []   # per-minibatch, flushed per epoch
        self._action_sats: list[float] = []  # per-minibatch, flushed per epoch
        self._dof_active = None              # per-DOF activation count (usage weights), flushed/epoch
        self._adv_mean: float | None = None  # per-epoch (set in prepare_dataset)
        self._adv_std: float | None = None
        self._morph_meta = None  # None=undetected, False=single-morph, dict=multi-morph metadata
        self._steps_since_resample = 0  # env-steps since last morphology resample (full ant only)
        self._resample_count = 0        # rebuilds performed; bounds one Algorithm.run() (_run_unit)
        self._train_finished = False    # set by _train_iter at exit, read by Algorithm.is_finished
        self._on_iteration = None       # package iteration tick (D26); set by CodesignAlgorithm
        self._defer_train = False       # True -> train() returns without looping (see train())
        # Own stream for the body draw, seeded off the global one the Task seeded at setup, so a
        # run's resample sequence is reproducible without plumbing the seed down here.
        self._morph_rng = random.Random(random.randrange(2 ** 32))
        # opt-in synced phase timing (config.timing); shared with PPGAgent. Off -> stock path.
        self._timing = self.config.get('timing', False)
        self._timings = {}
        self._tics = {}
        # opt-in per-node memory profiling (config.mem_profile); own flag, own syncs -> kept
        # independent of timing so fps and peak are each measured without the other's perturbation.
        # Records, per _prof region: persist Delta (memory held past the region) + peak Delta
        # (transient spike). reset_peak per region clobbers the global peak stat, so track the
        # epoch peak ourselves (self._peak_running, absolute bytes) for perf/peak_mem_mib.
        self._mem_profile = self.config.get('mem_profile', False)
        self._mems = {}         # key -> [persist_sum_bytes, count, peakdelta_max_bytes]
        self._peak_running = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()   # perf/peak_mem_mib measured from train start

    # ---- opt-in phase timing (cuda.synchronize + perf_counter); shared w/ PPGAgent ----

    def _tic(self, key):
        if self._timing:
            torch.cuda.synchronize()
            self._tics[key] = time.perf_counter()

    def _toc(self, key):
        if self._timing:
            torch.cuda.synchronize()
            self._timings[key] = self._timings.get(key, 0.0) + (time.perf_counter() - self._tics[key])

    def _env(self):
        return getattr(getattr(self.vec_env, 'envs', None), 'env', None)

    @contextlib.contextmanager
    def _prof(self, key):
        """Region profiler: accumulates synced time (perf/t_<key>) and, under mem_profile,
        persist Delta + peak Delta for the cost tree. Zero overhead when both flags are off."""
        if not (self._timing or self._mem_profile):
            yield
            return
        a0 = 0
        if self._mem_profile:
            torch.cuda.synchronize()
            a0 = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        t0 = 0.0
        if self._timing:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._timing:
                torch.cuda.synchronize()
                tk = 'perf/t_' + key
                self._timings[tk] = self._timings.get(tk, 0.0) + (time.perf_counter() - t0)
            if self._mem_profile:
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                m = self._mems.setdefault(key, [0.0, 0, 0.0])
                m[0] += torch.cuda.memory_allocated() - a0   # persist Delta (held past region)
                m[1] += 1
                m[2] = max(m[2], peak - a0)                   # peak Delta (transient spike)
                self._peak_running = max(self._peak_running, peak)

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

    # ---- the training loop, as a generator (package Phase 8) -------------------------
    # rl_games' train() is one monolithic loop, so an outer driver that wants control back
    # periodically -- codesigner.optimize, which fires a progress tick and checkpoints after every
    # Algorithm.run() -- has nowhere to take it. This is that loop, copied once and reshaped to
    # yield at each resample boundary. train() below just drains it, so the script entry point is
    # behaviourally unchanged and neither path can drift from the other.

    def _run_unit(self) -> int:
        """The counter whose increments bound one `Algorithm.run()`. Resamples performed, i.e. the
        rebuild boundary -- already a synchronization point, and where a crash costs the most (D24).
        A run with resampling off never increments it, so `run()` is then the whole of training,
        which is the right degenerate answer rather than a special case."""
        return self._resample_count

    def _train_iter(self):
        """`ContinuousA2CBase.train()`, yielding `(mean_rewards, epoch_num)` at every resample boundary
        and once more when training exits.

        The yielded reward is the CURRENT mean episode return, not `last_mean_rewards` -- that one
        is rl_games' best-ever, and it stays pinned at its -1e9 sentinel until `save_best_after`, so
        a driver tracking progress from it would see nothing for the first hundreds of epochs.

        A faithful copy but for the yields and one simplification: multi-GPU is dropped (asserted
        against) rather than carried, since every path in this repo is single-GPU and the broadcast
        bookkeeping is the bulk of what would be copied. Note in particular that `frame` is read
        BEFORE `self.frame` is advanced -- the continuous loop differs from the discrete one there,
        and every scalar logged against `frame` would shift by one epoch if this were "tidied".
        Keep it aligned with rl_games on upgrade.
        """
        assert not self.multi_gpu, "single-GPU only; see rl_games train() for the broadcast path"
        self.init_tensors()
        self.last_mean_rewards = -1000000000
        start_time = time.perf_counter()
        total_time = 0
        rep_count = 0                                        # noqa: F841 (parity with rl_games)
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs
        last_unit = self._run_unit()

        while True:
            epoch_num = self.update_epoch()
            (step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies,
             kls, last_lr, lr_mul) = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            self.dataset.update_values_dict(None)            # cleaning memory to optimize space
            should_exit = False

            self.diagnostics.epoch(self, current_epoch=epoch_num)
            scaled_time = self.num_agents * sum_time
            scaled_play_time = self.num_agents * play_time
            curr_frames = self.curr_frames
            self.frame += curr_frames

            a2c_common.print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time,
                                        scaled_time, epoch_num, self.max_epochs, frame,
                                        self.max_frames)
            self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                             a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame,
                             scaled_time, scaled_play_time, curr_frames)

            if len(b_losses) > 0:
                self.writer.add_scalar('losses/bounds_loss',
                                       torch_ext.mean_list(b_losses).item(), frame)

            # Iteration tick (D26): the epoch is this algorithm's inner cadence -- far finer than
            # the window that bounds a run(). Best-effort by contract, so an unattached driver
            # costs nothing. Reward is withheld until an episode has actually completed, rather
            # than reported as rl_games' -1e9 sentinel.
            if self._on_iteration is not None:
                have_reward = self.game_rewards.current_size > 0
                self._on_iteration(Progress(
                    tick=epoch_num,
                    reward=float(self.game_rewards.get_mean()[0]) if have_reward else None,
                    wall_time=time.perf_counter() - start_time))

            if self.game_rewards.current_size > 0:
                mean_rewards = self.game_rewards.get_mean()
                mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                mean_lengths = self.game_lengths.get_mean()
                self.mean_rewards = mean_rewards[0]

                for i in range(self.value_size):
                    rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                    self.writer.add_scalar(rewards_name + '/step', mean_rewards[i], frame)
                    self.writer.add_scalar(rewards_name + '/iter', mean_rewards[i], epoch_num)
                    self.writer.add_scalar(rewards_name + '/time', mean_rewards[i], total_time)
                    self.writer.add_scalar('shaped_' + rewards_name + '/step',
                                           mean_shaped_rewards[i], frame)
                    self.writer.add_scalar('shaped_' + rewards_name + '/iter',
                                           mean_shaped_rewards[i], epoch_num)
                    self.writer.add_scalar('shaped_' + rewards_name + '/time',
                                           mean_shaped_rewards[i], total_time)

                self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                if self.has_self_play_config:
                    self.self_play_manager.update(self)

                checkpoint_name = (self.config['name'] + '_ep_' + str(epoch_num) + '_rew_'
                                   + str(mean_rewards[0]))

                if self.save_freq > 0 and epoch_num % self.save_freq == 0:
                    self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                    print('saving next best rewards: ', mean_rewards)
                    self.last_mean_rewards = mean_rewards[0]
                    self.save(os.path.join(self.nn_dir, self.config['name']))

                    if 'score_to_win' in self.config and \
                            self.last_mean_rewards > self.config['score_to_win']:
                        print('Maximum reward achieved. Network won!')
                        self.save(os.path.join(self.nn_dir, checkpoint_name))
                        should_exit = True

            if epoch_num >= self.max_epochs and self.max_epochs != -1:
                if self.game_rewards.current_size == 0:
                    print('WARNING: Max epochs reached before any env terminated at least once')
                    mean_rewards = -np.inf
                self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_'
                                       + str(epoch_num) + '_rew_'
                                       + str(mean_rewards).replace('[', '_').replace(']', '_')))
                print('MAX EPOCHS NUM!')
                should_exit = True

            if self.frame >= self.max_frames and self.max_frames != -1:
                if self.game_rewards.current_size == 0:
                    print('WARNING: Max frames reached before any env terminated at least once')
                    mean_rewards = -np.inf
                self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_'
                                       + str(self.frame) + '_rew_'
                                       + str(mean_rewards).replace('[', '_').replace(']', '_')))
                print('MAX FRAMES NUM!')
                should_exit = True

            update_time = 0                                  # noqa: F841 (parity with rl_games)

            if should_exit:
                self._train_finished = True
                yield self.mean_rewards, epoch_num
                return
            # A window closed during this epoch -- hand control back to whatever is driving.
            if self._run_unit() != last_unit:
                last_unit = self._run_unit()
                yield self.mean_rewards, epoch_num
    def train(self):
        """rl_games' entry point: drain the generator. Same loop, same result.

        `_defer_train` makes this a no-op, which is how `CodesignAlgorithm` gets a fully
        constructed, restored and compiled agent out of `Runner.run_train` without training it --
        it then drives `_train_iter` a window at a time.
        """
        if self._defer_train:
            return None
        result = None
        for result in self._train_iter():
            pass
        return result

    def _maybe_resample(self):
        """Every resample_interval episodes, draw a fresh morphology set (full gym rebuild) and
        refresh the agent's cached obs. No-op unless the env samples morphologies and the knob is set.
        See docs/guides/morphology_resampling_cost.md."""
        interval = self.config.get('resample_interval', 0)  # episodes between resamples; 0 = off
        if not interval:
            return
        env = getattr(getattr(self.vec_env, 'envs', None), 'env', None)
        if env is None:
            return
        self._steps_since_resample += self.horizon_length
        if self._steps_since_resample < interval * env.max_episode_length:
            return
        print(f"[resample] new morphology set (every {interval} episodes)", flush=True)
        # The Task no longer draws bodies -- it is handed them (D10c), so the draw is the agent's.
        # This agent has no generator, so it draws random stable topologies; CodesignAgent overrides
        # this whole method and supplies its generator's designs instead.
        env.resample(sample_bodies(env.module_library, env.n_morphs, self._morph_rng))
        self.obs = self.env_reset()             # rebuilt env -> refresh stale rollout-start obs
        self.current_rewards.zero_()            # the hard reset ends all episodes; drop partials
        self.current_lengths.zero_()
        self._morph_meta = None                 # morphs changed -> re-detect per-morph logging labels
        self._steps_since_resample = 0
        self._resample_count += 1

    # ---- raw advantage scale logging (health/adv_*) ---------------------------------

    def prepare_dataset(self, batch_dict):
        # Raw advantage scale, before rl_games normalizes it (mirrors a2c_common:1030,1038).
        with torch.no_grad():
            adv = (batch_dict['returns'] - batch_dict['values']).sum(dim=1)
            self._adv_mean = adv.mean().item()
            self._adv_std = adv.std().item()
        return super().prepare_dataset(batch_dict)

    def _log_action_sat(self):
        # train_result = (a_loss, c_loss, entropy, kl, lr, lr_mul, mu, sigma, b_loss)
        with torch.no_grad():
            mu = self.train_result[6]
            # Inactive dims are masked to exactly 0 in our nets, so |mu|>eps selects active dims
            # (and is all dims for an MLP) -> saturation measured over active dims only.
            active = mu.abs() > 1e-6                    # (minibatch, n_act)
            saturated = (mu.abs() > 0.99).float().sum()
            self._action_sats.append((saturated / active.float().sum().clamp(min=1.0)).item())
            # per-DOF activation frequency over the batch -> usage weights for control/sigma_* (M2:
            # a dead DOF -- depth > max_effectors, never holds an effector -> mu==0 -> weight 0).
            cnt = active.float().sum(dim=0)             # (n_act,)
            self._dof_active = cnt if self._dof_active is None else self._dof_active + cnt

    def calc_gradients(self, input_dict):
        super().calc_gradients(input_dict)
        self._log_action_sat()

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
            wt = self._dof_active                       # per-DOF usage weights over the epoch's batch
            if wt is not None and wt.numel() == sigma.numel() and float(wt.sum()) > 0:
                wt = wt.to(sigma.device)
                sig_mean = (sigma * wt).sum().item() / wt.sum().item()   # usage-weighted, active DOF
                sig_max = sigma[wt > 0].max().item()                     # active dims only (dead=1.0)
            else:                                        # MLP (mu never exactly 0) / pre-first-update
                sig_mean, sig_max = sigma.mean().item(), sigma.max().item()
            w.add_scalar('control/sigma_mean', sig_mean, frame)
            w.add_scalar('control/sigma_min', sigma.min().item(), frame)   # active dims are smallest
            w.add_scalar('control/sigma_max', sig_max, frame)
        self._dof_active = None                          # flush per-epoch, like _grad_norms

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
        # per-node memory (mem_profile): mean persist Delta + max peak Delta, MiB.
        if self._mems:
            for k, (psum, cnt, pkd) in self._mems.items():
                w.add_scalar(f'perf/mem_persist_{k}', psum / max(cnt, 1) / 1e6, frame)
                w.add_scalar(f'perf/mem_peakd_{k}', pkd / 1e6, frame)
            self._mems = {}
        # peak GPU mem: passive read (no cuda.synchronize) -> always logged, no --timing needed.
        # Under mem_profile the global peak stat is reset per-region, so use our own running max.
        # Throughput comes free from rl_games' performance/step_inference_rl_update_fps.
        if torch.cuda.is_available():
            peak = self._peak_running if (self._mem_profile and self._peak_running) \
                else torch.cuda.max_memory_allocated()
            w.add_scalar('perf/peak_mem_mib', peak / 1e6, frame)
            self._peak_running = 0

        epoch_num = int(args[1]) if len(args) > 1 else kwargs.get('epoch_num', 0)
        self._log_morph_stats(w, frame, epoch_num)

    # ---- per-morphology performance (multi-morph codesign Tasks only) ------------

    def _morph_metadata(self):
        """Detect a multi-morph env and cache per-morph layout/labels. Returns dict or False."""
        if self._morph_meta is not None:
            return self._morph_meta
        env = getattr(getattr(self.vec_env, 'envs', None), 'env', None)
        if env is None or not hasattr(env, 'envs_per_morph') or not hasattr(env, 'groups'):
            self._morph_meta = False
            return False
        # occupied_slots, not the pre-migration `limbs`: 0-based now (see Morphology).
        morphs = [list(g['morph'].occupied_slots) for g in env.groups]
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
