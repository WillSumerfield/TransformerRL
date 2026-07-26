"""rl_games NetworkBuilders for ant envs."""
import torch
import torch.nn as nn
from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.algos_torch.models import ModelA2CContinuousLogStd

from .architectures import LimbTransformer, MultiMorphLimbTransformer
from .tokenize import (OBS_DIM_4 as OBS_DIM, MASK_DIM_4 as MASK_DIM,
                       OBS_DIM_8, LEN_DIM_8, MASK_DIM_8)


def _dyn_dims(net):
    """(mask_off, mask_dim, obs_total) for a MultiMorph net, from its tdims (single source of truth,
    parameterized by (n_limbs, max_limb_length)). Non-codesign (legacy 8-limb PPG/classic, out of
    scope since Phase 1) falls back to the old 107/16/16 layout."""
    d = getattr(net, "tdims", None)
    if d is None:
        return OBS_DIM_8 + LEN_DIM_8, MASK_DIM_8, OBS_DIM_8 + LEN_DIM_8 + MASK_DIM_8
    return d["mask_off"], d["mask_dim"], d["obs_total"]


def _raw_tail(net):
    """(offset, dim) of the block kept UNNORMALIZED: DOF mask + (Phase 5) is_cap + subtype one-hot.
    Every channel there is exactly {0,1} and is read back with a `> 0` threshold, which a normalized
    channel would break (a constant channel collapses to ~0 -> misread as absent)."""
    d = getattr(net, "tdims", None)
    if d is None:
        return OBS_DIM_8 + LEN_DIM_8, MASK_DIM_8
    return d["raw_tail_off"], d["raw_tail_dim"]


def _restore_mask_tail(normed, observation, off, dim, normalize_input):
    """Re-insert the raw {0,1} mask/type tail after normalization ([off : off+dim])."""
    if normalize_input:
        normed = normed.clone()
        normed[..., off:off + dim] = observation[..., off:off + dim]
    return normed


class LimbTransformerBuilder(NetworkBuilder):
    """rl_games builder for the classic fixed 4-limb ant."""

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            nn.Module.__init__(self)
            kwargs.pop('actions_num', None)
            kwargs.pop('input_shape', None)
            kwargs.pop('num_seqs', None)
            tc = params.get('transformer', {})
            self.net = LimbTransformer(**tc)
            self.log_std_param = nn.Parameter(torch.zeros(8))

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            out = self.net(obs)
            mu, value = out['mu'], out['value']
            if obs.shape[-1] >= OBS_DIM + MASK_DIM:
                mask_dof = (obs[..., OBS_DIM : OBS_DIM + MASK_DIM] > 0).float()
                # Inactive dims -> log_std 0 (sigma=1). The env masks inactive actions anyway,
                # so their sigma is irrelevant to dynamics; a moderate sigma keeps rl_games'
                # policy_kl well-conditioned (tiny sigma collapses its eps term, poisoning KL ->
                # the adaptive LR controller). mask_dof gates the gradient so inactive entries
                # never pull on log_std_param.
                log_std = mask_dof * self.log_std_param
            else:
                log_std = self.log_std_param.expand(obs.shape[0], -1)
            return mu, log_std, value, None

        def is_rnn(self):
            return False

    def build(self, name, **kwargs):
        return LimbTransformerBuilder.Network(self.params, **kwargs)


class MultiMorphLimbTransformerBuilder(NetworkBuilder):
    """rl_games builder for the 8-limb multi-morphology ant."""

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            nn.Module.__init__(self)
            kwargs.pop('actions_num', None)
            kwargs.pop('input_shape', None)
            kwargs.pop('num_seqs', None)
            kwargs.pop('value_size', None)               # single-net codesign: V1.0 is gencrit_head
            tc = params.get('transformer', {})
            self.net = MultiMorphLimbTransformer(**tc)
            self._mask_off, self._mask_dim, _ = _dyn_dims(self.net)   # 187/32 (Phase-1 codesign)
            self.log_std_param = nn.Parameter(torch.zeros(self._mask_dim))

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            out = self.net(obs, compute_value=obs_dict.get('compute_value', True),
                           detach_value=obs_dict.get('detach_value', False),
                           actions=obs_dict.get('prev_actions'))   # FD head sources its own action here
            mu, value = out['mu'], out.get('value')  # value None when aux head skipped
            if obs.shape[-1] >= self._mask_off + self._mask_dim:
                mask_dof = (obs[..., self._mask_off : self._mask_off + self._mask_dim] > 0).float()
                # Inactive dims -> log_std 0 (sigma=1). The env masks inactive actions anyway,
                # so their sigma is irrelevant to dynamics; a moderate sigma keeps rl_games'
                # policy_kl well-conditioned (tiny sigma collapses its eps term, poisoning KL ->
                # the adaptive LR controller). mask_dof gates the gradient so inactive entries
                # never pull on log_std_param.
                log_std = mask_dof * self.log_std_param
            else:
                log_std = self.log_std_param.expand(obs.shape[0], -1)
            return mu, log_std, value, None

        def get_aux_loss(self):
            return self.net.get_aux_loss()      # fused FD aux (2b); None when disarmed

        def is_rnn(self):
            return False

    def build(self, name, **kwargs):
        return MultiMorphLimbTransformerBuilder.Network(self.params, **kwargs)


class TransformerMaskedNorm(ModelA2CContinuousLogStd):
    """continuous_a2c_logstd whose input normalizer leaves the DOF mask dims raw.

    Standard RunningMeanStd collapses the constant mask to ~0 once running_mean
    rounds to 1.0 in fp32 (a hard cliff that flips every limb to "inactive"). We keep
    the stock normalizer and only override norm_obs to restore the raw mask tail.
    """

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)
            self._mask_off, self._mask_dim, exp_total = _dyn_dims(a2c_network.net)
            self._tail_off, self._tail_dim = _raw_tail(a2c_network.net)
            obs_total = self.obs_shape[0] if isinstance(self.obs_shape, (tuple, list)) else self.obs_shape
            assert obs_total == exp_total, (
                f"transformer_masked_a2c_logstd expects obs of {exp_total}, got {obs_total}"
            )

        def norm_obs(self, observation):
            return _restore_mask_tail(super().norm_obs(observation), observation,
                                      self._tail_off, self._tail_dim, self.normalize_input)


class MultiMorphValueBuilder(NetworkBuilder):
    """rl_games builder for the PPG disjoint value net (8-limb, value head only)."""

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            nn.Module.__init__(self)
            kwargs.pop('actions_num', None)
            kwargs.pop('input_shape', None)
            kwargs.pop('num_seqs', None)
            tc = params.get('transformer', {})
            self.net = MultiMorphLimbTransformer(policy_head=False, value_head=True, **tc)

        def forward(self, obs_dict):
            return self.net(obs_dict['obs'])['value']

        def is_rnn(self):
            return False

    def build(self, name, **kwargs):
        return MultiMorphValueBuilder.Network(self.params, **kwargs)


class TransformerMaskedValue(ModelA2CContinuousLogStd):
    """Value-only model for the PPG disjoint critic.

    Reuses ModelA2CContinuousLogStd's value_mean_std / denorm_value and the
    mask-restoring norm_obs, but its network has no policy head, so forward
    returns only a value (no action distribution / Normal).
    """

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)
            self._mask_off, self._mask_dim, exp_total = _dyn_dims(a2c_network.net)
            self._tail_off, self._tail_dim = _raw_tail(a2c_network.net)
            obs_total = self.obs_shape[0] if isinstance(self.obs_shape, (tuple, list)) else self.obs_shape
            assert obs_total == exp_total, (
                f"transformer_masked_value expects obs of {exp_total}, got {obs_total}"
            )

        def norm_obs(self, observation):
            return _restore_mask_tail(super().norm_obs(observation), observation,
                                      self._tail_off, self._tail_dim, self.normalize_input)

        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            input_dict['obs'] = self.norm_obs(input_dict['obs'])
            value = self.a2c_network(input_dict)
            return {'values': value if is_train else self.denorm_value(value)}
