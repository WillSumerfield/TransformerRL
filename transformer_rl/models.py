"""rl_games NetworkBuilders for ant envs."""
import torch
import torch.nn as nn
from rl_games.algos_torch.network_builder import NetworkBuilder
from rl_games.algos_torch.models import ModelA2CContinuousLogStd

from .architectures import LegTransformer, MultiMorphLegTransformer
from .tokenize import (
    OBS_DIM_4 as OBS_DIM, MASK_DIM_4 as MASK_DIM,
    OBS_DIM_8 as DYN_OBS_DIM, LEN_DIM_8 as DYN_LEN_DIM, MASK_DIM_8 as DYN_MASK_DIM,
)

_DYN_MASK_OFF = DYN_OBS_DIM + DYN_LEN_DIM   # mask follows the length block: obs[123:139]
_DYN_OBS_TOTAL = DYN_OBS_DIM + DYN_LEN_DIM + DYN_MASK_DIM  # 139


class LegTransformerBuilder(NetworkBuilder):
    """rl_games builder for 4-leg ant (ant / ant_adaptive)."""

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            nn.Module.__init__(self)
            kwargs.pop('actions_num', None)
            kwargs.pop('input_shape', None)
            kwargs.pop('num_seqs', None)
            tc = params.get('transformer', {})
            self.net = LegTransformer(**tc)
            self.log_std_param = nn.Parameter(torch.zeros(8))

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            mu, value = self.net(obs)
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
        return LegTransformerBuilder.Network(self.params, **kwargs)


class MultiMorphLegTransformerBuilder(NetworkBuilder):
    """rl_games builder for the 8-leg multi-morphology ant."""

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            nn.Module.__init__(self)
            kwargs.pop('actions_num', None)
            kwargs.pop('input_shape', None)
            kwargs.pop('num_seqs', None)
            tc = params.get('transformer', {})
            self.net = MultiMorphLegTransformer(**tc)
            self.log_std_param = nn.Parameter(torch.zeros(16))

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            mu, value = self.net(obs)
            if obs.shape[-1] >= _DYN_MASK_OFF + DYN_MASK_DIM:
                mask_dof = (obs[..., _DYN_MASK_OFF : _DYN_MASK_OFF + DYN_MASK_DIM] > 0).float()
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
        return MultiMorphLegTransformerBuilder.Network(self.params, **kwargs)


class TransformerMaskedNorm(ModelA2CContinuousLogStd):
    """continuous_a2c_logstd whose input normalizer leaves the DOF mask dims raw.

    Standard RunningMeanStd collapses the constant mask to ~0 once running_mean
    rounds to 1.0 in fp32 (a hard cliff that flips every leg to "inactive"). We keep
    the stock normalizer and only override norm_obs to restore the raw mask tail.
    """

    class Network(ModelA2CContinuousLogStd.Network):
        def __init__(self, a2c_network, **kwargs):
            super().__init__(a2c_network, **kwargs)
            obs_total = self.obs_shape[0] if isinstance(self.obs_shape, (tuple, list)) else self.obs_shape
            assert obs_total == _DYN_OBS_TOTAL, (
                f"transformer_masked_a2c_logstd expects obs of {_DYN_OBS_TOTAL} "
                f"(= {DYN_OBS_DIM} physical + {DYN_LEN_DIM} lengths + {DYN_MASK_DIM} mask), "
                f"got {obs_total}"
            )

        def norm_obs(self, observation):
            normed = super().norm_obs(observation)
            if self.normalize_input:
                normed = normed.clone()
                normed[..., -DYN_MASK_DIM:] = observation[..., -DYN_MASK_DIM:]
            return normed
