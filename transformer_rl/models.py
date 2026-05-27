"""rl_games NetworkBuilders for ant envs."""
import torch
import torch.nn as nn
from rl_games.algos_torch.network_builder import NetworkBuilder

from .architectures import LegTransformer, DynamicLegTransformer
from .tokenize import OBS_DIM_4 as OBS_DIM, MASK_DIM_4 as MASK_DIM, OBS_DIM_8 as DYN_OBS_DIM, MASK_DIM_8 as DYN_MASK_DIM


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
                mask_dof = (obs[..., OBS_DIM : OBS_DIM + MASK_DIM] > 0.5).float()
                log_std = self.log_std_param - 10.0 * (1.0 - mask_dof)
            else:
                log_std = self.log_std_param.expand(obs.shape[0], -1)
            return mu, log_std, value, None

        def is_rnn(self):
            return False

    def build(self, name, **kwargs):
        return LegTransformerBuilder.Network(self.params, **kwargs)


class DynamicLegTransformerBuilder(NetworkBuilder):
    """rl_games builder for 8-leg ant (ant_dynamic)."""

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            nn.Module.__init__(self)
            kwargs.pop('actions_num', None)
            kwargs.pop('input_shape', None)
            kwargs.pop('num_seqs', None)
            tc = params.get('transformer', {})
            self.net = DynamicLegTransformer(**tc)
            self.log_std_param = nn.Parameter(torch.zeros(16))

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            mu, value = self.net(obs)
            if obs.shape[-1] >= DYN_OBS_DIM + DYN_MASK_DIM:
                mask_dof = (obs[..., DYN_OBS_DIM : DYN_OBS_DIM + DYN_MASK_DIM] > 0.5).float()
                log_std = self.log_std_param - 10.0 * (1.0 - mask_dof)
            else:
                log_std = self.log_std_param.expand(obs.shape[0], -1)
            return mu, log_std, value, None

        def is_rnn(self):
            return False

    def build(self, name, **kwargs):
        return DynamicLegTransformerBuilder.Network(self.params, **kwargs)
