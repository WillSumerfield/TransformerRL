"""skrl Policy (Gaussian) + Value (Deterministic) wrapping LegTransformer."""
import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

from .transformer import LegTransformer


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device=None,
                 clip_actions=True, clip_log_std=True,
                 min_log_std=-5.0, max_log_std=2.0):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions,
                               clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        self.net = LegTransformer("policy")
        self.log_std_param = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role=""):
        mean = self.net(inputs["observations"])
        # broadcast state-independent log_std to batch shape expected by skrl
        log_std = self.log_std_param.expand_as(mean)
        return mean, {"log_std": log_std}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device=None,
                 clip_actions=False):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        DeterministicMixin.__init__(self, clip_actions=clip_actions)
        self.net = LegTransformer("value")

    def compute(self, inputs, role=""):
        return self.net(inputs["observations"]), {}


if __name__ == "__main__":
    import gymnasium as gym
    env = gym.make("Ant-v5")
    o_space, a_space = env.observation_space, env.action_space

    pol = Policy(o_space, a_space, device="cpu")
    val = Value(o_space, a_space, device="cpu")

    obs, _ = env.reset(seed=0)
    states = torch.from_numpy(obs).float().unsqueeze(0)  # [1, 105]

    # sample from policy
    actions, outputs = pol.act({"states": states}, role="policy")
    v, _ = val.act({"states": states}, role="value")
    print("sampled action:", actions.shape, "range:", actions.min().item(), actions.max().item())
    print("log_prob:", outputs["log_prob"].shape, "value:", v.shape, v.item())
    assert actions.shape == (1, 8) and v.shape == (1, 1)
    assert actions.min() >= -1.0 - 1e-6 and actions.max() <= 1.0 + 1e-6, "clip_actions failed"

    # one env step end-to-end
    env.step(actions.detach().numpy()[0])
    print("end-to-end env step ok")
    env.close()
