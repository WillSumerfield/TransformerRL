"""skrl Policy (Gaussian) + Value (Deterministic) wrapping LegTransformer."""
import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin

from .transformer import LegTransformer
from .obs_tokenize import OBS_DIM

# qpos order [h1,a1,h2,a2,h3,a3,h4,a4] → actuator order [h4,a4,h1,a1,h2,a2,h3,a3]
_QPOS_TO_ACT = torch.tensor([6, 7, 0, 1, 2, 3, 4, 5], dtype=torch.long)


class Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device=None,
                 clip_actions=True, clip_log_std=True,
                 min_log_std=-5.0, max_log_std=2.0):
        Model.__init__(self, observation_space=observation_space,
                       action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions,
                               clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std,
                               reduction="none")
        self.net = LegTransformer("policy")
        self.log_std_param = nn.Parameter(torch.zeros(self.num_actions))
        self.register_buffer("_qpos_to_act", _QPOS_TO_ACT)
        self._active_mask: torch.Tensor | None = None

    def compute(self, inputs, role=""):
        mean = self.net(inputs["observations"])
        log_std = self.log_std_param.expand_as(mean)
        return mean, {"log_std": log_std}

    def _extract_mask(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.shape[-1] > OBS_DIM:
            mask_qpos = (obs[..., OBS_DIM:OBS_DIM + 8] > 0.5).float()
            return mask_qpos[..., self._qpos_to_act]
        return torch.ones(obs.shape[0], 8, device=obs.device)

    def act(self, inputs, role=""):
        actions, outputs = super().act(inputs, role=role)
        obs = inputs.get("observations")
        if obs is None:
            obs = inputs.get("states")
        mask = self._extract_mask(obs)
        self._active_mask = mask
        outputs["log_prob"] = (outputs["log_prob"] * mask).sum(-1, keepdim=True)
        return actions, outputs

    def get_entropy(self, *, role=""):
        ent = self._g_distribution.entropy().to(self.device)  # [B, 8]
        if self._active_mask is not None:
            ent = ent * self._active_mask
        return ent


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
    actions, outputs = pol.act({"observations": states}, role="policy")
    v, _ = val.act({"observations": states}, role="value")
    print("sampled action:", actions.shape, "range:", actions.min().item(), actions.max().item())
    print("log_prob:", outputs["log_prob"].shape, "value:", v.shape, v.item())
    assert actions.shape == (1, 8) and v.shape == (1, 1)
    assert actions.min() >= -1.0 - 1e-6 and actions.max() <= 1.0 + 1e-6, "clip_actions failed"

    # one env step end-to-end
    env.step(actions.detach().numpy()[0])
    print("end-to-end env step ok")
    env.close()
