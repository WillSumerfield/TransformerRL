"""Controller training on bodies from the uniform grammar-action policy."""
from __future__ import annotations

import torch

from .codesign_agent import CodesignAgent
from .logging_agent import _LIMB_CODE


class UniformActionAgent(CodesignAgent):
    """Reuse CoDesign control learning without learning the morphology policy.

    The initial window uses the shared ``[1, 4, 6]`` body. At every later
    resample boundary, ``mode="uniform"`` chooses uniformly among the valid
    actions in the same grammar MDP used by CoDesign. No generator loss,
    generator critic fit, or control-cloning update is run.
    """

    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        self._uniform_log = None

    @torch.no_grad()
    def _maybe_resample(self):
        interval = self.config.get("resample_interval", 0)
        if not interval:
            return
        environment = self._env()
        if environment is None or not getattr(environment, "_sample_morphs", False):
            return

        self._steps_since_resample += self.horizon_length
        if self._steps_since_resample < interval * environment.max_episode_length:
            return

        returns = self._window_Ri() * self._r_scale
        trace = self._net().sample(environment.total_num_envs, mode="uniform")
        counts = trace["counts"].long()

        self._gen_window += 1
        self._last_R = returns
        self._cur_trace = trace
        self._cur_counts = counts
        self._cur_eff = trace["eff_sub"]
        self._cur_cap = trace["cap_sub"]
        self._uniform_log = {
            "return_mean": returns.mean().item(),
            "return_std": returns.std(unbiased=False).item(),
        }

        environment.set_next(counts, self._cur_eff, self._cur_cap)
        print(
            f"[uniform resample #{self._gen_window} | epoch {self.epoch_num}] "
            f"R_mean={returns.mean().item():.3f} "
            f"limbcount={(counts > 0).sum(1).float().mean().item():.2f} "
            f"modules={counts.sum(1).float().mean().item():.2f}",
            flush=True,
        )
        environment.resample()
        self.obs = self.env_reset()
        self.current_rewards.zero_()
        self.current_lengths.zero_()
        self._ep_ret.zero_()
        self._win_ret_sum.zero_()
        self._win_ret_cnt.zero_()
        self._morph_meta = None
        self._steps_since_resample = 0

    def write_stats(self, *args, **kwargs):
        """Keep normal PPO/aux logs and add the uniform body-source diagnostics."""
        super().write_stats(*args, **kwargs)
        if self.writer is None or self._uniform_log is None:
            return

        frame = args[11] if len(args) > 11 else kwargs.get("frame")
        counts = self._cur_counts
        prefix = "method/uniform_action"
        self.writer.add_scalar(
            f"{prefix}/return_mean",
            self._uniform_log["return_mean"],
            frame,
        )
        self.writer.add_scalar(
            f"{prefix}/return_std",
            self._uniform_log["return_std"],
            frame,
        )
        self.writer.add_scalar(
            f"{prefix}/mean_limbs",
            (counts > 0).sum(1).float().mean().item(),
            frame,
        )
        self.writer.add_scalar(
            f"{prefix}/mean_effectors",
            counts.sum(1).float().mean().item(),
            frame,
        )
        rates = (counts > 0).float().mean(0)
        for index, rate in enumerate(rates):
            self.writer.add_scalar(
                f"{prefix}/limb_presence/{_LIMB_CODE[index + 1]}",
                rate.item(),
                frame,
            )
        self._uniform_log = None
