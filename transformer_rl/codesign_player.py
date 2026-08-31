"""CodesignPlayer: play mode that samples bodies from the TRAINED single-network generator (the
shared trunk's GenAct head) instead of the fixed base morph. On each episode reset it unrolls the
generator (net.sample), rebuilds the env to those bodies, and runs the control policy on them -- so
you watch the morphology distribution the generator learned. Falls back to the env base build if the
BUILT net has no codesign generator (the flag is a property of the architecture, not of the loaded
file) -- and `play` refuses to start without a checkpoint, so restore() always runs. See
codesign_agent.py."""
import torch
from rl_games.algos_torch.players import PpoPlayerContinuous

from .morphology import designs_from_arrays
from .policy_switch import SwitchMixin


class SwitchablePlayer(SwitchMixin, PpoPlayerContinuous):
    """Standard continuous player with live checkpoint switching (a2c / ppg play)."""


class CodesignPlayer(SwitchMixin, PpoPlayerContinuous):
    def restore(self, fn):
        super().restore(fn)                            # control model + obs/value normalizers
        net = self.model.a2c_network.net
        self._gen = getattr(net, 'codesign_tokens', False)
        print(f"[codesign-play] {'sampling fresh bodies from the trained generator each episode' if self._gen else 'no codesign generator in checkpoint; using env base morph'}",
              flush=True)

    def _env(self):
        base = getattr(self.env, 'envs', self.env)     # VlearnEnv -> NewToOldAPI
        return getattr(base, 'env', base)              # -> the codesigner Task

    @torch.no_grad()
    def env_reset(self, env):
        e = self._env()
        if self._gen:
            tr = self.model.a2c_network.net.sample(e.total_num_envs)
            counts = tr['counts'].long()
            morphs = designs_from_arrays(e.module_library, counts, tr['eff_sub'], tr['cap_sub'])
            e.resample(morphs)                         # full rebuild to the sampled bodies
            print(f"[codesign-play] sampled bodies: mean #limbs="
                  f"{(counts > 0).sum(1).float().mean().item():.2f} "
                  f"modules={counts.sum(1).float().mean().item():.2f}", flush=True)
        return super().env_reset(env)
