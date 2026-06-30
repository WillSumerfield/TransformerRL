"""CodesignPlayer: play mode that samples bodies from the TRAINED single-network generator (the
shared trunk's GenAct head) instead of the fixed base morph. On each episode reset it unrolls the
generator (net.sample), rebuilds the env to those bodies, and runs the control policy on them -- so
you watch the morphology distribution the generator learned. Falls back to the env base build if the
checkpoint's net has no codesign generator. See codesign_agent.py."""
import torch
from rl_games.algos_torch.players import PpoPlayerContinuous


class CodesignPlayer(PpoPlayerContinuous):
    def restore(self, fn):
        super().restore(fn)                            # control model + obs/value normalizers
        net = self.model.a2c_network.net
        self._gen = getattr(net, 'codesign_tokens', False)
        print(f"[codesign-play] {'sampling fresh bodies from the trained generator each episode' if self._gen else 'no codesign generator in checkpoint; using env base morph'}",
              flush=True)

    def _env(self):
        base = getattr(self.env, 'envs', self.env)     # VlearnEnv -> NewToOldAPI
        return getattr(base, 'env', base)              # -> AntCodesignEnv

    @torch.no_grad()
    def env_reset(self, env):
        e = self._env()
        if self._gen and getattr(e, '_sample_morphs', False):
            presence = self.model.a2c_network.net.sample(e.total_num_envs)['presence']
            e.set_next(presence)
            e.resample()                               # full rebuild to the sampled bodies
            print(f"[codesign-play] sampled bodies: mean #legs="
                  f"{presence.sum(1).mean().item():.2f}", flush=True)
        return super().env_reset(env)
