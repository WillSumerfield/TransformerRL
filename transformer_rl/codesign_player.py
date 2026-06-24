"""CodesignPlayer: play mode that samples bodies from the TRAINED generator distribution (not the
fixed base morph). On each episode reset it draws fresh per-env presence from the restored Generator,
rebuilds the env to those bodies, and runs the control policy on them -- so you watch the morphology
distribution the generator learned. Falls back to the env's base build if the checkpoint has no
generator state. See codesign_agent.py / ADR-0010."""
import torch
from rl_games.algos_torch.players import PpoPlayerContinuous

from .generator import Generator


class CodesignPlayer(PpoPlayerContinuous):
    def restore(self, fn):
        super().restore(fn)                       # control model + obs/value normalizers
        ck = torch.load(fn, map_location=self.device, weights_only=False)
        if 'gen' in ck:
            self.gen = Generator(device=self.device)
            self.gen.load_state_dict(ck['gen'])
            p = torch.sigmoid(self.gen.logits).tolist()
            print(f"[codesign-play] generator restored; p(leg)={[round(x, 3) for x in p]} "
                  f"-> sampling fresh bodies each episode", flush=True)
        else:
            self.gen = None
            print("[codesign-play] no generator in checkpoint; using env base morph", flush=True)

    def _env(self):
        base = getattr(self.env, 'envs', self.env)     # VlearnEnv -> NewToOldAPI
        return getattr(base, 'env', base)              # -> AntCodesignEnv

    @torch.no_grad()
    def env_reset(self, env):
        e = self._env()
        if self.gen is not None and getattr(e, '_sample_morphs', False):
            presence, _ = self.gen.sample(e.total_num_envs)
            e.set_next(presence)
            e.resample()                               # full rebuild to the sampled bodies
            print(f"[codesign-play] sampled bodies: mean #legs="
                  f"{presence.sum(1).mean().item():.2f}", flush=True)
        return super().env_reset(env)
