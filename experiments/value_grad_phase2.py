"""Value-gradient Phase 2 — does dV/dp survive a continuous-p generator? (docs/value_gradient_propagation.md)

A small transformer (the morphology-generator analogue) is handed each leg's presence probability p
SCATTERED across tokens and must attend to gather it, regressing per-leg p_hat in [0,1] (MSE). We
pretrain it, freeze it, compose it in front of the frozen value net (its 8 p_hat outputs -> V's 8 tied
length slots, fed DIRECTLY as p), and backprop V to the generator's scattered-p inputs.

Puzzle input (B, 8 tokens, 16 dims): [0:8] one-hot of the token's own leg index; [8:16] scattered p —
for each leg j, exactly one random token holds the continuous value p_j at slot 8+j (others 0). To read
leg i, token i must attend across tokens to find p at its own slot 8+i.

Why the v2 recast fixes the v1 failure: targets are CONTINUOUS p (marginally uniform on [0,1]), so the
sigmoid head sits at interior probabilities — unsaturated — instead of being driven to the 0/1 rails by
binary BCE. Sigmoid still guarantees p_hat in [0,1] = exactly V's training domain.

Headlines (conditioning check, not mere existence):
  (a) SIGN-ALIGNMENT — composed per-leg grad (sum over tokens of dV/d scattered-p slot) agrees in sign
      with the DIRECT dV/dp at the same state;
  (b) NON-VANISHING NORM CHAIN — backward grad L2 norm from the generator OUTPUT (dV/dp_hat) back
      through the layers to the raw INPUT (dV/dx) does not collapse.
Writes data/value_grad_prop/phase2.npz (consumed by notebooks/value_grad_prop.ipynb).
"""
import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

import numpy as np
import torch
import torch.nn as nn
import yaml

import experiments.value_grad_prop as vg
from envs.ant_envs.ant_binary_leg import AntBinaryLegEnv

RUN            = "s43_nobern"  # value-net seed to compose against (override with argv[1])
N_PATTERNS     = 512        # reference states to align over
GEN_LAYERS     = 1
GEN_D          = 32
GEN_HEADS      = 4
PRETRAIN_STEPS = 3000
PRETRAIN_BS    = 2048
_LEN_OFF       = vg._LEN_OFF        # 107
_LEN_END       = _LEN_OFF + vg.LEN_DIM_8   # 123
_MASK          = vg._MASK_DIM      # 16


class PuzzleGen(nn.Module):
    """Transformer that gathers scattered p -> per-leg p_hat in [0,1] (sigmoid). forward_acts retains
    the full activation chain (raw input handled by the caller's leaf) so we can read backward grad
    norms: input embedding -> per-layer -> output p_hat."""
    def __init__(self, d=GEN_D, heads=GEN_HEADS, layers=GEN_LAYERS, ff=128):
        super().__init__()
        self.inp = nn.Linear(16, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, activation="gelu",
                                       batch_first=True, norm_first=True) for _ in range(layers)
        ])
        self.head = nn.Linear(d, 1)

    def _run(self, x, retain=False):
        h = self.inp(x); acts = []
        if retain: h.retain_grad(); acts.append(h)
        for lyr in self.layers:
            h = lyr(h)
            if retain: h.retain_grad(); acts.append(h)
        phat = torch.sigmoid(self.head(h).squeeze(-1))
        if retain: phat.retain_grad(); acts.append(phat)
        return phat, acts                                   # acts: [embed, post-layers..., p_hat]

    def forward(self, x):
        return self._run(x)[0]

    def forward_acts(self, x):
        return self._run(x, retain=True)


def make_puzzle(p: torch.Tensor):
    """p (B,8) float in [0,1] -> puzzle input (B,8,16). One-hot self + one random token per leg's p."""
    B, dev = p.shape[0], p.device
    x = torch.zeros(B, 8, 16, device=dev)
    idx = torch.arange(8, device=dev)
    x[:, idx, idx] = 1.0                                        # one-hot self index
    t = torch.randint(0, 8, (B, 8), device=dev)                 # random token holding leg j's p
    bb = torch.arange(B, device=dev)[:, None].expand(B, 8)
    x[bb, t, 8 + idx[None, :].expand(B, 8)] = p                 # scatter continuous p
    return x


@torch.no_grad()
def _pred_mse(gen, device):
    p = torch.rand(8192, 8, device=device)
    return ((gen(make_puzzle(p)) - p) ** 2).mean().item()


def composed_grad(gen, net, obs_norm, val_norm, ref_obs, p_true, reward_scale):
    """Per-leg composed grad (sum over tokens of dV/d scattered-p slot) + the output->input grad-norm
    chain (raw input dV/dx, then embed, per-layer, p_hat)."""
    x = make_puzzle(p_true).detach().requires_grad_(True)
    phat, acts = gen.forward_acts(x)
    obs_len = torch.cat([phat, phat], dim=1)                    # both length slots = p_hat, fed direct
    o = ref_obs.detach()
    full = torch.cat([o[:, :_LEN_OFF], obs_len, o[:, _LEN_END:]], dim=1)
    normed = obs_norm(full).clone()
    normed[..., -_MASK:] = full[..., -_MASK:]                   # mask tail raw (TransformerMaskedNorm)
    _, _, value, _ = net({"obs": normed})
    val = val_norm(value.view(-1, 1), denorm=True).squeeze(-1) / reward_scale
    val.sum().backward()
    per_leg = x.grad[:, :, 8:16].sum(dim=1)                     # (B,8) composed sensitivity to leg p
    # chain output->input: p_hat, post-layers (reversed), embed, raw input
    norms = [a.grad.norm().item() for a in reversed(acts)] + [x.grad.norm().item()]
    return per_leg.detach(), phat.detach(), np.array(norms, dtype=np.float32)


def main():
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    run = sys.argv[1] if len(sys.argv) > 1 else RUN
    cfg = yaml.safe_load(vg.CONFIG.read_text())
    reward_scale = cfg["params"]["config"].get("reward_shaper", {}).get("scale_value", 1.0)
    off_scale = cfg.get("env", {}).get("off_scale", 0.05)
    net, obs_norm, val_norm = vg._load(vg._checkpoint(run), cfg, device)

    # Reference states: t=0 reset of a sampled-p population; direct dV/dp at the env's true p.
    env = AntBinaryLegEnv(N_PATTERNS, device, off_scale=off_scale, rendering=False,
                          raise_exception=False, seed=321, with_window=False)
    ref_obs, _ = env.reset()
    p_true = torch.as_tensor(env._p, dtype=torch.float32, device=device)        # (N,8)
    direct = vg._grad_value(net, obs_norm, val_norm, ref_obs, reward_scale)[0].cpu().numpy()  # dV/dp

    # Pretrain the continuous-p generator (MSE regression). One converged operating point — no sweep.
    torch.manual_seed(0)
    gen = PuzzleGen().to(device)
    opt = torch.optim.Adam(gen.parameters(), 1e-3)
    mse = nn.MSELoss()
    for step in range(PRETRAIN_STEPS):
        p = torch.rand(PRETRAIN_BS, 8, device=device)          # uniform = marginal of the √U mix
        opt.zero_grad(); mse(gen(make_puzzle(p)), p).backward(); opt.step()
    pred_mse = _pred_mse(gen, device)

    composed, phat, chain = composed_grad(gen, net, obs_norm, val_norm, ref_obs, p_true, reward_scale)
    composed = composed.cpu().numpy()
    sign_match = float((np.sign(direct) == np.sign(composed)).mean())
    corr = float(np.corrcoef(direct.ravel(), composed.ravel())[0, 1])

    out = _ROOT / "data" / "value_grad_prop" / "phase2.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, direct=direct, composed=composed, phat=phat.cpu().numpy(),
                        p_true=p_true.cpu().numpy(), chain_norms=chain, run=run,
                        reward_scale=reward_scale, pred_mse=pred_mse,
                        sign_match=sign_match, corr=corr)
    print(f"[phase2] generator pred MSE {pred_mse:.4f}  (continuous-p regression)")
    print(f"[phase2] sign-align(composed vs direct dV/dp) = {sign_match:.3f}   corr = {corr:+.3f}")
    print(f"[phase2] grad-norm chain output->input (p_hat, layers, embed, raw-in):")
    print(f"         {np.array2string(chain, precision=3)}  (input-side not collapsed = no vanishing)")
    print(f"[phase2] wrote -> {out}")


if __name__ == "__main__":
    main()
