"""Experiment 4's measurement E: the learned attention map over the module tokens.

    python experiments/harness/attnmap.py attention          # -> data/paper/attn_attention.npz
    python experiments/harness/attnmap.py attention --arms full,self --ckpts 1 --dry-run

**This is the interpretability gate, not a nice-to-have.** If `full` beats `self_cls` on asymptotic
return, the claim is that cross-limb information did the work -- and a near-diagonal learned map
would mean the gap came from something else and the result is not yet explained. The number that
decides it is `attn_offdiag`: the share of a present module token's attention that lands on the OTHER
present module tokens. Its floor is what `self_cls` structurally cannot exceed, which is 0, so the
same pass run on the ablated arms is also the check that the mask does what it claims.

**Eval-time only.** `F.scaled_dot_product_attention` is a fused kernel and returns no weights, so the
map has to be recomputed as an explicit `softmax(QK^T/sqrt(d) + mask)`. The training forward is not
touched and must not be -- it stays on the fused path.

**The layer input is captured, not recomputed.** A forward pre-hook on the encoder layer takes the
exact tensor the trunk was about to attend over, so the embed stack (content one-hots, additive
pos/depth, RoPE phases) is the training one by construction rather than by a second implementation
that agrees today. Everything after the hook -- the QKV projection, the rotary application, the
scale -- is read off the layer's own modules and buffers for the same reason.

Averaged over real states: the policy is rolled at mu on the run's fixed body and the weights
accumulated per step, because a map read at reset would describe the ant standing still.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from experiments.harness.evalpass import forward, load_net, open_task    # noqa: E402
from experiments.harness.launch import STUDIES, study_runs               # noqa: E402
from experiments.harness.slots import _ckpt_epoch                        # noqa: E402
from transformer_rl.architectures import _rotate_half                    # noqa: E402

ENVS = 64               # states are averaged over, so a wide batch is cheaper than a long rollout
STEPS = 64              # env steps the map is averaged over
CKPTS = 3               # checkpoints per run, evenly spaced over what the run saved
LAYER = 0               # `n_layers: 1` in the series config, so layer 0 is the only round of mixing


def sidecar(study: str) -> Path:
    return _ROOT / "data" / "paper" / f"attn_{study}.npz"


# ---- the weights the fused kernel does not return ----------------------------------

class Weights:
    """Attention weights of one encoder layer, accumulated over however many forwards happen.

    Installed as a forward pre-hook so it sees the layer's input rather than rebuilding it. The
    layer is `_CustomEncoderLayer`, which is the only encoder experiment 4 runs on -- `attn_scope`
    refuses the stock `nn.TransformerEncoder` at construction, for the polarity reason documented
    there, so there is no second case to handle.
    """

    def __init__(self, encoder, layer: int = LAYER):
        self.layer = encoder.layers[layer]
        self.cos, self.sin = encoder.cos, encoder.sin
        self.sum = None
        self.n = 0
        self._h = self.layer.register_forward_pre_hook(self._hook, with_kwargs=True)

    def _hook(self, module, args, kwargs):
        x = args[0]
        mask = kwargs.get("attn_mask")
        B, T, D = x.shape
        nh, hd = module.nhead, module.head_dim
        h = module.norm1(x)
        qkv = module.qkv_proj(h).view(B, T, 3, nh, hd).permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        q = q * self.cos + _rotate_half(q) * self.sin       # the same rotary the layer applies
        k = k * self.cos + _rotate_half(k) * self.sin
        scores = (q @ k.transpose(-1, -2)) / (hd ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        a = F.softmax(scores, dim=-1).mean(0)               # state-averaged -> (n_heads, T, T)
        self.sum = a if self.sum is None else self.sum + a
        self.n += 1

    @property
    def mean(self):
        return (self.sum / self.n).float().cpu().numpy()

    def close(self):
        self._h.remove()


def offdiag_share(a: np.ndarray, present: np.ndarray, content_start: int) -> float:
    """Share of a present module token's attention that lands on the OTHER present module tokens.

    Restricted to present tokens both as query and as key: pad slots deeper than a limb's cap are
    never attention-masked (matching phase 1), so leaving them in would report the network's opinion
    of tokens that are not part of the body. Self-attention is excluded -- a token attending to
    itself is exactly the information `self` already has, and this number is what `full` has beyond
    it.
    """
    idx = content_start + np.flatnonzero(present)
    if idx.size < 2:
        return float("nan")
    block = a[:, idx][:, :, idx]                            # (heads, P, P)
    eye = np.eye(idx.size, dtype=bool)
    return float(block[:, ~eye].sum() / block.sum())


# ---- one checkpoint ----------------------------------------------------------------

@torch.no_grad()
def map_one(net, obs_norm, env, device, *, steps: int = STEPS, layer: int = LAYER):
    """(state-averaged (n_heads, T, T) map, off-diagonal share) for one loaded policy."""
    w = Weights(net.net.encoder, layer)
    obs, _ = env.reset()
    for _ in range(steps):
        mu, _ = forward(net, obs_norm, obs)                 # the hook fires inside this
        obs, *_ = env.step(mu)
    w.close()
    a = w.mean
    _, _, active, cap, _ = net.net._tokenize_modules(obs)
    present = (active + cap)[0].bool().cpu().numpy()        # same body in every env
    return a, offdiag_share(a, present, net.net._content_start)


def _checkpoints(run, n: int) -> list[Path]:
    """`n` of a run's periodic saves, evenly spaced and always including the last."""
    have = sorted((p for p in run.nn_dir.glob("last_*.pth") if _ckpt_epoch(p) > 0),
                  key=_ckpt_epoch)
    if len(have) <= n:
        return have
    return [have[int(round(i))] for i in np.linspace(0, len(have) - 1, n)]


# ---- the study ---------------------------------------------------------------------

def run_study(study: str, *, arms=None, seeds=None, envs: int = ENVS, steps: int = STEPS,
              n_ckpts: int = CKPTS, layer: int = LAYER, device: str = "cuda:0") -> Path:
    spec = STUDIES[study]
    arms = list(arms or spec.arms)
    seeds = list(seeds or spec.seeds)
    dev = torch.device(device)
    env = layout = None
    maps: dict[tuple, list] = {}
    epochs = np.zeros((len(arms), len(seeds), n_ckpts), int)
    share = np.full((len(arms), len(seeds), n_ckpts), np.nan)

    for run in study_runs(study, arms, tuple(seeds)):
        ckpts = _checkpoints(run, n_ckpts)
        if not ckpts:
            print(f"[attnmap] no checkpoints for {run.name}", flush=True)
            continue
        cfg = yaml.safe_load((run.run_dir / "config.yaml").read_text())
        if env is None:
            env, _, layout = open_task(cfg, envs, device=dev)
        i = (arms.index(run.meta["arm"]), seeds.index(run.meta["seed"]))
        for c, ckpt in enumerate(ckpts):
            net, obs_norm = load_net(ckpt, cfg, layout, dev)
            a, off = map_one(net, obs_norm, env, dev, steps=steps, layer=layer)
            maps[(*i, c)] = a
            share[(*i, c)] = off
            epochs[(*i, c)] = _ckpt_epoch(ckpt)
            print(f"[attnmap] {run.name} @ epoch {epochs[(*i, c)]}: "
                  f"off-diagonal module mass {off:.3f}", flush=True)
            del net, obs_norm
            torch.cuda.empty_cache()

    if not maps:
        raise SystemExit(f"[attnmap] no checkpoints found for '{study}'")
    heads, T, _ = next(iter(maps.values())).shape
    attn = np.full((len(arms), len(seeds), n_ckpts, heads, T, T), np.nan, np.float32)
    for key, a in maps.items():
        attn[key] = a
    path = sidecar(study)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, attn_map=attn, attn_offdiag=share, attn_epochs=epochs,
             arms=np.array(arms), seeds=np.array(seeds), steps=steps, envs=envs, layer=layer)
    print(f"[attnmap] {len(maps)} map(s) -> {path}", flush=True)
    return path


def main():
    ap = argparse.ArgumentParser(description="Experiment 4 measurement E: the learned attention map")
    ap.add_argument("study", choices=sorted(STUDIES))
    ap.add_argument("--arms", default="full",
                    help="comma-separated; default `full`, the only arm the measurement is about. "
                         "Pass the ablated arms to verify the mask instead")
    ap.add_argument("--seeds", default=None, help="comma-separated; default the study's")
    ap.add_argument("--envs", type=int, default=ENVS)
    ap.add_argument("--steps", type=int, default=STEPS, help="env steps the map is averaged over")
    ap.add_argument("--ckpts", type=int, default=CKPTS, help="checkpoints per run")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    args = ap.parse_args()

    arms = args.arms.split(",") if args.arms else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    if args.dry_run:
        for run in study_runs(args.study, arms, tuple(seeds) if seeds else None):
            got = _checkpoints(run, args.ckpts)
            print(f"  {run.name:30} {[_ckpt_epoch(p) for p in got] or 'MISSING'}")
        return
    run_study(args.study, arms=arms, seeds=seeds, envs=args.envs, steps=args.steps,
              n_ckpts=args.ckpts, layer=args.layer)


if __name__ == "__main__":
    main()
