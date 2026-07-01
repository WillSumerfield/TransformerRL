"""Collect per-token attention weights over episodes for a trained LimbTransformer policy."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

import argparse
import numpy as np
import torch
import yaml
from tqdm import tqdm
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from transformer_rl.models import LimbTransformerBuilder
from envs.ant_envs import AntEnv


def _load(checkpoint: Path, config: dict, device: torch.device):
    network = LimbTransformerBuilder.Network(
        params=config["params"]["network"],
        actions_num=8,
        input_shape=(59,),
        num_seqs=1,
    )

    raw = torch.load(checkpoint, map_location=device, weights_only=False)["model"]
    state = {k.removeprefix("_orig_mod."): v for k, v in raw.items()}

    net_state = {k.removeprefix("a2c_network."): v
                 for k, v in state.items() if k.startswith("a2c_network.")}
    network.load_state_dict(net_state)

    obs_norm = RunningMeanStd(59).to(device)
    rms_state = {k.removeprefix("running_mean_std."): v
                 for k, v in state.items() if k.startswith("running_mean_std.")}
    obs_norm.load_state_dict(rms_state)
    obs_norm.eval()

    # Train mode forces the Python path in TransformerEncoder, bypassing the C++ fast
    # path that would skip _sa_block. dropout=0.0 so numerics are identical.
    network.train()

    return network.to(device), obs_norm


def _patch_attn(encoder):
    captures = [[] for _ in encoder.layers]
    for i, layer in enumerate(encoder.layers):
        def _make(lay, cap):
            def patched(x, attn_mask, key_padding_mask, is_causal=False):
                out, w = lay.self_attn(x, x, x,
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                    is_causal=is_causal)
                cap.append(w.detach().cpu())
                return lay.dropout1(out)
            return patched
        layer._sa_block = _make(layer, captures[i])
    return captures


def run(seed: int, episodes: int, checkpoint: Path, config: Path, output: Path):
    cfg = yaml.safe_load(config.read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    network, obs_norm = _load(checkpoint, cfg, device)
    captures = _patch_attn(network.net.encoder)
    n_layers = len(captures)

    env = AntEnv(1, device, rendering=False, raise_exception=False, seed=seed)

    all_attn, all_rewards, episode_ends = [], [], []

    obs, _ = env.reset()
    for _ in tqdm(range(episodes), desc="episodes"):
        ep_done = False
        while not ep_done:
            with torch.no_grad():
                mu, _, _, _ = network({"obs": obs_norm(obs)})
            obs, rew, term, trunc, _ = env.step(mu)
            all_attn.append(
                np.stack([captures[l][-1][0].numpy() for l in range(n_layers)])
            )  # (n_layers, n_heads, n_tokens, n_tokens)
            all_rewards.append(float(rew[0]))
            ep_done = bool((term | trunc)[0].item())
        episode_ends.append(len(all_attn))

    output.parent.mkdir(parents=True, exist_ok=True)
    attn_arr = np.array(all_attn, dtype=np.float32)
    np.savez_compressed(
        output,
        actor_attn   = attn_arr,
        critic_attn  = attn_arr,   # shared encoder; stored twice for notebook compat
        rewards      = np.array(all_rewards,  dtype=np.float32),
        episode_ends = np.array(episode_ends, dtype=np.int64),
    )
    print(f"saved {len(all_attn)} steps, {episodes} episodes → {output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed",       type=int,  required=True)
    p.add_argument("--episodes",   type=int,  default=100)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--config",     type=Path, default=Path("configs/ppo_ant.yaml"))
    p.add_argument("--output",     type=Path, default=None)
    args = p.parse_args()

    if args.output is None:
        args.output = Path(f"data/attention_over_time_seed{args.seed}.npz")

    run(args.seed, args.episodes, args.checkpoint, args.config, args.output)
