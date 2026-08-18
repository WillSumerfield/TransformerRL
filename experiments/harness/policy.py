"""Checkpoint -> (net, obs normalizer), the read-only load path every analysis/eval script shares.

Extracted from the retired experiments/ppg_parity.py. Read-only w.r.t. training: it rebuilds the
network from the run's stamped `params.network` and loads the checkpoint's weights, nothing else.
obs_base / n_act are REQUIRED -- both come from the Task's published obs_layout(), never guessed.
"""
import torch
from rl_games.algos_torch.running_mean_std import RunningMeanStd

from transformer_rl.models import MultiMorphLimbTransformerBuilder

# Network config keys retired by the CoDesigner migration: slot count, chain depth and the module
# vocabulary are the live library's now, so a stamped config from an older run still names them.
# Dropped rather than honoured -- the library is the authority, and a checkpoint that disagrees with
# it would not load anyway.
_RETIRED_NET_KEYS = ("n_limbs", "max_limb_length", "module_library", "module_library_kwargs")


def _load_policy(ckpt, net_params: dict, device, *, obs_base: int, n_act: int, value_size: int = 1):
    net_params = dict(net_params)
    net_params["transformer"] = {k: val for k, val in net_params.get("transformer", {}).items()
                                 if k not in _RETIRED_NET_KEYS}
    obs_total = obs_base + (1 if value_size == 2 else 0)     # +progress dim for the V1.0 head
    net = MultiMorphLimbTransformerBuilder.Network(
        params=net_params, actions_num=n_act, input_shape=(obs_total,), num_seqs=1,
        value_size=value_size)
    raw = torch.load(ckpt, map_location=device, weights_only=False)["model"]
    state = {k.removeprefix("_orig_mod."): v for k, v in raw.items()}
    sub = lambda p: {k.removeprefix(p): v for k, v in state.items() if k.startswith(p)}
    net.load_state_dict(sub("a2c_network."))
    obs_norm = RunningMeanStd(obs_total).to(device)
    obs_norm.load_state_dict(sub("running_mean_std."))
    net.to(device).eval(); obs_norm.eval()
    return net, obs_norm
