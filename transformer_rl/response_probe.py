"""Detached, opt-in capture for the static morphology -> physical response probe.

No environment stepping, optimizer access, or random sampling. Rows are actuated
modules from up to eight late valid transitions per selected environment.
"""
from pathlib import Path
import hashlib
import time

import numpy as np
import torch
import torch.nn.functional as F

from .tokenize import tokenize_modules, token_dims


@torch.no_grad()
def capture_response_probe(agent, directory, max_bodies=256, samples=8):
    started = time.perf_counter()
    net = agent._net()
    buf = agent.experience_buffer.tensor_dict
    obs, dones = buf['obses'], buf['dones'].bool()
    dones = dones.reshape(obs.shape[:2])
    n, depth = net.n_limbs, net.max_limb_length
    dims = token_dims(n, depth)
    if obs.shape[-1] != dims['obs_total']:
        raise ValueError('Response probe requires the Phase-5 typed observation layout')
    if samples < 1 or max_bodies < 1:
        raise ValueError('samples and max_bodies must be positive')
    count = torch.as_tensor(agent._cur_counts, device=obs.device).long()
    eff = torch.as_tensor(agent._cur_eff, device=obs.device).long()
    cap = torch.as_tensor(agent._cur_cap, device=obs.device).long()
    hashes = [hashlib.sha256(torch.cat((count[b].flatten(), eff[b].flatten(),
              cap[b].flatten())).cpu().numpy().tobytes()).hexdigest() for b in range(len(count))]
    # Select unique bodies deterministically, avoiding population-duplicate weighting.
    selected, seen = [], set()
    for b, key in enumerate(hashes):
        if key not in seen:
            selected.append(b)
            seen.add(key)
        if len(selected) >= max_bodies:
            break
    fields = {k: [] for k in ('interaction', 'target', 'metadata', 'pre', 'post',
                              'body_id', 'module_id', 'match_group')}
    flags = [(m, m.training) for m in net.modules()]
    net.eval()
    try:
        for b in selected:
            # dones[t+1] denotes a reset observation, as in the existing FD path.
            valid = torch.nonzero(~dones[1:, b], as_tuple=False).flatten()
            valid = valid[valid >= (obs.shape[0] - 1) // 2]
            if not len(valid):
                continue
            take = torch.linspace(0, len(valid)-1, min(samples, len(valid)),
                                  device=obs.device).long()
            ts = valid[take]
            current, nxt = obs[ts, b], obs[ts+1, b]
            _, local, active, _, sub = tokenize_modules(current, n, depth)
            _, future, future_active, _, _ = tokenize_modules(nxt, n, depth)
            # Physical responses only for actuated modules (caps have masked kinematics).
            keep = (active > 0) & (future_active > 0)
            # Capture the actual pre-attention static input, including positional embedding.
            pre = []
            handle = net.encoder.register_forward_pre_hook(lambda m, args: pre.append(args[0].detach()))
            try:
                post = net._encode_design(count[b:b+1], cap[b:b+1], eff[b:b+1]).detach()
            finally:
                handle.remove()
            pre = pre[0][:, net._content_start:]
            post = post[:, net._content_start:]
            slots = torch.arange(n*depth, device=obs.device)
            limb, dep = slots % n, slots // n
            # Applied actions use the trainer's exact clamp/rescale path.
            actions = torch.as_tensor(agent.preprocess_actions(buf['actions'][ts, b]), device=obs.device)
            contact = current[:, dims['sens_off']:dims['obs_base']].reshape(-1, n, 6)[:, limb]
            # Local state (including previous action), current applied action and limb contact.
            interaction = torch.cat((local, actions.unsqueeze(-1), contact), -1)
            target = torch.cat((future[..., 2:3]-local[..., 2:3],
                                future[..., 19:25]-local[..., 19:25]), -1)
            parent_sub = sub.clone().zero_()
            parent_sub[:, n:] = sub[:, :-n]
            metadata = torch.cat((sub, parent_sub,
                F.one_hot(dep, depth).float().expand(len(ts), -1, -1),
                F.one_hot(limb, n).float().expand(len(ts), -1, -1)), -1)
            for name, value in [('interaction', interaction), ('target', target),
                                ('metadata', metadata), ('pre', pre.expand(len(ts), -1, -1)),
                                ('post', post.expand(len(ts), -1, -1))]:
                fields[name].append(value[keep].detach().float().cpu().numpy())
            mslots = slots.expand(len(ts), -1)[keep].cpu().numpy()
            fields['body_id'].append(np.repeat(hashes[b], len(mslots)))
            fields['module_id'].append(mslots)
            fields['match_group'].append((sub.argmax(-1)*depth + dep)[keep].cpu().numpy())
    finally:
        for module, training in flags:
            module.training = training
    if not fields['target']:
        raise ValueError('No valid late transitions available for response probe')
    result = {k: np.concatenate(v) for k, v in fields.items()}
    result['capture_seconds'] = np.array(time.perf_counter()-started)
    result['epoch'] = np.array(agent.epoch_num)
    result['window'] = np.array(agent._gen_window)
    result['schema_version'] = np.array(1)
    path = Path(directory) / f'response_e{agent.epoch_num:06d}_w{agent._gen_window:04d}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    # Refuse to replace a prior capture when a run is resumed.
    with path.open('xb') as stream:
        np.savez_compressed(stream, **result)
    return path
