"""Offline body-held-out response probe. Run with --help for arguments."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn


def split_bodies(ids, seed):
    bodies = np.unique(ids)
    if len(bodies) < 10:
        raise ValueError('Need at least 10 unique morphologies for train/validation/test')
    bodies = np.random.default_rng(seed).permutation(bodies)
    a, b = int(.7*len(bodies)), int(.85*len(bodies))
    return [np.flatnonzero(np.isin(ids, group)) for group in (bodies[:a], bodies[a:b], bodies[b:])]


def swap_context(data, rows, seed):
    """Swap at module level, same subtype/depth, different body; repeat across time."""
    rng = np.random.default_rng(seed)
    donors = rows.copy()
    eligible = np.zeros(len(rows), bool)
    groups = data['match_group'][rows]
    bodies = data['body_id'][rows]
    modules = data['module_id'][rows]
    for body, module in sorted(set(zip(bodies, modules))):
        positions = np.flatnonzero((bodies == body) & (modules == module))
        candidates = np.flatnonzero((groups == groups[positions[0]]) & (bodies != body))
        if len(candidates):
            donors[positions] = rows[rng.choice(candidates)]
            eligible[positions] = True
    return donors, eligible


def evaluate(data, seed=42, epochs=100):
    if epochs < 1:
        raise ValueError('epochs must be positive')
    for key in ('interaction', 'target', 'metadata', 'pre', 'post'):
        if not np.isfinite(data[key]).all():
            raise ValueError(f'Non-finite values in {key}')
    torch.set_num_threads(2)
    train, val, test = split_bodies(data['body_id'], seed)
    target = data['target'].astype(np.float32)
    mean, std = target[train].mean(0), target[train].std(0).clip(1e-4)
    y = torch.from_numpy((target-mean)/std)
    width = max(data[k].shape[1] for k in ('pre', 'post'))
    report = {'protocol': 'metadata_plus_context_v1', 'epochs': epochs,
              'seed': seed, 'splits': {name: {'bodies': len(np.unique(data['body_id'][idx])),
              'rows': len(idx)} for name, idx in zip(('train','validation','test'), (train,val,test))},
              'models': {}}
    # Zero physical delta is a strong baseline for adjacent simulation steps.
    persistence = ((target[test]/std)**2).mean(1)
    report['zero_delta_body_mean_normalized_mse'] = float(np.mean([
        persistence[data['body_id'][test] == body].mean()
        for body in np.unique(data['body_id'][test])]))
    body_errors = {}
    for name in ('interaction_only', 'metadata', 'pre', 'post'):
        morph = np.zeros((len(target), width), np.float32)
        metadata = np.zeros_like(data['metadata'])
        if name != 'interaction_only':
            metadata = data['metadata']
        if name in ('pre', 'post'):
            morph[:, :data[name].shape[1]] = data[name]
        x_raw = np.concatenate((data['interaction'], metadata, morph), 1).astype(np.float32)
        xm, xs = x_raw[train].mean(0), x_raw[train].std(0).clip(1e-4)
        x = torch.from_numpy((x_raw-xm)/xs)
        # Equal input width, architecture, initialization and minibatch ordering.
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(x.shape[1],64), nn.SiLU(), nn.Linear(64,64),
                              nn.SiLU(), nn.Linear(64,y.shape[1]))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        rng = np.random.default_rng(seed)
        best, best_state, best_epoch = float('inf'), None, 0
        for epoch in range(epochs):
            for batch in np.array_split(rng.permutation(train), max(1, (len(train)+511)//512)):
                loss = (model(x[batch])-y[batch]).square().mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                score = (model(x[val])-y[val]).square().mean().item()
            if score < best:
                best, best_epoch = score, epoch+1
                best_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_state)
        with torch.no_grad():
            pred = model(x[test]).numpy()
        error = ((pred-y[test].numpy())**2).mean(1)
        bodies = np.unique(data['body_id'][test])
        per_body = np.array([error[data['body_id'][test] == b].mean() for b in bodies])
        body_errors[name] = per_body
        metrics = {'test_body_mean_normalized_mse': float(per_body.mean()),
                   'inputs': ['interaction'] + ([] if name == 'interaction_only' else ['metadata'])
                             + ([name] if name in ('pre', 'post') else []),
                   'best_validation_mse': best, 'best_epoch': best_epoch,
                   'parameters': sum(p.numel() for p in model.parameters()),
                   'test_per_dimension_mse': ((pred-y[test].numpy())**2).mean(0).tolist()}
        if name == 'post':
            donors, eligible = swap_context(data, test, seed)
            swapped = x_raw[test].copy()
            swapped[:, -width:] = morph[donors]
            with torch.no_grad():
                sp = model(torch.from_numpy((swapped-xm)/xs)).numpy()
            se = ((sp-y[test].numpy())**2).mean(1)
            metrics['swap_eligible_rows'] = int(eligible.sum())
            metrics['swap_mse_increase_matched_rows'] = float((se-error)[eligible].mean()) if eligible.any() else None
        report['models'][name] = metrics
    for name in ('metadata', 'pre', 'interaction_only'):
        delta = body_errors[name]-body_errors['post']
        rng = np.random.default_rng(seed)
        boot = np.array([rng.choice(delta, len(delta)).mean() for _ in range(2000)])
        report[f'{name}_minus_post'] = {'body_mean_mse_gain':float(delta.mean()),
                                      'body_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist()}
    report['decision'] = 'Feasibility only: replicate across controller seeds before integration.'
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('captures', nargs='+', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=100)
    args = p.parse_args()
    if args.epochs < 1:
        p.error('--epochs must be positive')
    started = time.perf_counter()
    keys = ('interaction','target','metadata','pre','post','body_id','module_id','match_group')
    chunks = []
    for path in args.captures:
        with np.load(path, allow_pickle=False) as f:
            chunks.append({k:f[k] for k in keys})
    data = {k:np.concatenate([c[k] for c in chunks]) for k in keys}
    report = evaluate(data,args.seed,args.epochs)
    report['sources'] = [str(p) for p in args.captures]
    report['seconds'] = time.perf_counter()-started
    report['limitations'] = ['Observational prediction, not causal intervention.',
        'Actuated modules only; predicts joint velocity and relative velocity deltas.',
        'Different controller checkpoints may confound morphology comparisons.',
        'Local conditioning does not include all other joints actions or whole-body state.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
