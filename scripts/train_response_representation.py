"""Offline supervised morphology encoders; no PPO or simulator access."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformer_rl.architectures import MultiMorphLimbTransformer
from scripts.probe_static_response import split_bodies


def load_bodies(data, archive):
    with np.load(archive, allow_pickle=False) as f:
        arrays = [f[k].astype(np.int64) for k in ('counts', 'eff_sub', 'cap_sub')]
    lookup = {}
    for i in range(len(arrays[0])):
        key = hashlib.sha256(np.concatenate([a[i].flatten() for a in arrays]).tobytes()).hexdigest()
        lookup[key] = i
    bodies, inverse = np.unique(data['body_id'], return_inverse=True)
    missing = set(bodies)-lookup.keys()
    if missing:
        raise ValueError(f'{len(missing)} capture bodies missing from morphology archive')
    take = [lookup[b] for b in bodies]
    return [torch.from_numpy(a[take]) for a in arrays], inverse


class ResponseModel(nn.Module):
    def __init__(self, architecture, inputs, metadata_dim, arm, pretrained):
        super().__init__()
        self.arm = arm
        width = architecture['d_model']
        if arm in ('fresh', 'initialized'):
            self.body = MultiMorphLimbTransformer(**architecture)
            if arm == 'initialized':
                self.body.load_state_dict(pretrained, strict=True)
            for name, parameter in self.body.named_parameters():
                parameter.requires_grad_(name.startswith(('encoder.', 'embed_module.',
                    'angle_proj.', 'pos_emb.', 'depth_emb.', 'cls_design')))
        elif arm == 'local_learned':
            self.local = nn.Sequential(nn.Linear(metadata_dim,384),nn.SiLU(),
                                       nn.Linear(384,384),nn.SiLU(),nn.Linear(384,width))
        # Same decoder initialization and dimensions across all arms.
        torch.manual_seed(self.decoder_seed)
        self.decoder = nn.Sequential(nn.Linear(inputs+width,64),nn.SiLU(),
                                     nn.Linear(64,64),nn.SiLU(),nn.Linear(64,7))
        self.width = width

    def forward(self, x, metadata, structures, body_index, module_index):
        if self.arm in ('fresh','initialized'):
            unique, inverse = torch.unique(body_index, return_inverse=True)
            count, eff, cap = [a[unique] for a in structures]
            h = self.body._encode_design(count,cap,eff)[:,self.body._content_start:]
            z = h[inverse,module_index]
        elif self.arm == 'local_learned':
            z = self.local(metadata)
        else:
            z = x.new_zeros(len(x),self.width)
        return self.decoder(torch.cat((x,z),-1))


def run(args):
    torch.set_num_threads(2)
    with np.load(args.capture,allow_pickle=False) as f:
        data = {k:f[k] for k in ('interaction','metadata','target','body_id','module_id')}
        capture_epoch = int(f['epoch'])
    structures, inverse = load_bodies(data,args.morphologies)
    config = yaml.safe_load(args.config.read_text())
    architecture = config['params']['network']['transformer']
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['epoch'] >= capture_epoch:
        raise ValueError('Use a checkpoint strictly earlier than the capture')
    state = {k.split('a2c_network.net.',1)[1]:v for k,v in checkpoint['model'].items()
             if 'a2c_network.net.' in k}
    train,val,test = split_bodies(data['body_id'],args.seed)
    raw = np.concatenate((data['interaction'],data['metadata']),1).astype(np.float32)
    xm,xs = raw[train].mean(0),raw[train].std(0).clip(1e-4)
    ym,ys = data['target'][train].mean(0),data['target'][train].std(0).clip(1e-4)
    x = torch.from_numpy((raw-xm)/xs)
    y = torch.from_numpy((data['target']-ym)/ys)
    meta = x[:,-data['metadata'].shape[1]:]
    bi = torch.from_numpy(inverse)
    mi = torch.from_numpy(data['module_id'].astype(np.int64))
    split_counts = {k:len(np.unique(inverse[idx])) for k,idx in zip(('train','validation','test'),(train,val,test))}
    errors = {}
    report = {'protocol':'learned_static_response_v1','seed':args.seed,'epochs':args.epochs,
              'capture':str(args.capture),'checkpoint':str(args.checkpoint),
              'checkpoint_epoch':checkpoint['epoch'],'capture_epoch':capture_epoch,
              'split_bodies':split_counts,'arms':{}}
    # Every update uses all sampled rows from a group of bodies, with equal body weight.
    def batches(rows, shuffle=False, rng=None):
        bodies = np.unique(inverse[rows])
        if shuffle:
            bodies = rng.permutation(bodies)
        for start in range(0,len(bodies),32):
            yield rows[np.isin(inverse[rows],bodies[start:start+32])]

    def body_error(pred,rows):
        e = (pred-y[rows]).square().mean(-1)
        return torch.stack([e[bi[rows] == b].mean() for b in bi[rows].unique()])

    for arm in ('metadata','local_learned','fresh','initialized'):
        started = time.perf_counter()
        torch.manual_seed(args.seed)
        ResponseModel.decoder_seed = args.seed
        model = ResponseModel(architecture,x.shape[1],meta.shape[1],arm,state)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
        rng = np.random.default_rng(args.seed)
        best,best_epoch,best_state = float('inf'),0,None
        for epoch in range(args.epochs):
            model.train()
            for rows in batches(train,True,rng):
                pred = model(x[rows],meta[rows],structures,bi[rows],mi[rows])
                loss = body_error(pred,rows).mean()
                optimizer.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                ve = torch.cat([body_error(model(x[r],meta[r],structures,bi[r],mi[r]),r)
                                for r in batches(val)]).mean().item()
            if ve < best:
                best,best_epoch,best_state = ve,epoch+1,copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            error = torch.cat([body_error(model(x[r],meta[r],structures,bi[r],mi[r]),r)
                               for r in batches(test)]).numpy()
        errors[arm] = error
        report['arms'][arm] = {'test_body_nmse':float(error.mean()),'validation_body_nmse':best,
                              'best_epoch':best_epoch,'seconds':time.perf_counter()-started,
                              'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
        print(arm,json.dumps(report['arms'][arm]),flush=True)
    report['comparisons'] = {}
    for baseline in ('metadata','local_learned','fresh'):
        for treatment in ('fresh','initialized'):
            if baseline == treatment:
                continue
            delta = errors[baseline]-errors[treatment]
            rng = np.random.default_rng(args.seed)
            boot = [rng.choice(delta,len(delta)).mean() for _ in range(2000)]
            report['comparisons'][baseline+'_minus_'+treatment] = {
                'gain':float(delta.mean()),'body_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist()}
    report['limitations'] = ['Observational one-step prediction; no co-design improvement tested.',
        'Random body holdout, not held-out structural combinations.',
        'Earlier policy checkpoint may have encountered some held-out morphologies.',
        'Same decoder but encoder parameter counts differ; local learned control addresses extra capacity partially.',
        'One controller run; probe seeds change both split and initialization.']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report,f,indent=2)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('capture','morphologies','checkpoint','config','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=200)
    args=p.parse_args()
    if args.epochs < 1 or args.output.exists():
        p.error('Positive epochs and a new output path are required')
    run(args)
