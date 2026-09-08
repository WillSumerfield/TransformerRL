from types import SimpleNamespace

import numpy as np
import torch

from transformer_rl.architectures import MultiMorphLimbTransformer
from transformer_rl.response_probe import capture_response_probe
from transformer_rl.tokenize import token_dims
from scripts.probe_static_response import split_bodies, swap_context, evaluate


def test_capture_frozen_alignment_and_resets(tmp_path):
    n, depth, horizon, batch = 8, 4, 8, 2
    net = MultiMorphLimbTransformer(n_limbs=n, d_model=32, n_heads=4,
                                    codesign_tokens=True, max_limb_length=depth)
    dims = token_dims(n, depth)
    obs = torch.zeros(horizon,batch,dims['obs_total'])
    obs[:,:,dims['mask_off']] = 1
    obs[:,:,dims['sub_off']] = 1
    # velocity increases by two each step; reset transition carries a huge jump.
    obs[:,:,13+2*n*depth] = torch.arange(horizon)[:,None]*2
    dones = torch.zeros(horizon,batch,dtype=torch.bool)
    dones[6] = True
    actions = torch.full((horizon,batch,n*depth), 2.)
    counts = torch.zeros(batch,n,dtype=torch.long); counts[:,0] = 1
    eff = torch.full((batch,n,depth),-1,dtype=torch.long); eff[:,0,0] = 0
    cap = torch.zeros(batch,n,dtype=torch.long)
    agent = SimpleNamespace(_net=lambda:net, _cur_counts=counts, _cur_eff=eff,
        _cur_cap=cap, epoch_num=7, _gen_window=0,
        preprocess_actions=lambda a:a.clamp(-1,1),
        experience_buffer=SimpleNamespace(tensor_dict={'obses':obs,'dones':dones,'actions':actions}))
    before = {k:v.clone() for k,v in net.state_dict().items()}
    rng = torch.get_rng_state().clone()
    path = capture_response_probe(agent,tmp_path)
    assert net.training
    assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,before[k]) for k,v in net.state_dict().items())
    assert all(p.grad is None for p in net.parameters())
    with np.load(path) as data:
        assert len(data['target']) == 3  # late t=3,4,6; t=5 crosses reset; duplicate body removed
        np.testing.assert_allclose(data['target'][:,0],2)
        np.testing.assert_allclose(data['interaction'][:,25],1)
        assert data['post'].shape == data['pre'].shape == (3,32)


def test_body_split_and_swap():
    ids = np.repeat([f'b{i}' for i in range(20)],3)
    splits = split_bodies(ids,42)
    sets = [set(ids[idx]) for idx in splits]
    assert not sets[0]&sets[1] and not sets[1]&sets[2] and not sets[0]&sets[2]
    data = {'body_id':ids,'module_id':np.zeros(60,int),'match_group':np.zeros(60,int)}
    rows = splits[-1]
    donors, valid = swap_context(data,rows,42)
    assert valid.all()
    assert (ids[donors] != ids[rows]).all()
    assert np.isin(donors,rows).all()


def test_offline_pipeline_smoke():
    rng = np.random.default_rng(1)
    data = {'body_id':np.repeat([f'b{i}' for i in range(20)],3),
            'module_id':np.zeros(60,int),'match_group':np.zeros(60,int)}
    for key,width in [('interaction',32),('metadata',20),('pre',32),('post',32),('target',7)]:
        data[key] = rng.normal(size=(60,width)).astype(np.float32)
    report = evaluate(data,epochs=2)
    assert len({v['parameters'] for v in report['models'].values()}) == 1
    assert np.isfinite(report['models']['post']['test_body_mean_normalized_mse'])
    assert report['models']['metadata']['inputs'] == ['interaction', 'metadata']
    assert report['models']['pre']['inputs'] == ['interaction', 'metadata', 'pre']
    assert report['models']['post']['inputs'] == ['interaction', 'metadata', 'post']
    assert report['protocol'] == 'metadata_plus_context_v1'
