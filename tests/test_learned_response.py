import hashlib

import numpy as np
import pytest
import torch

from scripts.train_response_representation import ResponseModel, load_bodies


def test_body_join_requires_exact_full_morphology(tmp_path):
    counts = np.ones((1,8),np.int64)
    eff = np.zeros((1,8,4),np.int64)
    cap = np.zeros((1,8),np.int64)
    key = hashlib.sha256(np.concatenate([x.flatten() for x in (counts,eff,cap)]).tobytes()).hexdigest()
    path = tmp_path/'bodies.npz'
    np.savez(path,counts=counts,eff_sub=eff,cap_sub=cap)
    structures, inverse = load_bodies({'body_id':np.array([key,key])},path)
    assert structures[0].shape == (1,8)
    assert inverse.tolist() == [0,0]
    with pytest.raises(ValueError,match='missing'):
        load_bodies({'body_id':np.array(['incorrect'])},path)


def test_response_loss_trains_static_encoder_without_dynamic_inputs():
    torch.set_num_threads(2)
    ResponseModel.decoder_seed = 42
    model = ResponseModel(dict(d_model=32,n_heads=4,n_layers=1,ffn=64,
        codesign_tokens=True,max_limb_length=4,use_rope=True),52,20,'fresh',{})
    counts = torch.ones(2,8,dtype=torch.long)
    eff = torch.zeros(2,8,4,dtype=torch.long)
    cap = torch.zeros(2,8,dtype=torch.long)
    x = torch.randn(4,52)
    out = model(x,x[:,-20:],[counts,eff,cap],torch.tensor([0,0,1,1]),torch.tensor([0,1,0,1]))
    out.square().mean().backward()
    assert model.body.encoder.layers[0].qkv_proj.weight.grad.abs().sum() > 0
    assert model.body.embed_module.weight.grad.abs().sum() > 0
    assert model.body.joint_head.weight.grad is None
