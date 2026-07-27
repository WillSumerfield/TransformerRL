#!/usr/bin/env python3
"""Train the CoDesign controller on uniform grammar-policy bodies."""
import os
import sys
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from envs.ant_envs.ant_codesign import AntCodesignEnv
from transformer_rl import MultiMorphLimbTransformerBuilder
from transformer_rl.train_utils import run_training


run_training(
    default_config="ppo_ant_uniform_action.yaml",
    train_dir="runs/benchmarks/uniform_action",
    name="uniform_action_transformer",
    env_class=AntCodesignEnv,
    env_name="ant-uniform-action-env",
    network=("multimorph_limb_transformer", MultiMorphLimbTransformerBuilder),
    model="transformer_masked_a2c_logstd",
)
