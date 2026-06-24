"""Codesign training: combined PPO control + unconditional morphology generator (CodesignAgent)."""
import os; os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from transformer_rl.train_utils import run_training
from envs.ant_envs.ant_codesign import AntCodesignEnv
from transformer_rl import MultiMorphLegTransformerBuilder

run_training(
    default_config="ppo_ant_codesign.yaml",
    train_dir="runs/ant_codesign",
    name="ant_codesign_transformer",
    env_class=AntCodesignEnv,
    env_name="ant-codesign-env",
    network=("multimorph_leg_transformer", MultiMorphLegTransformerBuilder),
    model="transformer_masked_a2c_logstd",
)
