"""PPO training on AntMultiMorphEnv: one EnvironmentGroup per stable morphology."""
import os; os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT.parent / "vlearn-main" / "train"))

from transformer_rl.train_utils import run_training
from envs.ant_envs.ant_multimorph import AntMultiMorphEnv, _stable_morphologies
from transformer_rl import MultiMorphLegTransformerBuilder

run_training(
    default_config="ppo_ant_full.yaml",
    train_dir="runs/ant_full",
    env_class=AntMultiMorphEnv,
    env_name="ant-multimorph-env",
    network=("multimorph_leg_transformer", MultiMorphLegTransformerBuilder),
    morphology_set=_stable_morphologies(),
)
